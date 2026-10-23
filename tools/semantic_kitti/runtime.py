"""PTv3 v1.5.1 with explicit native-PyTorch attention/pooling compatibility.

This runtime is a new hardware port, not the original 4090/FlashAttention runtime.
Source state dictionaries remain strict; no upstream source file is modified.
"""
import importlib
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

POINTCEPT_COMMIT = '72a799353c568c29a3365c02a15129ebe7fe637a'


def segment_csr(src, indptr, reduce='sum'):
    if src.ndim != 2 or indptr.ndim != 1 or reduce not in ('sum', 'mean', 'min', 'max'):
        raise ValueError('Only PTv3 nonbatched CSR reductions are supported')
    return torch.segment_reduce(src, reduce=reduce, offsets=indptr)


def attention(qkv, cu_seqlens, max_seqlen, dropout_p=0., softmax_scale=None):
    """Same packed per-segment attention equation, implemented by PyTorch SDPA.

PTv3 pads regular segments to K, with a shorter segment only for a small cloud.
Consecutive equal-length segments are batched; no cross-segment attention occurs.
"""
    if qkv.ndim != 4 or qkv.shape[1] != 3:
        raise ValueError('Expected packed [tokens,3,heads,channels]')
    boundaries = cu_seqlens.detach().cpu().tolist()
    if boundaries[0] != 0 or boundaries[-1] != len(qkv):
        raise ValueError('Packed-attention boundaries do not cover the tokens')
    lengths = [b - a for a, b in zip(boundaries, boundaries[1:])]
    if not lengths or min(lengths) <= 0 or max(lengths) > max_seqlen:
        raise ValueError('Invalid packed-attention segment lengths')
    blocks = []
    i = 0
    while i < len(lengths):
        j = i + 1
        while j < len(lengths) and lengths[j] == lengths[i]:
            j += 1
        value = qkv[boundaries[i]:boundaries[j]].reshape(j-i, lengths[i], 3, *qkv.shape[2:])
        q, k, v = value.permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=softmax_scale)
        blocks.append(y.transpose(1, 2).reshape(-1, *qkv.shape[2:]))
        i = j
    return torch.cat(blocks, 0)


def _package(name, path):
    if name in sys.modules:
        old = sys.modules[name]
        if str(path) not in getattr(old, '__path__', []):
            raise RuntimeError('Different Pointcept runtime already imported: ' + name)
        return old
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def load_backbone_class(pointcept):
    pointcept = Path(pointcept).resolve()
    rev = subprocess.check_output(['git', '-C', str(pointcept), 'rev-parse', 'HEAD'], text=True).strip()
    if rev != POINTCEPT_COMMIT:
        raise ValueError('Use Pointcept v1.5.1 at ' + POINTCEPT_COMMIT)
    if subprocess.check_output(['git', '-C', str(pointcept), 'diff', '--name-only', 'HEAD'], text=True).strip():
        raise ValueError('Pointcept tracked source must be unmodified')
    # Import only the actual PTv3 dependency tree. Other Pointcept architectures
    # have unrelated CUDA extensions; none are replaced with no-op implementations.
    _package('pointcept', pointcept / 'pointcept')
    _package('pointcept.models', pointcept / 'pointcept/models')
    prompt = _package('pointcept.models.point_prompt_training', pointcept / 'pointcept/models/point_prompt_training')
    prompt.PDNorm = importlib.import_module('pointcept.models.point_prompt_training.prompt_driven_normalization').PDNorm
    scatter = types.ModuleType('torch_scatter')
    scatter.__file__ = __file__
    scatter.segment_csr = segment_csr
    if 'torch_scatter' in sys.modules and getattr(sys.modules['torch_scatter'], 'segment_csr', None) is not segment_csr:
        raise RuntimeError('Run this isolated native-pooling runtime in a fresh process')
    sys.modules['torch_scatter'] = scatter
    module = importlib.import_module('pointcept.models.point_transformer_v3.point_transformer_v3m1_base')
    module.flash_attn = types.SimpleNamespace(flash_attn_varlen_qkvpacked_func=attention)
    return module.PointTransformerV3


def build(pointcept, classes=19, training=False):
    if classes not in (16, 19):
        raise ValueError('Expected 16-class source or 19-class target model')
    backbone_type = load_backbone_class(pointcept)
    model = nn.Module()
    model.backbone = backbone_type(
        in_channels=4, order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2), enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512), enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024,) * 5, dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256), dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024,) * 4, mlp_ratio=4, qkv_bias=True,
        attn_drop=0., proj_drop=0., drop_path=.3,
        shuffle_orders=training, pre_norm=True, enable_rpe=False,
        enable_flash=True, upcast_attention=False, upcast_softmax=False,
        cls_mode=False, pdnorm_bn=False, pdnorm_ln=False,
        pdnorm_decouple=True, pdnorm_adaptive=False, pdnorm_affine=True,
        pdnorm_conditions=('nuScenes', 'SemanticKITTI', 'Waymo'))
    model.seg_head = nn.Linear(64, classes)
    model.train(training)
    return model


def load_source(model, checkpoint, new_head=False):
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    source = source.get('state_dict', source)
    source = {(k[7:] if k.startswith('module.') else k): v for k, v in source.items()}
    if any(k.startswith('frozen_') for k in source):
        raise ValueError('Use the extracted student, not a full distillation checkpoint')
    if new_head:
        if source['seg_head.weight'].shape[0] != 16 or model.seg_head.out_features != 19:
            raise ValueError('Expected explicit 16 -> 19 head replacement')
        backbone = {k[len('backbone.'):]: v for k, v in source.items() if k.startswith('backbone.')}
        if set(source) != {'backbone.' + k for k in backbone} | {'seg_head.weight', 'seg_head.bias'}:
            raise ValueError('Unexpected source checkpoint keys')
        model.backbone.load_state_dict(backbone, strict=True)
        return dict(backbone_tensors=len(backbone), new_random_head=True, source_classes=16, target_classes=19)
    model.load_state_dict(source, strict=True)
    return dict(tensors=len(source), new_random_head=False)


def prepare(points, device='cuda', half=False, grid=.05, intensity_scale=.2):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected a nonempty finite Nx4 LiDAR scan')
    coord = np.ascontiguousarray(points[:, :3])
    g = np.floor(coord / grid).astype(np.int64)
    g -= g.min(0)
    keys = (g[:, 0] * (g[:, 1].max() + 1) + g[:, 1]) * (g[:, 2].max() + 1) + g[:, 2]
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    dtype = torch.float16 if half else torch.float32
    c = torch.as_tensor(coord[first], device=device, dtype=dtype)
    s = torch.as_tensor(points[first, 3:4] * intensity_scale, device=device, dtype=dtype)
    d = dict(coord=c, grid_coord=torch.as_tensor(g[first], device=device),
             feat=torch.cat([c, s], -1), offset=torch.tensor([len(c)], device=device))
    return d, first, inverse


def logits(model, batch):
    from pointcept.models.utils.structure import Point
    return model.seg_head(model.backbone(Point(dict(batch))).feat)
