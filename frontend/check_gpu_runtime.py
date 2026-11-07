"""Validate the restored CUDA kernels and strict PTv3 load before timing it."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import torch


def check():
    import torch_scatter
    import spconv.pytorch as spconv
    from flash_attn import flash_attn_varlen_qkvpacked_func

    torch.manual_seed(0)
    result = dict(
        gpu=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0),
        torch=torch.__version__, cuda=torch.version.cuda,
        packages={p: importlib.metadata.version(p) for p in
                  ['spconv-cu126', 'cumm-cu126', 'torch-scatter', 'flash-attn']},
        checks={},
    )
    # A GPU reduction must agree with a separate CPU accumulation.
    values = torch.randn(1000, 8)
    index = torch.randint(0, 37, (1000,))
    expected = torch.zeros(37, 8).index_add_(0, index, values)
    actual = torch_scatter.scatter_sum(values.cuda(), index.cuda(), dim=0).cpu()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    result['checks']['scatter_max_abs'] = float((actual - expected).abs().max())

    # Sparse convolution is compared to an FP32 dense convolution at the same sites.
    indices = torch.cartesian_prod(torch.tensor([0]), torch.arange(7),
                                  torch.arange(7), torch.arange(7))
    indices = indices[torch.randperm(len(indices))[:240]].int().cuda()
    features = torch.randn(len(indices), 16, device='cuda', dtype=torch.float16)
    sparse = spconv.SparseConvTensor(features, indices, [7, 7, 7], 1)
    layer = spconv.SubMConv3d(16, 32, 3, padding=1, bias=False).cuda().half()
    with torch.no_grad():
        actual = layer(sparse)
        weight = layer.weight.float().permute(0, 4, 1, 2, 3).contiguous()
        dense = torch.nn.functional.conv3d(sparse.dense().float(), weight, padding=1)
        b, z, y, x = actual.indices.long().unbind(1)
        expected = dense[b, :, z, y, x]
        torch.testing.assert_close(actual.features.float(), expected, atol=.005, rtol=.01)
    result['checks']['spconv_max_abs'] = float((actual.features.float() - expected).abs().max())

    # Test ragged attention and the head dimension used by every frozen PTv3 stage.
    errors = []
    for heads in [2, 32]:
        lengths = [31, 127, 1024]
        offsets = torch.tensor([0, 31, 158, 1182], dtype=torch.int32, device='cuda')
        qkv = torch.randn(sum(lengths), 3, heads, 16, device='cuda', dtype=torch.float16)
        with torch.no_grad():
            actual = flash_attn_varlen_qkvpacked_func(qkv, offsets, max(lengths), 0., causal=False)
            expected = []
            for part in qkv.split(lengths):
                q, k, v = [a.float().transpose(0, 1) for a in part.unbind(1)]
                expected.append(((q @ k.transpose(-1, -2) / 4).softmax(-1) @ v).transpose(0, 1))
            expected = torch.cat(expected)
            torch.testing.assert_close(actual.float(), expected, atol=.005, rtol=.01)
        errors.append(float((actual.float() - expected).abs().max()))
    result['checks']['flash_max_abs'] = errors

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from ptv3_loader_verified import build_ptv3, CKPT_PATH, CONFIG_PATH
    model, config = build_ptv3(half=True, shuffle_orders=False)
    assert config.model.num_classes == 16 and config.model.backbone.enable_flash
    result['model'] = dict(
        strict_load=True, state_tensors=len(model.state_dict()),
        parameters=sum(p.numel() for p in model.parameters()), classes=16,
        checkpoint_sha256=hashlib.sha256(Path(CKPT_PATH).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(Path(CONFIG_PATH).read_bytes()).hexdigest(),
        source_root=os.environ['PTV3_POINTCEPT'],
    )
    assert result['model']['state_tensors'] == 488
    assert result['model']['checkpoint_sha256'] == '35f5d3aec63ae1c383f4a4146e157aa054860cb12b178d899571287e6f97e5f1'
    torch.cuda.synchronize()
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = check()
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
