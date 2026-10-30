"""nuScenes adapter of the released GLFS lookup-weight fusion (not a new head).

Reference: joaomcm/Semantic-3D-Mapping-Uncertainty-Calibration, ed5c59f,
StreamlinedWeightEstimator + CombinedTrainer.forward. See the experiment protocol
for dataset adaptations. All observations are supplied explicitly; this module
does not associate objects, select observations, or consume labels at inference.
"""
from pathlib import Path
import importlib.util
import numpy as np
import torch
from torch import nn


def verify_sources(root, protocol):
    import hashlib
    project = Path(__file__).resolve().parents[2]
    for base, expected in ((Path(root), protocol['reference_hashes']), (project, protocol['source_hashes'])):
        for relative, sha in expected.items():
            with (base/relative).open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != sha:
                raise ValueError('Pinned experiment source changed: '+relative)


class GLFS(nn.Module):
    def __init__(self, classes=16, depth_edges=(0, 10, 20, 30, 40, 50),
                 angle_edges=(0, 30, 60, 90), temperature=.4):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.full((classes,), float(np.log(temperature))))
        # The released implementation initializes these lookup parameters in f64.
        self.weights = nn.Parameter(torch.ones(classes, len(depth_edges), len(angle_edges), dtype=torch.float64))
        self.register_buffer('depth_edges', torch.tensor(depth_edges, dtype=torch.float32))
        self.register_buffer('angle_edges', torch.tensor(angle_edges, dtype=torch.float32))

    def forward(self, logits, depth, angle, valid):
        if logits.ndim != 3 or valid.shape != logits.shape[:2]:
            raise ValueError('Expected [query, observation, class] and explicit mask')
        if valid.dtype != torch.bool or not bool(valid[:, 0].all()):
            raise ValueError('Observation zero must be the valid independent current prediction')
        if depth.shape != valid.shape or angle.shape != valid.shape:
            raise ValueError('Geometry must describe each source observation')
        if not all(bool(torch.isfinite(x).all()) for x in (logits, depth, angle)):
            raise ValueError('Nonfinite inputs, including padding')
        p = (logits.float() / self.log_temperature.exp()).softmax(-1)
        p = p + .0001
        p = p / p.sum(-1, keepdim=True)
        with torch.no_grad():
            d = torch.bucketize(depth.float().contiguous(), self.depth_edges).clamp(0, len(self.depth_edges)-1)
            a = torch.bucketize(angle.float().clamp(0, 90).contiguous(), self.angle_edges).clamp(0, len(self.angle_edges)-1)
        w = (self.weights[p.argmax(-1), d, a].relu() + .0001) * valid
        w = w.float()
        # Released training defaults L=epsilon=0: normalized arithmetic integration.
        result = (w[..., None] * p).sum(1) / w.sum(1, keepdim=True)
        return result / result.sum(-1, keepdim=True)


def probability_average(logits, valid):
    p = logits.float().softmax(-1)
    return (p * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)


def official_mdece(path, device='cpu'):
    """Use the authors' pinned loss, with only its hard-coded accumulator device fixed.

    nuScenes has 16 real classes numbered 0..15; filter ignore=-1 outside this
    loss, and do not silently exclude class zero (barrier) as ScanNet does.
    """
    spec = importlib.util.spec_from_file_location('glfs_author_dece', Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class MeanDECE(module.DECE):
        def forward(self, prediction, target):
            classes = torch.unique(target)
            if not len(classes):
                return prediction.sum() * 0
            return torch.stack([super(MeanDECE, self).forward(prediction[target == c], target[target == c])
                                for c in classes]).mean()
    return MeanDECE(device, 10, t_a=100, t_b=.01)


def observation_geometry(xyz):
    """Sensor range and acute PCA-normal/ray incidence, no pose or label target.

    Fixed 16-nearest-neighbour covariance. This is an outdoor LiDAR geometry
    adapter, not the authors' RGB-D normal renderer. Sparse/nonplanar surface
    statistics are recorded by the extractor instead of hidden as GT normals.
    """
    from scipy.spatial import cKDTree
    xyz = np.asarray(xyz, np.float64)
    if len(xyz) < 16:
        raise ValueError('Too few points for the fixed normal estimator')
    distance, neighbors = cKDTree(xyz).query(xyz, k=16, workers=2)
    local = xyz[neighbors]
    local -= local.mean(1, keepdims=True)
    values, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', local, local)/16)
    depth = np.linalg.norm(xyz, axis=1)
    if np.any(depth <= 0) or not np.isfinite(values).all():
        raise ValueError('Undefined source incidence')
    cosine = np.abs((vectors[:, :, 0] * xyz).sum(1) / depth).clip(0, 1)
    angle = np.degrees(np.arccos(cosine))
    stats = dict(points=len(xyz), range_over_50=int((depth > 50).sum()),
                 neighbor16_over_2m=int((distance[:, -1] > 2).sum()),
                 normal_degenerate=int((values[:, 1] < 1e-10).sum()))
    return depth.astype(np.float32), angle.astype(np.float32), stats
