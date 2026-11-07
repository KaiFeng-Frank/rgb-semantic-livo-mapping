"""Bounded, causal semantic evidence. Raw observations are the only memory writes.

No ROS or Pointcept dependencies. Torch tensors stay on the caller's device.
This module does not estimate motion, object identity, or calibrated uncertainty.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MemoryConfig:
    feature_dim: int = 64
    classes: int = 16
    capacity: int = 16384
    cell_size: float = 0.5
    records_per_cell: int = 2
    neighbors: int = 8
    max_age_s: float = 1.0
    radius: float = 0.5
    min_cosine: float = 0.5
    gate_threshold: float = 0.5
    chunk_size: int = 8192
    hidden_dim: int = 32
    max_points: int = 200000
    ray_width: int = 2048
    ray_height: int = 512
    ray_tolerance: float = 0.003
    clearance: float = 0.3

    def __post_init__(self):
        for name in ('feature_dim', 'classes', 'capacity', 'records_per_cell',
                     'neighbors', 'chunk_size', 'hidden_dim', 'max_points',
                     'ray_width', 'ray_height'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.classes < 2 or self.records_per_cell > self.capacity:
            raise ValueError('Invalid classes or per-cell capacity')
        for name in ('cell_size', 'max_age_s', 'radius', 'ray_tolerance', 'clearance'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.radius > self.cell_size or not -1 <= self.min_cosine <= 1:
            raise ValueError('Radius must fit the 27-cell search; invalid cosine')
        if not math.isfinite(self.gate_threshold) or not 0 <= self.gate_threshold <= 1:
            raise ValueError('gate_threshold must lie in [0, 1]')


def _keys(xyz, size, check=True):
    """Collision-free keys within +/- 2**20 cells, checked at the boundary."""
    grid = torch.floor(xyz / size).long()
    if check and bool(((grid < -(1 << 20) + 1) | (grid >= (1 << 20) - 1)).any()):
        raise ValueError('Coordinates outside memory key range')
    grid = grid + (1 << 20)
    return (grid[..., 0] * (1 << 21) + grid[..., 1]) * (1 << 21) + grid[..., 2]


def _ray_bins(xyz, cfg):
    ranges = torch.linalg.vector_norm(xyz, dim=1)
    unit = xyz / ranges.clamp_min(1e-6)[:, None]
    az = torch.atan2(unit[:, 1], unit[:, 0])
    el = torch.asin(unit[:, 2].clamp(-1, 1))
    u = torch.floor((az + math.pi) * cfg.ray_width / (2 * math.pi)).long()
    v = torch.floor((el + math.pi / 2) * cfg.ray_height / math.pi).long()
    return v.clamp(0, cfg.ray_height - 1) * cfg.ray_width + u.remainder(cfg.ray_width), ranges, unit


@torch.no_grad()
def visibility(memory_sensor, scan_sensor, cfg):
    """0 unknown, 1 compatible surface, 2 occluded, 3 observed free space.

    Only measured rays with a close angular match can invalidate evidence.
    Missing returns and no view NEVER mean free space. This is a conservative
    sampled-ray check, not an occupancy reconstruction or object tracker.
    """
    out = torch.zeros(len(memory_sensor), dtype=torch.uint8, device=memory_sensor.device)
    if not len(memory_sensor) or not len(scan_sensor):
        return out
    bins, depth, unit = _ray_bins(scan_sensor, cfg)
    order = depth.argsort(stable=True)
    order = order[bins[order].argsort(stable=True)]
    sb = bins[order]
    first = torch.ones(len(order), dtype=torch.bool, device=order.device)
    first[1:] = sb[1:] != sb[:-1]
    representatives = order[first]
    sb = bins[representatives]
    mb, md, mu = _ray_bins(memory_sensor, cfg)
    j = torch.searchsorted(sb, mb).clamp_max(len(sb) - 1)
    idx = representatives[j]
    matched = (sb[j] == mb) & (md > 1e-3) & (depth[idx] > 1e-3)
    matched &= (mu * unit[idx]).sum(1) >= math.cos(cfg.ray_tolerance)
    delta = depth[idx] - md
    out[matched] = 1
    out[matched & (delta < -cfg.clearance)] = 2
    out[matched & (delta > cfg.clearance)] = 3
    return out


class EvidenceMemory:
    """Finite observation ledger, sorted by cell; no fused predictions accepted."""
    fields = ('xyz', 'features', 'probabilities', 'stamp', 'frame', 'key')

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, torch.empty(0, device=device).device
        self.offsets = torch.cartesian_prod(*(torch.arange(-1, 2, device=self.device) for _ in range(3)))
        self.reset()

    def reset(self):
        c, d = self.cfg, self.device
        self.xyz = torch.empty((0, 3), device=d)
        self.features = torch.empty((0, c.feature_dim), device=d, dtype=torch.float16)
        self.probabilities = torch.empty((0, c.classes), device=d, dtype=torch.float16)
        self.stamp = torch.empty(0, device=d, dtype=torch.float64)
        self.frame = torch.empty(0, device=d, dtype=torch.long)
        self.key = torch.empty(0, device=d, dtype=torch.long)
        self.readable = torch.empty(0, device=d, dtype=torch.bool)
        self.last_stamp, self.last_frame, self.epoch = None, None, None

    def _select(self, idx):
        for field in self.fields:
            setattr(self, field, getattr(self, field)[idx])

    @torch.no_grad()
    def prepare(self, stamp, transform, scan_sensor):
        old = len(self.xyz)
        self._select(stamp - self.stamp <= self.cfg.max_age_s)
        expired = old - len(self.xyz)
        sensor = (self.xyz - transform[:3, 3]) @ transform[:3, :3]
        vis = visibility(sensor, scan_sensor, self.cfg)
        free = vis == 3
        revoked = int(free.sum())
        self._select(~free)
        self.readable = vis[~free] != 2
        return dict(expired=expired, revoked_free_space=revoked,
                    occluded=int((vis == 2).sum()))

    @torch.no_grad()
    def retrieve(self, xyz, features, stamp):
        c, n = self.cfg, len(xyz)
        shape = (n, c.neighbors)
        if not len(self.xyz):
            return dict(features=features.new_zeros((*shape, c.feature_dim)),
                        probabilities=features.new_zeros((*shape, c.classes)),
                        delta=xyz.new_zeros((*shape, 3)), age=xyz.new_zeros(shape),
                        valid=torch.zeros(shape, dtype=torch.bool, device=xyz.device),
                        frame=torch.full(shape, -1, dtype=torch.long, device=xyz.device))
        # Fixed 27 cells x records_per_cell, never an N x memory-size distance matrix.
        grid = torch.floor(xyz / c.cell_size).long()
        neighbor_xyz = (grid[:, None, :] + self.offsets[None]) * c.cell_size
        keys = _keys(neighbor_xyz, c.cell_size, check=False)
        start = torch.searchsorted(self.key, keys)
        idx = start[..., None] + torch.arange(c.records_per_cell, device=xyz.device)
        expected = keys[..., None].expand_as(idx).reshape(n, -1)
        valid = idx.reshape(n, -1) < len(self.key)
        idx = idx.reshape(n, -1).clamp_max(len(self.key) - 1)
        valid &= self.key[idx] == expected
        valid &= self.readable[idx] & (self.stamp[idx] < stamp)
        delta = self.xyz[idx] - xyz[:, None]
        distance2 = delta.square().sum(-1)
        valid &= distance2 <= c.radius ** 2
        # Cheap spatial shortlist precedes feature gathering.
        k = min(c.neighbors, idx.shape[1])
        _, sel = distance2.masked_fill(~valid, float('inf')).topk(k, largest=False)
        idx, valid = idx.gather(1, sel), valid.gather(1, sel)
        hist = self.features[idx].to(features.dtype)
        cosine = F.cosine_similarity(features[:, None].float(), hist.float(), dim=-1)
        valid &= cosine >= c.min_cosine
        return dict(features=hist, probabilities=self.probabilities[idx].float(),
                    delta=self.xyz[idx] - xyz[:, None],
                    age=(stamp - self.stamp[idx]).float(), valid=valid, frame=self.frame[idx])

    @torch.no_grad()
    def write_raw(self, xyz, features, probabilities, stamp, frame):
        c = self.cfg
        if not len(xyz):
            return
        keys = _keys(xyz, c.cell_size, check=False)
        order = keys.argsort(stable=True)
        # One representative per cell per frame: dense same-frame points cannot
        # become independent repeated votes. Fine detail remains in current path.
        first = torch.ones(len(order), dtype=torch.bool, device=xyz.device)
        first[1:] = keys[order[1:]] != keys[order[:-1]]
        idx = order[first]
        # Uniform spatial coverage when the single frame exceeds the entire cap.
        if len(idx) > c.capacity:
            idx = idx[torch.linspace(0, len(idx) - 1, c.capacity, device=xyz.device).long()]
        new = dict(xyz=xyz[idx].detach().float(), features=features[idx].detach().half(),
                   probabilities=probabilities[idx].detach().half(),
                   stamp=torch.full((len(idx),), stamp, dtype=torch.float64, device=xyz.device),
                   frame=torch.full((len(idx),), frame, dtype=torch.long, device=xyz.device), key=keys[idx])
        for field in self.fields:
            setattr(self, field, torch.cat([new[field], getattr(self, field)]))
        # Newest records first within each cell, bounded without permanent lock-in.
        order = self.stamp.argsort(descending=True, stable=True)
        order = order[self.key[order].argsort(stable=True)]
        sk = self.key[order]
        first = torch.ones(len(order), dtype=torch.bool, device=xyz.device)
        first[1:] = sk[1:] != sk[:-1]
        positions = torch.arange(len(order), device=xyz.device)
        begin = torch.where(first, positions, 0).cummax(0).values
        order = order[(positions - begin) < c.records_per_cell]
        if len(order) > c.capacity:
            # Retain new observations before older ones. Ties already spatially ordered.
            order = order[self.stamp[order].argsort(descending=True, stable=True)[:c.capacity]]
        order = order[self.key[order].argsort(stable=True)]
        self._select(order)
        self.readable = torch.ones(len(order), dtype=torch.bool, device=xyz.device)


class HistoryFusion(nn.Module):
    """Local feature attention plus a learned history-use gate; not uncertainty."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, h, k = cfg.feature_dim, cfg.hidden_dim, cfg.classes
        self.query = nn.Linear(d, h)
        self.key = nn.Linear(d, h)
        self.value = nn.Linear(d, h)
        self.context = nn.Sequential(nn.Linear(5, h), nn.ReLU(), nn.Linear(h, 1))
        self.delta = nn.Linear(h, k, bias=False)
        self.gate = nn.Sequential(nn.Linear(d + h + 3, h), nn.ReLU(), nn.Linear(h, 1))
        # Conservative initialization. Deployment still requires an explicit trained checkpoint.
        nn.init.constant_(self.gate[-1].bias, -2.)

    def forward(self, features, base_logits, candidates):
        present = candidates['valid'].any(1)
        active = present.nonzero().flatten()
        base = base_logits.float()
        if not len(active):
            zero = self.delta.weight.sum() * 0
            return dict(logits=base + zero, candidate_logits=base + zero,
                        base_logits=base, gate=base.new_zeros(len(base)) + zero, has_history=present)
        # Unsupported points use the current path without running the MLPs or
        # attention. This also avoids allocating dense N*K hidden activations.
        candidates = {k: v[active] for k, v in candidates.items()}
        f, hist = features[active].float(), candidates['features'].float()
        valid = candidates['valid']
        rel = torch.cat([candidates['delta'] / self.cfg.radius,
                         candidates['age'][..., None] / self.cfg.max_age_s,
                         F.cosine_similarity(f[:, None], hist, dim=-1)[..., None]], dim=-1)
        scores = (self.query(f)[:, None] * self.key(hist)).sum(-1) / math.sqrt(self.cfg.hidden_dim)
        scores = scores + self.context(rel).squeeze(-1)
        # Explicit zero weight for empty rows; no softmax(-inf,...,-inf) NaNs.
        weights = torch.softmax(scores.masked_fill(~valid, -1e4), dim=1) * valid
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        h = (weights[..., None] * self.value(hist)).sum(1)
        p = base[active].softmax(-1)
        hp = (weights[..., None] * candidates['probabilities'].float()).sum(1)
        entropy = -(p * p.clamp_min(1e-8).log()).sum(1)
        disagreement = (p - hp).abs().sum(1)
        age = (weights * candidates['age']).sum(1)
        gate = torch.sigmoid(self.gate(torch.cat([f, h, entropy[:, None],
                                                disagreement[:, None], age[:, None]], 1))).squeeze(1)
        residual = self.delta(h)
        # Soft selection during training; explicit baseline fallback at inference
        # when predicted utility is insufficient. This is not a correctness guarantee.
        applied = gate if self.training else torch.where(gate >= self.cfg.gate_threshold, gate, 0.)
        return dict(logits=base.index_add(0, active, applied[:, None] * residual),
                    candidate_logits=base.index_add(0, active, residual),
                    base_logits=base, gate=base.new_zeros(len(base)).index_copy(0, active, gate),
                    has_history=present)


class CausalSemanticMemory(nn.Module):
    def __init__(self, cfg=MemoryConfig(), device='cpu'):
        super().__init__()
        self.cfg = cfg
        self.fusion = HistoryFusion(cfg).to(device)
        self.memory = EvidenceMemory(cfg, device)
        self.last_status = {}

    def reset(self):
        self.memory.reset()

    def _validate(self, xyz, features, logits, stamp, frame, transform, epoch):
        c, m = self.cfg, self.memory
        if type(frame) is not int or frame < 0 or type(epoch) is not int or epoch < 0:
            raise ValueError('Nonnegative integer frame/epoch required')
        if not math.isfinite(stamp):
            raise ValueError('Finite sensor timestamp required')
        n = len(xyz)
        if n > c.max_points or xyz.shape != (n, 3) or features.shape != (n, c.feature_dim) or logits.shape != (n, c.classes):
            raise ValueError('Invalid raw observation shapes/capacity')
        if any(x.device != m.device for x in (xyz, features, logits, transform)):
            raise ValueError('All tensors must be on the memory device')
        if transform.shape != (4, 4) or not all(bool(torch.isfinite(x).all()) for x in (xyz, features, logits, transform)):
            raise ValueError('Invalid/nonfinite observation or transform')
        # Validate the actual computation/storage dtypes before mutating history.
        if not all(bool(torch.isfinite(x.to(dtype)).all()) for x, dtype in (
                (xyz, torch.float32), (features, torch.float16),
                (logits, torch.float32), (transform, torch.float32))):
            raise ValueError('Observation overflows memory storage or prediction dtype')
        eye = torch.eye(3, device=xyz.device)
        rot = transform[:3, :3].float()
        if not torch.allclose(rot.T @ rot, eye, atol=1e-4, rtol=0) or not torch.allclose(torch.det(rot), rot.new_tensor(1.), atol=1e-4, rtol=0):
            raise ValueError('Transform rotation must be SO(3)')
        if not torch.allclose(transform[3].float(), rot.new_tensor([0, 0, 0, 1]), atol=1e-6, rtol=0):
            raise ValueError('Invalid homogeneous transform')
        if m.epoch is not None and epoch < m.epoch:
            raise ValueError('Old coordinate epoch; reset explicitly for a new stream')
        if m.last_stamp is not None and (stamp <= m.last_stamp or frame <= m.last_frame):
            raise ValueError('Duplicate/backward frame or sensor time')

    def forward(self, xyz_sensor, features, base_logits, *, stamp, frame, transform, epoch=0, shadow=False):
        """Transform maps this scan's sensor frame to a consistent memory frame.

        Caller supplies deskewed geometry at that timestamp. Frames are strictly
        chronological. An increased epoch invalidates all old coordinate records.
        GT is never an input. All memory records are detached raw observations.
        """
        self._validate(xyz_sensor, features, base_logits, stamp, frame, transform, epoch)
        xyz = xyz_sensor.float() @ transform[:3, :3].float().T + transform[:3, 3].float()
        _keys(xyz, self.cfg.cell_size)  # reject invalid keys before any state changes
        m = self.memory
        reset = m.epoch is not None and m.epoch != epoch
        if reset:
            m.reset()
        status = m.prepare(stamp, transform.float(), xyz_sensor.float())
        outputs = []
        for start in range(0, len(xyz), self.cfg.chunk_size):
            stop = start + self.cfg.chunk_size
            candidates = m.retrieve(xyz[start:stop], features[start:stop], stamp)
            outputs.append(self.fusion(features[start:stop], base_logits[start:stop], candidates))
        if outputs:
            result = {k: torch.cat([o[k] for o in outputs]) for k in outputs[0]}
        else:
            result = dict(logits=base_logits.float(), candidate_logits=base_logits.float(),
                          base_logits=base_logits.float(), gate=base_logits.new_empty(0),
                          has_history=torch.empty(0, dtype=torch.bool, device=xyz.device))
        # Crucial: do not feed result['logits'] or attention-modified features back.
        m.write_raw(xyz, features, base_logits.detach().float().softmax(-1), stamp, frame)
        m.last_stamp, m.last_frame, m.epoch = stamp, frame, epoch
        if shadow:
            result['logits'] = base_logits.float()
        self.last_status = dict(status, records=len(m.xyz), epoch=epoch, epoch_reset=reset,
                                frame=frame, stamp=stamp, shadow=shadow,
                                history_points=int(result['has_history'].sum()))
        return result


def training_loss(output, labels, margin=0.02):
    """Train the selector using detached loss improvement on TRAIN labels only.

    Both prediction branches share one backbone pass. Corrupt/absent history
    examples must be supplied by the caller. No temporal consistency penalty
    forces a correct current prediction to repeat an old error.
    """
    valid = labels >= 0
    if not bool(valid.any()):
        raise ValueError('Training frame has no supervised labels')
    y = labels[valid].long()
    fused = F.cross_entropy(output['logits'][valid], y)
    candidate = F.cross_entropy(output['candidate_logits'][valid], y, reduction='none')
    base = F.cross_entropy(output['base_logits'][valid], y, reduction='none')
    has = output['has_history'][valid]
    target = (base.detach() - candidate.detach() > margin).float()
    gate_loss = F.binary_cross_entropy(output['gate'][valid][has].clamp(1e-6, 1 - 1e-6), target[has]) if bool(has.any()) else fused * 0
    # Auxiliary candidate training prevents the closed gate from starving the residual.
    auxiliary = candidate[has].mean() if bool(has.any()) else fused * 0
    return fused + 0.2 * gate_loss + 0.2 * auxiliary, dict(
        segmentation=float(fused.detach()), gate=float(gate_loss.detach()),
        auxiliary=float(auxiliary.detach()), supervised=int(valid.sum()),
        history_supervised=int(has.sum()))


def checkpoint(cfg, fusion, *, provenance, trained=False):
    return dict(schema=1, config=asdict(cfg), fusion=fusion.state_dict(),
                provenance=provenance, trained=bool(trained))
