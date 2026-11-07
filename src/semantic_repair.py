"""Experimental current-anchored repair from one previous raw observation.

Motion-tolerant retrieval is NOT motion estimation or instance association.
The module has a bounded query budget and never writes its corrected output.
Raw-keyframe pilot weights must not activate the deskewed production worker.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from semantic_memory import CausalSemanticMemory, EvidenceMemory, MemoryConfig, _keys, visibility


@dataclass(frozen=True)
class RepairConfig:
    max_queries: int = 4096
    confidence: float = .9
    static_radius: float = .5
    moving_radius: float = 1.5
    residual_limit: float = 2.
    # Native nuScenes-16 ontology, fixed before evaluation.
    moving_classes: tuple = (1, 2, 3, 4, 5, 6, 8, 9)

    def __post_init__(self):
        if type(self.max_queries) is not int or self.max_queries < 1:
            raise ValueError('Positive query budget required')
        if not 0 < self.confidence <= 1:
            raise ValueError('Invalid confidence threshold')
        if not all(math.isfinite(x) and x > 0 for x in
                   (self.static_radius, self.moving_radius, self.residual_limit)):
            raise ValueError('Invalid radius/residual limit')
        if self.static_radius > self.moving_radius:
            raise ValueError('Static radius exceeds moving radius')


def repair_memory_config():
    return MemoryConfig(cell_size=1.5, radius=1.5, records_per_cell=4,
                        neighbors=4, min_cosine=.7, max_age_s=.75)


class PreviousEvidence(EvidenceMemory):
    """One raw representative per predicted class/cell, from the last frame only.

    A small class in a cell cannot be suppressed merely because a road point
    occurs earlier in the input. Predicted classes select representatives, not
    permissible query classes; a wrong current label can still be corrected.
    """
    def __init__(self, cfg, device, moving_classes):
        super().__init__(cfg, device)
        self.moving_classes = torch.tensor(moving_classes, device=device)

    @torch.no_grad()
    def prepare(self, stamp, transform, scan_sensor):
        old = len(self.xyz)
        self._select(stamp - self.stamp <= self.cfg.max_age_s)
        expired = old - len(self.xyz)
        sensor = (self.xyz - transform[:3, 3]) @ transform[:3, :3]
        vis = visibility(sensor, scan_sensor, self.cfg)
        mobile = torch.isin(self.probabilities.argmax(-1), self.moving_classes)
        # A vacated former position does not invalidate a moving object's raw
        # observation. Static free-space evidence still revokes static records.
        free = (vis == 3) & ~mobile
        self._select(~free)
        self.readable = ((vis != 2) | mobile)[~free]
        return dict(expired=expired, revoked_free_space=int(free.sum()),
                    occluded=int(((vis == 2) & ~mobile).sum()))

    @torch.no_grad()
    def write_raw(self, xyz, features, probabilities, stamp, frame):
        c = self.cfg
        keys = _keys(xyz, c.cell_size, check=False)
        confidence, labels = probabilities.max(-1)
        order = confidence.argsort(descending=True, stable=True)
        order = order[labels[order].argsort(stable=True)]
        order = order[keys[order].argsort(stable=True)]
        first = torch.ones(len(order), dtype=torch.bool, device=xyz.device)
        first[1:] = ((keys[order[1:]] != keys[order[:-1]]) |
                     (labels[order[1:]] != labels[order[:-1]]))
        order = order[first]
        order = order[confidence[order].argsort(descending=True, stable=True)]
        order = order[keys[order].argsort(stable=True)]
        first = torch.ones(len(order), dtype=torch.bool, device=xyz.device)
        first[1:] = keys[order[1:]] != keys[order[:-1]]
        positions = torch.arange(len(order), device=xyz.device)
        begin = torch.where(first, positions, 0).cummax(0).values
        order = order[positions - begin < c.records_per_cell]
        if len(order) > c.capacity:
            order = order[torch.linspace(0, len(order)-1, c.capacity, device=xyz.device).long()]
        # Replace the previous scan, including on empty scans. No fused feedback.
        self.xyz, self.key = xyz[order].detach().float(), keys[order]
        self.features = features[order].detach().half()
        self.probabilities = probabilities[order].detach().half()
        self.stamp = torch.full((len(order),), stamp, dtype=torch.float64, device=xyz.device)
        self.frame = torch.full((len(order),), frame, dtype=torch.long, device=xyz.device)
        self.readable = torch.ones(len(order), dtype=torch.bool, device=xyz.device)


class JointRepair(nn.Module):
    def __init__(self, cfg, repair):
        super().__init__()
        self.cfg, self.repair = cfg, repair
        self.net = nn.Sequential(nn.Linear(3*cfg.feature_dim + 2*cfg.classes + 5, 64),
                                 nn.ReLU(), nn.Linear(64, cfg.classes))
        # Start as the exact trained single-frame anchor, never an arbitrary delta.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, anchor, candidates):
        valid = candidates['valid']
        present = valid.any(1)
        hist = candidates['features'].float()
        cosine = F.cosine_similarity(features[:, None].float(), hist, dim=-1)
        # Fixed similarity weighting: no extra learned confidence-only utility gate.
        weights = torch.softmax((4*cosine).masked_fill(~valid, -1e4), dim=1) * valid
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        h = (hist * weights[..., None]).sum(1)
        hp = (candidates['probabilities'].float() * weights[..., None]).sum(1)
        delta = (candidates['delta'] * weights[..., None]).sum(1) / self.cfg.radius
        age = (candidates['age'] * weights).sum(1, keepdim=True) / self.cfg.max_age_s
        similarity = (cosine * weights).sum(1, keepdim=True)
        p, f = anchor.float().softmax(-1), features.float()
        # Every selected point retains a learnable current-only path. Replacing
        # a missing token by the current token uses the SAME head/parameters as
        # the matched current control. Historical coverage cannot censor training.
        h = torch.where(present[:, None], h, f)
        hp = torch.where(present[:, None], hp, p)
        similarity = torch.where(present[:, None], similarity, torch.ones_like(similarity))
        joint = torch.cat((f, h, h-f, p, hp-p, delta, age, similarity), dim=1)
        correction = self.repair.residual_limit * torch.tanh(self.net(joint))
        return anchor.float() + correction


class CurrentAnchoredRepair(CausalSemanticMemory):
    """Caller provides a frozen, independently trained current-frame anchor.

    Both temporal and matched current-only modes select identical queries from
    that anchor. Labels, boxes and instance IDs are not accepted as inputs.
    """
    def __init__(self, cfg=None, repair=RepairConfig(), device='cpu'):
        cfg = cfg or repair_memory_config()
        super().__init__(cfg, device)
        if cfg.radius != repair.moving_radius:
            raise ValueError('Retrieval radius must equal moving radius')
        if any(type(c) is not int or not 0 <= c < cfg.classes for c in repair.moving_classes):
            raise ValueError('Moving classes outside ontology')
        self.repair = repair
        self.memory = PreviousEvidence(cfg, device, repair.moving_classes)
        self.fusion = JointRepair(cfg, repair).to(device)

    @torch.no_grad()
    def queries(self, anchor):
        p = anchor.float().softmax(-1)
        confidence = p.max(-1).values
        ids = (confidence < self.repair.confidence).nonzero().flatten()
        if len(ids) > self.repair.max_queries:
            ids = ids[confidence[ids].topk(self.repair.max_queries, largest=False).indices]
        return ids, p

    def forward(self, xyz_sensor, features, base_logits, *, anchor_logits, stamp,
                frame, transform, epoch=0, current_only=False, disable_history=False,
                shuffle_history=False, narrow_history=False):
        self._validate(xyz_sensor, features, base_logits, stamp, frame, transform, epoch)
        if (anchor_logits.shape != base_logits.shape or anchor_logits.device != features.device or
                not bool(torch.isfinite(anchor_logits.float()).all())):
            raise ValueError('Invalid single-frame anchor')
        xyz = xyz_sensor.float() @ transform[:3, :3].float().T + transform[:3, 3].float()
        _keys(xyz, self.cfg.cell_size)
        m = self.memory
        if m.epoch is not None and m.epoch != epoch:
            m.reset()
        ids, p = self.queries(anchor_logits)
        status = dict(expired=0, revoked_free_space=0, occluded=0)
        f = features[ids]
        if current_only:
            n = len(ids)
            candidates = dict(features=f[:, None], probabilities=p[ids, None],
                delta=xyz.new_zeros((n,1,3)), age=xyz.new_zeros((n,1)),
                valid=torch.zeros((n,1), dtype=torch.bool, device=xyz.device))
        else:
            status = m.prepare(stamp, transform.float(), xyz_sensor.float())
            candidates = m.retrieve(xyz[ids], f, stamp)
            # Top-two current predictions avoid locking association to a single
            # possibly wrong class. A wider radius merely admits motion candidates.
            top = p[ids].topk(2, dim=1).indices
            mobile = torch.isin(top, m.moving_classes).any(1)
            radius = torch.where(mobile, self.repair.moving_radius, self.repair.static_radius)
            if narrow_history:
                radius.fill_(self.repair.static_radius)
            candidates['valid'] &= candidates['delta'].square().sum(-1) <= radius[:, None]**2
            if shuffle_history and len(ids):
                # Deterministic diagnostic: wrong past content, unchanged geometry
                # and eligibility. Never uses future frames or annotation inputs.
                for key in ('features', 'probabilities'):
                    candidates[key] = candidates[key].roll(len(ids)//2, dims=0)
            if disable_history:
                candidates['valid'].zero_()
        selected = self.fusion(f, anchor_logits[ids], candidates)
        logits = anchor_logits.float().index_copy(0, ids, selected)
        present = candidates['valid'].any(1)
        if not current_only:
            m.write_raw(xyz, features, base_logits.detach().float().softmax(-1), stamp, frame)
        m.last_stamp, m.last_frame, m.epoch = stamp, frame, epoch
        self.last_status = dict(status, queries=len(ids), usable=int(present.sum()), records=len(m.xyz))
        return dict(logits=logits, anchor_logits=anchor_logits.float(), query_indices=ids,
                    has_history=present, selected_logits=selected)


def repair_loss(output, labels, class_weights):
    """Classification plus error-repair and correct-class margin preservation.

    Targets depend on training labels only. The same objective applies to both
    learned arms. No confidence-improvement pseudo-label or inference threshold.
    """
    ids = output['query_indices']
    valid = labels[ids] >= 0
    logits = output['selected_logits'][valid]
    anchor = output['anchor_logits'][ids][valid].detach()
    y = labels[ids][valid].long()
    if not len(y):
        return logits.sum() * 0
    weight = class_weights[y]
    ce = F.cross_entropy(logits, y, reduction='none')
    mask = F.one_hot(y, logits.shape[1]).bool()
    margin = logits.gather(1, y[:, None]).squeeze(1) - logits.masked_fill(mask, -1e4).max(1).values
    old_margin = anchor.gather(1, y[:, None]).squeeze(1) - anchor.masked_fill(mask, -1e4).max(1).values
    correct = anchor.argmax(-1) == y
    fix = (~correct) * F.relu(.5 - margin)
    retain = correct * F.relu(old_margin.clamp(max=2.) - margin)
    return (weight * (ce + .5*fix + .5*retain)).sum() / weight.sum()
