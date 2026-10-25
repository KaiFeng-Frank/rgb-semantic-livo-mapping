"""Causal short-term consistency; independent of the pose/front-end thread.

Only the previous RAW observation is retained. This is a nonlearned baseline,
not a training loss, object tracker, or long-term semantic-map feedback module.
Geometric gates reduce association errors; they cannot certify correctness.
"""
from dataclasses import asdict, dataclass
import math
import time

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ConsistencyConfig:
    classes: int = 16
    max_points: int = 60000
    max_age_s: float = 1.5
    radius_m: float = .25
    radial_tolerance_m: float = .15
    history_confidence: float = .8
    confidence_advantage: float = .1
    current_veto_confidence: float = .8
    history_weight: float = .35
    decay_s: float = 2.
    excluded_classes: tuple = (1, 2, 3, 4, 5, 6, 8, 9)

    def __post_init__(self):
        if type(self.classes) is not int or self.classes < 2 or type(self.max_points) is not int or self.max_points < 1:
            raise ValueError('Invalid class/point bounds')
        for k in ('max_age_s', 'radius_m', 'radial_tolerance_m', 'decay_s'):
            if not math.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError('Invalid geometric/time bounds')
        for k in ('history_confidence', 'confidence_advantage', 'current_veto_confidence', 'history_weight'):
            if not math.isfinite(getattr(self, k)) or not 0 <= getattr(self, k) <= 1:
                raise ValueError('Invalid confidence/weight')
        if any(type(c) is not int or not 0 <= c < self.classes for c in self.excluded_classes):
            raise ValueError('Excluded class outside vocabulary')


def rigid(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-6, rtol=0)):
        raise ValueError('Expected a rigid sensor-to-world transform')
    return pose


def associate(current, previous, current_pose, previous_pose, cfg):
    """Reciprocal nearest neighbours with bounded Euclidean and ray-depth error.

    All points participate before semantic filtering; a moving foreground cannot
    be removed to expose a farther static match. This is NOT full occlusion or
    motion reasoning. Returns independent, one-to-one original point indices.
    """
    # Work in current sensor coordinates, keeping large global offsets out of trees.
    relative = np.linalg.inv(current_pose) @ previous_pose
    past = previous @ relative[:3, :3].T + relative[:3, 3]
    past_tree, current_tree = cKDTree(past), cKDTree(current)
    distances, near = past_tree.query(current, distance_upper_bound=cfg.radius_m, workers=1)
    ci = np.flatnonzero(np.isfinite(distances))
    if not len(ci):
        return ci, ci.copy()
    hi = near[ci]
    _, back = current_tree.query(past[hi], distance_upper_bound=cfg.radius_m, workers=1)
    mutual = back == ci
    ci, hi = ci[mutual], hi[mutual]
    delta = current[ci] - past[hi]
    current_ray = current[ci]
    history_ray = past[hi] - relative[:3, 3]
    cr = np.linalg.norm(current_ray, axis=1)
    hr = np.linalg.norm(history_ray, axis=1)
    valid = (cr > 1e-6) & (hr > 1e-6)
    for ray, length in ((current_ray, cr), (history_ray, hr)):
        radial = np.abs(np.einsum('ij,ij->i', delta, ray)) / np.maximum(length, 1e-6)
        valid &= radial <= cfg.radial_tolerance_m
    return ci[valid], hi[valid]


def fuse(current, history, age_s, cfg, *, conditional=True):
    """Soft posterior mixture on ALREADY geometrically matched pairs."""
    weight = np.full(len(current), .5, np.float32)
    if conditional:
        cc, hc = current.max(1), history.max(1)
        allowed = (~np.isin(current.argmax(1), cfg.excluded_classes)
                   & ~np.isin(history.argmax(1), cfg.excluded_classes)
                   & (hc >= cfg.history_confidence)
                   & (hc >= cc + cfg.confidence_advantage)
                   & (cc < cfg.current_veto_confidence))
        weight[:] = cfg.history_weight * math.exp(-age_s / cfg.decay_s)
        weight[~allowed] = 0.
    output = current.copy()
    active = weight > 0
    output[active] = ((1-weight[active, None])*current[active]
                      + weight[active, None]*history[active])
    return output, weight


class CausalConsistency:
    def __init__(self, config=None):
        self.cfg = config or ConsistencyConfig()
        self.history = None
        self.last_stamp_ns = None
        self.last_status = {}

    def reset(self):
        self.history = None
        self.last_stamp_ns = None

    def revise_pose(self, times_ns, poses):
        """Re-anchor retained raw evidence using an already received graph only."""
        if self.history is None:
            return
        i = int(np.searchsorted(times_ns, self.history['stamp_ns']))
        if i >= len(times_ns) or times_ns[i] != self.history['stamp_ns']:
            self.history = None  # Never use unanchored history.
        else:
            self.history['pose'] = rigid(poses[i]).copy()

    def step(self, xyz, probabilities, *, stamp_ns, pose, diagnostics=False):
        start = time.perf_counter()
        xyz = np.asarray(xyz, dtype=np.float64)
        probabilities = np.asarray(probabilities, dtype=np.float32)
        pose = rigid(pose)
        if type(stamp_ns) is not int or stamp_ns < 0:
            raise ValueError('Expected a nonnegative integer sensor timestamp')
        if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
            raise ValueError('Reset explicitly before a new or reordered sequence')
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all()
                or probabilities.shape != (len(xyz), self.cfg.classes)
                or not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or not np.allclose(probabilities.sum(1), 1., atol=2e-5, rtol=0)):
            raise ValueError('Expected finite XYZ and full normalized class probabilities')
        output = probabilities.copy()
        ci, hi = np.empty(0, np.int64), np.empty(0, np.int64)
        weight = np.empty(0, np.float32)
        past_prob = np.empty((0, self.cfg.classes), np.float32)
        reason, age = 'no_history', None
        if len(xyz) > self.cfg.max_points:
            # Exact raw fallback and bounded memory, not an unbounded search.
            reason = 'point_budget'
        elif self.history is not None:
            age = (stamp_ns - self.history['stamp_ns']) / 1e9
            reason = 'expired' if age > self.cfg.max_age_s else 'matched'
            if reason == 'matched':
                ci, hi = associate(xyz, self.history['xyz'], pose, self.history['pose'], self.cfg)
                past_prob = self.history['probabilities'][hi]
                output[ci], weight = fuse(probabilities[ci], past_prob, age, self.cfg)
        # Write RAW inputs only, also after expiry. Never consume enhanced outputs.
        self.history = (dict(stamp_ns=stamp_ns, xyz=xyz.copy(), pose=pose.copy(),
                             probabilities=probabilities.copy())
                        if len(xyz) <= self.cfg.max_points else None)
        self.last_stamp_ns = stamp_ns
        changed = output.argmax(1) != probabilities.argmax(1)
        status = dict(reason=reason, age_s=age, pairs=len(ci), used=int((weight > 0).sum()),
                      changed=int(changed.sum()), points=len(xyz),
                      retained_bytes=sum(v.nbytes for v in self.history.values() if isinstance(v, np.ndarray)) if self.history else 0,
                      elapsed_ms=(time.perf_counter()-start)*1000)
        self.last_status = status
        result = dict(probabilities=output, status=status)
        if diagnostics:
            result.update(current_indices=ci, history_indices=hi, history_probabilities=past_prob,
                          weights=weight)
        return result

    def config_dict(self):
        return asdict(self.cfg)
