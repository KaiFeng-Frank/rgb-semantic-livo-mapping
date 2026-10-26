"""Bounded, revisable instance semantics from independent current observations.

Tracker IDs are proposals. A geometric/appearance veto starts a new generation;
no semantic class is used to decide whether two objects have the same identity.
Only raw frame evidence is retained, never the enhanced point predictions.
"""
from collections import Counter, deque
from dataclasses import asdict, dataclass
import time

import numpy as np


@dataclass(frozen=True)
class Config:
    classes: int = 16
    min_points: int = 6
    max_object_points: int = 2048
    max_tracks: int = 128
    history_frames: int = 4
    ttl_s: float = 2.
    component_fraction: float = .6
    visual_weight: float = .25
    posterior_min: float = .6
    posterior_margin: float = .15
    decisive_current: float = .85
    history_decay_s: float = 1.5
    initial_speed_mps: float = 2.
    position_slack_m: float = .75
    acceleration_mps2: float = 3.
    identity_margin_m: float = .35
    appearance_distance: float = .45


THING = np.array([1, 2, 3, 5, 6, 9])


def component(xyz, cfg):
    """Largest connected surface, with bounded neighbor edges and ambiguity veto."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if not cfg.min_points <= len(xyz) <= cfg.max_object_points:
        return None
    radius = float(np.clip(np.median(np.linalg.norm(xyz, axis=1))*.025, .25, 1.))
    d, j = cKDTree(xyz).query(xyz, k=min(12, len(xyz)), distance_upper_bound=radius, workers=1)
    valid = np.isfinite(d)
    i = np.broadcast_to(np.arange(len(xyz))[:, None], j.shape)
    graph = coo_matrix((np.ones(int(valid.sum())), (i[valid], j[valid])), shape=(len(xyz),)*2)
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    best = int(sizes.argmax())
    if sizes[best] < cfg.min_points or sizes[best]/len(xyz) < cfg.component_fraction:
        return None
    return np.flatnonzero(labels == best)


def world(obs):
    return obs['pose'][:3, :3] @ obs['center'] + obs['pose'][:3, 3]


class InstanceSemantics:
    def __init__(self, cfg=None, *, temporal=True, guard=True, conflict_guard=True):
        self.cfg = cfg or Config()
        self.temporal, self.guard = temporal, guard
        self.conflict_guard = conflict_guard
        self.states, self.session = {}, None
        self.last_stamp, self.generation = -1, 0

    def revise_poses(self, times, poses):
        """Reanchor all retained centers to the same received graph revision."""
        times, poses = np.asarray(times), np.asarray(poses)
        for tid, state in list(self.states.items()):
            for obs in state['history']:
                i = int(np.searchsorted(times, obs['stamp_ns']))
                if i >= len(times) or times[i] != obs['stamp_ns']:
                    del self.states[tid]
                    break
                obs['pose'] = poses[i].copy()

    def prediction(self, state, stamp):
        h = state['history']; last = h[-1]
        dt = (stamp-last['stamp_ns'])*1e-9
        center = world(last)
        tolerance = self.cfg.position_slack_m + self.cfg.initial_speed_mps*dt
        if len(h) >= 2:
            elapsed = (last['stamp_ns']-h[-2]['stamp_ns'])*1e-9
            velocity = (center-world(h[-2]))/elapsed
            if np.linalg.norm(velocity) <= 20.:
                center = center+velocity*dt
                tolerance = self.cfg.position_slack_m+.5*self.cfg.acceleration_mps2*dt*dt
        return center, tolerance

    def step(self, xyz, probabilities, association, *, stamp_ns, pose):
        started = time.perf_counter(); cfg = self.cfg
        xyz = np.asarray(xyz, dtype=np.float64); p = np.asarray(probabilities, dtype=np.float32)
        pose = np.asarray(pose, dtype=np.float64)
        if (type(stamp_ns) is not int or stamp_ns <= self.last_stamp or xyz.shape != (len(p), 3)
                or p.ndim != 2 or p.shape[1] != cfg.classes or not np.isfinite(xyz).all()
                or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1., atol=2e-3)
                or pose.shape != (4,4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0,0,0,1])
                or not np.isclose(np.linalg.det(pose[:3,:3]), 1., atol=1e-4)
                or not np.allclose(pose[:3,:3].T@pose[:3,:3], np.eye(3), atol=1e-4)):
            raise ValueError('Invalid instance semantic observation')
        self.last_stamp = stamp_ns
        raw_labels = p.argmax(1).astype(np.uint16); raw_conf = p.max(1)
        labels, confidence = raw_labels.copy(), raw_conf.copy()
        n = len(p)
        result = dict(labels=labels, confidence=confidence, raw_labels=raw_labels, raw_confidence=raw_conf,
                      valid=np.zeros(n, np.uint8), epochs=np.zeros(n, np.uint32),
                      instance_labels=np.full(n, 65535, np.uint16), instance_confidence=np.zeros(n, np.float32))
        session = association['info'].get('session')
        if session is not None and session != self.session:
            self.states.clear(); self.session = session
        self.states = {tid:s for tid,s in self.states.items()
                       if (stamp_ns-s['history'][-1]['stamp_ns'])*1e-9 <= cfg.ttl_s}
        counts = Counter(); observations = {}
        image_stamp = association['info'].get('camera_stamp_ns', stamp_ns)
        if image_stamp > stamp_ns:
            raise ValueError('Future visual evidence')
        ids = np.asarray(association['ids'])
        if ids.shape != (n,) or np.any(ids < 0) or len(np.unique(ids[ids > 0])) > cfg.max_tracks:
            raise ValueError('Invalid instance association IDs')
        for tid in np.unique(ids[ids > 0]) if session else []:
            at = np.flatnonzero(ids == tid)
            chosen = component(xyz[at], cfg)
            if chosen is None:
                counts['surface_rejected'] += 1
                continue
            at = at[chosen]
            cls = int(association['labels'][at[0]])
            if cls not in THING:
                counts['class_rejected'] += 1
                continue
            score = float(np.median(association['scores'][at]))
            if not 0 <= score <= 1:
                raise ValueError('Invalid visual score')
            obs_p = p[at].mean(0).astype(np.float64)
            alpha = cfg.visual_weight*score
            obs_p *= 1-alpha; obs_p[cls] += alpha
            obs_p /= obs_p.sum()
            observations[int(tid)] = dict(at=at, center=np.median(xyz[at], axis=0),
                pose=pose.copy(), stamp_ns=stamp_ns, image_stamp=image_stamp, posterior=obs_p,
                appearance=association.get('appearance', {}).get(int(tid)))
        predictions = {tid:self.prediction(s, stamp_ns) for tid,s in self.states.items()}
        current_world = {tid:world(o) for tid,o in observations.items()}
        decisions = {}
        for tid, obs in observations.items():
            old = self.states.get(tid); reason = 'new'
            if old is not None and self.temporal:
                last = old['history'][-1]; reason = 'accepted'
                if image_stamp <= last['image_stamp']:
                    reason = 'reused_camera'
                elif self.guard:
                    predicted, tolerance = predictions[tid]
                    distance = float(np.linalg.norm(current_world[tid]-predicted))
                    if distance > tolerance:
                        reason = 'motion'
                    elif any(np.linalg.norm(current_world[tid]-v[0]) < distance+cfg.identity_margin_m
                             for other,v in predictions.items() if other != tid):
                        reason = 'competing_history'
                    elif any(np.linalg.norm(v-predicted) < distance+cfg.identity_margin_m
                             for other,v in current_world.items() if other != tid):
                        reason = 'competing_current'
                    elif obs['appearance'] is not None and last['appearance'] is not None:
                        similarity = np.sqrt(np.asarray(obs['appearance'])*last['appearance']).sum()
                        if np.sqrt(max(0., 1-float(similarity))) > cfg.appearance_distance:
                            reason = 'appearance'
            decisions[tid] = reason
        details = []
        for tid, obs in observations.items():
            reason = decisions[tid]; old = self.states.get(tid)
            counts[reason] += 1
            reuse = reason == 'accepted'
            state_p = obs['posterior'].copy()
            if reuse:
                old_p = old['posterior']
                # Decisive new raw evidence can correct a wrong initial state.
                decisive = state_p.argmax() != old_p.argmax() and state_p.max() >= cfg.decisive_current
                if decisive:
                    history = deque(maxlen=cfg.history_frames); counts['semantic_reset'] += 1
                else:
                    history = old['history']
                    weights = np.array([np.exp(-(stamp_ns-o['stamp_ns'])*1e-9/cfg.history_decay_s) for o in history])
                    weights /= max(1., weights.sum())  # History has at most one frame's weight.
                    state_p = (state_p+sum(w*o['posterior'] for w,o in zip(weights, history)))/(1+weights.sum())
                    counts['history_instances'] += 1
                epoch = old['epoch']
            else:
                history = deque(maxlen=cfg.history_frames)
                self.generation += 1; epoch = self.generation
            best = int(state_p.argmax()); top = np.sort(state_p)[-2:]
            accepted = best in THING and top[-1] >= cfg.posterior_min and top[-1]-top[-2] >= cfg.posterior_margin
            at = obs.pop('at')
            # History must not manufacture confidence over current counterevidence.
            # If the current instance already supports this same decision, keep
            # ordinary single-frame pooling. Otherwise contrary confident points
            # veto the history-dependent decision for the whole component.
            contrary = ((raw_labels[at] != best) & (raw_conf[at] >= cfg.posterior_min)
                        & (raw_conf[at]-p[at,best] >= cfg.posterior_margin))
            current_top = np.sort(obs['posterior'])[-2:]
            current_supports = (obs['posterior'].argmax() == best and current_top[-1] >= cfg.posterior_min
                                and current_top[-1]-current_top[-2] >= cfg.posterior_margin)
            conflict = bool(self.conflict_guard and accepted and reuse and not current_supports and np.any(contrary))
            if conflict:
                accepted = False
                counts['semantic_conflict'] += 1
                history = deque(maxlen=cfg.history_frames)
                state_p = obs['posterior'].copy()
                best = int(state_p.argmax())
            result['epochs'][at] = epoch
            if accepted:
                labels[at], confidence[at] = best, state_p[best]
                result['valid'][at] = 1
                result['instance_labels'][at], result['instance_confidence'][at] = best, state_p[best]
                counts['unified_points'] += len(at); counts['unified_instances'] += 1
            # Only the current raw observation is stored, never state_p or enhanced points.
            history.append(obs)
            self.states[tid] = dict(history=history, posterior=state_p.copy(), epoch=epoch)
            details.append(dict(track_id=tid, epoch=epoch, identity=reason, points=len(at),
                                label=best, confidence=float(state_p[best]), unified=bool(accepted),
                                semantic_conflict=conflict))
        if len(self.states) > cfg.max_tracks:
            keep = sorted(self.states, key=lambda t:self.states[t]['history'][-1]['stamp_ns'], reverse=True)[:cfg.max_tracks]
            self.states = {t:self.states[t] for t in keep}
        result['info'] = dict(counts, changed_points=int(np.count_nonzero(labels != raw_labels)),
            tracks=len(self.states), history_observations=sum(len(s['history']) for s in self.states.values()),
            instances=details, elapsed_ms=(time.perf_counter()-started)*1000,
            identity_key='(visual_session, track_id, semantic_epoch)', semantic_update=True)
        return result

    def configuration(self):
        return dict(asdict(self.cfg), temporal=self.temporal, identity_guard=self.guard,
                    conflict_guard=self.conflict_guard,
                    evidence='raw frame posterior means + bounded current visual vote; no enhanced-output feedback')
