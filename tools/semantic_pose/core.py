"""NumPy/SciPy weighted point-to-plane ICP. No files, GT or model inference.

G/D/S/T share the same solver, points, nearest-neighbour association and robust
kernel. Labels adjust influence only; they never create geometric matches.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def transform(x, t):
    return x @ t[:3, :3].T + t[:3, 3]


def confidence_strength(c, cfg):
    start = cfg['confidence_start']
    return np.clip((np.asarray(c)-start)/(1-start), 0., 1.)


def groups(labels, cfg):
    result = np.full(np.shape(labels), -1, dtype=np.int32)
    for i, members in enumerate(cfg['compatibility_groups']):
        result[np.isin(labels, members)] = i
    return result


def weights(source, target, src_idx, dst_idx, arm, cfg):
    """Unknown/low-confidence evidence is neutral, not discarded."""
    if arm not in ('G', 'D', 'S', 'T'):
        raise ValueError(arm)
    w = np.ones(len(src_idx))
    if arm == 'G':
        return w
    ls, lt = source['label'][src_idx], target['label'][dst_idx]
    cs = confidence_strength(source['confidence'][src_idx], cfg)
    ct = confidence_strength(target['confidence'][dst_idx], cfg)
    valid_s, valid_t = (ls >= 0) & (ls < 16), (lt >= 0) & (lt < 16)
    cs, ct = cs*valid_s, ct*valid_t
    risk = np.maximum(cs*np.isin(ls, cfg['movable_classes']),
                      ct*np.isin(lt, cfg['movable_classes']))
    w *= 1-(1-cfg['movable_weight_floor'])*risk
    if arm in ('S', 'T'):
        gs, gt = groups(ls, cfg), groups(lt, cfg)
        conflict = (gs >= 0) & (gt >= 0) & (gs != gt)
        w *= 1-(1-cfg['incompatible_weight_floor'])*np.minimum(cs, ct)*conflict
    if arm == 'T':
        # Other-frame evidence is fixed before ICP. Missing history stays neutral.
        for features, idx in ((source, src_idx), (target, dst_idx)):
            known = features['temporal_count'][idx] >= cfg['temporal_minimum_other_frames']
            w *= 1-(1-cfg['temporal_weight_floor'])*known*(1-features['temporal_stability'][idx])
    return w


def temporal_evidence(reference, histories, cfg):
    """At most one nearest geometric observation per distinct earlier frame.

    Histories are separate scans, in reference coordinates, from fixed initial
    odometry. No transformed pose from this ICP or fused correction is used.
    No observation within the radius means unknown, not proof of movement.
    """
    frames = [int(other['frame']) for other in histories]
    if len(set(frames)) != len(frames):
        raise ValueError('Repeated frame cannot count as independent history')
    n = len(reference['xyz'])
    counts, total = np.zeros(n, dtype=np.int32), np.zeros(n)
    for other in histories:
        distance, idx = cKDTree(other['xyz']).query(reference['xyz'], workers=1)
        known = (distance <= cfg['temporal_radius_m'])
        known &= (reference['confidence'] >= cfg['confidence_start'])
        known &= (other['confidence'][idx] >= cfg['confidence_start'])
        known &= (reference['label'] >= 0) & (other['label'][idx] >= 0)
        same = reference['label'] == other['label'][idx]
        stable = same & (distance <= cfg['temporal_stable_distance_m'])
        counts += known
        total += known*stable
    return counts, np.divide(total, counts, out=np.ones(n), where=counts > 0)


def jacobian(points, normals):
    # Left increment: R(delta) p + dt, with [rotation, translation] ordering.
    return np.column_stack((np.cross(points, normals), normals))


def register(source, target, initial, arm, cfg, geometric):
    tree = cKDTree(target['xyz'])
    pose = initial.copy()
    history = []
    failure = None
    for threshold, iterations in zip(geometric['icp_correspondence_m'], geometric['icp_iterations']):
        previous = None
        for _ in range(iterations):
            moved = transform(source['xyz'], pose)
            distance, nearest = tree.query(moved, workers=1)
            src_idx = np.flatnonzero(distance < threshold)
            dst_idx = nearest[src_idx]
            if len(src_idx) < 6:
                failure = 'insufficient_correspondences'
                break
            normal = target['normal'][dst_idx]
            residual = np.einsum('ij,ij->i', normal, moved[src_idx]-target['xyz'][dst_idx])
            semantic = weights(source, target, src_idx, dst_idx, arm, cfg)
            robust = np.minimum(1., geometric['icp_huber_m']/np.maximum(np.abs(residual), 1e-15))
            w = robust*semantic
            j = jacobian(moved[src_idx], normal)
            h = j.T @ (w[:, None]*j)
            b = j.T @ (w*residual)
            scale = np.sqrt(np.maximum(np.diag(h), 1e-30))
            normalized = h/scale[:, None]/scale[None, :]
            eigenvalues = np.linalg.eigvalsh(normalized)
            condition = float(eigenvalues[-1]/max(eigenvalues[0], 1e-30))
            effective = float(w.sum()**2/max(np.sum(w*w), 1e-30))
            if eigenvalues[0] <= 0 or condition > cfg['maximum_normalized_condition']:
                failure = 'geometric_degeneracy'
                break
            if effective < cfg['minimum_effective_correspondences']:
                failure = 'insufficient_effective_correspondences'
                break
            step = np.linalg.solve(normalized, -b/scale)/scale
            delta = np.eye(4)
            delta[:3, :3] = Rotation.from_rotvec(step[:3]).as_matrix()
            delta[:3, 3] = step[3:]
            pose = delta @ pose
            dd = tree.query(transform(source['xyz'], pose), workers=1)[0]
            inlier = dd < threshold
            fitness = float(inlier.mean())
            rmse = float(np.sqrt(np.mean(dd[inlier]**2))) if inlier.any() else float('inf')
            history.append(dict(threshold_m=threshold, fitness=fitness, rmse_m=rmse,
                effective_correspondences=effective, condition=condition,
                downweighted_fraction=float(np.mean(semantic < 1-1e-12))))
            if previous is not None and abs(previous[0]-fitness)<1e-7 and abs(previous[1]-rmse)<1e-7:
                break
            previous = fitness, rmse
        if failure:
            break
    distance, nearest = tree.query(transform(source['xyz'], pose), workers=1)
    keep = distance < geometric['icp_correspondence_m'][-1]
    idx = np.flatnonzero(keep)
    final_weights = weights(source, target, idx, nearest[idx], arm, cfg)
    correction = pose @ np.linalg.inv(initial)
    stats = dict(fitness=float(keep.mean()), rmse_m=float(np.sqrt(np.mean(distance[keep]**2))) if keep.any() else None,
        correspondences=int(keep.sum()), correction_translation_m=float(np.linalg.norm(correction[:3, 3])),
        correction_rotation_deg=float(np.rad2deg(Rotation.from_matrix(correction[:3, :3]).magnitude())),
        iterations=len(history), failure=failure, history=history,
        downweighted_fraction=float(np.mean(final_weights < 1-1e-12)) if len(idx) else 0.,
        mean_semantic_weight=float(np.mean(final_weights)) if len(idx) else 1.)
    stats['accepted'] = bool(failure is None and np.isfinite(pose).all()
        and stats['fitness'] >= geometric['minimum_icp_fitness']
        and stats['rmse_m'] <= geometric['maximum_icp_rmse_m']
        and stats['correspondences'] >= geometric['minimum_correspondences']
        and stats['correction_translation_m'] <= geometric['maximum_correction_translation_m']
        and stats['correction_rotation_deg'] <= geometric['maximum_correction_rotation_deg'])
    return pose, stats


def guard(candidate, stats, baseline, cfg):
    difference = candidate @ np.linalg.inv(baseline)
    dt = float(np.linalg.norm(difference[:3, 3]))
    dr = float(np.rad2deg(Rotation.from_matrix(difference[:3, :3]).magnitude()))
    use = bool(stats['accepted'] and dt <= cfg['maximum_semantic_delta_translation_m']
               and dr <= cfg['maximum_semantic_delta_rotation_deg'])
    return (candidate if use else baseline).copy(), dict(fallback=not use,
        delta_from_G_translation_m=dt, delta_from_G_rotation_deg=dr)
