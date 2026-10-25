"""Offline Scan Context-style retrieval -> Open3D ICP -> GTSAM Pose3 graph.

The compact NumPy descriptor adapter implements the published ring/sector height
representation and cosine distance; it is not the authors' original C++ module.
Reference: https://github.com/gisbi-kim/scancontext . Registration and nonlinear
optimization use the unmodified Open3D and GTSAM libraries. This module never
loads ground-truth poses, labels, predictions, or evaluation scores.
"""
import argparse
from pathlib import Path
import time

import gtsam
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from .common import REPO, Scans, corrected_tum, digest, protocol, save, transform


def descriptor(points, cfg):
    rings, sectors = cfg['scan_context_rings'], cfg['scan_context_sectors']
    radius = np.linalg.norm(points[:, :2], axis=1)
    valid = np.isfinite(points).all(1) & (radius > 0) & (radius <= cfg['scan_context_radius_m'])
    p, radius = points[valid], radius[valid]
    ri = np.clip(np.ceil(radius / cfg['scan_context_radius_m'] * rings).astype(int)-1, 0, rings-1)
    angle = np.mod(np.arctan2(p[:, 1], p[:, 0]), 2*np.pi)
    si = np.clip(np.ceil(angle / (2*np.pi) * sectors).astype(int)-1, 0, sectors-1)
    d = np.full((rings, sectors), -np.inf)
    np.maximum.at(d, (ri, si), p[:, 2]+cfg['scan_context_height_offset_m'])
    d[~np.isfinite(d)] = 0
    return d


def descriptor_distance(a, b):
    """Exhaustive yaw shifts (small offline database), ignoring empty sectors."""
    distances = []
    na = np.linalg.norm(a, axis=0)
    for shift in range(a.shape[1]):
        shifted = np.roll(b, shift, axis=1)
        denom = na * np.linalg.norm(shifted, axis=0)
        valid = denom > 1e-12
        distances.append(float(1-np.mean(np.sum(a[:, valid]*shifted[:, valid], axis=0)/denom[valid]))
                         if valid.any() else 1.)
    best = int(np.argmin(distances))
    return distances[best], best


def cloud(points, voxel):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, np.float64)))
    return pc.voxel_down_sample(voxel)


def register(source, target, initial, cfg):
    """Return T_target_source; GTSAM Between(target, source) has this direction."""
    reg = o3d.pipelines.registration
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.2, max_nn=30))
    estimation = reg.TransformationEstimationPointToPlane(reg.HuberLoss(cfg['icp_huber_m']))
    result = None
    pose = initial.copy()
    for distance, iterations in zip(cfg['icp_correspondence_m'], cfg['icp_iterations']):
        result = reg.registration_icp(source, target, distance, pose, estimation,
                    reg.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=iterations))
        pose = result.transformation
    delta = pose @ np.linalg.inv(initial)
    metrics = dict(fitness=float(result.fitness), rmse_m=float(result.inlier_rmse),
                   correspondences=len(result.correspondence_set),
                   correction_translation_m=float(np.linalg.norm(delta[:3, 3])),
                   correction_rotation_deg=float(np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude())))
    metrics['accepted'] = bool(np.isfinite(pose).all()
        and metrics['fitness'] >= cfg['minimum_icp_fitness']
        and metrics['rmse_m'] <= cfg['maximum_icp_rmse_m']
        and metrics['correspondences'] >= cfg['minimum_correspondences']
        and metrics['correction_translation_m'] <= cfg['maximum_correction_translation_m']
        and metrics['correction_rotation_deg'] <= cfg['maximum_correction_rotation_deg'])
    return pose, metrics


def optimize(poses, edges, cfg):
    graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    diag = gtsam.noiseModel.Diagonal.Sigmas
    odo_noise = diag(np.r_[np.full(3, np.deg2rad(cfg['odometry_rotation_sigma_deg'])),
                           np.full(3, cfg['odometry_translation_sigma_m'])])
    loop_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(cfg['loop_huber_sigma']),
        diag(np.r_[np.full(3, np.deg2rad(cfg['loop_rotation_sigma_deg'])),
                    np.full(3, cfg['loop_translation_sigma_m'])]))
    for j, pose in enumerate(poses):
        values.insert(j, gtsam.Pose3(pose))
        if j:
            graph.add(gtsam.BetweenFactorPose3(j-1, j, gtsam.Pose3(np.linalg.inv(poses[j-1])@pose), odo_noise))
    graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(poses[0]), diag(np.full(6, 1e-6))))
    for i, j, measurement in edges:
        graph.add(gtsam.BetweenFactorPose3(i, j, gtsam.Pose3(measurement), loop_noise))
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    result = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()
    optimized = np.stack([result.atPose3(j).matrix() for j in range(len(poses))])
    return optimized, dict(error_before=float(graph.error(values)), error_after=float(graph.error(result)))


def run(root, out):
    cfg = protocol()
    out.mkdir(parents=True, exist_ok=True)
    trajectory = REPO / cfg['baseline_trajectory']
    scans = Scans(root, trajectory)
    frames, poses, pcs, descriptions = [], [], [], []
    t0 = time.monotonic()
    for f in range(cfg['keyframe_stride'], cfg['frames'], cfg['keyframe_stride']):
        if not scans.traj.valid([scans.images[f]])[0]:
            continue
        points, pose = scans.local(f)
        points = points[np.isfinite(points).all(1) & (np.linalg.norm(points[:, :2], axis=1)<80)]
        frames.append(f)
        poses.append(pose)
        descriptions.append(descriptor(points, cfg))
        pcs.append(cloud(points, cfg['icp_voxel_m']))
        if len(frames) % 20 == 0:
            print('KEYFRAMES', len(frames), 'frame', f, flush=True)
    poses, descriptions = np.asarray(poses), np.asarray(descriptions)
    key_times = scans.images[frames]
    ringkeys = descriptions.mean(axis=2)
    edges, attempts = [], []
    last = -100
    for j in range(len(frames)):
        if j-last < cfg['loop_cooldown_keyframes']:
            continue
        eligible = np.flatnonzero((key_times[j]-key_times > cfg['minimum_loop_seconds']) &
            (np.linalg.norm(poses[:, :3, 3]-poses[j, :3, 3], axis=1)<cfg['maximum_candidate_distance_m']))
        if not len(eligible):
            continue
        candidates = eligible[np.argsort(np.linalg.norm(ringkeys[eligible]-ringkeys[j], axis=1))[:cfg['ring_candidates']]]
        scored = sorted(((*descriptor_distance(descriptions[i], descriptions[j]), int(i)) for i in candidates))
        for distance, shift, i in scored:
            record = dict(historical_frame=frames[i], current_frame=frames[j], i=i, j=j,
                          descriptor_distance=distance, descriptor_yaw_shift=shift, accepted=False)
            if distance > cfg['scan_context_threshold']:
                record['rejection'] = 'descriptor'
                attempts.append(record)
                continue
            pieces = []
            half = cfg['submap_half_window']
            for k in range(max(0, i-half), min(len(frames), i+half+1)):
                # No temporally adjacent-to-current scans in the historical submap.
                if key_times[j]-key_times[k] > cfg['minimum_loop_seconds']:
                    pieces.append(transform(np.asarray(pcs[k].points), np.linalg.inv(poses[i])@poses[k]))
            target = cloud(np.concatenate(pieces), cfg['icp_voxel_m'])
            measurement, stats = register(pcs[j], target, np.linalg.inv(poses[i])@poses[j], cfg)
            record.update(stats)
            record['T_historical_current'] = measurement.tolist()
            attempts.append(record)
            print('LOOP_ATTEMPT', frames[i], frames[j], stats, flush=True)
            if stats['accepted']:
                edges.append((i, j, measurement))
                last = j
                break
    # An actual graph solve with odometry only must retain its initialization.
    identity, identity_stats = optimize(poses, [], cfg)
    control_error = float(np.max(np.abs(identity-poses)))
    assert control_error < 1e-6, control_error
    optimized, graph_stats = optimize(poses, edges, cfg)
    tum = np.loadtxt(trajectory)
    corrected = corrected_tum(tum, key_times, poses, optimized if edges else poses)
    np.savetxt(out/'corrected_imu_tum.txt', corrected, fmt='%.10f')
    np.savetxt(out/'baseline_imu_tum.txt', tum, fmt='%.10f')
    np.savez_compressed(out/'keyframes.npz', frames=frames, times=key_times, baseline=poses, corrected=optimized)
    save(out/'protocol.json', cfg)
    save(out/'loops.json', dict(keyframes=len(frames), accepted_loops=len(edges), attempts=attempts,
        no_loop_max_matrix_error=control_error, no_loop_graph=identity_stats, graph=graph_stats,
        baseline_sha256=digest(trajectory), corrected_sha256=digest(out/'corrected_imu_tum.txt'),
        protocol_sha256=digest(Path(__file__).with_name('protocol.json')), wall_seconds=time.monotonic()-t0,
        libraries=dict(open3d=o3d.__version__, gtsam='4.2'), ground_truth_used=False))
    print('BACKEND_COMPLETE', len(edges), 'loops', graph_stats, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    run(a.root, a.out)
