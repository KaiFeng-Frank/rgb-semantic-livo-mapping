"""Build paired loop measurements and pose graphs without evaluation GT."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from tools.loop_closure.common import REPO, S, Scans, save, digest, rigid_pose, corrected_tum
from tools.loop_closure.backend import cloud, register as original_register, optimize
from .core import register, temporal_evidence, guard, transform


def configuration():
    return json.loads(Path(__file__).with_name('protocol.json').read_text())


def geometry_configuration(cfg):
    return json.loads((REPO/cfg['geometric_protocol']).read_text())


def aggregate(points, evidence, voxel):
    """Attach averaged evidence to EXACT original Open3D centroid sampling."""
    original = cloud(points, voxel)
    dense = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    traced, _, members = dense.voxel_down_sample_and_trace(
        voxel, dense.get_min_bound()-voxel*.5, dense.get_max_bound()+voxel*.5, False)
    xyz = np.asarray(original.points).copy()
    error, order = cKDTree(np.asarray(traced.points)).query(xyz, workers=1)
    if len(xyz) != len(members) or len(np.unique(order)) != len(order) or error.max() > 1e-9:
        raise ValueError('Evidence downsampling differs from geometric baseline')
    mean = np.stack([np.mean(evidence[np.asarray(members[i])], axis=0) for i in order])
    label = mean.argmax(axis=1).astype(np.int32)
    confidence = mean.max(axis=1)
    label[confidence == 0] = -1
    return dict(xyz=xyz, evidence=mean, label=label, confidence=confidence)


def raw_scan(scans, frame):
    pts, order, times = scans.read(frame)
    valid = scans.traj.valid(times)
    world, idx = scans.world(pts[order, :3], times, valid)
    pose, ok = scans.traj.query_one(scans.images[frame])
    if not ok:
        raise ValueError('Missing pose')
    pose = rigid_pose(pose @ S.T_I_L)
    xyz = transform(world, np.linalg.inv(pose))
    keep = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz[:, :2], axis=1)<80)
    # One indexing chain for coordinates and both prediction sources.
    return xyz[keep], order[idx][keep], len(pts), pose


def scan_features(root, scans, frame, model, voxel):
    xyz, raw_idx, n, pose = raw_scan(scans, frame)
    pred = root/f'out/v06_score/pred/{model}_07_r1/f{frame:06d}.npz'
    with np.load(pred) as z:
        labels, confidence = z['lab'], z['conf']
        if len(labels) != n or len(confidence) != n:
            raise ValueError('Prediction/scan point count mismatch')
        labels, confidence = labels[raw_idx].astype(np.int32), confidence[raw_idx].astype(np.float64)
    if not np.isfinite(confidence).all() or np.any((confidence<0) | (confidence>1)):
        raise ValueError('Invalid confidence')
    evidence = np.zeros((len(xyz), 16))
    good = (labels>=0) & (labels<16)
    evidence[np.flatnonzero(good), labels[good]] = confidence[good]
    result = aggregate(xyz, evidence, voxel)
    result['pose'], result['frame'] = pose, frame
    return result


def in_coordinates(features, pose):
    result = dict(features)
    result['xyz'] = transform(features['xyz'], np.linalg.inv(pose) @ features['pose'])
    return result


def features(root, scans, model, cfg, geo):
    frames = cfg['target_frames']+cfg['source_history_frames']+[cfg['current_frame']]
    if len(set(frames)) != len(frames) or max(cfg['target_frames']) >= min(cfg['source_history_frames']):
        raise ValueError('Historical reference and source history must be disjoint')
    if max(cfg['source_history_frames']) >= cfg['current_frame']:
        raise ValueError('Source history must be causal')
    scans_f = {f:scan_features(root, scans, f, model, geo['icp_voxel_m']) for f in frames}
    source = scans_f[cfg['current_frame']]
    target_pose = scans_f[cfg['historical_frame']]['pose']
    historical = [in_coordinates(scans_f[f], target_pose) for f in cfg['target_frames']]
    target = aggregate(np.concatenate([f['xyz'] for f in historical]),
                       np.concatenate([f['evidence'] for f in historical]), geo['icp_voxel_m'])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target['xyz']))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.2, max_nn=30))
    target['normal'] = np.asarray(pc.normals).copy()
    target['temporal_count'], target['temporal_stability'] = temporal_evidence(target, historical, cfg)
    source_history = [in_coordinates(scans_f[f], source['pose']) for f in cfg['source_history_frames']]
    source['temporal_count'], source['temporal_stability'] = temporal_evidence(source, source_history, cfg)
    return source, target, np.linalg.inv(target_pose) @ source['pose']


def save_graph(out, name, measurement, frames, poses, times, tum, geo):
    i, j = frames.index(10), frames.index(1060)
    optimized, stats = optimize(poses, [(i, j, measurement)], geo)
    corrected = corrected_tum(tum, times, poses, optimized)
    path = out/f'{name}_imu_tum.txt'
    np.savetxt(path, corrected, fmt='%.10f')
    return dict(trajectory_sha256=digest(path), graph=stats)


def run(root, out):
    cfg = configuration()
    geo = geometry_configuration(cfg)
    if (out/'measurements.json').exists():
        raise ValueError('Refusing to overwrite a completed run')
    out.mkdir(parents=True, exist_ok=True)
    save(out/'protocol.json', cfg)
    save(out/'geometry-protocol.json', geo)
    source_path = REPO/cfg['source_trajectory']
    scans = Scans(root, source_path)
    frozen = json.loads((REPO/cfg['historical_results']/'loops.json').read_text())
    assert digest(source_path) == frozen['baseline_sha256']
    accepted = [x for x in frozen['attempts'] if x['accepted']]
    assert len(accepted) == 1
    assert (accepted[0]['historical_frame'], accepted[0]['current_frame']) == (cfg['historical_frame'], cfg['current_frame'])
    frames = [f for f in range(geo['keyframe_stride'], geo['frames'], geo['keyframe_stride'])
              if scans.traj.valid([scans.images[f]])[0]]
    assert len(frames) == frozen['keyframes']
    times = scans.images[frames]
    poses = np.stack([rigid_pose(scans.traj.query_one(t)[0] @ S.T_I_L) for t in times])
    identity, _ = optimize(poses, [], geo)
    assert np.max(np.abs(identity-poses)) < 1e-6
    tum = np.loadtxt(source_path)
    np.savetxt(out/'baseline_imu_tum.txt', tum, fmt='%.10f')
    results = dict(ground_truth_used=False, protocol_sha256=digest(Path(__file__).with_name('protocol.json')),
                   initial_trajectory_sha256=digest(source_path), frames=frames, arms={}, models={}, checks={})
    archive = dict(keyframe_poses=poses, keyframe_times=times, keyframe_frames=frames)
    for model in cfg['models']:
        start = time.monotonic()
        source, target, initial = features(root, scans, model, cfg, geo)
        for side, feat in [('source', source), ('target', target)]:
            for field in ['label', 'confidence', 'temporal_count', 'temporal_stability']:
                archive[f'{model}_{side}_{field}'] = feat[field]
        if 'source_xyz' in archive:
            for key, array in [('source_xyz', source['xyz']), ('target_xyz', target['xyz']), ('normal', target['normal']), ('initial', initial)]:
                np.testing.assert_array_equal(archive[key], array)
        else:
            archive.update(source_xyz=source['xyz'], target_xyz=target['xyz'], normal=target['normal'], initial=initial)
            s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source['xyz']))
            t = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target['xyz']))
            historical, historical_stats = original_register(s, t, initial, geo)
            error = float(np.max(np.abs(historical-np.asarray(accepted[0]['T_historical_current']))))
            save(out/'control-progress.json', dict(historical_open3d_max_matrix_error=error,
                measurement=historical.tolist(), registration=historical_stats))
            assert error < cfg['historical_reproduction_matrix_tolerance'], error
            results['checks']['historical_open3d_max_matrix_error'] = error
            results['arms']['open3d'] = dict(measurement=historical.tolist(), registration=historical_stats,
                **save_graph(out, 'open3d', historical, frames, poses, times, tum, geo))
            baseline, stats = register(source, target, initial, 'G', cfg, geo)
            assert stats['accepted'], stats
            delta = baseline @ np.linalg.inv(historical)
            dt = float(np.linalg.norm(delta[:3, 3]))
            dr = float(np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude()))
            save(out/'control-progress.json', dict(historical_open3d_max_matrix_error=error,
                G_vs_open3d_translation_m=dt, G_vs_open3d_rotation_deg=dr,
                G_measurement=baseline.tolist(), G_registration=stats))
            assert dt < cfg['unweighted_open3d_parity_translation_m'], dt
            assert dr < cfg['unweighted_open3d_parity_rotation_deg'], dr
            results['checks']['G_vs_open3d'] = dict(translation_m=dt, rotation_deg=dr)
            results['arms']['G'] = dict(measurement=baseline.tolist(), registration=stats, fallback=False,
                **save_graph(out, 'G', baseline, frames, poses, times, tum, geo))
        for arm in cfg['arms'][1:]:
            tick = time.monotonic()
            name = f'{model}_{arm}'
            candidate, stats = register(source, target, initial, arm, cfg, geo)
            used, checks = guard(candidate, stats, baseline, cfg)
            item = dict(candidate_measurement=candidate.tolist(), measurement=used.tolist(), registration=stats,
                **checks, **save_graph(out, name, used, frames, poses, times, tum, geo))
            # Retain rejected candidate trajectories too; no selection on GT.
            item['candidate_graph'] = save_graph(out, name+'_candidate', candidate, frames, poses, times, tum, geo)
            item['seconds'] = time.monotonic()-tick
            results['arms'][name] = item
            print('ARM_COMPLETE', name, 'fallback', item['fallback'], 'seconds', item['seconds'], flush=True)
        results['models'][model] = dict(seconds=time.monotonic()-start,
            source_points=len(source['xyz']), target_points=len(target['xyz']),
            source_temporal_supported=int(np.sum(source['temporal_count']>=2)),
            target_temporal_supported=int(np.sum(target['temporal_count']>=2)))
    results['checks']['identical_geometry_between_models'] = True
    np.savez_compressed(out/'registration-inputs.npz', **archive)
    results['registration_inputs_sha256'] = digest(out/'registration-inputs.npz')
    save(out/'measurements.json', results)
    print('ALL_TRAJECTORIES_FROZEN_BEFORE_SCORING', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    run(args.root, args.out)
