"""Evaluate the frozen output; no evaluator feedback goes into loop selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from .common import REPO, S, Scans, interpolate, matrices, protocol, save, transform, check_time_tables


def align(source, target):
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source-a).T@(target-b)/len(source))
    d = np.eye(3)
    d[2, 2] = np.linalg.det(vt.T@u.T)
    rot = vt.T@d@u.T
    return source@rot.T+b-rot@a


def voxel_sample(points, size):
    _, indices = np.unique(np.floor(points/size).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(indices)]


def geometry(scans, cfg, center):
    groups = []
    for group in ['early_frames', 'late_frames']:
        pieces = []
        for f in cfg[group]:
            pts, order, times = scans.read(f)
            raw = np.fromfile(scans.root/f'evaluation/labels/{f:06d}.label', np.uint32)&0xFFFF
            valid = scans.traj.valid(times)
            if valid.sum() < 100:
                continue
            world, idx = scans.world(pts[order, :3], times, valid)
            selected = np.isin(raw[order][idx], cfg['static_raw_labels'])
            selected &= np.linalg.norm(pts[order][idx, :3], axis=1) < cfg['range_m']
            # Shared fixed world ROI; no arm-specific ICP/alignment for this metric.
            selected &= np.linalg.norm(world[:, :2]-center[:2], axis=1) < 25.
            pieces.append(world[selected])
        groups.append(voxel_sample(np.concatenate(pieces), cfg['voxel_m']))
    early, late = groups
    assert len(early)>100 and len(late)>100, (len(early), len(late))
    e2l = cKDTree(late).query(early, workers=2)[0]
    l2e = cKDTree(early).query(late, workers=2)[0]
    both = np.r_[e2l, l2e]
    scores = {}
    for threshold in cfg['distance_thresholds_m']:
        p, r = float(np.mean(e2l<threshold)), float(np.mean(l2e<threshold))
        scores[str(threshold)] = dict(early_to_late=p, late_to_early=r, fscore=2*p*r/max(p+r, 1e-12))
    return dict(early_points=len(early), late_points=len(late),
                symmetric_median_m=float(np.median(both)), symmetric_p90_m=float(np.percentile(both, 90)),
                distances=scores, fitting_applied=False), groups


def run(root, out):
    cfg = protocol()
    paths = {'baseline':REPO/cfg['baseline_trajectory'], 'corrected':out/'corrected_imu_tum.txt'}
    scans = {arm:Scans(root, path) for arm,path in paths.items()}
    tr = np.eye(4)
    for line in (root/'evaluation/calib.txt').read_text().splitlines():
        if line.startswith('Tr:'):
            tr[:3] = np.array(list(map(float, line.split()[1:]))).reshape(3, 4)
    raw = np.loadtxt(root/'evaluation/kitti_official_07.txt').reshape(-1, 3, 4)
    gt = np.tile(np.eye(4), (len(raw), 1, 1))
    gt[:, :3] = raw
    gt = np.linalg.inv(tr)@gt@tr
    ts = scans['baseline'].images[:len(gt)]
    rel = np.loadtxt(root/'evaluation/times.txt')
    clock_check = check_time_tables(rel, ts)
    result = dict(reference=cfg['evaluation_reference'], time_association=clock_check,
                  trajectory={}, revisit_geometry={}, semantics={})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    center_pose, _ = scans['baseline'].traj.query_one(ts[10])
    center = (center_pose@S.T_I_L)[:3, 3]
    geometry_fig, ga = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for arm, color in [('baseline', '#d75a44'), ('corrected', '#277eb6')]:
        tum = np.loadtxt(paths[arm])
        valid = (tum[:, 0]>=ts[0]) & (tum[:, 0]<=ts[-1])
        tum = tum[valid]
        truth = interpolate(ts, gt, tum[:, 0])
        estimated = matrices(tum)@S.T_I_L
        positions = align(estimated[:, :3, 3], truth[:, :3, 3])
        error = np.linalg.norm(positions-truth[:, :3, 3], axis=1)
        # Fixed one-second RPE, using full relative SE(3), no alignment needed.
        future_times = tum[:, 0]+1.
        rv = future_times<=tum[-1, 0]
        future_est = interpolate(tum[:, 0], estimated, future_times[rv])
        future_gt = interpolate(ts, gt, future_times[rv])
        re = np.linalg.inv(estimated[rv])@future_est
        rt = np.linalg.inv(truth[rv])@future_gt
        rpe = np.linalg.norm((np.linalg.inv(rt)@re)[:, :3, 3], axis=1)
        stats = dict(pairs=len(error), ate_rmse_m=float(np.sqrt(np.mean(error**2))),
                     ate_median_m=float(np.median(error)), ate_max_m=float(error.max()),
                     rpe_1s_translation_rmse_m=float(np.sqrt(np.mean(rpe**2))))
        result['trajectory'][arm] = stats
        if arm == 'baseline':
            axes[0].plot(truth[:, 0, 3], truth[:, 1, 3], 'k-', lw=1.2, label='KITTI official GT')
        axes[0].plot(positions[:, 0], positions[:, 1], color=color, lw=1, label=f'{arm}: {stats["ate_rmse_m"]:.3f} m')
        axes[1].plot(tum[:, 0]-tum[0, 0], error, color=color, label=arm)
        geo, groups = geometry(scans[arm], cfg['geometry_evaluation'], center)
        result['revisit_geometry'][arm] = geo
        ax = ga[0 if arm == 'baseline' else 1]
        for pts, c, label in zip(groups, ['#277eb6', '#e59a30'], ['early pass', 'late pass']):
            ax.scatter(pts[:, 0]-center[0], pts[:, 1]-center[1], s=.15, c=c, label=label, rasterized=True)
        ax.set_title(f'{arm}: held-out static revisits\nF@0.2m = {geo["distances"]["0.2"]["fscore"]:.3f}')
        ax.set_aspect('equal')
        ax.set_xlabel('world x relative to start [m]')
        ax.set_ylabel('world y relative to start [m]')
        ax.legend(markerscale=8)
        semantic_path = out/f'{arm}_semantics.json'
        if semantic_path.exists():
            data = json.loads(semantic_path.read_text())
            result['semantics'][arm] = dict(map_voxels=data['map_voxels'],
                outside_scan=data['offline_all']['outside'], outside_map=data['map_all_lookup']['outside'],
                outside_map_complete=data['map_all_complete']['outside'])
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    axes[0].set_title('Same frozen frontend; one SE(3) alignment per arm')
    axes[1].set_xlabel('time [s]')
    axes[1].set_ylabel('position error [m]')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/'trajectory_comparison.png', dpi=170)
    geometry_fig.tight_layout()
    geometry_fig.savefig(out/'heldout_revisit_comparison.png', dpi=170)
    plt.close('all')
    if len(result['semantics']) == 2:
        a, b = result['semantics']['baseline'], result['semantics']['corrected']
        assert a['outside_scan'] == b['outside_scan'], 'Fixed network predictions must score identically'
        result['fixed_predictions_check'] = True
    save(out/'comparison.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'semantics'}, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    run(a.root, a.out)
