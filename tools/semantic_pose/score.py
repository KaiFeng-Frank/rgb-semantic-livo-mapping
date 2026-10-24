"""Independent evaluator: opens GT only after every measurement is frozen."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from tools.loop_closure.common import REPO, S, Scans, interpolate, matrices, save, digest, check_time_tables
from tools.loop_closure.evaluate import align, geometry


def trajectory_metrics(tum, times, gt):
    truth = interpolate(times, gt, tum[:, 0])
    estimated = matrices(tum) @ S.T_I_L
    positions = align(estimated[:, :3, 3], truth[:, :3, 3])
    error = np.linalg.norm(positions-truth[:, :3, 3], axis=1)
    future = tum[:, 0]+1.
    keep = future <= tum[-1, 0]
    re = np.linalg.inv(estimated[keep]) @ interpolate(tum[:, 0], estimated, future[keep])
    rt = np.linalg.inv(truth[keep]) @ interpolate(times, gt, future[keep])
    difference = np.linalg.inv(rt) @ re
    trans = np.linalg.norm(difference[:, :3, 3], axis=1)
    rot = np.rad2deg(Rotation.from_matrix(difference[:, :3, :3]).magnitude())
    return dict(pairs=len(tum), ate_rmse_m=float(np.sqrt(np.mean(error**2))),
                ate_median_m=float(np.median(error)), ate_max_m=float(error.max()),
                rpe_pairs=int(keep.sum()), rpe_1s_translation_rmse_m=float(np.sqrt(np.mean(trans**2))),
                rpe_1s_rotation_rmse_deg=float(np.sqrt(np.mean(rot**2)))), positions, error


def relative_error(estimated, truth):
    error = np.linalg.inv(truth) @ estimated
    return dict(translation_m=float(np.linalg.norm(error[:3, 3])),
                rotation_deg=float(np.rad2deg(Rotation.from_matrix(error[:3, :3]).magnitude())))


def score(root, out):
    cfg = json.loads((out/'protocol.json').read_text())
    assert cfg == json.loads(Path(__file__).with_name('protocol.json').read_text())
    geo = json.loads((out/'geometry-protocol.json').read_text())
    assert geo == json.loads((REPO/cfg['geometric_protocol']).read_text())
    measured = json.loads((out/'measurements.json').read_text())
    assert digest(Path(__file__).with_name('protocol.json')) == measured['protocol_sha256']
    assert digest(out/'registration-inputs.npz') == measured['registration_inputs_sha256']
    for name, item in measured['arms'].items():
        assert digest(out/f'{name}_imu_tum.txt') == item['trajectory_sha256']
        if 'candidate_graph' in item:
            assert digest(out/f'{name}_candidate_imu_tum.txt') == item['candidate_graph']['trajectory_sha256']
    # All outputs above verified before the first GT read below.
    tr = np.eye(4)
    for line in (root/'evaluation/calib.txt').read_text().splitlines():
        if line.startswith('Tr:'):
            tr[:3] = np.array(list(map(float, line.split()[1:]))).reshape(3, 4)
    raw = np.loadtxt(root/'evaluation/kitti_official_07.txt').reshape(-1, 3, 4)
    gt = np.tile(np.eye(4), (len(raw), 1, 1))
    gt[:, :3] = raw
    gt = np.linalg.inv(tr) @ gt @ tr
    baseline_scans = Scans(root, out/'baseline_imu_tum.txt')
    times = baseline_scans.images[:len(gt)]
    clock = check_time_tables(np.loadtxt(root/'evaluation/times.txt'), times)
    reference = np.linalg.inv(gt[cfg['historical_frame']]) @ gt[cfg['current_frame']]
    center_pose, _ = baseline_scans.traj.query_one(times[cfg['historical_frame']])
    center = (center_pose @ S.T_I_L)[:3, 3]
    baseline = np.loadtxt(out/'baseline_imu_tum.txt')
    support = (baseline[:, 0]>=times[0]) & (baseline[:, 0]<=times[-1])
    query_times = baseline[support, 0]
    report = dict(reference=cfg['evaluation'], time_association=clock, arms={}, candidate_trajectories={}, checks={})
    plotted = {}
    names = ['baseline']+list(measured['arms'])
    for name in names:
        path = out/f'{name}_imu_tum.txt'
        tum = np.loadtxt(path)
        np.testing.assert_array_equal(tum[:, 0], baseline[:, 0])
        metrics, pos, error = trajectory_metrics(tum[support], times, gt)
        if name == 'baseline':
            with np.load(out/'registration-inputs.npz') as data:
                measurement = data['initial']
        else:
            measurement = np.asarray(measured['arms'][name]['measurement'])
        metrics['loop_error'] = relative_error(measurement, reference)
        metrics['revisit_geometry'], _ = geometry(Scans(root, path), geo['geometry_evaluation'], center)
        metrics['fallback'] = measured['arms'].get(name, {}).get('fallback', False)
        report['arms'][name] = metrics
        plotted[name] = pos, error
        if name in measured['arms'] and 'candidate_graph' in measured['arms'][name]:
            item = measured['arms'][name]
            candidate = np.loadtxt(out/f'{name}_candidate_imu_tum.txt')
            np.testing.assert_array_equal(candidate[:, 0], baseline[:, 0])
            cm, _, _ = trajectory_metrics(candidate[support], times, gt)
            cm['loop_error'] = relative_error(np.asarray(item['candidate_measurement']), reference)
            report['candidate_trajectories'][name] = cm
        print('SCORED', name, metrics['ate_rmse_m'], metrics['loop_error'], flush=True)
    old = json.loads((REPO/cfg['historical_results']/'comparison.json').read_text())
    for a, b in [('baseline', 'baseline'), ('open3d', 'corrected')]:
        for field in ['ate_rmse_m', 'rpe_1s_translation_rmse_m']:
            assert abs(report['arms'][a][field]-old['trajectory'][b][field])<1e-5
    report['checks'].update(historical_scores_reproduced=True, identical_timestamps=True,
                           registration_outputs_verified_before_GT=True)
    g = report['arms']['G']
    report['delta_vs_G'] = {k:dict(ate_m=v['ate_rmse_m']-g['ate_rmse_m'],
        rpe_translation_m=v['rpe_1s_translation_rmse_m']-g['rpe_1s_translation_rmse_m'],
        rpe_rotation_deg=v['rpe_1s_rotation_rmse_deg']-g['rpe_1s_rotation_rmse_deg'],
        loop_translation_m=v['loop_error']['translation_m']-g['loop_error']['translation_m'],
        loop_rotation_deg=v['loop_error']['rotation_deg']-g['loop_error']['rotation_deg'])
        for k,v in report['arms'].items() if k not in ('baseline', 'open3d', 'G')}
    save(out/'comparison.json', report)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    displayed = list(report['arms'])
    y = np.arange(len(displayed))
    for ax, title, values in [
        (axes[0], 'Full trajectory ATE [m], lower is better', [report['arms'][n]['ate_rmse_m'] for n in displayed]),
        (axes[1], 'Loop translation error [m]', [report['arms'][n]['loop_error']['translation_m'] for n in displayed]),
        (axes[2], 'Loop rotation error [deg]', [report['arms'][n]['loop_error']['rotation_deg'] for n in displayed])]:
        ax.barh(y, values, color=['#999999' if n in ('baseline', 'open3d', 'G') else '#3079aa' if n.startswith('ZS') else '#bb6d36' for n in displayed])
        ax.set_yticks(y, displayed)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, max(values)*1.23)
        for i, value in enumerate(values):
            ax.text(value+max(values)*.015, i, f'{value:.4f}', va='center', fontsize=9)
    fig.suptitle('Frozen seq07 loop: paired semantic contribution, no GT tuning', fontsize=13)
    fig.tight_layout()
    fig.savefig(out/'pose-comparison.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    for name in ['G', 'ZS_T', 'armB0_s1_T']:
        ax.plot(query_times-query_times[0], plotted[name][1], label=name, alpha=.85)
    ax.set(xlabel='Time [s]', ylabel='Aligned position error [m]', title='Every trajectory sample, same SE(3)-alignment protocol')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/'error-over-time.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    score(a.root, a.out)
