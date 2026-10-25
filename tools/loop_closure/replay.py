"""Two-pass, bounded-memory replay of unchanged B0 predictions into a fresh map.

Uses the original sem_core fusion and map_eval deskew/gating contract. A second
streaming pass queries the final map; no per-point history is kept in memory.
The legacy metric excludes partially out-of-trajectory points; this omission is
reported explicitly and an additional complete-coverage reading counts them as
abstentions. No RGB is fetched, so the saved map is semantic, not RGB textured.
"""
import argparse
import gc
from pathlib import Path
import time

import numpy as np

from .common import REPO, S, Scans, digest, protocol, save
from kitti_calib import KittiCalib
import label_spaces as LS

SK = np.asarray(LS.sk_lut(), np.int32)
NU = np.asarray(LS.nusc_lut(), np.int32)
K = len(LS.COARSE)


class Counts:
    def __init__(self):
        # Last two columns: prediction outside vocabulary, no map answer.
        self.counts = np.zeros((K, K+2), np.int64)

    def add(self, gt, pred):
        valid = gt >= 0
        gt, pred = gt[valid], pred[valid]
        col = np.where(pred == -2, K, np.where(pred == -1, K+1, pred))
        self.counts += np.bincount(gt*(K+2)+col, minlength=K*(K+2)).reshape(K, K+2)

    def metrics(self):
        c, a = self.counts[:, :K+1], self.counts[:, K+1]
        row, col, diag = c.sum(1), c[:, :K].sum(0), np.diag(c[:, :K])
        present = row+a > 0
        union = row+col-diag
        iou = np.divide(diag*100., union, out=np.zeros(K), where=union>0)
        strict = np.divide(diag*100., union+a, out=np.zeros(K), where=union+a>0)
        n = int(self.counts.sum())
        return dict(n_eval=n, n_labelled=int(c.sum()), n_abstained=int(a.sum()),
                    coverage=float(c.sum()/max(1, n)), miou9=float(iou[present].mean()),
                    miou9_abstain_wrong=float(strict[present].mean()),
                    per_class_iou={name:float(iou[k]) for k, name in enumerate(LS.COARSE) if present[k]},
                    confusion_gt_pred_unmapped_abstain=self.counts.tolist())


def subsets():
    return {s:Counts() for s in ['outside', 'frustum', 'global']}


def add_subsets(c, gt, pred, frustum):
    c['outside'].add(gt[~frustum], pred[~frustum])
    c['frustum'].add(gt[frustum], pred[frustum])
    c['global'].add(gt, pred)


def prepare(scans, f, calibration, cfg):
    pts, order, times = scans.read(f)
    label_path = scans.root/f'evaluation/labels/{f:06d}.label'
    gt = SK[np.fromfile(label_path, np.uint32) & 0xFFFF][order]
    pred_path = scans.root / f'out/v06_score/pred/armB0_s1_07_r1/f{f:06d}.npz'
    with np.load(pred_path) as z:
        lab, conf = z['lab'][order].astype(np.int32), z['conf'][order].astype(np.float32)
    assert len(lab) == len(pts) == len(gt), f
    # Exact P3 arithmetic from the frozen Projector.
    hom = np.column_stack([pts[:, :3].astype(np.float64), np.ones(len(pts))])
    uvw = hom @ (calibration.P_rect_02 @ calibration.T_velo_rect0).T
    depth = uvw[:, 2]
    uv = np.full((len(pts), 2), np.nan)
    good = depth > .5
    uv[good] = uvw[good, :2]/depth[good, None]
    w, h = calibration.img_size
    inside = (good & (uv[:, 0]>=0) & (uv[:, 0]<=w-1) & (uv[:, 1]>=0) & (uv[:, 1]<=h-1))[order]
    valid = scans.traj.valid(times)
    keep = valid & (conf >= cfg['map_confidence_gate'])
    if valid.sum() < 100:
        return gt, lab, conf, inside, keep, np.empty((0, 3)), np.empty(0, int)
    if not keep.any():
        raise RuntimeError(f'No confident points at frame {f}; deskew contract has no endpoints')
    world, idx = scans.world(pts[order, :3], times, keep)
    return gt, lab, conf, inside, keep, world, idx


def run(root, out, arm):
    cfg = protocol()
    traj = REPO/cfg['baseline_trajectory'] if arm == 'baseline' else out/'corrected_imu_tum.txt'
    scans = Scans(root, traj)
    cal = KittiCalib(root/'data/raw/2011_09_30')
    vm = S.SemanticVoxelMap(voxel=cfg['map_voxel_m'], cap0=1<<22, hash_cap=1<<24, dyn=False)
    t0 = time.monotonic()
    offline = subsets()
    frames_no_pose = 0
    for f in range(cfg['frames']):
        gt, lab, conf, inside, keep, world, idx = prepare(scans, f, cal, cfg)
        add_subsets(offline, gt, NU[lab], inside)
        if not len(idx):
            frames_no_pose += 1
        else:
            selected = keep[idx]
            vm.insert(np.ascontiguousarray(world[selected]), lab[idx][selected], conf[idx][selected])
        if f % 100 == 0:
            print('FUSE', arm, f, 'voxels', vm.n, 'seconds', round(time.monotonic()-t0), flush=True)
    n = vm.n
    order = np.argsort(vm.key[:n], kind='stable')
    keys = vm.key[:n][order]
    classes = np.argmax(vm.score[:n], axis=1).astype(np.uint8)
    labels = NU[classes][order]
    xyz = (vm.xyz[:n]/np.maximum(vm.n_obs[:n], 1)[:, None]).astype(np.float32)
    np.savez_compressed(out/f'{arm}_semantic_map.npz', xyz=xyz, cls=classes,
                        key=vm.key[:n], n_obs=vm.n_obs[:n], voxel=cfg['map_voxel_m'])
    del vm, xyz, classes, order
    gc.collect()
    mapped, complete = subsets(), subsets()
    omitted = 0
    for f in range(cfg['frames']):
        gt, lab, conf, inside, keep, world, idx = prepare(scans, f, cal, cfg)
        answers = np.full(len(gt), -1, np.int32)
        if len(idx):
            query = S.voxel_key(world, cfg['map_voxel_m'])
            rows = np.minimum(np.searchsorted(keys, query), n-1)
            hit = keys[rows] == query
            answers[idx[hit]] = labels[rows[hit]]
            # Match archived map_eval's exact denominator for historical parity.
            add_subsets(mapped, gt[idx], answers[idx], inside[idx])
            omitted += int((gt>=0).sum()-(gt[idx]>=0).sum())
        else:
            add_subsets(mapped, gt, answers, inside)
        add_subsets(complete, gt, answers, inside)
        if f % 200 == 0:
            print('SCORE', arm, f, flush=True)
    result = dict(arm=arm, frames=cfg['frames'], frames_no_pose=frames_no_pose,
                  map_voxels=n, trajectory_sha256=digest(traj), wall_seconds=time.monotonic()-t0,
                  legacy_partial_pose_points_omitted=omitted,
                  offline_all={s:c.metrics() for s,c in offline.items()},
                  map_all_lookup={s:c.metrics() for s,c in mapped.items()},
                  map_all_complete={s:c.metrics() for s,c in complete.items()})
    save(out/f'{arm}_semantics.json', result)
    if arm == 'baseline':
        import json
        historical = next(r for r in json.loads((REPO/'results/v06/scoring/runs_summary.json').read_text())['runs']
                          if r['tag'] == 'armB0_s1_07_r1')
        checks = dict(voxel_count_matches=n == historical['map_voxels'],
            offline_miou_difference=result['offline_all']['outside']['miou9']-historical['outside']['offline_all']['miou9'],
            map_miou_difference=result['map_all_lookup']['outside']['miou9']-historical['outside']['map_all_lookup']['miou9'],
            map_n_eval_matches=result['map_all_lookup']['outside']['n_eval']==historical['outside']['map_all_lookup']['n_eval'])
        checks['passed'] = bool(checks['voxel_count_matches'] and checks['map_n_eval_matches']
            and abs(checks['offline_miou_difference'])<1e-8 and abs(checks['map_miou_difference'])<1e-8)
        save(out/'baseline-parity.json', checks)
        assert checks['passed'], checks
    print('REPLAY_COMPLETE', arm, result['map_all_lookup']['outside']['miou9'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--arm', choices=['baseline', 'corrected'], required=True)
    a = ap.parse_args()
    run(a.root, a.out, a.arm)
