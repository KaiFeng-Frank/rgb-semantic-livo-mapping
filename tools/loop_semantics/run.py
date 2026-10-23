"""Paired offline semantic refinement with immutable geometry and observations."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tools.loop_closure.common import REPO, S, Scans, digest, save
from tools.loop_closure.replay import Counts, NU, SK, K
from kitti_calib import KittiCalib
import label_spaces as LS
from .fusion import Visit, propose, apply_proposal

HERE = Path(__file__).resolve().parent


def configuration():
    return json.loads((HERE / 'protocol.json').read_text())


def heldout(frame, cfg):
    return frame % cfg['heldout_stride'] == cfg['heldout_remainder']


def read_observation(scans, frame, cfg):
    """No GT access. Preserve original point ordering and deskew arithmetic."""
    pts, order, times = scans.read(frame)
    path = scans.root / f"out/v06_score/pred/{cfg['prediction_arm']}/f{frame:06d}.npz"
    with np.load(path) as z:
        label = z['lab'][order].astype(np.int32)
        confidence = z['conf'][order].astype(np.float32)
    if len(label) != len(pts) or len(confidence) != len(pts):
        raise ValueError('Archived prediction point count differs from scan')
    valid = scans.traj.valid(times)
    keep = valid & (confidence >= cfg['confidence_gate'])
    if valid.sum() < 100:
        world, idx = np.empty((0, 3)), np.empty(0, np.int64)
    else:
        if not keep.any():
            raise ValueError('Missing confidence-gate endpoints for the frozen deskew contract')
        world, idx = scans.world(pts[order, :3], times, keep)
    return dict(pts=pts, order=order, label=label, confidence=confidence, world=world,
                idx=idx, keep=keep, ranges=np.linalg.norm(pts[order, :3], axis=1))


def setup(root, cfg):
    source = REPO / cfg['geometry_source']
    trajectory = source / 'corrected_imu_tum.txt'
    if digest(trajectory) != cfg['corrected_trajectory_sha256']:
        raise ValueError('Fixed geometry changed')
    manifest = {r['name']: r for r in json.loads((source / 'manifest.json').read_text())['assets']}
    if digest(source / 'loops.json') != manifest['loops.json']['sha256']:
        raise ValueError('Archived loop decisions changed')
    loops = [r for r in json.loads((source / 'loops.json').read_text())['attempts'] if r['accepted']]
    # This feasibility protocol is explicitly scoped to the one verified seq07 loop.
    # Do not silently introduce arbitrary pair ordering for overlapping loops.
    if len(loops) != 1:
        raise ValueError('This frozen experiment expects exactly one verified geometric loop')
    loop = loops[0]
    frames = [loop['historical_frame'], loop['current_frame']]
    if frames[1] - frames[0] < cfg['minimum_visit_gap_frames']:
        raise ValueError('Revisit is too close in time')
    scans = Scans(root, trajectory)
    centers = []
    for frame in frames:
        pose, ok = scans.traj.query_one(scans.images[frame])
        if not ok:
            raise ValueError('Loop endpoint outside trajectory')
        centers.append((pose @ S.T_I_L)[:3, 3])
    center = np.mean(centers, axis=0)
    half = cfg['visit_half_window_frames']
    windows = [(max(0, f-half), min(cfg['frames']-1, f+half)) for f in frames]
    if windows[0][1] >= windows[1][0]:
        raise ValueError('Visit windows overlap')
    return scans, center, windows


def roi(world, center, cfg):
    return np.square(world[:, :2] - center[:2]).sum(1) <= cfg['roi_radius_m']**2


def build_maps(root, out):
    """Build every arm before opening a GT file. Read archived predictions only."""
    cfg = configuration()
    scans, center, windows = setup(root, cfg)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'build.json').exists():
        raise ValueError('Choose a new output directory; do not overwrite scored maps')
    save(out / 'protocol.json', cfg)
    maps = {mode: S.SemanticVoxelMap(voxel=cfg['voxel_m'], cap0=1<<22, hash_cap=1<<24, dyn=False)
            for mode in ['all', 'heldout']}
    visits = {mode: [Visit(16, cfg['vote_group_frames'], cfg['range_scale_m']) for _ in windows] for mode in maps}
    included = {mode: [] for mode in maps}
    t0 = time.monotonic()
    for frame in range(cfg['frames']):
        obs = read_observation(scans, frame, cfg)
        idx = obs['idx']
        if len(idx):
            selected = obs['keep'][idx]
            xyz, label, confidence = obs['world'][selected], obs['label'][idx][selected], obs['confidence'][idx][selected]
            region = roi(xyz, center, cfg)
            ranges = obs['ranges'][idx][selected]
        for mode, vm in maps.items():
            if mode == 'heldout' and heldout(frame, cfg):
                continue
            included[mode].append(frame)
            if not len(idx):
                continue
            vm.insert(np.ascontiguousarray(xyz), label, confidence)
            for visit, (lo, hi) in zip(visits[mode], windows):
                if lo <= frame <= hi:
                    visit.add(frame, S.voxel_key(xyz[region], cfg['voxel_m']), label[region], confidence[region],
                              xyz[region], ranges[region])
        if frame % 100 == 0:
            print('FUSE', frame, {k: v.n for k, v in maps.items()}, round(time.monotonic()-t0), flush=True)
    result = dict(trajectory_sha256=cfg['corrected_trajectory_sha256'], loop_windows=windows,
                  roi_center=center.tolist(), roi_radius_m=cfg['roi_radius_m'], ground_truth_read=False,
                  fixed_prediction_arm=cfg['prediction_arm'], new_inference=False, training=False,
                  primary_endpoint=cfg['primary_endpoint'], modes={})
    for mode in list(maps):
        vm = maps.pop(mode)
        order = np.argsort(vm.key[:vm.n], kind='stable')
        keys = vm.key[:vm.n][order]
        base = vm.score[:vm.n].argmax(1).astype(np.uint8)[order]
        protected = (vm.score[:vm.n, cfg['protected_source_classes']] > 0).any(1)[order]
        xyz = (vm.xyz[:vm.n] / np.maximum(vm.n_obs[:vm.n], 1)[:, None]).astype(np.float32)[order]
        first, second = [v.finish() for v in visits[mode]]
        proposal = propose(first, second, cfg)
        labels = {arm: apply_proposal(keys, base, protected, proposal, arm)
                  for arm in cfg['arms'] if arm != 'ordinary'}
        changed = {arm: int(np.count_nonzero(value != base)) for arm, value in labels.items()}
        for arm, value in labels.items():
            if not np.array_equal(value[protected], base[protected]):
                raise AssertionError('Protected voxel labels changed')
        path = out / f'{mode}-maps.npz'
        np.savez_compressed(path, key=keys, xyz=xyz, ordinary=base, protected=protected, **labels)
        # Save evidence for all loop correspondences, not only successful changes.
        evidence = {k: v for k, v in proposal.items() if isinstance(v, np.ndarray)}
        for arm in labels:
            evidence.update({arm + '_' + k: v for k, v in proposal[arm].items()})
        np.savez_compressed(out / f'{mode}-revisit-evidence.npz', **evidence)
        result['modes'][mode] = dict(map_voxels=len(keys), input_frames=included[mode],
            protected_voxels=int(protected.sum()), shared_visit_voxels=len(proposal['keys']),
            geometry_eligible_voxels=int(proposal['eligible'].sum()), changed_voxels=changed,
            map_sha256=digest(path), geometry_identical_between_arms=True,
            coverage_identical_between_arms=True, protected_labels_identical=True)
        del vm, first, second, proposal, keys, base, xyz, labels
        gc.collect()
    result['wall_seconds'] = time.monotonic() - t0
    save(out / 'build.json', result)
    print('BUILD_COMPLETE', result['wall_seconds'], flush=True)
    return result


def strict_report(counts):
    r = counts.metrics()
    matrix = counts.counts
    true_positive = np.diag(matrix[:, :K])
    union = matrix.sum(1) + matrix[:, :K].sum(0) - true_positive
    values = np.divide(true_positive * 100., union, out=np.zeros(K), where=union > 0)
    r['miou9_fixed_classes_abstain_wrong'] = float(values.mean())
    r['class_iou_abstain_wrong'] = dict(zip(LS.COARSE, values.tolist()))
    return r


def score(root, out):
    cfg = configuration()
    build = json.loads((out / 'build.json').read_text())
    if digest(out / 'protocol.json') != digest(HERE / 'protocol.json'):
        raise ValueError('Parameters changed after map construction')
    scans, center, windows = setup(root, cfg)
    cal = KittiCalib(Path(root) / 'data/raw/2011_09_30')
    maps = {}
    accum = {}
    effects = {}
    scope_names = ['full', 'outside', 'roi', 'static_roi']
    for mode in ['all', 'heldout']:
        if digest(out / f'{mode}-maps.npz') != build['modes'][mode]['map_sha256']:
            raise ValueError('Map changed before scoring')
        with np.load(out / f'{mode}-maps.npz') as z:
            maps[mode] = {k: z[k] for k in ['key'] + cfg['arms']}
        accum[mode] = {arm: {scope: Counts() for scope in scope_names} for arm in cfg['arms']}
        effects[mode] = {arm: dict(changed_valid_points=0, corrected_points=0, damaged_points=0,
                                   wrong_to_different_wrong=0) for arm in cfg['arms'][1:]}
    scan_metrics = Counts()
    query_hash = {mode: hashlib.sha256() for mode in maps}
    t0 = time.monotonic()
    for frame in range(cfg['frames']):
        obs = read_observation(scans, frame, cfg)
        # GT appears only in this evaluation stage, after immutable maps are built.
        raw = np.fromfile(Path(root) / f'evaluation/labels/{frame:06d}.label', dtype=np.uint32)
        if len(raw) != len(obs['pts']):
            raise ValueError('GT point count differs from scan')
        gt = SK[raw & 0xffff][obs['order']]
        scan_metrics.add(gt, NU[obs['label']])
        hom = np.column_stack([obs['pts'][:, :3].astype(np.float64), np.ones(len(raw))])
        uvw = hom @ (cal.P_rect_02 @ cal.T_velo_rect0).T
        depth = uvw[:, 2]
        uv = np.full((len(raw), 2), np.nan)
        visible = depth > .5
        uv[visible] = uvw[visible, :2] / depth[visible, None]
        width, height = cal.img_size
        inside = (visible & (uv[:, 0] >= 0) & (uv[:, 0] <= width-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height-1))[obs['order']]
        region = np.zeros(len(raw), bool)
        if any(lo <= frame <= hi for lo, hi in windows):
            region[obs['idx']] = roi(obs['world'], center, cfg)
        scopes = dict(full=np.ones(len(raw), bool), outside=~inside, roi=region,
                      static_roi=region & (gt >= 4))
        query_keys = S.voxel_key(obs['world'], cfg['voxel_m'])
        for mode, mapping in maps.items():
            if mode == 'heldout' and not heldout(frame, cfg):
                continue
            query_hash[mode].update(np.array([frame, len(raw)], dtype='<i8').tobytes())
            query_hash[mode].update(query_keys.astype('<i8').tobytes())
            rows = np.minimum(np.searchsorted(mapping['key'], query_keys), len(mapping['key']) - 1)
            hit = mapping['key'][rows] == query_keys
            answers = {}
            for arm in cfg['arms']:
                prediction = np.full(len(raw), -1, np.int32)
                prediction[obs['idx'][hit]] = NU[mapping[arm][rows[hit]]]
                answers[arm] = prediction
                for scope, mask in scopes.items():
                    accum[mode][arm][scope].add(gt[mask], prediction[mask])
            valid = gt >= 0
            for arm in cfg['arms'][1:]:
                changed = valid & (answers[arm] != answers['ordinary'])
                corrected = changed & (answers[arm] == gt)
                damaged = changed & (answers['ordinary'] == gt)
                e = effects[mode][arm]
                e['changed_valid_points'] += int(changed.sum())
                e['corrected_points'] += int(corrected.sum())
                e['damaged_points'] += int(damaged.sum())
                e['wrong_to_different_wrong'] += int((changed & ~corrected & ~damaged).sum())
        if frame % 200 == 0:
            print('SCORE', frame, round(time.monotonic()-t0), flush=True)
    reports = {mode: {arm: {scope: strict_report(c) for scope, c in scopes.items()}
                      for arm, scopes in arms.items()} for mode, arms in accum.items()}
    for mode, arms in accum.items():
        for scope in scope_names:
            reference = arms['ordinary'][scope].counts
            for arm in cfg['arms'][1:]:
                counts = arms[arm][scope].counts
                if not np.array_equal(counts.sum(1), reference.sum(1)) or not np.array_equal(counts[:, -1], reference[:, -1]):
                    raise AssertionError('Scoring denominator or abstention coverage changed')
    old = json.loads((REPO / cfg['geometry_source'] / 'corrected_semantics.json').read_text())
    actual = accum['all']['ordinary']['outside'].counts
    expected = np.asarray(old['map_all_complete']['outside']['confusion_gt_pred_unmapped_abstain'])
    if not np.array_equal(actual, expected):
        raise AssertionError('Ordinary map fails exact historical confusion-matrix parity')
    metric = 'miou9_fixed_classes_abstain_wrong'
    primary = {arm: reports['heldout'][arm]['roi'][metric] for arm in cfg['arms']}
    result = dict(primary_endpoint=cfg['primary_endpoint'], primary_percent=primary,
                  delta_revisit_vs_ordinary_pp=primary['revisit_balanced']-primary['ordinary'],
                  delta_revisit_vs_group_balanced_pp=primary['revisit_balanced']-primary['group_balanced'],
                  reports=reports, effects=effects, query_point_hashes={k: v.hexdigest() for k,v in query_hash.items()},
                  ordinary_historical_confusion_parity=True, scan_predictions_unchanged=True,
                  geometry_identical=True, coverage_identical=True, scan_metric=strict_report(scan_metrics),
                  leaderboard_score=False, single_scan_result=False, new_training=False,
                  heldout_semantics_excluded_from_mapping=True, heldout_geometry_independent=False,
                  caveat='One public sequence, fixed coarse taxonomy, offline future-aware mapping; no claim of 19-class leaderboard or unseen-scene generalization.',
                  score_wall_seconds=time.monotonic()-t0)
    save(out / 'comparison.json', result)
    print('PRIMARY', json.dumps(primary), flush=True)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stage', choices=['build', 'score', 'all'], default='all')
    a = ap.parse_args()
    if a.stage in ('build', 'all'):
        build_maps(a.root, a.out)
    if a.stage in ('score', 'all'):
        score(a.root, a.out)
