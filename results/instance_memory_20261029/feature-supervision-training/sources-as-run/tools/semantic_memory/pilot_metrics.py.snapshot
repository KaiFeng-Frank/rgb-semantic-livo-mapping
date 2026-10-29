"""Paired point metrics and GT-defined object-observation event diagnostics.

Boxes/instance IDs are evaluation-only. Visibility means >=5 correctly assigned
GT semantic points inside the annotated box; a gap is NOT proof of occlusion.
"""
from collections import defaultdict
import numpy as np


def confusion(labels, prediction, classes=16):
    valid = labels >= 0
    return np.bincount(labels[valid] * classes + prediction[valid], minlength=classes**2).reshape(classes, classes)


def scores(matrix):
    matrix = np.asarray(matrix)
    tp = matrix.diagonal()
    union = matrix.sum(0) + matrix.sum(1) - tp
    iou = np.divide(tp, union, out=np.zeros(len(tp), np.float64), where=union > 0)
    return dict(miou_fixed16=float(iou.mean()), miou_present=float(iou[union>0].mean()) if np.any(union>0) else None,
                accuracy=float(tp.sum()/matrix.sum()) if matrix.sum() else None,
                class_iou=[float(x) if n else None for x,n in zip(iou,union)],
                labeled_points=int(matrix.sum()), confusion=matrix.tolist())


def object_groups(xyz_world, labels, objects):
    from pyquaternion import Quaternion
    owner = np.full(len(labels), -1, np.int32)
    for j, obj in enumerate(objects):
        if obj['annotated_lidar_points'] < 5:
            continue
        r = Quaternion(obj['rotation']).rotation_matrix
        local = (xyz_world - np.asarray(obj['translation'])) @ r
        half = np.asarray(obj['size'])[[1,0,2]] / 2.
        mask = (np.abs(local) <= half+1e-5).all(1) & (labels == obj['label'])
        owner[mask] = np.where(owner[mask] == -1, j, -2)
    result = []
    for j,obj in enumerate(objects):
        indices = np.flatnonzero(owner == j)
        if len(indices) >= 5:
            result.append(dict(instance=obj['instance'], label=obj['label'], indices=indices))
    return result


def events(rows, arms):
    """Use the same observation and baseline-error episodes for every arm.

    Success: >=50% of an object's semantic GT points predicted correctly. Delays
    are observed at labeled keyframes (~0.5 s), not inference response latency.
    Failed/censored episodes stay in denominators; conditional latency is separate.
    """
    tracks = defaultdict(list)
    for row in rows:
        tracks[(row['scene'], row['instance'])].append(row)
    grouped = defaultdict(list)
    for key, track in tracks.items():
        track.sort(key=lambda r:r['frame'])
        runs = []
        for row in track:
            if not runs or row['frame'] != runs[-1][-1]['frame'] + 1:
                runs.append([])
            runs[-1].append(row)
        for j, run in enumerate(runs):
            # Scene-start tracks are left-censored, never invented as new arrivals.
            if j == 0 and run[0]['frame'] > 0:
                grouped['first_observed'].append(run)
            if j > 0:
                grouped['reappearance_after_observation_gap'].append(run)
            i = 0
            while i < len(run):
                if run[i]['correct_fraction']['baseline'] >= .5:
                    i += 1
                    continue
                start = i
                while i < len(run) and run[i]['correct_fraction']['baseline'] < .5:
                    i += 1
                # Include the first baseline-recovered frame, when present.
                grouped['baseline_error_recovery'].append(run[start:min(i+1,len(run))])
    result = {}
    for event in ('first_observed','reappearance_after_observation_gap','baseline_error_recovery'):
        episodes = grouped[event]
        result[event] = {}
        for arm in arms:
            delays, capped, immediate = [], [], 0
            for episode in episodes:
                start = episode[0]['stamp']
                success = next((r for r in episode if r['correct_fraction'][arm] >= .5), None)
                if success is not None:
                    delay = success['stamp'] - start
                    delays.append(delay)
                    immediate += delay == 0
                # Shared finite horizon: last observation plus one keyframe interval.
                horizon = episode[-1]['stamp']-start+.5
                capped.append(success['stamp']-start if success is not None else horizon)
            result[event][arm] = dict(episodes=len(episodes), successes=len(delays),
                censored=len(episodes)-len(delays), immediate=immediate,
                success_rate=len(delays)/len(episodes) if episodes else None,
                conditional_mean_delay_s=float(np.mean(delays)) if delays else None,
                mean_delay_with_censored_at_common_horizon_s=float(np.mean(capped)) if capped else None)
    return result


def paired_summary(labels, base, pred):
    valid = labels >= 0
    bc, pc = base[valid] == labels[valid], pred[valid] == labels[valid]
    return dict(rescued=int((~bc & pc).sum()), harmed=int((bc & ~pc).sum()),
                base_correct=int(bc.sum()), base_wrong=int((~bc).sum()))
