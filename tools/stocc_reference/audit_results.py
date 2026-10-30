"""Independently audit exported predictions, coverage and occupancy mIoU on CPU.

This does not call the upstream evaluator or run the neural network. Temporal
consistency remains the official model's metric, not an independently verified
instance or object-recovery score.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np


CLASSES = ['others', 'barrier', 'bicycle', 'bus', 'car',
           'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
           'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
           'terrain', 'manmade', 'vegetation', 'free']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(hist):
    intersection = np.diag(hist)
    union = hist.sum(0) + hist.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.full(18, np.nan), where=union > 0)
    return iou * 100, float(np.nanmean(iou[:17]) * 100)


def timing(rows):
    def stats(key):
        values = np.array([row[key] for row in rows])
        return dict(mean=float(values.mean()), p50=float(np.median(values)),
                    p95=float(np.percentile(values, 95)), maximum=float(values.max()))
    return dict(frames=len(rows), model_ms=stats('model_ms'), load_ms=stats('load_ms'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--expected-frames', type=int, default=6019)
    parser.add_argument('--expected-scenes', type=int, default=150)
    args = parser.parse_args()
    start = time.time()
    summary = json.loads((args.run/'summary.json').read_text())
    assert summary['status'] == 'complete'
    rows = [json.loads(line) for line in (args.run/'frames.jsonl').read_text().splitlines()]
    assert len(rows) == summary['frames'] == args.expected_frames
    with args.metadata.open('rb') as stream:
        infos = sorted(pickle.load(stream)['infos'], key=lambda info: info['timestamp'])
    assert len(infos) == len(rows)
    expected = {info['token']: info for info in infos}
    assert len(expected) == len(infos)
    assert len({row['index'] for row in rows}) == len(rows)
    assert {row['token'] for row in rows} == set(expected)
    assert {p.stem for p in (args.run/'predictions').glob('*.npz')} == set(expected)
    scene_counts = Counter(info['scene_name'] for info in infos)
    assert len(scene_counts) == args.expected_scenes
    hist = np.zeros((18, 18), dtype=np.int64)
    by_scene = {scene: np.zeros_like(hist) for scene in scene_counts}
    last = {}
    for count, row in enumerate(rows, 1):
        info = expected[row['token']]
        assert infos[row['index']]['token'] == row['token']
        assert info['scene_name'] == row['scene']
        assert info['global_idx'] == row['global_idx'] == last.get(row['scene'], -1) + 1
        last[row['scene']] = row['global_idx']
        with np.load(args.run/'predictions'/f"{row['token']}.npz") as saved:
            pred = saved['prediction']
        assert pred.shape == (200, 200, 16) and pred.dtype == np.uint8
        assert int(pred.max()) < 18
        assert hashlib.sha256(pred.tobytes()).hexdigest() == row['prediction_sha256']
        with np.load(args.data_root/'gts'/row['scene']/row['token']/'labels.npz') as label:
            mask = label['mask_camera'].astype(bool)
            gt = label['semantics'][mask].astype(np.int64)
        predicted = pred[mask].astype(np.int64)
        assert gt.min() >= 0 and gt.max() < 18
        frame_hist = np.bincount(18 * gt + predicted, minlength=18 * 18).reshape(18, 18)
        hist += frame_hist
        by_scene[row['scene']] += frame_hist
        if count % 500 == 0:
            print('audited_predictions', count, flush=True)
    assert {scene: index + 1 for scene, index in last.items()} == dict(scene_counts)
    iou, mean = scores(hist)
    official = summary['metrics']
    assert abs(mean - official['Overall']) <= .0050001, (mean, official['Overall'])
    for index, name in enumerate(CLASSES[:17]):
        if np.isfinite(iou[index]):
            assert abs(iou[index] - official[name]) <= .0050001, (name, iou[index], official[name])
        else:
            assert official[name] is None or not np.isfinite(official[name])
    per_scene = {scene: dict(frames=scene_counts[scene], mIoU=scores(h)[1],
                             confusion=h.tolist()) for scene, h in sorted(by_scene.items())}
    result = dict(
        completed=time.time(), elapsed_seconds=time.time()-start,
        frames=len(rows), scenes=len(scene_counts), complete_scene_coverage=True,
        all_prediction_hashes_verified=True, prediction_dtype='uint8',
        prediction_shape=[200, 200, 16], camera_mask=True, free_class_excluded_from_mean=True,
        class_names=CLASSES, confusion=hist.tolist(),
        class_iou_percent=[float(v) if np.isfinite(v) else None for v in iou],
        mIoU_percent=mean, official_rounded_mIoU_percent=official['Overall'],
        official_rounded_class_scores_match=True,
        official_mSTCV_percent=100 * official['mSTCV'],
        mSTCV_independently_recomputed=False,
        timing_all=timing(rows),
        timing_scene_first_frames=timing([row for row in rows if row['global_idx'] == 0]),
        timing_non_first_frames=timing([row for row in rows if row['global_idx'] > 0]),
        official_run_elapsed_seconds=summary['elapsed_seconds'],
        throughput_including_export_and_final_evaluation=len(rows)/summary['elapsed_seconds'],
        peak_allocated_bytes=summary['peak_allocated_bytes'],
        peak_reserved_bytes=summary['peak_reserved_bytes'],
        summary_sha256=sha(args.run/'summary.json'),
        metadata_sha256=sha(args.metadata), frames_sha256=sha(args.run/'frames.jsonl'),
        audit_source_sha256=sha(Path(__file__)), per_scene=per_scene,
        interpretation='Occupancy validation; not an instance-consistency or memory-causal-gain experiment.')
    (args.run/'independent-audit.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_scene', 'confusion']}), flush=True)


if __name__ == '__main__':
    main()
