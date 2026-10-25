"""Full-scan, 19-class SemanticKITTI evaluation. Default: complete sequence 08."""
import argparse
from pathlib import Path

import numpy as np

from .schema import FRAME_COUNTS, VALIDATION_SEQUENCES, NAMES, raw_to_learning, save, verify_vendor


class Confusion:
    def __init__(self):
        self.matrix = np.zeros((20, 20), dtype=np.int64)  # prediction x ground truth

    def add(self, prediction, target):
        p, t = np.asarray(prediction), np.asarray(target)
        if p.ndim != 1 or p.shape != t.shape or p.dtype.kind not in 'iu' or t.dtype.kind not in 'iu':
            raise ValueError('Point count, shape or integer dtype mismatch')
        if np.any((p < 0) | (p >= 20)) or np.any((t < 0) | (t >= 20)):
            raise ValueError('Learning IDs must be in 0..19')
        self.matrix += np.bincount(p.astype(np.int64) * 20 + t, minlength=400).reshape(20, 20)

    def report(self):
        c = self.matrix.copy()
        c[:, 0] = 0  # Only ignored GROUND TRUTH is removed. Prediction 0 is retained.
        tp = np.diag(c)
        fp = c.sum(1) - tp
        fn = c.sum(0) - tp
        iou = tp / (tp + fp + fn + 1e-15)
        count = int(c.sum())
        return dict(
            miou19_percent=float(iou[1:].mean() * 100),
            evaluated_points=count, ignored_gt_points=int(self.matrix[:, 0].sum()),
            abstentions_on_evaluated_points=int(c[0].sum()),
            point_accuracy_including_abstentions_percent=float(100 * tp[1:].sum() / max(1, count)),
            class_iou_percent={n: float(iou[i] * 100) for i, n in enumerate(NAMES, 1)},
            class_gt_points={n: int(c[:, i].sum()) for i, n in enumerate(NAMES, 1)},
            confusion_prediction_rows_gt_columns=self.matrix.tolist(),
            metric='Official 19-class mIoU, full scan, no confidence/range/FOV filtering',
            denominator='All 19 classes, including absent/unpredicted classes; '
                        'only official ignored ground-truth IDs are excluded')


def evaluate(dataset, predictions, sequence='08'):
    if sequence not in VALIDATION_SEQUENCES:
        raise ValueError('Local validation is restricted to 07/08/09, never hidden test labels')
    expected = [f'{i:06d}' for i in range(FRAME_COUNTS[sequence])]
    base = Path(dataset) / 'sequences' / sequence
    pred_dir = Path(predictions) / 'sequences' / sequence / 'predictions'
    for folder, suffix in [(base / 'velodyne', '.bin'), (base / 'labels', '.label'), (pred_dir, '.label')]:
        actual = sorted(p.stem for p in folder.glob('*' + suffix))
        if actual != expected:
            raise ValueError(f'Incomplete or extra frames in {folder}: {len(actual)} vs {len(expected)}')
    accum = Confusion()
    for frame in expected:
        nbytes = (base / 'velodyne' / (frame + '.bin')).stat().st_size
        if nbytes == 0 or nbytes % 16:
            raise ValueError('Invalid scan size: ' + frame)
        arrays = []
        for folder in [base / 'labels', pred_dir]:
            p = folder / (frame + '.label')
            if p.stat().st_size != nbytes // 4:
                raise ValueError('Label/scan size mismatch: ' + str(p))
            arrays.append(np.fromfile(p, dtype='<u4'))
        accum.add(raw_to_learning(arrays[1], prediction=True), raw_to_learning(arrays[0]))
    result = accum.report()
    result.update(sequence=sequence, frames=len(expected),
                  role='official_validation' if sequence == '08' else 'public_sequence_diagnostic',
                  leaderboard_test_score=False, official_source=verify_vendor()['commit'])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--sequence', choices=VALIDATION_SEQUENCES, default='08')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    result = evaluate(a.dataset, a.predictions, a.sequence)
    save(a.out, result)
    print(f"{a.sequence}: full-scan mIoU-19 = {result['miou19_percent']:.4f}%")


if __name__ == '__main__':
    main()
