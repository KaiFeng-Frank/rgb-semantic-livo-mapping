"""Revisit evidence balancing. No GT, file reads, pose estimation or learning."""
import numpy as np


def reduce_records(keys, scores, xyz, ranges):
    """Average records for each key, returning its actual record count.

    Applying this once per frame and once per temporal group removes point
    density and frame-count multiplication within each vote group.
    """
    keys = np.asarray(keys, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    ranges = np.asarray(ranges, dtype=np.float64)
    if scores.ndim != 2 or xyz.shape != (len(keys), 3) or ranges.shape != keys.shape or len(scores) != len(keys):
        raise ValueError('Mismatched evidence shapes')
    if not np.isfinite(scores).all() or not np.isfinite(xyz).all() or not np.isfinite(ranges).all():
        raise ValueError('Nonfinite evidence')
    if np.any(scores < 0) or np.any(ranges < 0):
        raise ValueError('Negative evidence or range')
    unique, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n = len(unique)
    summed = np.zeros((n, scores.shape[1]), dtype=np.float64)
    np.add.at(summed, inv, scores)
    center = np.column_stack([np.bincount(inv, weights=xyz[:, k], minlength=n) for k in range(3)])
    distance = np.bincount(inv, weights=ranges, minlength=n)
    return dict(keys=unique, score=summed / counts[:, None], xyz=center / counts[:, None],
                ranges=distance / counts, count=counts)


class Visit:
    """Mean per voxel/frame -> mean per group -> bounded visit evidence."""
    def __init__(self, classes, group_frames, range_scale):
        self.classes, self.group_frames, self.range_scale = classes, group_frames, range_scale
        self.current_group = None
        self.pending = []
        self.packets = []
        self.last_frame = -1

    def add(self, frame, keys, labels, confidence, xyz, ranges):
        if frame <= self.last_frame:
            raise ValueError('Each frame must be added once, in increasing order')
        self.last_frame = frame
        labels, confidence = np.asarray(labels), np.asarray(confidence)
        if labels.shape != (len(keys),) or confidence.shape != labels.shape:
            raise ValueError('Prediction/point mismatch')
        if np.any((labels < 0) | (labels >= self.classes)) or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError('Invalid predicted class or confidence')
        group = frame // self.group_frames
        if self.current_group != group:
            self.flush()
            self.current_group = group
        if not len(keys):
            return
        scores = np.zeros((len(keys), self.classes), dtype=np.float64)
        scores[np.arange(len(keys)), labels] = confidence
        self.pending.append(reduce_records(keys, scores, xyz, ranges))

    def flush(self):
        if not self.pending:
            return
        fields = {key: np.concatenate([r[key] for r in self.pending]) for key in ['keys', 'score', 'xyz', 'ranges']}
        packet = reduce_records(fields['keys'], fields['score'], fields['xyz'], fields['ranges'])
        reliability = 1. / (1. + (packet['ranges'] / self.range_scale) ** 2)
        packet['score'] *= reliability[:, None]
        self.packets.append(packet)
        self.pending = []

    def finish(self):
        self.flush()
        if not self.packets:
            return dict(keys=np.empty(0, np.int64), score=np.zeros((0, self.classes)),
                        xyz=np.empty((0, 3)), ranges=np.empty(0), count=np.empty(0, np.int64))
        fields = {key: np.concatenate([r[key] for r in self.packets]) for key in ['keys', 'score', 'xyz', 'ranges']}
        result = reduce_records(fields['keys'], fields['score'], fields['xyz'], fields['ranges'])
        self.packets = []
        return result


def propose(first, second, cfg):
    """Both arms use identical observations and eligibility, only weights differ.

    group_balanced retains the duration advantage of the longer visit.
    revisit_balanced assigns each visit one (range-quality-weighted) mean vote.
    No probability is invented for classes absent from archived predictions.
    """
    keys, ia, ib = np.intersect1d(first['keys'], second['keys'], assume_unique=True, return_indices=True)
    a, b = first['score'][ia], second['score'][ib]
    na, nb = first['count'][ia], second['count'][ib]
    distance = np.linalg.norm(first['xyz'][ia] - second['xyz'][ib], axis=1)
    eligible = ((na >= cfg['minimum_groups_per_visit']) & (nb >= cfg['minimum_groups_per_visit']) &
                (distance <= cfg['maximum_centroid_distance_m']))
    protected = cfg['protected_source_classes']
    # Any potentially moving-class evidence in either visit vetoes revision.
    eligible &= ~((a[:, protected] > 0).any(1) | (b[:, protected] > 0).any(1))
    result = dict(keys=keys, eligible=eligible, groups_first=na, groups_second=nb,
                  centroid_distance=distance, score_first=a, score_second=b)
    for arm, evidence in [('group_balanced', a * na[:, None] + b * nb[:, None]),
                          ('revisit_balanced', a + b)]:
        winner = evidence.argmax(1)
        ordered = np.sort(evidence, axis=1)
        total = evidence.sum(1)
        share = np.divide(ordered[:, -1], total, out=np.zeros(len(keys)), where=total > 0)
        margin = np.divide(ordered[:, -1] - ordered[:, -2], total, out=np.zeros(len(keys)), where=total > 0)
        accept = eligible & (share >= cfg['minimum_winner_share']) & (margin >= cfg['minimum_winner_margin'])
        result[arm] = dict(label=winner.astype(np.uint8), accept=accept, share=share, margin=margin)
    return result


def apply_proposal(sorted_keys, baseline_labels, protected_voxels, proposal, arm):
    """No geometry/coverage changes; unknown or protected voxels keep baseline."""
    output = np.array(baseline_labels, copy=True)
    keys = proposal['keys']
    if not len(sorted_keys) or not len(keys):
        return output
    rows = np.searchsorted(sorted_keys, keys)
    in_bounds = rows < len(sorted_keys)
    rows = np.minimum(rows, len(sorted_keys) - 1)
    selected = in_bounds & (sorted_keys[rows] == keys) & proposal[arm]['accept'] & ~protected_voxels[rows]
    output[rows[selected]] = proposal[arm]['label'][selected]
    return output
