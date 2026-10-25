"""Pinned official labels and a deliberately lossy, diagnostic source-label bridge."""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VENDOR = HERE / 'vendor'
CONFIG = yaml.safe_load((VENDOR / 'semantic-kitti.yaml').read_text())
NAMES = [CONFIG['labels'][CONFIG['learning_map_inv'][i]] for i in range(1, 20)]
RAW_IDS = np.array([CONFIG['learning_map_inv'][i] for i in range(20)], dtype=np.uint32)
LUT = np.zeros(65536, dtype=np.int64)
for raw, target in CONFIG['learning_map'].items():
    LUT[raw] = target

# Frozen BEFORE scoring. This is not a learned 19-class head. It cannot split
# manmade into building/pole/sign, vegetation into foliage/trunk, or road/parking.
# Barrier -> fence and other_flat -> other-ground are explicit approximations.
# Traffic cones have no official counterpart: prediction 0 remains a false
# negative against every non-ignored target, never a point-exclusion mask.
NUSC16_TO_SK = np.array([14, 2, 5, 1, 5, 3, 6, 0, 5, 4, 9, 12, 11, 17, 13, 15])
SOURCE_NAMES = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
                'manmade', 'vegetation']
FRAME_COUNTS = {r['sequence']: r['frames'] for r in
                json.loads((HERE / 'frame_counts.json').read_text())['sequences']}
VALIDATION_SEQUENCES = ('07', '08', '09')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_vendor():
    manifest = json.loads((VENDOR / 'manifest.json').read_text())
    for item in manifest['files']:
        if digest(VENDOR / item['path']) != item['sha256']:
            raise ValueError('Official source changed: ' + item['path'])
    return manifest


def raw_to_learning(raw, prediction=False):
    raw = np.asarray(raw)
    if raw.ndim != 1 or raw.dtype.kind not in 'ui':
        raise ValueError('Expected a one-dimensional integer label array')
    if np.any(raw < 0) or np.any(raw > np.iinfo(np.uint32).max):
        raise ValueError('Labels must fit uint32')
    sem = raw.astype(np.uint32) & 0xffff
    valid = RAW_IDS if prediction else np.array(list(CONFIG['learning_map']))
    if prediction and np.any(raw != sem):
        raise ValueError('Single-scan predictions must not contain instance IDs')
    if not np.isin(sem, valid).all():
        raise ValueError('Unknown/noncanonical semantic label IDs')
    return LUT[sem]


def learning_to_raw(pred):
    pred = np.asarray(pred)
    if pred.ndim != 1 or pred.dtype.kind not in 'ui' or np.any((pred < 0) | (pred > 19)):
        raise ValueError('Expected learning IDs 0..19')
    return RAW_IDS[pred].astype('<u4')


def bridge(pred):
    pred = np.asarray(pred)
    if pred.ndim != 1 or pred.dtype.kind not in 'ui' or np.any((pred < 0) | (pred > 15)):
        raise ValueError('Expected nuScenes IDs 0..15; common-9 caches are invalid')
    return NUSC16_TO_SK[pred]


def taxonomy_audit():
    reachable = set(NUSC16_TO_SK.tolist()) - {0}
    absent = [NAMES[i - 1] for i in range(1, 20) if i not in reachable]
    return dict(
        native_19_class_model=False, source_classes=16, evaluated_classes=19,
        reachable_classes=len(reachable), unrepresentable_classes=absent,
        fixed_bridge_miou_ceiling_percent=100 * len(reachable) / 19,
        mapping=[dict(source=s, target_learning_id=int(t),
                      target='abstain' if t == 0 else NAMES[t - 1])
                 for s, t in zip(SOURCE_NAMES, NUSC16_TO_SK)],
        caveat='Fixed coarse-to-fine heuristic, not an adapted 19-class model. '
               'No target labels are used to select this mapping. All 19 classes '
               'stay in the mean; unsupported predictions count as errors.')
