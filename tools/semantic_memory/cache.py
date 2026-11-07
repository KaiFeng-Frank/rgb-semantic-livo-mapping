"""Export frozen, unfused PTv3 decoder features from a declared scan manifest.

Input frames name .bin float32 Nx4 scans, already deskewed in the stated sensor
frame, an explicit 4x4 transform, timestamp and frame index. Optional labels are
.npy int arrays in the model's declared native label space (no silent taxonomy
conversion). Output is accepted by train.py. No datasets are downloaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))


def export(source, output):
    import torch
    from ptv3_worker import Segmenter
    from ptv3_loader_verified import CKPT_PATH
    from tools.semantic_memory.train import (META_KEYS, TRAIN_IDS, VAL_IDS, TEST_IDS,
                                             feature_contract, load_manifest, preflight)
    from semantic_memory import MemoryConfig
    spec = json.loads(source.read_text())
    if spec.get('schema') != 1 or spec.get('geometry') != 'deskewed_sensor':
        raise ValueError('Explicit schema=1 and deskewed_sensor geometry required')
    if spec.get('label_space') not in ('native16', 'semantic-kitti19'):
        raise ValueError('Explicit native label space required')
    split = spec.get('split')
    allowed = {'train': TRAIN_IDS, 'val': VAL_IDS, 'test': TEST_IDS}.get(split)
    if allowed is None or not spec.get('sequences'):
        raise ValueError('Explicit split and nonempty sequences required')
    synthetic = spec.get('synthetic', False)
    if type(synthetic) is not bool:
        raise ValueError('synthetic must be a boolean')
    ids = [s['id'] for s in spec['sequences']]
    if len(set(ids)) != len(ids) or any(s not in allowed for s in ids):
        raise ValueError('Duplicate or forbidden sequence in this split')
    segmenter = Segmenter(half=True, shuffle=False)
    classes = 16 if spec['label_space'] == 'native16' else 19
    if classes != segmenter.num_classes:
        raise ValueError('The frozen checkpoint does not implement this label space')
    with open(CKPT_PATH, 'rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    contract = feature_contract(spec['label_space'])
    if segmenter.grid_size != contract['grid_size'] or segmenter.intensity_scale != contract['intensity_scale']:
        raise ValueError('Extractor differs from the declared training feature contract')
    fingerprint = hashlib.sha256()
    for p in (Path(__file__), ROOT/'src/ptv3_worker.py', ROOT/'src/ptv3_loader_verified.py'):
        fingerprint.update(p.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema=1, split=split, label_space=spec['label_space'],
                  feature_dim=segmenter.model.seg_head.in_features,
                  backbone_sha256=digest, model_config_sha256=segmenter.memory_model_sha256(),
                  feature_contract=contract,
                  source_fingerprint=fingerprint.hexdigest(), synthetic=synthetic, sequences=[])
    try:
        for sequence in spec['sequences']:
            sid = sequence['id']
            if not sequence['frames']:
                raise ValueError('Empty sequence')
            rows = []
            folder = output / sid
            folder.mkdir()
            last_frame, last_stamp = -1, -float('inf')
            for item in sequence['frames']:
                frame, stamp = item['frame'], item['stamp']
                if type(frame) is not int or frame <= last_frame or not np.isfinite(stamp) or stamp <= last_stamp:
                    raise ValueError('Frames must be chronological')
                points = np.fromfile(source.parent / item['path'], dtype='<f4').reshape(-1, 4)
                transform = np.asarray(item['transform'], dtype=np.float32)
                if transform.shape != (4, 4) or not np.isfinite(transform).all():
                    raise ValueError('Invalid explicit transform')
                rot = transform[:3, :3]
                if (not np.allclose(rot.T @ rot, np.eye(3), atol=1e-4, rtol=0)
                        or not np.isclose(np.linalg.det(rot), 1., atol=1e-4, rtol=0)
                        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6, rtol=0)):
                    raise ValueError('Transform must be sensor-to-memory SE(3)')
                raw = segmenter.segment(points, return_features=True)
                data = {k: raw[k].cpu().numpy() for k in ('xyz', 'features', 'logits')}
                data['transform'] = transform
                data['metadata'] = np.array(json.dumps({k: result[k] for k in META_KEYS}))
                if item.get('labels'):
                    labels = np.load(source.parent / item['labels'], allow_pickle=False)
                    if labels.shape != (len(points),) or labels.dtype.kind not in 'iu' or np.any((labels < -1) | (labels >= classes)):
                        raise ValueError('Labels must align with raw points and native taxonomy')
                    inv = raw['inverse'].cpu().numpy()
                    valid = labels >= 0
                    counts = np.bincount(inv[valid] * classes + labels[valid],
                                         minlength=len(data['xyz']) * classes).reshape(-1, classes)
                    # No arbitrary single-point label when a voxel has conflicting annotations.
                    data['label'] = np.where(counts.sum(1) > 0, counts.argmax(1), -1).astype(np.int64)
                path = folder / f'{frame:06d}.npz'
                np.savez_compressed(path, **data)
                rows.append(dict(frame=frame, stamp=stamp, path=str(path.relative_to(output)),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                last_frame, last_stamp = frame, stamp
                del raw, data
            result['sequences'].append(dict(id=sid, frames=rows))
        (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
        preflight(load_manifest(output / 'manifest.json'), MemoryConfig(classes=classes))
        (output / 'source.json').write_text(json.dumps(dict(
            input_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            point_to_voxel_labels='majority of valid native labels'), indent=2)+'\n')
    except Exception as exc:
        (output / 'failure.json').write_text(json.dumps(dict(error=str(exc))) + '\n')
        raise
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    export(a.manifest.resolve(), a.out.resolve())
