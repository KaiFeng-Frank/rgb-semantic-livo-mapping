"""Full-point native-19 single-scan inference; no labels, camera or map inputs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .runtime import build, prepare, logits, POINTCEPT_COMMIT
from .schema import FRAME_COUNTS, NAMES, learning_to_raw, digest, save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--pointcept', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sequences', default='08')
    ap.add_argument('--partial-diagnostic', action='store_true')
    a = ap.parse_args()
    sequences = a.sequences.split(',')
    if len(set(sequences)) != len(sequences) or any(s not in FRAME_COUNTS for s in sequences):
        raise ValueError('Invalid/duplicate sequences')
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('Choose a new output directory; predictions are immutable')
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if saved.get('native19') is not True or saved.get('class_names') != NAMES:
        raise ValueError('A native-19 trained checkpoint with exact taxonomy is required')
    if not saved.get('training_data_disclosure') or saved.get('config', {}).get('pointcept') != POINTCEPT_COMMIT:
        raise ValueError('Missing training disclosure or incompatible architecture')
    if any(s in saved['config']['sequences'] for s in sequences):
        raise ValueError('Refusing held-out inference on sequences used to train this model')
    if saved['config']['mode'] == 'pilot' and not a.partial_diagnostic:
        raise ValueError('Pilot checkpoints can only produce explicit diagnostic outputs')
    scans = []
    for seq in sequences:
        items = sorted((a.dataset / 'sequences' / seq / 'velodyne').glob('*.bin'))
        if not items or (not a.partial_diagnostic and [p.stem for p in items] != [f'{i:06d}' for i in range(FRAME_COUNTS[seq])]):
            raise ValueError('Empty/incomplete inference sequence: ' + seq)
        scans.extend((seq, p) for p in items)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = build(a.pointcept, classes=19).cuda()
    model.load_state_dict(saved['state_dict'], strict=True)
    model = model.eval().half()
    del saved['state_dict'], saved['optimizer']
    meta = dict(task='single_scan_semantic_segmentation', num_classes=19, label_space='SemanticKITTI19',
                uses_future_scans=False, map_fusion=False, frozen_bridge=False, sequences=sequences,
                partial_diagnostic=a.partial_diagnostic, training_mode=saved['config']['mode'],
                checkpoint_sha256=digest(a.checkpoint), training_data_disclosure=saved['training_data_disclosure'],
                pointcept=POINTCEPT_COMMIT, torch=torch.__version__, attention='torch SDPA',
                predictions=[], complete=False, leaderboard_test_score=False)
    a.out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for seq, scan in scans:
            if scan.stat().st_size == 0 or scan.stat().st_size % 16:
                raise ValueError('Malformed scan: ' + str(scan))
            points = np.fromfile(scan, dtype='<f4').reshape(-1, 4)
            batch, _, inv = prepare(points, half=True)
            torch.cuda.synchronize()
            start = time.monotonic()
            pred = logits(model, batch).argmax(-1).cpu().numpy()[inv] + 1
            seconds = time.monotonic() - start
            raw = learning_to_raw(pred)
            rel = f'sequences/{seq}/predictions/{scan.stem}.label'
            dest = a.out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw.tofile(dest)
            meta['predictions'].append(dict(path=rel, sha256=digest(dest), points=len(points),
                                            scan_sha256=digest(scan), inference_seconds=seconds))
            if len(meta['predictions']) % 100 == 0:
                save(a.out / 'inference.json', meta)
                print('INFERRED', len(meta['predictions']), '/', len(scans), flush=True)
            del batch
    meta['complete'] = True
    save(a.out / 'inference.json', meta)
    print('SAVED', len(scans), 'full-point predictions to', a.out, flush=True)


if __name__ == '__main__':
    main()
