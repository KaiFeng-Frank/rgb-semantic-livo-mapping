"""Build a strictly checked single-scan submission ZIP; never uploads it."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from .schema import FRAME_COUNTS, raw_to_learning, digest, save, verify_vendor

TEST_SEQUENCES = tuple(f'{i:02d}' for i in range(11, 22))


def build_submission(dataset, predictions, provenance, destination):
    """Require complete test scans and a native single-scan inference manifest."""
    meta = json.loads(Path(provenance).read_text())
    required = dict(task='single_scan_semantic_segmentation', num_classes=19,
                    label_space='SemanticKITTI19', uses_future_scans=False,
                    map_fusion=False, frozen_bridge=False)
    for key, value in required.items():
        if type(meta.get(key)) is not type(value) or meta[key] != value:
            raise ValueError(f'Invalid/missing inference contract: {key} must be {value!r}')
    if sorted(meta.get('sequences', [])) != list(TEST_SEQUENCES):
        raise ValueError('Manifest must cover exactly test sequences 11..21')
    if not meta.get('checkpoint_sha256') or not meta.get('training_data_disclosure'):
        raise ValueError('Checkpoint hash and training-data disclosure are required')
    records = {r['path']: r for r in meta.get('predictions', [])}
    if len(records) != len(meta.get('predictions', [])):
        raise ValueError('Duplicate predictions in inference manifest')
    dataset, predictions, destination = Path(dataset), Path(predictions), Path(destination)
    if destination.exists() or destination.suffix != '.zip':
        raise ValueError('Choose a new .zip output path')
    inventory = []
    for seq in TEST_SEQUENCES:
        scans = sorted((dataset / 'sequences' / seq / 'velodyne').glob('*.bin'))
        if [p.stem for p in scans] != [f'{i:06d}' for i in range(FRAME_COUNTS[seq])]:
            raise ValueError('Missing or noncontiguous scans: ' + seq)
        pred_dir = predictions / 'sequences' / seq / 'predictions'
        if sorted(p.stem for p in pred_dir.glob('*.label')) != [p.stem for p in scans]:
            raise ValueError('Prediction/scan frame set mismatch: ' + seq)
        for scan in scans:
            rel = f'sequences/{seq}/predictions/{scan.stem}.label'
            pred = predictions / rel
            nbytes = scan.stat().st_size
            if nbytes == 0 or nbytes % 16 or pred.stat().st_size != nbytes // 4:
                raise ValueError('Prediction/scan point count mismatch: ' + rel)
            record = records.get(rel)
            checksum = digest(pred)
            if record is None or record.get('sha256') != checksum or record.get('points') != nbytes // 16:
                raise ValueError('Prediction changed or missing from manifest: ' + rel)
            raw = np.fromfile(pred, dtype='<u4')
            mapped = raw_to_learning(raw, prediction=True)
            if np.any(mapped == 0):
                raise ValueError('Native 19-class predictions must label every point: ' + rel)
            inventory.append(dict(path=rel, sha256=checksum, points=len(raw)))
    if set(records) != {r['path'] for r in inventory}:
        raise ValueError('Extra prediction records outside test set')
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix('.zip.partial')
    if tmp.exists():
        raise ValueError('Partial output already exists: ' + str(tmp))
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        # Explicit directory entries are required by the official validator.
        z.writestr('sequences/', b'')
        for seq in TEST_SEQUENCES:
            z.writestr(f'sequences/{seq}/', b'')
            z.writestr(f'sequences/{seq}/predictions/', b'')
        for record in inventory:
            z.write(predictions / record['path'], record['path'])
    tmp.replace(destination)
    report = dict(file=destination.name, sha256=digest(destination), frames=len(inventory),
                  points=sum(r['points'] for r in inventory), manifest_sha256=digest(provenance),
                  official_source=verify_vendor()['commit'], uploaded=False,
                  note='Local structural check only. Organizer admission and current terms are separate.')
    save(destination.with_suffix('.validation.json'), report)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(build_submission(a.dataset, a.predictions, a.manifest, a.out), indent=2))
