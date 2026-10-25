"""CPU-only 19-class diagnostic of archived ZS/B0 on 07/09. Not a leaderboard run."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import zipfile

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .evaluate import Confusion
from .schema import REPO, FRAME_COUNTS, bridge, raw_to_learning, digest, save, taxonomy_audit, verify_vendor

RELEASE = 'https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/releases/download/v0.6-artifacts-20260928/'
LABELS = 'https://www.semantic-kitti.org/assets/data_odometry_labels.zip'


def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and expected and digest(path) == expected:
        return
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])))
    tmp = path.with_suffix(path.suffix + '.partial')
    with session.get(url, timeout=(20, 120), stream=True) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for chunk in r.iter_content(4 * 1024**2):
                f.write(chunk)
    if expected and digest(tmp) != expected:
        raise ValueError('Checksum mismatch: ' + path.name)
    tmp.replace(path)


def run(root, out):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    official = verify_vendor()
    protocol = json.loads((Path(__file__).parent / 'protocol.json').read_text())
    assets = {a['name']: a for a in json.loads(
        (REPO / 'results/v06/artifacts_20260928/artifact-manifest.json').read_text())['assets']}
    save(out / 'taxonomy-audit.json', taxonomy_audit())
    save(out / 'protocol.json', protocol)
    labels = root / 'data_odometry_labels.zip'
    download(LABELS, labels)
    sources = dict(labels=dict(url=LABELS, bytes=labels.stat().st_size, sha256=digest(labels)),
                   archives=[], label_frames=[], official=official)
    results = {}
    with zipfile.ZipFile(labels) as z:
        for seq in protocol['sequences']:
            prefix = f'dataset/sequences/{seq}/labels/'
            wanted = [prefix + f'{i:06d}.label' for i in range(FRAME_COUNTS[seq])]
            actual = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith('.label'))
            if wanted != actual:
                raise ValueError('Label archive frame mismatch: ' + seq)
            for arm in protocol['arms']:
                name = f'v06-pred-{arm}-seq{seq}-r1.tar'
                for asset in [name, name + '.index.json']:
                    download(RELEASE + asset, root / asset, assets[asset]['sha256'])
                sources['archives'].append(dict(name=name, url=RELEASE + name,
                                                sha256=assets[name]['sha256']))
                records = json.loads((root / (name + '.index.json')).read_text())
                index = {r['path']: r for r in records}
                accum = Confusion()
                with tarfile.open(root / name) as tf:
                    expected_paths = {f'out/v06_score/pred/{arm}_{seq}_r1/f{i:06d}.npz'
                                      for i in range(FRAME_COUNTS[seq])}
                    members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith('.npz')]
                    if len(members) != len(expected_paths) or {m.name for m in members} != expected_paths:
                        raise ValueError('Prediction archive frame mismatch')
                    for i, label_name in enumerate(wanted):
                        rel = f'out/v06_score/pred/{arm}_{seq}_r1/f{i:06d}.npz'
                        payload = tf.extractfile(rel).read()
                        if hashlib.sha256(payload).hexdigest() != index[rel]['sha256']:
                            raise ValueError('Prediction member checksum mismatch: ' + rel)
                        with np.load(io.BytesIO(payload), allow_pickle=False) as cached:
                            p = bridge(cached['lab'])
                        data = z.read(label_name)  # ZIP also validates each member CRC.
                        if len(data) != len(p) * 4:
                            raise ValueError('Prediction/GT point count mismatch: ' + rel)
                        t = raw_to_learning(np.frombuffer(data, dtype='<u4'))
                        accum.add(p, t)
                        if arm == protocol['arms'][0]:
                            sources['label_frames'].append(dict(path=label_name,
                                points=len(t), sha256=hashlib.sha256(data).hexdigest()))
                        if (i + 1) % 250 == 0:
                            print('SCORED', seq, arm, i + 1, flush=True)
                report = accum.report()
                report.update(sequence=seq, frames=FRAME_COUNTS[seq], arm=arm,
                              source_space='nuScenes16', native_19_class_model=False,
                              role='frozen_bridge_diagnostic', leaderboard_test_score=False,
                              new_inference=False, map_fusion=False)
                results[f'{arm}_{seq}'] = report
                save(out / f'{arm}_{seq}.json', report)
                print('RESULT', seq, arm, report['miou19_percent'], flush=True)
                # Only this run's downloaded archive; immutable release is retained.
                (root / name).unlink()
    summary = dict(protocol=protocol, taxonomy=taxonomy_audit(), runs=results,
                   paired_gain_pp={seq: results[f'armB0_s1_{seq}']['miou19_percent'] -
                                   results[f'ZS_{seq}']['miou19_percent'] for seq in protocol['sequences']},
                   seconds=time.monotonic() - start, seq08_validation_completed=False,
                   test_11_21_completed=False, native19_training_completed=False,
                   caveat='All 19 classes and all GT-valid points scored. Public 07/09 diagnostic '
                          'of a fixed lossy bridge only; not official validation or a test submission.')
    save(out / 'comparison.json', summary)
    save(out / 'provenance.json', sources)
    lines = ['# Frozen source-label bridge: 19-class diagnostic', '', summary['caveat'], '',
             '| Public sequence | ZS mIoU-19 | B0 s1 mIoU-19 | Difference (pp) |',
             '|---|---:|---:|---:|']
    for seq in protocol['sequences']:
        lines.append(f"| {seq} | {results[f'ZS_{seq}']['miou19_percent']:.4f} | "
                     f"{results[f'armB0_s1_{seq}']['miou19_percent']:.4f} | {summary['paired_gain_pp'][seq]:+.4f} |")
    lines += ['', 'These are full-scan scores, NOT the old out-of-frustum common-9 scores.',
              'B0 uses one archived training seed and one inference pass. No new inference or training.',
              '', 'Unrepresentable classes: ' + ', '.join(taxonomy_audit()['unrepresentable_classes']),
              '', 'Sequence 08 validation, native 19-class training and test submission remain incomplete.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print('AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    run(a.root, a.out)
