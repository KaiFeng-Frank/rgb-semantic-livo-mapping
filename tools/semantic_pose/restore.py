"""Recover only the frozen registration/diagnostic scans on an ephemeral runner."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile

from tools.loop_closure.common import REPO, DRIVE, digest, save

sys.path.insert(0, str(REPO/'tools'))
from restore_driving_comparison import RemoteZip, client, download
from compare_driving_semantics import RELEASE

KITTI = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/'


def restore(root, out):
    cfg = json.loads(Path(__file__).with_name('protocol.json').read_text())
    geo = json.loads((REPO/cfg['geometric_protocol']).read_text())
    registration = sorted(set(cfg['target_frames']+cfg['source_history_frames']+[cfg['current_frame']]))
    diagnostic = geo['geometry_evaluation']['early_frames']+geo['geometry_evaluation']['late_frames']
    assert not set(registration).intersection(diagnostic)
    frames = sorted(set(registration+diagnostic))
    root.mkdir(parents=True, exist_ok=True)
    assets = {a['name']:a for a in json.loads((REPO/'results/v06/artifacts_20260928/artifact-manifest.json').read_text())['assets']}
    session = client()
    evidence = dict(registration_frames=registration, diagnostic_frames=diagnostic,
                    prediction_archives={}, sources=[], new_inference=False, raw_data_uploaded=False)
    for model in cfg['models']:
        name = f'v06-pred-{model}-seq07-r1.tar'
        for asset in [name, name+'.index.json']:
            download(session, RELEASE+asset, root/asset, assets[asset]['sha256'])
        index = {r['path']:r for r in json.loads((root/(name+'.index.json')).read_text())}
        with tarfile.open(root/name) as tf:
            for f in registration:
                rel = f'out/v06_score/pred/{model}_07_r1/f{f:06d}.npz'
                data = tf.extractfile(tf.getmember(rel)).read()
                assert len(data) == index[rel]['bytes']
                assert hashlib.sha256(data).hexdigest() == index[rel]['sha256']
                p = root/rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
        evidence['prediction_archives'][model] = assets[name]['sha256']
        print('PREDICTIONS_VERIFIED', model, len(registration), flush=True)
    # Only evaluator input, never opened by the registration module.
    for archive, members in [
        ('data_odometry_calib.zip', [('dataset/sequences/07/calib.txt', 'evaluation/calib.txt'),
                                    ('dataset/sequences/07/times.txt', 'evaluation/times.txt')]),
        ('data_odometry_poses.zip', [('dataset/poses/07.txt', 'evaluation/kitti_official_07.txt')])]:
        r = session.get(KITTI+archive, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for rel, dest in members:
                p = root/dest
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(z.read(rel))
        evidence['sources'].append(dict(url=KITTI+archive, sha256=hashlib.sha256(r.content).hexdigest()))
    raw_url = KITTI+'raw_data/2011_09_30_drive_0027/'+DRIVE+'.zip'
    previous = json.loads((REPO/cfg['historical_results']/'input-recovery.json').read_text())
    hashes = {r['path']:r['sha256'] for r in previous['scans_and_timestamps']}
    def shard(worker):
        remote = RemoteZip(raw_url)
        records = []
        names = [f'velodyne_points/data/{f:010d}.bin' for f in frames[worker::4]]
        if worker == 0:
            names += ['velodyne_points/timestamps_start.txt', 'velodyne_points/timestamps_end.txt', 'image_02/timestamps.txt']
        with zipfile.ZipFile(remote) as z:
            for rel in names:
                data = z.read('2011_09_30/'+DRIVE+'/'+rel)
                sha = hashlib.sha256(data).hexdigest()
                assert sha == hashes[rel]
                p = root/'data/raw/2011_09_30'/DRIVE/rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
                records.append(dict(path=rel, sha256=sha, bytes=len(data)))
        return records, remote.fetched
    with ThreadPoolExecutor(max_workers=4) as pool:
        shards = list(pool.map(shard, range(4)))
    evidence['scans_and_timestamps'] = [r for records, _ in shards for r in records]
    evidence['sources'].append(dict(url=raw_url, range_bytes=sum(b for _, b in shards)))
    labels = RemoteZip('https://www.semantic-kitti.org/assets/data_odometry_labels.zip')
    with zipfile.ZipFile(labels) as z:
        for f in diagnostic:
            data = z.read(f'dataset/sequences/07/labels/{f:06d}.label')
            assert hashlib.sha256(data).hexdigest() == previous['labels_sha256'][f'{f:06d}.label']
            p = root/f'evaluation/labels/{f:06d}.label'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    out.mkdir(parents=True, exist_ok=True)
    save(out/'input-recovery.json', evidence)
    print('RESTORE_COMPLETE', len(frames), 'scans', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    restore(a.root, a.out)
