"""Restore seq07 on an ephemeral CPU runner. Raw KITTI data is never published."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import zipfile

from .common import REPO, DRIVE, digest, save, protocol

sys.path.insert(0, str(REPO / 'tools'))
from restore_driving_comparison import RemoteZip, client, download
from compare_driving_semantics import RELEASE

KITTI = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/'


def restore(root, out):
    import json
    root.mkdir(parents=True, exist_ok=True)
    assets = {a['name']: a for a in json.loads((REPO / 'results/v06/artifacts_20260928/artifact-manifest.json').read_text())['assets']}
    session = client()
    name = 'v06-pred-armB0_s1-seq07-r1.tar'
    for asset in [name, name+'.index.json']:
        download(session, RELEASE+asset, root/asset, assets[asset]['sha256'])
    index = {r['path']: r for r in json.loads((root/(name+'.index.json')).read_text())}
    with tarfile.open(root/name) as tf:
        for f in range(protocol()['frames']):
            rel = f'out/v06_score/pred/armB0_s1_07_r1/f{f:06d}.npz'
            member = tf.getmember(rel)
            assert member.isfile()
            data = tf.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest() == index[rel]['sha256']
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    evidence = dict(prediction_sha256=assets[name]['sha256'], prediction_files=1101,
                    new_inference=False, raw_data_uploaded=False, sources=[])
    # Small official ZIPs: extract only explicit members; never extractall.
    small = [
        ('raw_data/2011_09_30_calib.zip', [(f'2011_09_30/{n}', f'data/raw/2011_09_30/{n}') for n in
          ['calib_cam_to_cam.txt', 'calib_velo_to_cam.txt', 'calib_imu_to_velo.txt']]),
        ('data_odometry_calib.zip', [(f'dataset/sequences/07/{n}', f'evaluation/{n}') for n in ['calib.txt', 'times.txt']]),
        ('data_odometry_poses.zip', [('dataset/poses/07.txt', 'evaluation/kitti_official_07.txt')])]
    for rel, members in small:
        response = session.get(KITTI+rel, timeout=120)
        response.raise_for_status()
        evidence['sources'].append(dict(url=KITTI+rel, sha256=hashlib.sha256(response.content).hexdigest()))
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for member, dest in members:
                p = root/dest
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(z.read(member))
    url = 'https://www.semantic-kitti.org/assets/data_odometry_labels.zip'
    remote = RemoteZip(url)
    with zipfile.ZipFile(remote) as z:
        for f in range(1101):
            p = root/f'evaluation/labels/{f:06d}.label'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read(f'dataset/sequences/07/labels/{f:06d}.label'))
            if f % 200 == 0:
                print('LABELS', f, flush=True)
    evidence['sources'].append(dict(url=url, archive_bytes=remote.size, range_bytes=remote.fetched))
    url = KITTI+'raw_data/2011_09_30_drive_0027/'+DRIVE+'.zip'
    prefix = '2011_09_30/'+DRIVE+'/'
    base = root/'data/raw/2011_09_30'/DRIVE
    # Each worker has its own HTTP range reader / ZIP position.
    def shard(worker):
        remote = RemoteZip(url)
        records = []
        with zipfile.ZipFile(remote) as z:
            names = [f'velodyne_points/data/{f:010d}.bin' for f in range(worker, 1101, 6)]
            if worker == 0:
                names += ['velodyne_points/timestamps_start.txt', 'velodyne_points/timestamps_end.txt', 'image_02/timestamps.txt']
            for i, rel in enumerate(names):
                p = base/rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(z.read(prefix+rel))
                records.append(dict(path=rel, sha256=digest(p), bytes=p.stat().st_size))
                if i % 50 == 0:
                    print('RAW', worker, i, '/', len(names), flush=True)
        return records, remote.fetched
    with ThreadPoolExecutor(max_workers=6) as pool:
        shards = list(pool.map(shard, range(6)))
    evidence['scans_and_timestamps'] = sorted([r for s, _ in shards for r in s], key=lambda r:r['path'])
    evidence['sources'].append(dict(url=url, range_bytes=sum(n for _, n in shards)))
    evidence['labels_sha256'] = {p.name:digest(p) for p in sorted((root/'evaluation/labels').glob('*.label'))}
    save(out/'input-recovery.json', evidence)
    print('RESTORE_COMPLETE', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    restore(a.root, a.out)
