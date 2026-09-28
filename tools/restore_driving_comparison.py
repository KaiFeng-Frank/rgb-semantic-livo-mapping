#!/usr/bin/env python3
"""Recover only the existing driving clip's inputs on a remote CPU render host.

Predictions come from the verified GitHub archive, not a new model run. Raw inputs
are range-read from KITTI's official ZIP, never uploaded to this repository.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from compare_driving_semantics import ARMS, RELEASE, digest, save

KITTI = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'
DRIVE = '2011_09_30_drive_0027_sync'


def client():
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    return session


class RemoteZip(io.RawIOBase):
    """Seekable HTTP range reader; refuses any fallback to downloading a whole ZIP."""
    def __init__(self, url):
        self.session, self.url = client(), url
        self.pos, self.fetched = 0, 0
        self.cache_at, self.cache = 0, b''
        with self.session.get(url, headers={'Range': 'bytes=0-0'}, timeout=60, stream=True) as r:
            r.raise_for_status()
            assert r.status_code == 206
            self.size = int(r.headers['Content-Range'].split('/')[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        if not 0 <= self.pos <= self.size:
            raise ValueError('Seek outside ZIP')
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n == 0:
            return b''
        if n > 32 * 1024**2:
            raise ValueError('Unexpected large ZIP read')
        end = self.pos + n
        if not (self.cache_at <= self.pos and end <= self.cache_at + len(self.cache)):
            last = min(self.size, self.pos + max(n, 65536)) - 1
            with self.session.get(self.url, headers={'Range': f'bytes={self.pos}-{last}'},
                                  timeout=90, stream=True) as r:
                r.raise_for_status()
                assert r.status_code == 206
                assert r.headers['Content-Range'] == f'bytes {self.pos}-{last}/{self.size}'
                data = r.content
                assert len(data) == last - self.pos + 1
            self.fetched += len(data)
            self.cache_at, self.cache = self.pos, data
        start = self.pos - self.cache_at
        self.pos = end
        return self.cache[start:start + n]


def download(session, url, path, expected):
    if path.exists() and digest(path) == expected:
        return
    tmp = path.with_suffix(path.suffix + '.partial')
    with session.get(url, timeout=120, stream=True) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for chunk in r.iter_content(4 * 1024**2):
                f.write(chunk)
    assert digest(tmp) == expected, path.name
    tmp.replace(path)


def restore(root, out):
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / 'results/v06/artifacts_20260928/artifact-manifest.json').read_text())
    assets = {a['name']: a for a in manifest['assets']}
    historical = json.loads((repo / 'results/v06/driving_demo_20260928/measurement.json').read_text())
    rows = historical['timeline']
    frames = [r['frame'] for r in rows]
    assert frames == list(range(970, 1101))
    session = client()
    archives = root / 'out/preservation_20260928'
    archives.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    recovered = []
    for arm in ARMS:
        name = f'v06-pred-{arm}-seq07-r1.tar'
        for asset in [name, name + '.index.json']:
            print('RESTORING', asset, flush=True)
            download(session, RELEASE + asset, archives / asset, assets[asset]['sha256'])
        index = {r['path']: r for r in json.loads((archives / (name + '.index.json')).read_text())}
        with tarfile.open(archives / name) as tf:
            for frame in frames:
                rel = f'out/v06_score/pred/{arm}_07_r1/f{frame:06d}.npz'
                member = tf.getmember(rel)
                assert member.isfile()
                data = tf.extractfile(member).read()
                assert len(data) == index[rel]['bytes']
                assert hashlib.sha256(data).hexdigest() == index[rel]['sha256']
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                recovered.append(rel)
        print('PREDICTIONS_VERIFIED', arm, len(frames), flush=True)
    # Preserve the exact scoring scripts as well, so the shared preprocessing can
    # be reviewed even after the original experiment host has been released.
    runtime = 'semantic-runtime-and-metadata.tar'
    for asset in [runtime, runtime + '.index.json']:
        download(session, RELEASE + asset, archives / asset, assets[asset]['sha256'])
    runtime_index = {r['path']: r for r in json.loads((archives / (runtime + '.index.json')).read_text())}
    sources = {}
    with tarfile.open(archives / runtime) as tf:
        for rel in ['tools/cache_ptv3.py', 'tools/cache_trained.py']:
            data = tf.extractfile(tf.getmember(rel)).read()
            sha = hashlib.sha256(data).hexdigest()
            assert sha == runtime_index[rel]['sha256']
            sources[rel] = dict(sha256=sha, source=data.decode())
    save(out / 'scoring-sources.json', sources)
    print('SCORING_SOURCES_VERIFIED', {k: v['sha256'] for k, v in sources.items()}, flush=True)
    calib_url = KITTI + '2011_09_30_calib.zip'
    r = session.get(calib_url, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in ['calib_cam_to_cam.txt', 'calib_velo_to_cam.txt', 'calib_imu_to_velo.txt']:
            p = root / 'data/raw/2011_09_30' / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read('2011_09_30/' + name))
    raw_url = KITTI + '2011_09_30_drive_0027/' + DRIVE + '.zip'
    remote = RemoteZip(raw_url)
    raw_dir = root / 'data/raw/2011_09_30' / DRIVE
    prefix = '2011_09_30/' + DRIVE + '/'
    with zipfile.ZipFile(remote) as z:
        for rel in ['image_02/timestamps.txt']:
            dest = raw_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(prefix + rel))
        for i, row in enumerate(rows):
            frame = row['frame']
            for rel in [f'image_02/data/{frame:010d}.png', f'velodyne_points/data/{frame:010d}.bin']:
                dest = raw_dir / rel
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(z.read(prefix + rel))
                if rel.endswith('.bin'):
                    assert digest(dest) == row['points_sha256'], ('Original scan mismatch', frame)
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print('RAW_INPUTS_VERIFIED', i + 1, '/', len(rows), 'range bytes', remote.fetched, flush=True)
    save(out / 'input-recovery.json', dict(
        raw_zip_url=raw_url, raw_zip_bytes=remote.size, raw_range_bytes_downloaded=remote.fetched,
        calibration_url=calib_url, scans_matching_original_demo_sha256=len(rows),
        frames=frames, prediction_files=len(recovered), prediction_files_verified=True,
        prediction_archives={arm: dict(url=RELEASE + f'v06-pred-{arm}-seq07-r1.tar',
            sha256=assets[f'v06-pred-{arm}-seq07-r1.tar']['sha256']) for arm in ARMS},
        raw_data_uploaded=False, new_inference=False))
    print('RESTORE_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    restore(args.root, args.out)
