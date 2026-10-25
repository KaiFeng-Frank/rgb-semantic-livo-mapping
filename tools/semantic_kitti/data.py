"""Fetch explicit KITTI members by HTTP range, with disk budget and checksums."""
import argparse
from contextlib import ExitStack
import io
import json
from pathlib import Path
import shutil
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .schema import FRAME_COUNTS, digest, save

SCAN_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip'
LABEL_URL = 'https://www.semantic-kitti.org/assets/data_odometry_labels.zip'


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.session = requests.Session()
        self.session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                           status_forcelist=[429, 500, 502, 503, 504])))
        self.url, self.pos, self.fetched = url, 0, 0
        self.cache_start, self.cache = 0, b''
        with self.session.get(url, headers={'Range': 'bytes=0-0', 'Accept-Encoding': 'identity'},
                              timeout=(20, 90), stream=True) as r:
            r.raise_for_status()
            if r.status_code != 206:
                raise ValueError('Server must support ranges; refusing a whole-archive download')
            self.size = int(r.headers['Content-Range'].split('/')[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        if not 0 <= position <= self.size:
            raise ValueError('Seek outside archive')
        self.pos = position
        return position

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n == 0:
            return b''
        if n > 32 * 1024**2:
            raise ValueError('Unexpected range over 32 MiB')
        end = self.pos + n
        if not (self.cache_start <= self.pos and end <= self.cache_start + len(self.cache)):
            last = min(self.size, self.pos + max(n, 65536)) - 1
            with self.session.get(self.url, headers={'Range': f'bytes={self.pos}-{last}',
                                  'Accept-Encoding': 'identity'}, timeout=(20, 120), stream=True) as r:
                r.raise_for_status()
                if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {self.pos}-{last}/{self.size}':
                    raise ValueError('Invalid range response')
                content = r.content
            if len(content) != last - self.pos + 1:
                raise ValueError('Truncated HTTP range')
            self.cache_start, self.cache = self.pos, content
            self.fetched += len(content)
        start = self.pos - self.cache_start
        self.pos = end
        return self.cache[start:start+n]

    def close(self):
        self.session.close()
        super().close()


def download_verified(url, path, sha256):
    """Resume immutable weights/wheels using small ranges, verify the whole file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) == sha256:
            return
        raise ValueError('Existing destination checksum mismatch: ' + str(path))
    partial = path.with_suffix(path.suffix + '.partial')
    with RangeReader(url) as remote, partial.open('ab') as dest:
        remote.seek(partial.stat().st_size)
        while remote.tell() < remote.size:
            for attempt in range(5):
                try:
                    block = remote.read(min(1024**2, remote.size - remote.tell()))
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
            dest.write(block)
            dest.flush()
            if remote.tell() % (16 * 1024**2) == 0:
                print('DOWNLOAD_BYTES', path.name, remote.tell(), remote.size, flush=True)
    if digest(partial) != sha256:
        raise ValueError('Downloaded file checksum mismatch: ' + str(partial))
    partial.replace(path)


def fetch(root, sequence, frames, with_labels=False, max_gib=10., reserve_gib=15.):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if len(frames) != len(set(frames)) or not frames or any(i < 0 or i >= FRAME_COUNTS[sequence] for i in frames):
        raise ValueError('Invalid frame list')
    if with_labels and int(sequence) > 10:
        raise ValueError('Hidden test labels must never be requested')
    manifest_path = root / f'recovery-{sequence}.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(sequence=sequence, members={})
    spec = [(SCAN_URL, 'velodyne', 'bin')]
    if with_labels:
        spec.append((LABEL_URL, 'labels', 'label'))
    with ExitStack() as stack:
        plans = []
        total = needed = 0
        for url, kind, suffix in spec:
            remote = stack.enter_context(RangeReader(url))
            z = stack.enter_context(zipfile.ZipFile(remote))
            requested = [(f'sequences/{sequence}/{kind}/{i:06d}.{suffix}', i) for i in frames]
            size = sum(z.getinfo('dataset/' + rel).file_size for rel, _ in requested)
            total += size
            needed += sum(z.getinfo('dataset/' + rel).file_size for rel, _ in requested if not (root / rel).exists())
            plans.append((url, kind, remote, z, requested, size))
        if total > max_gib * 1024**3:
            raise ValueError(f'Request needs {total/1024**3:.2f} GiB, over the explicit budget')
        if shutil.disk_usage(root).free - needed < reserve_gib * 1024**3:
            raise ValueError('Insufficient free disk after preserving the reserve')
        for url, kind, remote, z, requested, size in plans:
            print('FETCH', sequence, kind, 'frames', len(frames), 'GiB', round(size/1024**3, 3), flush=True)
            for count, (rel, _) in enumerate(requested, 1):
                dest = root / rel
                old = manifest['members'].get(rel)
                if old and dest.exists() and digest(dest) == old['sha256']:
                    continue
                content = z.read('dataset/' + rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + '.partial')
                tmp.write_bytes(content)
                tmp.replace(dest)
                manifest['members'][rel] = dict(bytes=len(content), sha256=digest(dest), url=url)
                save(manifest_path, manifest)
                if count % 100 == 0:
                    print('FETCHED', sequence, kind, count, '/', len(frames), flush=True)
            save(manifest_path, manifest)
            print('HTTP_BYTES', remote.fetched, flush=True)
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--sequence', choices=list(FRAME_COUNTS), default='08')
    ap.add_argument('--frames', default='0,1000,2000')
    ap.add_argument('--labels', action='store_true')
    ap.add_argument('--max-gib', type=float, default=10.)
    ap.add_argument('--reserve-gib', type=float, default=15.)
    a = ap.parse_args()
    frames = list(range(FRAME_COUNTS[a.sequence])) if a.frames == 'all' else [int(f) for f in a.frames.split(',')]
    fetch(a.root, a.sequence, frames, a.labels, a.max_gib, a.reserve_gib)
