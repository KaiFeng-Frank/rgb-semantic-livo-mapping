"""Retain only diagnostic panoptic targets from a pinned public transfer mirror."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time

ROOT = Path('/home/wuyou/stocc_reference/history-diagnostic')
REV = 'ce32017d6de629ad5b98f16168d12e9ab2598059'
NAME = 'nuScenes-panoptic-v1.0-all.tar.gz'
URL = 'https://hf-mirror.com/datasets/haozhanglife/nuscenes/resolve/' + REV + '/' + NAME
SHA = 'dab40d850a0c35024f2422a8d2a9ae508544bfefc4b2cbc50a84813dcd07844d'
SIZE = 172581558


def main():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    wanted = {row['lidar_token'] + '_panoptic.npz' for row in protocol['frames']}
    dest = ROOT/'targets/panoptic'
    dest.mkdir(parents=True, exist_ok=True)
    state = dict(started=time.time(), url=URL, revision=REV, archive_sha256=SHA,
                 archive_bytes=SIZE, provenance='Pinned public mirror of panoptic labels; archive identity checked against mirror LFS metadata, not independently against the original author archive.')
    path = ROOT/NAME
    env = {k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    if not path.exists() or path.stat().st_size != SIZE or hashlib.sha256(path.read_bytes()).hexdigest() != SHA:
        subprocess.run(['aria2c', '--no-conf=true', '--continue=true', '--split=8',
                    '--max-connection-per-server=8', '--file-allocation=none',
                    '--max-tries=10', '--retry-wait=3', '--timeout=25',
                    '--auto-file-renaming=false', '--allow-overwrite=false',
                    '--summary-interval=10', '--console-log-level=warn',
                    '--checksum=sha-256='+SHA, '--dir='+str(ROOT), '--out='+NAME, URL],
                       env=env, check=True)
    assert path.stat().st_size == SIZE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA
    manifest = {}
    duplicates = 0
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name).name
            if name not in wanted:
                continue
            assert member.isfile() and member.size < 2*1024*1024
            content = archive.extractfile(member).read()
            assert len(content) == member.size
            digest = hashlib.sha256(content).hexdigest()
            if name in manifest:
                # Mini and trainval contain byte-identical copies of shared frames.
                assert manifest[name]['sha256'] == digest
                duplicates += 1
                continue
            (dest/name).write_bytes(content)
            manifest[name] = dict(bytes=len(content), sha256=digest)
    assert set(manifest) == wanted
    (ROOT/'panoptic-files.json').write_text(json.dumps(manifest, indent=2))
    state.update(status='complete', files=len(manifest), identical_archive_duplicates=duplicates, completed=time.time())
    (ROOT/'panoptic-acquisition.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
