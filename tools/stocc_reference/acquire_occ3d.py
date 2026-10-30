"""Download a pinned public Occ3D mirror on the remote machine, retaining val labels.

The mirror's LFS digest authenticates the mirror bytes, not independent identity
with the authors' Google Drive archive. Keep that provenance distinction explicit.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time

ROOT = Path('/home/wuyou/stocc_reference')
REV = 'b679e0ace1dc03d0a86737cb73fa790df03a0ad5'
BASE = 'https://hf-mirror.com/datasets/RoyYao233/occ3d-nuscenes-transfer/resolve/' + REV
FILES = {
    'gts.tar.gz': (2737816708, '0635d1383d8b99cce26a0344ec3ca3c4346f53907a06e82ea525acf7b1abba53'),
    'annotations.json': (150463571, '87cc81f04ef1cd82bd72647c32110b84de8e5620652ecb2b5beae83ee65a1ba2'),
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    directory = ROOT/'downloads/occ3d'
    directory.mkdir(parents=True, exist_ok=True)
    state_path = ROOT/'occ3d-acquisition.json'
    state = dict(status='downloading', started=time.time(), revision=REV,
                 provenance='Public transfer mirror; SHA-256 checked against pinned LFS metadata; not independently compared to author Drive bytes.', files={})
    save(state_path, state)
    try:
        for name, (size, digest) in FILES.items():
            dest = directory/name
            if not dest.exists() or dest.stat().st_size != size or sha(dest) != digest:
                subprocess.run(['aria2c', '--continue=true', '--max-connection-per-server=8',
                                '--split=8', '--min-split-size=8M', '--file-allocation=none',
                                '--max-tries=20', '--retry-wait=5', '--timeout=30',
                                '--summary-interval=30', '--console-log-level=warn',
                                '--auto-file-renaming=false', '--allow-overwrite=false',
                                '--checksum=sha-256='+digest, '--dir='+str(directory),
                                '--out='+name, BASE+'/'+name], check=True)
            if dest.stat().st_size != size or sha(dest) != digest:
                raise ValueError('Mirror checksum mismatch: '+name)
            state['files'][name] = dict(bytes=size, sha256=digest, url=BASE+'/'+name)
            save(state_path, state)
        plan = json.loads((ROOT/'camera-plan.json').read_text())
        scenes = set(plan['scenes'])
        data = ROOT/'data/nuscenes'
        state['status'] = 'extracting';save(state_path,state)
        records = []
        with tarfile.open(directory/'gts.tar.gz', 'r|gz') as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                selected = next((i for i, p in enumerate(parts) if p in scenes), None)
                if selected is None or not member.isfile():
                    continue
                rel = parts[selected:]
                if len(rel) != 3 or rel[2] != 'labels.npz' or '..' in rel or member.size > 20*1024**2:
                    raise ValueError('Unexpected label member: '+member.name)
                content = archive.extractfile(member).read()
                if len(content) != member.size:
                    raise ValueError('Truncated label')
                dest = data/'gts'/Path(*rel)
                digest = hashlib.sha256(content).hexdigest()
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():
                    assert sha(dest) == digest
                else:
                    tmp=dest.with_suffix('.partial');tmp.write_bytes(content);tmp.replace(dest)
                records.append(dict(path=str(dest.relative_to(data)), bytes=member.size, sha256=digest))
        assert len(records) == plan['samples'], len(records)
        save(ROOT/'occ3d-label-manifest.json', records)
        state.update(status='complete', val_frames=len(records), val_scenes=len(scenes), finished=time.time())
        save(state_path,state)
        print(json.dumps(state),flush=True)
    except Exception as error:
        state.update(status='failed', error=repr(error), updated=time.time());save(state_path,state)
        raise


if __name__ == '__main__':
    main()
