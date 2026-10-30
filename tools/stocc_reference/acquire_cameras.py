"""Acquire original nuScenes validation keyframe cameras on the remote host.

Reuses the existing verified official-S3 downloader, with output paths isolated
from the running TASeg experiment. Each temporary archive is removed only after
checksum verification, extraction and a per-file manifest have succeeded.
"""
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
import ast

ROOT = Path('/home/wuyou/stocc_reference')
SOURCE = Path('/home/wuyou/taseg_reference/data/nuscenes')
HELPER = Path(__file__).with_name('official_s3.py')
if not HELPER.exists():
    HELPER = Path('/home/wuyou/taseg_reference/tools/remote_data_queue.py')
SPLITS = Path('/home/wuyou/taseg_reference/runtime/venv/lib/python3.12/site-packages/nuscenes/utils/splits.py')
# Read the official literal without importing unrelated SDK visualization dependencies.
val = next(ast.literal_eval(n.value) for n in ast.parse(SPLITS.read_text()).body
           if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'val' for t in n.targets))
spec = importlib.util.spec_from_file_location('official_s3_downloader', HELPER)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)
download.ROOT = ROOT
download.DOWNLOADS = ROOT/'downloads/cameras'
download.DOWNLOADS.mkdir(parents=True, exist_ok=True)
download.RESERVE = 20*1024**3


def main():
    meta = SOURCE/'v1.0-trainval'
    scenes = {s['token']:s['name'] for s in json.loads((meta/'scene.json').read_text()) if s['name'] in val}
    samples = {s['token']:s['scene_token'] for s in json.loads((meta/'sample.json').read_text()) if s['scene_token'] in scenes}
    required = {s['filename'] for s in json.loads((meta/'sample_data.json').read_text())
                if s['sample_token'] in samples and s['is_key_frame'] and s['filename'].startswith('samples/CAM_')}
    assert len(scenes) == 150 and len(samples) == 6019 and len(required) == 36114
    data = ROOT/'data/nuscenes'
    (data/'samples').mkdir(parents=True, exist_ok=True)
    for name in ('v1.0-trainval','maps','lidarseg','samples/LIDAR_TOP'):
        p = data/name
        if not p.exists():
            p.symlink_to(SOURCE/name, target_is_directory=True)
    download.save(ROOT/'camera-plan.json', dict(scenes=sorted(scenes.values()), samples=len(samples),
        required=sorted(required), helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        archive_policy='Two concurrent official keyframe archives; verify S3 ETag, retain validation images only, remove verified archive after extraction.'))

    def one(index):
        directory = ROOT/f'camera-acquisition/{index:02d}'
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory/'state.json'
        if state_path.exists() and json.loads(state_path.read_text()).get('status') == 'complete':
            return json.loads(state_path.read_text())
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(index=index, started=time.time())
        state.update(status='starting', resumed=time.time())
        download.save(state_path,state)
        try:
            archive = download.acquire(f'public/v1.0/v1.0-trainval{index:02d}_keyframes.tgz', None, state, state_path)
            state.update(status='extracting');download.save(state_path,state)
            records = []
            with tarfile.open(archive, 'r|gz') as t:
                for member in t:
                    name = str(PurePosixPath(member.name))
                    if name not in required:
                        continue
                    if not member.isfile() or member.size > 10*1024**2:
                        raise ValueError('Unexpected camera archive member')
                    content = t.extractfile(member).read()
                    if len(content) != member.size or not content.startswith(b'\xff\xd8'):
                        raise ValueError('Truncated or non-JPEG camera')
                    dest = data/name
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    sha = hashlib.sha256(content).hexdigest()
                    if dest.exists():
                        assert hashlib.sha256(dest.read_bytes()).hexdigest()==sha
                    else:
                        tmp=dest.with_suffix('.partial');tmp.write_bytes(content);tmp.replace(dest)
                    records.append(dict(path=name, bytes=member.size, sha256=sha))
            download.save(directory/'files.json',records)
            state.update(status='complete',files=len(records),bytes_extracted=sum(x['bytes'] for x in records),finished=time.time())
            state.pop('error', None)
            download.save(state_path,state)
            archive.unlink()
            print('completed_camera_archive',index,len(records),flush=True)
            return state
        except Exception as error:
            state.update(status='failed',error=repr(error));download.save(state_path,state)
            raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results=list(ex.map(one,range(1,11)))
    found={p['path'] for i in range(1,11) for p in json.loads((ROOT/f'camera-acquisition/{i:02d}/files.json').read_text())}
    assert found==required
    download.save(ROOT/'camera-acquisition/complete.json',dict(complete=True,images=len(found),archives=results))


if __name__=='__main__':
    main()
