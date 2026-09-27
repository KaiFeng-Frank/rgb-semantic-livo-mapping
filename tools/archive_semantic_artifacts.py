#!/usr/bin/env python3
"""Preserve semantic experiment artefacts as verified GitHub Release assets.

prepare runs on the experiment host. upload reads a GitHub token from stdin only;
it never saves it or sends it to any host other than api.github.com/uploads.github.com.
Raw datasets and the withdrawn v0.6 seed-2 checkpoint are excluded.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import threading
import time
from urllib.parse import quote

REPO = 'KaiFeng-Frank/rgb-semantic-livo-mapping'
SOURCE_COMMIT = 'fb36c69337dce37cb1195defafcbf0723d3d7693'
ARMS = ['armB0_s1', 'armB0_s2', 'armB0_s3', 'armB0_noKL_s1',
        'armB0_noKL_s2', 'armB0_noKL_s3', 'armRprime_noKL_s1']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def prepare(root, dest):
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema=1, repository=REPO, source_commit=SOURCE_COMMIT,
                    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    excluded=['raw downloadable datasets and ROS bags',
                              'withdrawn exp/sk2/armB0_noKL_s2 checkpoint',
                              'reconstructible full prediction caches other than the specified demo arms'],
                    assets=[])

    def add(path, name, category, restore=None):
        assert path.is_file(), path
        size = path.stat().st_size
        assert 0 < size < 2 * 1024**3, (path, size)
        entry = dict(name=name, source=str(path.relative_to(root)), bytes=size,
                     sha256=digest(path), category=category)
        if restore:
            entry['restore_to'] = restore
        manifest['assets'].append(entry)
        print('PREPARED', name, size, entry['sha256'], flush=True)
        return entry

    # These names deliberately match the clean release's exact inference tensors.
    for arm in ARMS:
        report = json.loads((root / ('out/v06_score/extract_' + arm + '.json')).read_text())
        p = root / ('weights/v06/' + arm + '_student.pth')
        e = add(p, 'v06-' + arm + '-student.pth', 'v06-inference', str(p.relative_to(root)))
        assert report['PASS'] and e['sha256'] == report['out_sha256'], ('student provenance', arm)
        trained = arm + '_clean' if arm == 'armB0_noKL_s2' else arm
        best = root / ('exp/sk2/' + trained + '/model/model_best.pth')
        assert digest(best) == report['source']['sha256'], ('checkpoint provenance', arm)
    for tag in ['B0', 'Rprime_noKL', 'ZS']:
        p = root / ('weights/v05/' + tag + '_student.pth')
        add(p, 'v05-' + tag + '-student.pth', 'v05-online-and-figure-model', str(p.relative_to(root)))
    for arm in ARMS:
        trained = arm + '_clean' if arm == 'armB0_noKL_s2' else arm
        for kind in ['best', 'last']:
            p = root / ('exp/sk2/' + trained + '/model/model_' + kind + '.pth')
            add(p, 'v06-' + arm + '-' + kind + '.pth', 'v06-training-checkpoint', str(p.relative_to(root)))
    # Earlier component comparisons remain usable without rerunning training.
    for arm in ['armB0', 'armB1', 'armC', 'armD', 'armD_noKL', 'armR', 'armRprime_noKL']:
        p = root / ('exp/sk/' + arm + '/model/model_best.pth')
        if p.exists():
            add(p, 'v04-' + arm + '-best.pth', 'v04-component-reference', str(p.relative_to(root)))

    def bundle(name, files, category):
        paths = sorted(set(p for p in files if p.is_file() and not p.is_symlink()))
        assert paths, name
        index = []
        target = dest / name
        partial = target.with_suffix(target.suffix + '.partial')
        with tarfile.open(partial, 'w') as tf:
            for p in paths:
                assert root.resolve() in p.resolve().parents, p
                rel = str(p.relative_to(root))
                assert '/.git/' not in rel and p.name not in ('.env', 'id_rsa', 'id_ed25519')
                index.append(dict(path=rel, bytes=p.stat().st_size, sha256=digest(p)))
                info = tf.gettarinfo(str(p), arcname=rel)
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                with p.open('rb') as f:
                    tf.addfile(info, f)
        partial.replace(target)
        save(dest / (name + '.index.json'), index)
        e = add(target, name, category)
        e['archive_members'] = len(index)
        e['archive_index'] = name + '.index.json'

    online = [p for p in (root / 'out/v06/runs').iterdir() if p.is_file()]
    bundle('v06-online-evidence.tar', online, 'online-poses-latency-maps')
    older = [p for p in (root / 'out/v05').iterdir() if p.is_file()]
    bundle('v05-map-and-figure-evidence.tar', older, 'v05-map-figures')
    for seq in ['07', '09']:
        for arm in ['ZS', 'armB0_s1', 'armB0_noKL_s1']:
            p = root / ('out/v06_score/pred/' + arm + '_' + seq + '_r1')
            assert (p / '.done').exists(), p
            bundle('v06-pred-' + arm + '-seq' + seq + '-r1.tar', list(p.iterdir()),
                   'paired-video-predictions')

    # Small source/configuration and result metadata, no credentials or data inputs.
    meta = []
    for rel in ['config', 'opt', 'tools', 'src', 'out/v06_train']:
        for p in (root / rel).rglob('*'):
            parts = p.relative_to(root).parts
            if any(x in ('.git', '__pycache__', 'build', 'dist', '.cache') for x in parts):
                continue
            if p.is_file() and (p.suffix in ('.py', '.sh', '.yaml', '.yml', '.json', '.txt', '.md', '.log', '.patch', '.diff', '.hpp', '.h', '.cpp', '.cu', '.cuh') or p.name.startswith(('LICENSE', 'COPYING'))) and p.stat().st_size < 12 * 1024**2:
                meta.append(p)
    for p in (root / 'out/v06_score').rglob('*'):
        if 'pred' in p.relative_to(root / 'out/v06_score').parts:
            continue
        if p.is_file() and p.suffix in ('.json', '.txt', '.log', '.md'):
            meta.append(p)
    for rel in ['out/v06', 'out/v04']:
        meta.extend(p for p in (root / rel).iterdir() if p.is_file() and p.suffix in ('.json', '.txt', '.log', '.md', '.png'))
    meta += [root / 'CRITICAL_CONSTRAINTS.md', root / 'HANDOFF_kitti_bag.md', root / 'HANDOFF_fastlivo2_trajectory.md']
    meta += list((root / 'weights/nuscenes-semseg-pt-v3m1-0-base').glob('*.py'))
    # Training config/logs, with the invalid checkpoint directory excluded.
    for parent in ['exp/sk', 'exp/sk2']:
        for arm in (root / parent).iterdir():
            if parent == 'exp/sk2' and arm.name == 'armB0_noKL_s2':
                continue
            meta.extend(p for p in arm.iterdir() if p.is_file() and p.suffix in ('.py', '.json', '.txt', '.log'))
    secret = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    for p in meta:
        if secret.search(p.read_bytes()):
            raise RuntimeError('Potential credential in proposed archive: ' + str(p.relative_to(root)))
    envs = {}
    for name in ['ptv3', 'ags']:
        py = Path('/data/wuyou/miniconda3/envs') / name / 'bin/python'
        code = 'import importlib.metadata as m,json,sys; print(json.dumps(dict(python=sys.version,packages=sorted([(d.metadata.get("Name",""),d.version) for d in m.distributions()]))))'
        envs[name] = json.loads(subprocess.check_output([str(py), '-c', code], text=True))
    save(dest / 'environment.json', dict(platform=platform.platform(), environments=envs))
    meta.append(dest / 'environment.json')
    bundle('semantic-runtime-and-metadata.tar', meta, 'source-configuration-provenance')
    for path in sorted(dest.glob('*.index.json')):
        add(path, path.name, 'archive-member-checksums')
    add(dest / 'environment.json', 'environment.json', 'environment')
    manifest['total_bytes'] = sum(a['bytes'] for a in manifest['assets'])
    save(dest / 'artifact-manifest.json', manifest)
    (dest / 'SHA256SUMS').write_text(''.join(a['sha256'] + '  ' + a['name'] + '\n' for a in manifest['assets']))
    print('READY', len(manifest['assets']), manifest['total_bytes'], flush=True)


def upload(root, dest, release_id, workers):
    import requests
    token = sys.stdin.readline().strip()
    assert token and '\n' not in token
    headers = dict(Authorization='Bearer ' + token, Accept='application/vnd.github+json',
                   **{'X-GitHub-Api-Version': '2022-11-28'})
    api = 'https://api.github.com/repos/' + REPO
    url = 'https://uploads.github.com/repos/' + REPO + '/releases/' + str(release_id) + '/assets'
    release = requests.get(api + '/releases/' + str(release_id), headers=headers, timeout=30)
    release.raise_for_status()
    download_base = ('https://github.com/' + REPO + '/releases/download/' +
                     quote(release.json()['tag_name'], safe='') + '/')
    manifest = json.loads((dest / 'artifact-manifest.json').read_text())
    entries = list(manifest['assets'])
    for name in ['artifact-manifest.json', 'SHA256SUMS', 'RESTORE.md']:
        path = dest / name
        entries.append(dict(name=name, source=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=digest(path)))
    existing = []
    for page in range(1, 10):
        res = requests.get(api + '/releases/' + str(release_id) + '/assets', headers=headers,
                           params=dict(per_page=100, page=page), timeout=30)
        res.raise_for_status()
        found = res.json()
        existing.extend(found)
        if len(found) < 100:
            break
    existing = {a['name']: a for a in existing}
    lock = threading.Lock()
    receipt = dict(repository=REPO, release_id=release_id, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), assets=[])

    def check(a, entry):
        assert a['state'] == 'uploaded' and a['size'] == entry['bytes'], entry['name']
        assert a.get('digest') == 'sha256:' + entry['sha256'], ('GitHub digest mismatch', entry['name'], a.get('digest'))
        return dict(name=entry['name'], id=a['id'], bytes=a['size'], sha256=entry['sha256'],
                    github_digest=a['digest'],
                    browser_download_url=download_base + quote(entry['name'], safe=''))

    def one(entry):
        started = time.monotonic()
        old = existing.get(entry['name'])
        if old and old['state'] == 'uploaded':
            verified = check(old, entry)
        else:
            path = root / entry['source']
            assert path.stat().st_size == entry['bytes']
            assert digest(path) == entry['sha256']
            print('UPLOADING', entry['name'], entry['bytes'], flush=True)
            # A single uploader must own this draft. Remove only an incomplete
            # asset with this exact manifest name when resuming a stopped upload.
            for attempt in range(4):
                if old:
                    if old['state'] == 'uploaded':
                        verified = check(old, entry)
                        break
                    assert old['state'] == 'starter', (entry['name'], old['state'])
                    res = requests.delete(api + '/releases/assets/' + str(old['id']), headers=headers, timeout=30)
                    res.raise_for_status()
                    old = None
                try:
                    with path.open('rb') as f:
                        res = requests.post(url, params=dict(name=entry['name']),
                                            headers={**headers, 'Content-Type': 'application/octet-stream',
                                                     'Content-Length': str(entry['bytes'])}, data=f,
                                            timeout=(30, 300))
                    if res.status_code != 201:
                        raise RuntimeError('HTTP ' + str(res.status_code))
                    verified = check(res.json(), entry)
                    break
                except (requests.RequestException, RuntimeError) as error:
                    if attempt == 3:
                        raise RuntimeError('Upload failed after retries: ' + entry['name']) from error
                    print('RETRY', entry['name'], attempt + 1, type(error).__name__, flush=True)
                    time.sleep(5 * (attempt + 1))
                    res = requests.get(api + '/releases/' + str(release_id) + '/assets', headers=headers,
                                       params=dict(per_page=100), timeout=30)
                    res.raise_for_status()
                    old = next((a for a in res.json() if a['name'] == entry['name']), None)
        with lock:
            receipt['assets'].append(verified)
            save(dest / 'upload-receipt.json', receipt)
        print('VERIFIED', entry['name'], 'seconds', round(time.monotonic() - started, 2), flush=True)

    # Save the main models first; metadata/large checkpoints follow in the declared order.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, entry) for entry in entries]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    receipt['complete'] = True
    receipt['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(dest / 'upload-receipt.json', receipt)
    print('ALL_VERIFIED', len(receipt['assets']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'upload'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--release-id', type=int)
    parser.add_argument('--workers', type=int, default=6)
    a = parser.parse_args()
    if a.action == 'prepare':
        prepare(a.root, a.dest)
    else:
        assert a.release_id
        assert 1 <= a.workers <= 8
        upload(a.root, a.dest, a.release_id, a.workers)


if __name__ == '__main__':
    main()
