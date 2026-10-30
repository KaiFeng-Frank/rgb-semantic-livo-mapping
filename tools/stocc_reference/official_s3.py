#!/usr/bin/env python3
"""Direct remote acquisition of official nuScenes files, with bounded disk use.

Retains all LiDAR keyframes and the frozen FSA sweep union. Raw point bytes are
never changed. A published-MD5 mismatch is explicitly recorded; an exact current
S3 multipart ETag can independently authenticate the downloaded object bytes.
"""
import concurrent.futures
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import re
import subprocess
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / 'downloads'
DATA = ROOT / 'data/nuscenes'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BASES = ['https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/',
         'https://d36yt3mvayqw5m.cloudfront.net/']
RESERVE = 20 * 1024**3


def aria2_transfer(key, part, size, etag, record, state_path):
    """Resume sequential prefixes or aria2 bitfields; never infer sparse-file progress from size."""
    mirrors = []
    for base in BASES:
        url = base + key
        try:
            with OPENER.open(urllib.request.Request(url, method='HEAD'), timeout=10) as head:
                if int(head.headers['Content-Length']) == size and head.headers['ETag'].strip('"') == etag:
                    mirrors.append(url)
        except Exception:
            continue
    if not mirrors:
        raise OSError('No mirror has the expected official object identity')
    inputs = state_path.parent/'aria2-sources.txt'
    # aria2 input-file entries need per-entry output options; global --out is
    # not applied to these entries. Keep the partial name/control bitmap paired.
    inputs.write_text('\t'.join(mirrors)+'\n  out='+part.name+'\n  dir='+str(DOWNLOADS)+'\n')
    command = ['aria2c', '--no-conf=true', '--continue=true', '--always-resume=true',
               '--auto-file-renaming=false', '--allow-overwrite=false', '--file-allocation=none',
               '--split=8', '--max-connection-per-server=8', '--min-split-size=16M',
               '--connect-timeout=10', '--timeout=15', '--lowest-speed-limit=64K',
               '--max-tries=20', '--retry-wait=3', '--auto-save-interval=5',
               '--summary-interval=5', '--human-readable=false', '--show-console-readout=false',
               '--console-log-level=notice', '--download-result=full', '--uri-selector=adaptive',
               '--header=If-Match: "'+etag+'"', '--dir='+str(DOWNLOADS), '--out='+part.name,
               '--input-file='+str(inputs)]
    env = os.environ.copy()
    for name in list(env):
        if name.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
            env.pop(name)
    log_path = state_path.parent/'aria2.log'
    with log_path.open('a') as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        record.update(status='downloading', engine='aria2', aria2_pid=process.pid,
                      mirror_sources=mirrors, aria2_log=str(log_path), updated=time.time())
        save(state_path, record)
        for line in process.stdout:
            log.write(line); log.flush()
            match = re.search(r'\[#[0-9a-f]+\s+(\d+)B?/(\d+)B?', line)
            if match:
                received, reported_size = map(int, match.groups())
                if reported_size == size and 0 <= received <= size:
                    record.update(bytes=received, updated=time.time(), progress_source='aria2-completed-bytes')
                    save(state_path, record)
        code = process.wait()
    record['aria2_exit_code'] = code
    if code or not part.exists() or part.stat().st_size != size or Path(str(part)+'.aria2').exists():
        raise OSError(f'aria2 did not finish: exit {code}; see {log_path}')
    record.update(bytes=size, updated=time.time())
    save(state_path, record)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def checksum_file(path):
    md5, sha, parts = hashlib.md5(), hashlib.sha256(), []
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b''):
            md5.update(chunk)
            sha.update(chunk)
            parts.append(hashlib.md5(chunk).digest())
    multipart = hashlib.md5(b''.join(parts)).hexdigest() + '-' + str(len(parts))
    return {'md5': md5.hexdigest(), 'sha256': sha.hexdigest(),
            'multipart_etag_8MiB': multipart}


def acquire(key, expected_md5, record, state_path):
    name = PurePosixPath(key).name
    final, part = DOWNLOADS / name, DOWNLOADS / (name + '.part')
    final_control = Path(str(final)+'.aria2')
    if final_control.exists():
        # An interrupted legacy transfer may have the final name and full
        # apparent size while still sparse. The aria2 bitmap is authoritative.
        if part.exists() or Path(str(part)+'.aria2').exists():
            raise RuntimeError('Ambiguous partial archives; inspect before resuming')
        if final.exists():
            final.replace(part)
        final_control.replace(Path(str(part)+'.aria2'))
    previous_etag = record.get('etag')
    for attempt in range(1, 31):
        url = BASES[(attempt - 1) % len(BASES)] + key
        try:
            with OPENER.open(urllib.request.Request(url, method='HEAD'), timeout=30) as head:
                size = int(head.headers['Content-Length'])
                etag = head.headers['ETag'].strip('"')
            if previous_etag and previous_etag != etag:
                raise ValueError('Source ETag changed across retries')
            previous_etag = etag
            record.update(source=url, etag=etag, size=size, attempt=attempt)
            candidate = final if final.exists() else part
            start = candidate.stat().st_size if candidate.exists() else 0
            if start > size:
                raise ValueError('Local object exceeds source length')
            control = Path(str(part)+'.aria2')
            if not final.exists() and shutil.which('aria2c') and (start < size or control.exists()):
                committed = min(start, record.get('bytes', start)) if control.exists() else start
                if shutil.disk_usage(ROOT).free < size - committed + RESERVE:
                    raise RuntimeError('Insufficient disk reserve for segmented download')
                aria2_transfer(key, part, size, etag, record, state_path)
                candidate = part
                start = size
            if start < size:
                if final.exists():
                    raise ValueError('Final archive is incomplete')
                if shutil.disk_usage(ROOT).free < size - start + RESERVE:
                    raise RuntimeError('Insufficient disk reserve for this archive')
                headers = {'Accept-Encoding': 'identity', 'If-Match': '"' + etag + '"'}
                if start:
                    headers['Range'] = f'bytes={start}-'
                with OPENER.open(urllib.request.Request(url, headers=headers), timeout=45) as response:
                    if start and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-')):
                        raise ValueError('Invalid range response')
                    if response.headers.get('ETag', '').strip('"') != etag:
                        raise ValueError('Download ETag differs from HEAD')
                    last = 0
                    with part.open('ab') as out:
                        for chunk in iter(lambda: response.read(1024**2), b''):
                            out.write(chunk)
                            start += len(chunk)
                            if time.time() - last >= 5:
                                out.flush()
                                record.update(status='downloading', bytes=start, updated=time.time())
                                save(state_path, record)
                                last = time.time()
                if part.stat().st_size != size:
                    raise OSError('Truncated download')
                candidate = part
            record.update(status='verifying', bytes=size, updated=time.time())
            save(state_path, record)
            hashes = checksum_file(candidate)
            published_match = hashes['md5'] == expected_md5 if expected_md5 else None
            etag_match = etag in (hashes['md5'], hashes['multipart_etag_8MiB'])
            record.update(hashes=hashes, published_md5=expected_md5,
                          published_md5_match=published_match, current_s3_etag_match=etag_match)
            if not (published_match or etag_match):
                raise ValueError('Neither published MD5 nor current S3 object ETag matches')
            record['integrity_basis'] = 'published-md5' if published_match else 'current-official-s3-etag'
            if expected_md5 and not published_match:
                record['checksum_discrepancy'] = 'Published MD5 differs; downloaded bytes match current official S3 object ETag exactly.'
            if candidate != final:
                candidate.replace(final)
            return final
        except (ValueError, RuntimeError):
            raise
        except Exception as error:
            record.update(status='retrying', error=repr(error), updated=time.time())
            save(state_path, record)
            if attempt == 30:
                raise
            time.sleep(min(20, attempt * 2))
