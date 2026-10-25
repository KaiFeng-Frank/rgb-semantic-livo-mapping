#!/usr/bin/env python3
"""Restore pinned small source files, apply the reviewed patch, prepare ROS deps.

Does not install system packages, retrieve datasets/models, or modify other workspaces.
The source manifest verifies Git blob hashes before writing any upstream file.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                return r.read()
        except Exception:
            if attempt == 2: raise


def restore(manifest, dest):
    def one(entry):
        p = dest/entry['path']
        if p.exists(): data = p.read_bytes()
        else:
            data = fetch('https://raw.githubusercontent.com/'+manifest['repository']+'/'+manifest['commit']+'/'+entry['path'])
        blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob != entry['sha']:
            raise ValueError('source differs from pinned revision; refusing to overwrite '+str(p))
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists(): p.write_bytes(data)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(one, manifest['files']))


def replace(path, old, new):
    s = path.read_text()
    if new in s: return
    if s.count(old) != 1: raise ValueError('build patch mismatch: '+str(path))
    path.write_text(s.replace(old, new))


def prepare(workspace):
    workspace = workspace.resolve()
    marker = workspace/'.geometric-backend-source.json'
    patch = HERE/'patches/fastlivo2-backend.patch'
    wanted = dict(upstream=json.loads((HERE/'upstream-files.json').read_text())['commit'],
                  patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest())
    source = workspace/'src/fast_livo'
    if marker.exists():
        if json.loads(marker.read_text()) != wanted:
            raise ValueError('workspace has a different patch; use a new isolated workspace')
    else:
        restore(json.loads((HERE/'upstream-files.json').read_text()), source)
        subprocess.run(['git','apply','--check',str(patch)], cwd=source, check=True)
        subprocess.run(['git','apply',str(patch)], cwd=source, check=True)
        marker.write_text(json.dumps(wanted, indent=2)+'\n')
    (source/'Log/result').mkdir(parents=True, exist_ok=True)
    vikit = workspace/'src/rpg_vikit'
    vikit_marker = vikit/'.prepared'
    if not vikit_marker.exists():
        restore(json.loads((HERE/'vikit-files.json').read_text()), vikit)
        for name in ('vikit_common','vikit_ros'):
            p=vikit/name/'CMakeLists.txt'
            find='FIND_PACKAGE(Sophus REQUIRED)' if name=='vikit_common' else 'find_package(Sophus REQUIRED)'
            replace(p,find,find+'\nif(NOT Sophus_INCLUDE_DIRS AND TARGET Sophus::Sophus)\n  get_target_property(Sophus_INCLUDE_DIRS Sophus::Sophus INTERFACE_INCLUDE_DIRECTORIES)\nendif()')
            if name=='vikit_common':
                s=p.read_text().replace('-std=c++0x','-std=c++17').replace('-std=c++11','-std=c++17')
                s=s.replace('-march=native','-march=x86-64').replace('-mtune=native','-mtune=generic');p.write_text(s)
                replace(p,'  ${Sophus_LIBRARIES}\n  fmt::fmt)','  ${Sophus_LIBRARIES}\n  Sophus::Sophus\n  fmt::fmt)')
            else:
                replace(p,'  ${cpp_typesupport_target}\n)','  ${cpp_typesupport_target}\n  Sophus::Sophus\n)')
        vikit_marker.write_text('4b7abc838f5d2ca9137f70f122eaaeff9eaf0f50\n')
    # Upstream omits this build dependency; a clean runner must build/export
    # the pinned Sophus before either vikit package is configured.
    for name in ('vikit_common','vikit_ros'):
        replace(vikit/name/'package.xml', '<name>'+name+'</name>',
                '<name>'+name+'</name>\n  <depend>sophus</depend>')
    # The original 100 ms discovery race can silently substitute zero intrinsics.
    # Match the existing reproduction's service-readiness fix; no estimator math.
    replace(vikit/'vikit_ros/include/vikit/params_helper.h',
            'wait_for_service(std::chrono::milliseconds(100))',
            'wait_for_service(std::chrono::milliseconds(10000))')
    sophus=workspace/'src/sophus'
    if not sophus.exists():
        lock=json.loads((HERE/'dependencies.json').read_text())['sophus']
        data=fetch(lock['url'])
        if hashlib.sha256(data).hexdigest()!=lock['sha256']:raise ValueError('Sophus archive hash mismatch')
        sophus.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tf:
            for member in tf.getmembers():
                rel=Path(member.name).parts[1:]
                if not rel:continue
                if '..' in rel or member.issym() or member.islnk():raise ValueError('unsafe source member')
                p=sophus.joinpath(*rel)
                if member.isdir():p.mkdir(parents=True,exist_ok=True)
                elif member.isfile():
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tf.extractfile(member).read())
    # Message-only Livox dependency; this frontend currently accepts Velodyne.
    livox=workspace/'src/livox_ros_driver2'
    for p in (HERE/'livox_msgs').rglob('*'):
        if p.is_file():
            out=livox/p.relative_to(HERE/'livox_msgs');out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists() and out.read_bytes()!=p.read_bytes():raise ValueError('different local Livox package')
            out.write_bytes(p.read_bytes())
    print(json.dumps(dict(workspace=str(workspace),**wanted),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    args = ap.parse_args()
    prepare(args.workspace)
