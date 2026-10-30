"""Prepare immutable source-scan geometry on CPU while the encoder runtime builds."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
from tools.instance_memory.data import read, save
from .glfs_baseline import observation_geometry
from .nuscenes_pilot_data import digest


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def prepare(root, data):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
    from ptv3_worker import voxelize
    index, protocol = read(root/'inputs/index.json'), read(root/'protocol.json')
    folder = root/'geometry'
    folder.mkdir(exist_ok=False)
    manifest = dict(frames=[], scope='Source-only geometry; no labels, backbone or GPU; measured cost remains part of the adapter')
    for scenes in protocol['parts'].values():
        for scene in scenes:
            (folder/scene).mkdir()
            for k, row in enumerate(index['sequences'][scene]):
                path = data/row['scan_path']
                if digest(path) != row['scan_sha256']:
                    raise ValueError('Input changed')
                scan = np.fromfile(path, dtype='<f4').reshape(-1, 5)
                xyz, _, _, inverse = voxelize(scan[:, :3], scan[:, 3:4]/255)
                start = time.perf_counter()
                depth, angle, stats = observation_geometry(xyz)
                seconds = time.perf_counter()-start
                dest = folder/scene/f'{k:03d}.npz'
                np.savez(dest, depth=depth, angle=angle, xyz_sha256=array_hash(xyz),
                         inverse_sha256=array_hash(inverse.astype(np.int32)), source_sha256=row['scan_sha256'])
                manifest['frames'].append(dict(scene=scene, frame=k, seconds=seconds, sha256=digest(dest), **stats))
            save(folder/'progress.json', dict(frames=len(manifest['frames']), scene=scene))
            print('geometry', scene, len(manifest['frames']), flush=True)
    save(folder/'manifest.json', manifest)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args()
    prepare(a.root, a.data)
