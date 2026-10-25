"""Extract one frozen forward per real scan to a bounded, temporary RAM cache."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
from .nuscenes_pilot_data import LABEL_MAP, digest
from .pilot_metrics import object_groups


def extract(index_path, cache):
    import torch
    from ptv3_worker import Segmenter
    from ptv3_loader_verified import CKPT_PATH, serialization_policy
    index = json.loads(index_path.read_text())
    cache.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(20261025)
    segmenter = Segmenter(half=True, shuffle=False, intensity_scale=1.)
    manifest = dict(index_sha256=digest(index_path), backbone_sha256=digest(CKPT_PATH),
        model_config_sha256=segmenter.memory_model_sha256(), synthetic=False,
        serialization_policy=serialization_policy(segmenter.model),
        geometry=index['geometry'], intensity=index['intensity'], files=[], source_codes={})
    for name in ('src/ptv3_worker.py','src/ptv3_loader_verified.py',
                 'tools/semantic_memory/nuscenes_pilot_cache.py','tools/semantic_memory/pilot_metrics.py'):
        manifest['source_codes'][name] = digest(REPO/name)
    total_bytes = 0
    for scene, rows in index['sequences'].items():
        folder = cache/scene
        folder.mkdir()
        origin_inv = np.linalg.inv(np.array(rows[0]['transform']))
        started = time.monotonic()
        for row in rows:
            if shutil.disk_usage(cache).free < 600*1024**2:
                raise RuntimeError('RAM cache has less than 600 MiB available; stopping before exhaustion')
            scan = Path(index['data_root'])/row['scan_path']
            label_path = Path(index['data_root'])/row['label_path']
            points = np.fromfile(scan, dtype='<f4').reshape(-1,5)[:,:4].copy()
            points[:,3] /= 255.
            raw_labels = np.fromfile(label_path, dtype=np.uint8)
            if len(raw_labels) != len(points) or np.any(raw_labels >= 32):
                raise ValueError('Invalid real labels')
            labels = LABEL_MAP[raw_labels]
            with torch.inference_mode():
                raw = segmenter.segment(points, return_features=True)
            data = {k:raw[k].detach().cpu().numpy() for k in ('xyz','features','logits','inverse')}
            data['inverse'] = data['inverse'].astype(np.int32)
            data['point_label'] = labels.astype(np.int8)
            valid = labels >= 0
            counts = np.bincount(data['inverse'][valid]*16+labels[valid], minlength=len(data['xyz'])*16).reshape(-1,16)
            data['label'] = np.where(counts.sum(1)>0, counts.argmax(1), -1).astype(np.int8)
            t = np.asarray(row['transform'])
            data['transform'] = (origin_inv @ t).astype(np.float32)
            world = points[:,:3] @ t[:3,:3].T + t[:3,3]
            groups = object_groups(world, labels, row['objects'])
            data['group_owner'] = np.full(len(points), -1, np.int32)
            for j,g in enumerate(groups):
                data['group_owner'][g['indices']] = j
            data['groups'] = np.array(json.dumps([{k:v for k,v in g.items() if k!='indices'} for g in groups]))
            path = folder/f"{row['frame']:03d}.npz"
            np.savez(path, **data)
            total_bytes += path.stat().st_size
            manifest['files'].append(dict(scene=scene, frame=row['frame'],
                path=str(path.relative_to(cache)), sha256=digest(path), bytes=path.stat().st_size,
                voxels=len(data['xyz']), points=len(points), objects=len(groups)))
            del raw, data
        print(json.dumps(dict(stage='features', scene=scene, frames=len(rows), elapsed_s=time.monotonic()-started,
                              total_cache_bytes=total_bytes)), flush=True)
    manifest['total_cache_bytes'] = total_bytes
    (cache/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copyfile(cache/'manifest.json', index_path.parent/'feature-manifest.json')
    print(json.dumps(dict(stage='features_complete', files=len(manifest['files']), bytes=total_bytes)), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    args=p.parse_args()
    extract(args.index,args.cache)
