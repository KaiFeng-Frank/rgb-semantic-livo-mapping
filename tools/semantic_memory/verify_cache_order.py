"""Check every frozen scan in reverse extraction order in a fresh process."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
from ptv3_worker import Segmenter
from .nuscenes_pilot_data import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('index','cache','out'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    index = json.loads(a.index.read_text())
    manifest = json.loads((a.cache/'manifest.json').read_text())
    torch.set_num_threads(1)
    torch.manual_seed(971)  # Deliberately differs from extraction's 20261025.
    segmenter = Segmenter(half=True, shuffle=False, intensity_scale=1.)
    if segmenter.memory_model_sha256() != manifest['model_config_sha256']:
        raise ValueError('Different effective feature policy')
    frames = []
    state = torch.get_rng_state().clone()
    for entry in reversed(manifest['files']):
        scene, frame = entry['scene'], entry['frame']
        row = next(r for r in index['sequences'][scene] if r['frame']==frame)
        points = np.fromfile(Path(index['data_root'])/row['scan_path'], dtype='<f4').reshape(-1,5)[:,:4].copy()
        points[:,3] /= 255.
        with torch.inference_mode():
            raw = segmenter.segment(points, return_features=True)
        now = {k:v.detach().cpu().numpy() for k,v in raw.items()}
        with np.load(a.cache/entry['path'], allow_pickle=False) as old:
            if not all(np.array_equal(now[k], old[k]) for k in ('xyz','inverse')):
                raise AssertionError('Voxelization differs')
            before, after = old['logits'].argmax(-1), now['logits'].argmax(-1)
            record = dict(scene=scene, frame=frame,
                equal={k:bool(np.array_equal(now[k],old[k])) for k in ('features','logits')},
                max_abs={k:float(np.abs(now[k].astype(np.float32)-old[k].astype(np.float32)).max()) for k in ('features','logits')},
                changed_voxels=int((before!=after).sum()),
                changed_points=int((before[old['inverse']]!=after[now['inverse']]).sum()))
        frames.append(record)
        if len(frames)%40==0:
            print(json.dumps(dict(stage='reverse_replay', frames=len(frames),
                changed_points=sum(r['changed_points'] for r in frames))), flush=True)
    out = dict(scope='all scans, reversed scene/frame order, separate process and different RNG seed',
        index_sha256=digest(a.index), cache_manifest_sha256=digest(a.cache/'manifest.json'),
        source_sha256={name:digest(REPO/name) for name in ('src/ptv3_loader_verified.py',
            'src/ptv3_worker.py','tools/semantic_memory/verify_cache_order.py')},
        frames=len(frames), exact_feature_frames=sum(r['equal']['features'] for r in frames),
        exact_logit_frames=sum(r['equal']['logits'] for r in frames),
        changed_points=sum(r['changed_points'] for r in frames),
        changed_voxels=sum(r['changed_voxels'] for r in frames),
        cpu_rng_consumed=not torch.equal(state,torch.get_rng_state()), records=frames)
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='records'}),flush=True)
    if out['cpu_rng_consumed'] or out['changed_points']:
        raise AssertionError('Unresolved inference repeatability failure; see saved evidence')


if __name__=='__main__':
    main()
