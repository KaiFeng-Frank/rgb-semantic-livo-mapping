"""Real-scan probe of serialization randomness, independent of temporal heads.

Reconstructs the historical root-off / pooling-on policy explicitly, then tests
the corrected all-off policy. A-A and A-B-A comparisons follow warmup; no GT is
used. Reports residual floating-point variation separately from RNG consumption.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
from ptv3_loader_verified import serialization_policy, set_serialization_policy, CKPT_PATH
from ptv3_worker import Segmenter
from .nuscenes_pilot_data import digest


def compare(a, b):
    if not (torch.equal(a['xyz'], b['xyz']) and torch.equal(a['inverse'], b['inverse'])):
        raise AssertionError('Input voxelization changed')
    lab_a, lab_b = a['logits'].argmax(-1), b['logits'].argmax(-1)
    out = dict(voxels=len(lab_a), points=len(a['inverse']),
               changed_voxel_labels=int((lab_a != lab_b).sum()),
               changed_point_labels=int((lab_a[a['inverse']] != lab_b[b['inverse']]).sum()))
    for key in ('logits', 'features'):
        delta = (a[key].float()-b[key].float()).abs()
        out[key] = dict(equal=torch.equal(a[key], b[key]), max_abs=float(delta.max()),
                        mean_abs=float(delta.mean()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    index = json.loads(a.index.read_text())
    torch.set_num_threads(1)
    torch.manual_seed(20261025)
    segmenter = Segmenter(half=True, shuffle=False, intensity_scale=1.)
    native = serialization_policy(segmenter.model)
    if not native or any(native.values()):
        raise AssertionError('Loader failed to disable all serialization randomness')
    # First/middle/last scene: selected by index order, before reading predictions.
    scenes = list(index['sequences'])
    scenes = [scenes[0], scenes[len(scenes)//2], scenes[-1]]
    results = []
    for scene in scenes:
        rows = index['sequences'][scene][:2]
        scans = []
        for row in rows:
            points = np.fromfile(Path(index['data_root'])/row['scan_path'], dtype='<f4').reshape(-1,5)[:,:4].copy()
            points[:,3] /= 255.
            scans.append(points)
        def forward(which):
            state = torch.get_rng_state().clone()
            with torch.inference_mode():
                raw = segmenter.segment(scans[which], return_features=True)
            raw = {k:v.detach().cpu().clone() for k,v in raw.items()}
            return raw, not torch.equal(state, torch.get_rng_state())
        for policy in ('legacy_root_off_pooling_on', 'all_off'):
            set_serialization_policy(segmenter.model, False)
            if policy.startswith('legacy'):
                for module in segmenter.model.modules():
                    if type(module).__name__ == 'SerializedPooling':
                        module.shuffle_orders = True
            torch.manual_seed(20261025)
            forward(0); forward(1)  # Warm both shapes before the comparisons.
            first, r0 = forward(0)
            second, r1 = forward(0)
            _, rb = forward(1)
            third, r2 = forward(0)
            result = dict(scene=scene, frames=[r['frame'] for r in rows], policy=policy,
                effective_policy=serialization_policy(segmenter.model),
                model_config_sha256=segmenter.memory_model_sha256(),
                cpu_rng_consumed=[r0,r1,rb,r2], adjacent=compare(first,second),
                intervening_scan=compare(second,third))
            if policy == 'all_off' and any(result['cpu_rng_consumed']):
                raise AssertionError('Fixed inference still consumes CPU RNG')
            results.append(result)
            print(json.dumps(result), flush=True)
    out = dict(scope='frozen backbone, real scans; no training or labels',
        native_loader_policy=native, index_sha256=digest(a.index),
        backbone_sha256=digest(CKPT_PATH),
        source_sha256={name:digest(REPO/name) for name in
                      ('src/ptv3_loader_verified.py','src/ptv3_worker.py',
                       'tools/semantic_memory/probe_serialization.py')}, results=results)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n')


if __name__ == '__main__':
    main()
