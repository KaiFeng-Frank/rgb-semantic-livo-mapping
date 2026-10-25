"""Engineering checks: same-forward feature capture, bounded GPU memory and IPC.

Uses existing nuScenes point files read-only. Repeats a fixed real scan with an
identity transform to isolate integration; this is NOT a driving/accuracy test.
No dataset, feature cache or training checkpoint is written by this tool.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def stats(values):
    return dict(n=len(values), median_ms=statistics.median(values),
                p95_ms=float(np.percentile(values, 95)), max_ms=max(values))


def run(scan, out, repetitions):
    import torch
    from ptv3_worker import Segmenter
    from ptv3_client import PTv3Client
    from semantic_memory import CausalSemanticMemory, MemoryConfig
    torch.set_num_threads(1)
    torch.manual_seed(20261025)
    out.mkdir(parents=True, exist_ok=False)
    points = np.fromfile(scan, dtype='<f4').reshape(-1, 5)[:, :4].copy()
    points[:, 3] /= 255.
    model = Segmenter(half=True, shuffle=False, intensity_scale=1.)
    # Check exactly this captured feature tensor against the logits from its own forward.
    raw = model.segment(points, return_features=True)
    with torch.inference_mode():
        torch.testing.assert_close(model.model.seg_head(raw['features']), raw['logits'], atol=0, rtol=0)
    capture = dict(raw_points=len(points), voxels=len(raw['xyz']), feature_dim=raw['features'].shape[1],
                   inverse_min=int(raw['inverse'].min()), inverse_max=int(raw['inverse'].max()),
                   same_forward_head_exact=True, state_tensors=len(model.model.state_dict()))
    cfg = MemoryConfig()
    memory = CausalSemanticMemory(cfg, 'cuda').eval()
    transform = torch.eye(4, device='cuda')
    overhead, changed = [], []
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    with torch.inference_mode():
        for i in range(repetitions + 5):
            torch.cuda.synchronize(); started = time.perf_counter()
            result = memory(raw['xyz'].float(), raw['features'], raw['logits'], frame=i,
                            stamp=i*.1, transform=transform, shadow=True)
            torch.cuda.synchronize(); elapsed = 1000*(time.perf_counter()-started)
            if not torch.equal(result['logits'], raw['logits'].float()):
                raise AssertionError('Shadow changed the same-forward logits')
            if i >= 5:
                overhead.append(elapsed)
                changed.append(int(result['has_history'].sum()))
    extra_peak = torch.cuda.max_memory_allocated() - allocated_before
    # Release main-process model before spawning the second process on a laptop GPU.
    del result, memory, raw, model
    torch.cuda.empty_cache()
    timings = {}
    for mode in ('baseline', 'shadow'):
        client = PTv3Client(half=True, shuffle=False, intensity_scale=1.,
                           memory_shadow=mode == 'shadow', stderr_path=str(out/f'{mode}-worker.log'))
        measurements = []
        try:
            for i in range(repetitions + 5):
                context = dict(stamp=i*.1, frame=i, epoch=0, transform=np.eye(4)) if mode == 'shadow' else None
                started = time.perf_counter()
                labels, confidence = client.segment(points, memory_context=context)
                elapsed = 1000*(time.perf_counter()-started)
                if labels.shape != (len(points),) or not np.isfinite(confidence).all() or np.any(labels >= 16):
                    raise AssertionError('Invalid IPC segmentation output')
                if i >= 5:
                    measurements.append(dict(client_ms=elapsed, worker_ms=client.last_infer_ms))
        finally:
            client.close()
        timings[mode] = dict(client=stats([v['client_ms'] for v in measurements]),
                             worker=stats([v['worker_ms'] for v in measurements]), records=measurements)
    result = dict(scope='engineering_fixed_real_scan_repeated_not_driving_not_accuracy',
                  source_sha256=hashlib.sha256(scan.read_bytes()).hexdigest(),
                  intensity='nuScenes remission /255; PTv3 intensity_scale=1.0',
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, capture=capture,
                  memory_only=stats(overhead), memory_extra_peak_bytes=extra_peak,
                  supported_points=changed, ipc=timings, trained=False,
                  same_forward_shadow_identity=True, raw_data_written=False)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--repetitions', type=int, default=20)
    a = ap.parse_args()
    if a.repetitions < 1:
        ap.error('positive repetitions required')
    run(a.scan, a.out, a.repetitions)
