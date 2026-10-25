"""Measure real PTv3 forward/backward memory; random targets are NOT training results."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .runtime import build, prepare, logits, load_source, POINTCEPT_COMMIT
from .schema import save, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pointcept', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--scan', type=Path)
    ap.add_argument('--points', type=int, default=24000)
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    torch.manual_seed(20261024)
    torch.set_num_threads(4)
    rng = np.random.default_rng(20261024)
    if a.scan:
        points = np.fromfile(a.scan, dtype='<f4').reshape(-1, 4)
    else:
        points = rng.uniform([-40, -40, -3, 0], [40, 40, 3, 1], (a.points, 4)).astype(np.float32)
    report = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                  pointcept=POINTCEPT_COMMIT, attention='torch SDPA packed segments',
                  pooling='torch.segment_reduce', raw_points=len(points),
                  scan_sha256=digest(a.scan) if a.scan else None,
                  real_scan=bool(a.scan), synthetic_targets=True, benchmark_score=False,
                  new_19_class_head=True, training_completed=False, steps=[])
    try:
        model = build(a.pointcept, classes=19, training=True)
        if a.checkpoint:
            report['load'] = load_source(model, a.checkpoint, new_head=True)
            report['checkpoint_sha256'] = digest(a.checkpoint)
        model = model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.005)
        scaler = torch.amp.GradScaler('cuda')
        report['parameters'] = sum(p.numel() for p in model.parameters())
        d, first, inv = prepare(points)
        report['voxels'] = len(first)
        target = torch.as_tensor(rng.integers(0, 19, len(first)), device='cuda')
        for step in range(a.steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.monotonic()
            with torch.autocast('cuda', dtype=torch.float16):
                output = logits(model, d)
                loss = F.cross_entropy(output.float(), target)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            record = dict(step=step, seconds=time.monotonic() - t0, loss=float(loss.detach()),
                          gradient_norm=float(norm), max_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          max_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            report['steps'].append(record)
            print(json.dumps(record), flush=True)
        del optimizer, d, output, loss, target
        gc.collect()
        model = model.eval().half()
        d, first, inv = prepare(points, half=True)
        with torch.inference_mode():
            pred = logits(model, d).argmax(-1).cpu().numpy()[inv]
        if len(pred) != len(points) or not ((pred >= 0) & (pred < 19)).all():
            raise ValueError('Invalid full-point inference output')
        report.update(success=True, full_point_prediction_count=len(pred))
    except Exception as e:
        report.update(success=False, error_type=type(e).__name__, error=str(e))
        save(a.out, report)
        raise
    save(a.out, report)


if __name__ == '__main__':
    main()
