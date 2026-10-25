"""Separate GT-supervised native-19 reference; never overwrites the weak B0 model.

The default run is a small feasibility pilot. Full training requires complete
official training sequences, validated before reading any labels. Sequence 08 is
never accepted for training. Output/checkpoints belong outside Git source.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .runtime import build, prepare, logits, load_source, POINTCEPT_COMMIT
from .schema import FRAME_COUNTS, NAMES, raw_to_learning, digest, save

TRAIN_SEQUENCES = tuple(f'{i:02d}' for i in range(11) if i != 8)


def inventory(dataset, sequences, full=False):
    if not sequences or len(set(sequences)) != len(sequences) or any(s not in TRAIN_SEQUENCES for s in sequences):
        raise ValueError('Train only on 00..10 excluding validation sequence 08')
    if full and sorted(sequences) != list(TRAIN_SEQUENCES):
        raise ValueError('Full reference training needs all official training sequences')
    records = []
    for seq in sequences:
        folder = Path(dataset) / 'sequences' / seq
        scans = sorted((folder / 'velodyne').glob('*.bin'))
        if not scans or (full and [p.stem for p in scans] != [f'{i:06d}' for i in range(FRAME_COUNTS[seq])]):
            raise ValueError('Empty/incomplete training sequence: ' + seq)
        if any(not p.stem.isdigit() or len(p.stem) != 6 or int(p.stem) >= FRAME_COUNTS[seq] for p in scans):
            raise ValueError('Invalid official frame ID: ' + seq)
        for scan in scans:
            label = folder / 'labels' / (scan.stem + '.label')
            if scan.stat().st_size == 0 or scan.stat().st_size % 16 or not label.exists() or label.stat().st_size != scan.stat().st_size // 4:
                raise ValueError('Missing or malformed scan/labels: ' + str(scan))
            records.append(dict(scan=str(scan), label=str(label), sequence=seq,
                                scan_sha256=digest(scan), label_sha256=digest(label)))
    return records


def training_sample(record, seed, step, max_voxels):
    rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
    points = np.fromfile(record['scan'], dtype='<f4').reshape(-1, 4).copy()
    labels = raw_to_learning(np.fromfile(record['label'], dtype='<u4'))
    # Shuffle before selecting one representative per voxel; labels follow points.
    perm = rng.permutation(len(points))
    points, labels = points[perm], labels[perm]
    theta = rng.uniform(-np.pi, np.pi)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    points[:, :2] = points[:, :2] @ rotation.T
    points[:, :3] *= rng.uniform(.95, 1.05)
    d, first, _ = prepare(points, device='cpu')
    target = labels[first] - 1  # official ignored 0 becomes CE ignore_index -1
    if len(first) > max_voxels:
        center = rng.integers(len(first))
        distance = np.square(points[first, :3] - points[first[center], :3]).sum(1)
        chosen = np.argpartition(distance, max_voxels - 1)[:max_voxels]
        for key in ('coord', 'feat', 'grid_coord'):
            d[key] = d[key][chosen]
        d['offset'] = torch.tensor([len(chosen)])
        target = target[chosen]
    if not (target >= 0).any():
        raise ValueError('Training crop has no evaluable labels')
    return {k: v.cuda() for k, v in d.items()}, torch.as_tensor(target, device='cuda')


def save_checkpoint(path, model, optimizer, scaler, step, config, inventory_hash):
    tmp = path.with_suffix('.pth.partial')
    torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), completed_steps=step, config=config,
                    inventory_sha256=inventory_hash, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(), native19=True, class_names=NAMES,
                    supervision='SemanticKITTI official training 3D GT',
                    training_data_disclosure='nuScenes-supervised source PTv3 backbone; SemanticKITTI '
                    '3D training GT, sequences ' + ','.join(config['sequences']) +
                    '; no validation/test GT training; no camera/future scans/map fusion.'), tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--pointcept', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sequences', default='04')
    ap.add_argument('--mode', choices=['pilot', 'full'], default='pilot')
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--accumulation', type=int, default=4)
    ap.add_argument('--max-voxels', type=int, default=80000)
    ap.add_argument('--lr', type=float, default=.0001)
    ap.add_argument('--seed', type=int, default=20261024)
    ap.add_argument('--save-every', type=int, default=25)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    if min(a.steps, a.accumulation, a.max_voxels, a.save_every) < 1 or a.lr <= 0:
        raise ValueError('Steps, accumulation, crop size, checkpoint interval and LR must be positive')
    records = inventory(a.dataset, a.sequences.split(','), a.mode == 'full')
    config = dict(mode=a.mode, sequences=a.sequences.split(','), steps=a.steps,
                  accumulation=a.accumulation, max_voxels=a.max_voxels, lr=a.lr, seed=a.seed,
                  source_sha256=digest(a.source), pointcept=POINTCEPT_COMMIT,
                  classes=19, loss='unweighted cross entropy, ignore_index=-1',
                  grid_size=.05, intensity_scale=.2, augmentation='random voxel representative, z rotation, scale .95..1.05',
                  attention='torch SDPA', pooling='torch.segment_reduce', torch=str(torch.__version__))
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.resume and any(a.out.iterdir()):
        raise ValueError('Use a new empty run directory or explicitly resume')
    inventory_path = a.out / 'training-inventory.json'
    if a.resume:
        if json.loads(inventory_path.read_text()) != records:
            raise ValueError('Training dataset changed since checkpoint')
    else:
        save(inventory_path, records)
        save(a.out / 'config.json', config)
    inventory_hash = digest(inventory_path)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(4)
    model = build(a.pointcept, classes=19, training=True).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.005)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    start = 0
    checkpoint = a.out / 'last.pth'
    if a.resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        old_config = dict(saved['config'])
        old_steps = old_config.pop('steps')
        new_config = dict(config)
        new_config.pop('steps')
        if old_config != new_config or a.steps < old_steps or saved['inventory_sha256'] != inventory_hash:
            raise ValueError('Resume configuration or dataset differs from checkpoint')
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        start = saved['completed_steps']
        save(a.out / 'config.json', config)
        # Remove uncheckpointed log entries after a crash before redoing those
        # updates. Preserve the exact optimizer/scaler/RNG state of last.pth.
        metric_path = a.out / 'metrics.jsonl'
        if metric_path.exists():
            kept = [line for line in metric_path.read_text().splitlines() if json.loads(line)['step'] <= start]
            metric_path.write_text('\n'.join(kept) + ('\n' if kept else ''))
        del saved
    else:
        load_source(model, a.source, new_head=True)
    optimizer.zero_grad(set_to_none=True)
    status = dict(mode=a.mode, full_training=a.mode == 'full', benchmark_score=False,
                  device=torch.cuda.get_device_name(0), training_frames=len(records),
                  native19=True, supervision='official training 3D GT', completed_steps=start,
                  requested_steps=a.steps, training_finished=False, skipped_updates=0)
    try:
        for step in range(start, a.steps):
            t0 = time.monotonic()
            losses = []
            torch.cuda.reset_peak_memory_stats()
            for micro in range(a.accumulation):
                index = step * a.accumulation + micro
                rng = np.random.default_rng(np.random.SeedSequence([a.seed, index]))
                record = records[int(rng.integers(len(records)))]
                batch, target = training_sample(record, a.seed, index, a.max_voxels)
                with torch.autocast('cuda', dtype=torch.float16):
                    output = logits(model, batch)
                    loss = F.cross_entropy(output.float(), target, ignore_index=-1)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                scaler.scale(loss / a.accumulation).backward()
                losses.append(float(loss.detach()))
                del batch, target, output, loss
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            status['skipped_updates'] += int(skipped)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            metric = dict(step=step+1, loss=float(np.mean(losses)), seconds=time.monotonic()-t0,
                          gradient_norm=float(norm) if torch.isfinite(norm) else None,
                          skipped_update=skipped, max_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            with (a.out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(metric, allow_nan=False) + '\n')
            status.update(completed_steps=step+1, last_metric=metric)
            save(a.out / 'status.json', status)
            if (step+1) % 10 == 0 or step == start:
                print(json.dumps(metric), flush=True)
            if (step+1) % a.save_every == 0 or step+1 == a.steps:
                save_checkpoint(checkpoint, model, optimizer, scaler, step+1, config, inventory_hash)
    except Exception as e:
        status.update(error_type=type(e).__name__, error=str(e))
        save(a.out / 'status.json', status)
        raise
    status.update(training_finished=True, checkpoint_sha256=digest(checkpoint))
    save(a.out / 'status.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
