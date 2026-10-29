"""Fixed official-compile audit: shared detections, paired masks, cold/warm timing.

No semantic/instance GT is loaded. Cached detector masks are current inputs,
not reference answers. Prewarming may specialize graph shapes, so these warm
measurements cannot by themselves establish online readiness.
"""
import argparse
import gc
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from tools.instance_memory.data import read, save
from tools.semantic_memory.nuscenes_pilot_data import DATA, digest

PROTOCOL = Path(__file__).resolve().parents[2]/'config/sam2_compile_probe_20261029.json'


def budget(path):
    if shutil.disk_usage(path).free < read(PROTOCOL)['min_free_disk_bytes']:
        raise RuntimeError('Disk reserve reached; stop without deleting evidence')


def prepare(frames, images, yolo, out):
    import cv2
    import torch
    from .fullrate import FullRateDetector
    from .kalman_cache import gpu_admission
    admission = gpu_admission()
    if out.exists():
        raise ValueError('Refuse overwrite')
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    detector = FullRateDetector(yolo)
    entries = []
    for i, row in enumerate(read(frames)['scenes']['scene-0218'][:32]):
        budget(out)
        path = DATA/row['path'] if row['keyframe'] is not None else images/Path(row['path']).name
        image = cv2.imread(str(path))
        f, timing = detector.step(image, int(row['timestamp_us']*1000))
        target = out/f'{i:03d}.npz'
        np.savez_compressed(target, index=f.index, ids=f.ids, labels=f.labels, scores=f.scores)
        entries.append(dict(frame=i, image=str(path), image_sha256=digest(path), stamp_ns=f.stamp_ns,
                            session=f.session, cache=target.name, cache_sha256=digest(target), detector=timing))
    save(out/'manifest.json', dict(entries=entries, admission=admission, detector=detector.meta,
                                  frames_index_sha256=digest(frames), protocol_sha256=digest(PROTOCOL)))


def evaluate(cache, source, weights, out, compiled):
    import cv2
    import torch
    from .geometry import InstanceFrame
    from .kalman_cache import gpu_admission
    from .sam2_memory import SAM2Pair
    protocol = read(PROTOCOL)
    admission = gpu_admission()
    if out.exists():
        raise ValueError('Refuse overwrite')
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    clean = not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    if commit != protocol['source_commit'] or not clean or digest(weights) != protocol['checkpoint_sha256']:
        raise ValueError('Official source/checkpoint pin mismatch')
    manifest = read(cache/'manifest.json')
    if manifest['protocol_sha256'] != digest(PROTOCOL):
        raise ValueError('Protocol changed after detector caching')
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    start = time.perf_counter()
    model = SAM2Pair(source, weights, image_size=512, official_compile=compiled)
    init_seconds = time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats()
    passes = []
    for repeat in range(4 if compiled else 1):
        # Independent replay from empty state; compilation cache alone survives.
        model.histories.clear()
        model.last_seen.clear()
        model.stamp = None
        model.session = None
        model.frame = 0
        rows = []
        for e in manifest['entries']:
            budget(out)
            if digest(e['image']) != e['image_sha256'] or digest(cache/e['cache']) != e['cache_sha256']:
                raise ValueError('Input changed')
            image = cv2.imread(e['image'])
            with np.load(cache/e['cache'], allow_pickle=False) as a:
                frame = InstanceFrame(e['stamp_ns'], e['session'], a['index'], a['ids'], a['labels'], a['scores'])
            wall = time.perf_counter()
            masks, timing = model.step(image, frame)
            elapsed = (time.perf_counter()-wall)*1000
            if not e['frame'] and np.any(masks['current'] != masks['temporal']):
                raise AssertionError('No-history current/temporal mismatch')
            # First pass records compact outputs for eager/compiled equivalence.
            # Later passes must also match that compiled pass exactly.
            path = out/f"{e['frame']:03d}.npz"
            packed = {k: np.packbits(v.reshape(len(v), -1), axis=1) for k, v in masks.items()}
            if repeat == 0:
                np.savez_compressed(path, **packed, shape=masks['current'].shape)
            else:
                with np.load(path, allow_pickle=False) as old:
                    for k, v in packed.items():
                        if not np.array_equal(v, old[k]):
                            raise AssertionError(f'Repeated compiled output changed: {repeat}, {e["frame"]}, {k}')
            r = dict(frame=e['frame'], repeat=repeat, wall_both_arms_ms=elapsed, **timing)
            rows.append(r)
            print(r, flush=True)
            # Incremental report is evidence if compiler exits/fails later.
            save(out/'progress.json', dict(repeat=repeat, frames=rows, elapsed_seconds=time.perf_counter()-start))
        values = [r['encoder_ms']+r['temporal_ms'] for r in rows]
        passes.append(dict(repeat=repeat, frames=rows, sam_only_ms=dict(median=float(np.median(values)),
                      p95=float(np.percentile(values, 95)), max=max(values))))
    save(out/'report.json', dict(passes=passes, metadata=model.meta, init_seconds=init_seconds,
         admission=admission, peak_allocated_bytes=torch.cuda.max_memory_allocated(), protocol=protocol,
         protocol_sha256=digest(PROTOCOL), input_manifest_sha256=digest(cache/'manifest.json'),
         source_sha256={p.name:digest(p) for p in [Path(__file__), Path(__file__).with_name('sam2_memory.py')]},
         scope='Current masks and IDs shared; no GT. First pass includes compile stalls, three repeats reset histories. Warm timing does not prove unseen-input or ROS latency.'))
    del model
    gc.collect()


def compare(eager, compiled, out):
    protocol = read(PROTOCOL)
    rows = []
    all_iou = []
    disagree = total = 0
    for ref in sorted(eager.glob('[0-9][0-9][0-9].npz')):
        with np.load(ref, allow_pickle=False) as a, np.load(compiled/ref.name, allow_pickle=False) as b:
            shape = tuple(a['shape'])
            if not np.array_equal(a['shape'], b['shape']):
                raise ValueError('Output shape mismatch')
            for arm in ('current', 'temporal'):
                n = int(np.prod(shape[1:]))
                x = np.unpackbits(a[arm], axis=1, count=n).astype(bool)
                y = np.unpackbits(b[arm], axis=1, count=n).astype(bool)
                intersection = (x & y).sum(1)
                union = (x | y).sum(1)
                iou = intersection[union > 0]/union[union > 0]
                diff = int((x != y).sum())
                disagree += diff
                total += x.size
                all_iou.extend(iou.tolist())
                rows.append(dict(frame=int(ref.stem), arm=arm, changed_pixels=diff, compared_pixels=x.size,
                                 nonempty_object_iou=iou.tolist()))
    if len(rows) != 64:
        raise ValueError('Incomplete 32-frame comparison')
    s = dict(pixel_disagreement_fraction=disagree/max(1,total), mean_nonempty_object_iou=float(np.mean(all_iou)),
             min_nonempty_object_iou=float(min(all_iou)))
    gate = protocol['accuracy_equivalence']
    s['equivalent'] = (s['pixel_disagreement_fraction'] <= gate['max_pixel_disagreement_fraction'] and
                      s['mean_nonempty_object_iou'] >= gate['min_mean_nonempty_object_iou'] and
                      s['min_nonempty_object_iou'] >= gate['min_nonempty_object_iou'])
    c = read(compiled/'report.json')
    runtime = all(p['sam_only_ms']['p95'] <= protocol['runtime_gate_ms'] for p in c['passes'][1:])
    save(out, dict(summary=s, warm_runtime_passed=runtime, eligible_for_further_validation=runtime and s['equivalent'],
                   rows=rows, eager_sha256=digest(eager/'report.json'), compiled_sha256=digest(compiled/'report.json')))
    print(s, 'warm_runtime_passed', runtime, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest='command', required=True)
    q = sp.add_parser('prepare')
    for k in ('frames', 'images', 'yolo', 'out'):
        q.add_argument('--'+k, type=Path, required=True)
    q = sp.add_parser('evaluate')
    for k in ('cache', 'source', 'weights', 'out'):
        q.add_argument('--'+k, type=Path, required=True)
    q.add_argument('--compiled', action='store_true')
    q = sp.add_parser('compare')
    for k in ('eager', 'compiled', 'out'):
        q.add_argument('--'+k, type=Path, required=True)
    a = vars(p.parse_args())
    command = a.pop('command')
    {'prepare':prepare, 'evaluate':evaluate, 'compare':compare}[command](**a)
