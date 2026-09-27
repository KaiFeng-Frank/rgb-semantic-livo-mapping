#!/usr/bin/env python3
"""Record fresh single-sweep inference and render its measured 1x timeline.

This is an inference replay, not a ROS/RViz end-to-end latency measurement.
Run measure with the experiment's ptv3 Python; render needs numpy and OpenCV.
Raw KITTI inputs stay on the experiment host.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def stamps(path):
    return np.asarray(path.read_text().splitlines(), dtype='datetime64[ns]').astype(np.int64)


def source(root):
    return root / 'data/raw/2011_09_30/2011_09_30_drive_0027_sync'


def measure(a):
    import torch
    from ptv3_worker import Segmenter

    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                   '--format=csv,noheader'], text=True).strip()
    if gpu:
        raise RuntimeError('GPU already has a compute process; do not disturb it.')
    torch.set_num_threads(1)
    raw = source(a.root)
    frames = list(range(a.first, a.last + 1))
    image_ns = stamps(raw / 'image_02/timestamps.txt')[frames]
    end_ns = stamps(raw / 'velodyne_points/timestamps_end.txt')[frames]
    image_t = (image_ns - image_ns[0]) / 1e9
    ready_t = (end_ns - image_ns[0]) / 1e9
    assert np.all(np.diff(image_t) > 0) and np.all(ready_t > image_t)
    points = [np.fromfile(raw / f'velodyne_points/data/{f:010d}.bin', np.float32).reshape(-1, 4)
              for f in frames]
    ckpt = a.root / 'weights/v06/armB0_s1_student.pth'
    expected_sha = '35f5d3aec63ae1c383f4a4146e157aa054860cb12b178d899571287e6f97e5f1'
    assert sha(ckpt) == expected_sha
    model = Segmenter(ckpt=str(ckpt), half=True, shuffle=False, tf32=False,
                      intensity_scale=0.2, grid_size=0.05, fast_voxel=True,
                      gpu_voxel=True, fast_hilbert=True)
    assert model.ckpt_sha256 == expected_sha and model.ckpt_tensors == 488
    for f in [0, 200, 400, 600, 800, 1000]:
        warm = np.fromfile(raw / f'velodyne_points/data/{f:010d}.bin', np.float32).reshape(-1, 4)
        model.segment(warm)
    torch.cuda.synchronize()
    recorded, predictions = [], []
    start = time.perf_counter() + 0.5
    for k, f in enumerate(frames):
        remaining = start + ready_t[k] - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        begin = time.perf_counter()
        lab, conf = model.segment(points[k])
        torch.cuda.synchronize()
        finish = time.perf_counter()
        assert len(lab) == len(points[k]) and len(conf) == len(lab)
        assert np.all(lab < 16) and np.isfinite(conf).all()
        recorded.append(dict(frame=f, points=len(lab), image_s=float(image_t[k]),
                             scan_available_s=float(ready_t[k]), start_s=begin-start,
                             done_s=finish-start, inference_ms=(finish-begin)*1000,
                             queue_ms=(begin-start-ready_t[k])*1000,
                             scan_to_result_ms=(finish-start-ready_t[k])*1000,
                             camera_to_result_ms=(finish-start-image_t[k])*1000))
        predictions.append((lab.copy(), conf.copy()))
    # All disk output happens after the timed replay has finished.
    a.out.mkdir(parents=True, exist_ok=True)
    for row, (lab, conf), pts in zip(recorded, predictions, points):
        np.savez_compressed(a.out / f'prediction_{row["frame"]:06d}.npz', label=lab, conf=conf)
        row['points_sha256'] = hashlib.sha256(pts.tobytes()).hexdigest()
        row['prediction_sha256'] = sha(a.out / f'prediction_{row["frame"]:06d}.npz')
    stats = {}
    for name in ['inference_ms', 'queue_ms', 'scan_to_result_ms', 'camera_to_result_ms']:
        v = [r[name] for r in recorded]
        stats[name] = dict(zip(['p50', 'p95', 'max'], map(float, np.percentile(v, [50, 95, 100]))))
    result = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  model='v06 armB0_s1', checkpoint_sha256=expected_sha,
                  gpu=torch.cuda.get_device_name(0), sequence='07', first=a.first, last=a.last,
                  frames=len(recorded), rate=1.0, input_preloaded=True, dropped_frames=0,
                  conf_gate=0.5, half=True, shuffle=False, warmup_frames=[0, 200, 400, 600, 800, 1000],
                  scope='Fresh inference including GPU voxelization and CPU result return; '
                        'FIFO replay at original sweep-end timestamps. Rendering is offline.',
                  excluded='Disk input, cold start, ROS transport, SLAM, map fusion, physical display.',
                  selection='Contiguous seq07 frames selected using scene contents (people and vehicles), '
                            'before inspecting this model output. This is not an accuracy benchmark.',
                  stats=stats, timeline=recorded)
    save(a.out / 'measurement.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'timeline'}), flush=True)


def render(a):
    import cv2
    from kitti_calib import KittiCalib

    cv2.setNumThreads(1)
    data = json.loads((a.out / 'measurement.json').read_text())
    rows = data['timeline']
    raw = source(a.root)
    cal = KittiCalib(str(a.root / 'data/raw/2011_09_30'))
    # BGR; highlight cars and people while keeping all predicted classes visible.
    colors = np.asarray([(180, 160, 120), (20, 170, 255), (245, 200, 0), (255, 190, 20),
                         (180, 180, 40), (25, 160, 245), (35, 85, 255), (40, 150, 240),
                         (200, 180, 40), (235, 170, 10), (190, 110, 155), (145, 120, 150),
                         (155, 130, 220), (80, 150, 150), (170, 175, 180), (80, 170, 80)], np.uint8)
    images, overlays, counts = [], [], []
    for r in rows:
        f = r['frame']
        im = cv2.imread(str(raw / f'image_02/data/{f:010d}.png'))
        assert im is not None and im.shape[:2] == (370, 1226)
        pts = np.fromfile(raw / f'velodyne_points/data/{f:010d}.bin', np.float32).reshape(-1, 4)
        assert hashlib.sha256(pts.tobytes()).hexdigest() == r['points_sha256']
        pred = a.out / f'prediction_{f:06d}.npz'
        assert sha(pred) == r['prediction_sha256']
        with np.load(pred) as z:
            lab, conf = z['label'], z['conf']
        uv, dep, valid = cal.project_velo_to_cam2(pts, img_shape=im.shape, return_mask=True)
        label, confidence = lab[valid], conf[valid]
        selected = np.flatnonzero(confidence >= data['conf_gate'])
        selected = selected[np.argsort(dep[selected])[::-1]]  # nearer returns drawn last
        overlay = (im * 0.58).astype(np.uint8)
        for k in selected:
            u, v = np.rint(uv[k]).astype(int)
            cls = int(label[k])
            cv2.circle(overlay, (u, v), 2 if cls in (3, 6) else 1,
                       tuple(map(int, colors[cls])), -1, cv2.LINE_AA)
        images.append(im)
        overlays.append(overlay)
        counts.append({name: int(((label == cls) & (confidence >= .5)).sum())
                       for name, cls in [('car_points', 3), ('person_points', 6)]})
    w, h, fps = 1280, 1056, 30
    movie = a.out / 'driving_semantics_seq07_measured.mp4'
    codec = None
    for tag in ['avc1', 'mp4v']:
        writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*tag), fps, (w, h))
        if writer.isOpened():
            codec = tag
            break
        writer.release()
    if codec is None:
        raise RuntimeError('No working MP4 encoder')
    image_t = np.asarray([r['image_s'] for r in rows])
    done_t = np.asarray([r['done_s'] for r in rows])
    count = int(np.ceil((max(image_t[-1], done_t[-1]) + .4) * fps))

    def text(canvas, s, x, y, scale=.66, color=(222, 229, 239)):
        cv2.putText(canvas, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    preview = None
    max_people_k = max(range(len(rows)), key=lambda k: counts[k]['person_points'])
    for v in range(count):
        t = v / fps
        camera = max(0, int(np.searchsorted(image_t, t, side='right') - 1))
        semantic = int(np.searchsorted(done_t, t, side='right') - 1)
        canvas = np.full((h, w, 3), (23, 19, 17), np.uint8)
        text(canvas, 'DRIVING VIEW / SEMANTIC POINT CLOUD', 27, 37, .87)
        text(canvas, '1x replay | KITTI 07 | v06 B0 seed 1 | RTX 4090 | measured inference', 27, 67, .61)
        text(canvas, f'CAMERA   frame {rows[camera]["frame"]}     replay t = {t:05.2f} s', 27, 106)
        canvas[121:491, 27:1253] = images[camera]
        if semantic >= 0:
            r = rows[semantic]
            assert r['done_s'] <= t + 1e-9
            text(canvas, f'SEMANTICS   latest completed frame {r["frame"]}', 27, 530)
            text(canvas, f'Scan available -> result: {r["scan_to_result_ms"]:.0f} ms     '
                         f'Image age now: {(t-r["image_s"])*1000:.0f} ms', 27, 560, .67, (118, 214, 255))
            canvas[578:948, 27:1253] = overlays[semantic]
        else:
            text(canvas, 'SEMANTICS   waiting for first scan / inference', 27, 530)
            text(canvas, 'No prediction shown before it was computed.', 27, 560)
        x = 27
        for label, cls in [('CAR', 3), ('PERSON', 6), ('ROAD', 10), ('VEGETATION', 15), ('STRUCTURE', 14)]:
            cv2.rectangle(canvas, (x, 970), (x+17, 987), tuple(map(int, colors[cls])), -1)
            text(canvas, label, x+25, 985, .56)
            x += 120 if cls in (3, 6, 10) else 205
        text(canvas, 'Offline display follows measured completion times; confidence >= 0.5.', 27, 1019, .59)
        text(canvas, 'No ROS / SLAM / display overhead measured. Points are projected, not dense image masks.', 27, 1042, .55)
        writer.write(canvas)
        if semantic == max_people_k and preview is None:
            preview = canvas.copy()
    writer.release()
    cap = cv2.VideoCapture(str(movie))
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (h, w, 3)
        decoded += 1
    cap.release()
    assert decoded == count and preview is not None
    cv2.imwrite(str(a.out / 'preview.jpg'), preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
    save(a.out / 'video-check.json', dict(file=movie.name, sha256=sha(movie), bytes=movie.stat().st_size,
        fps=fps, width=w, height=h, codec=codec, frames=decoded, duration_s=decoded/fps,
        no_future_prediction=True, preview_frame=rows[max_people_k]['frame'], visible_point_counts=counts))
    print('VIDEO_VERIFIED', movie, decoded, movie.stat().st_size, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['measure', 'render'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--first', type=int, default=970)
    p.add_argument('--last', type=int, default=1100)
    args = p.parse_args()
    assert 0 <= args.first <= args.last <= 1100
    sys.path.insert(0, str(args.root / 'src'))
    (measure if args.action == 'measure' else render)(args)
