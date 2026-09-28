#!/usr/bin/env python3
"""Render paired, archived ZS/B0 predictions on exactly the same camera/scan frames.

This is an offline visual comparison, not a new latency or accuracy measurement.
Uses the first archived scoring pass for both arms, with identical preprocessing.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kitti_calib import KittiCalib

ARMS = ['ZS', 'armB0_s1']
COLORS = np.asarray([
    (180, 160, 120), (20, 170, 255), (245, 200, 0), (255, 190, 20),
    (180, 180, 40), (25, 160, 245), (35, 85, 255), (40, 150, 240),
    (200, 180, 40), (235, 170, 10), (190, 110, 155), (145, 120, 150),
    (155, 130, 220), (80, 150, 150), (170, 175, 180), (80, 170, 80)
], dtype=np.uint8)  # BGR, identical to the original driving demo.
RELEASE = 'https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/releases/download/v0.6-artifacts-20260928/'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def overlay(im, uv, depth, valid, lab, conf):
    selected = np.flatnonzero(conf[valid] >= .5)
    selected = selected[np.argsort(depth[selected])[::-1]]
    result = (im * .58).astype(np.uint8)
    labels = lab[valid]
    for k in selected:
        u, v = np.rint(uv[k]).astype(int)
        cls = int(labels[k])
        cv2.circle(result, (u, v), 2 if cls in (3, 6) else 1,
                   tuple(map(int, COLORS[cls])), -1, cv2.LINE_AA)
    return result


def caption(canvas, text, x, y, scale=.85, thickness=1):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (222, 229, 239), thickness, cv2.LINE_AA)


class MovieWriter:
    def __init__(self, path, width, height, fps):
        self.process = None
        if shutil.which('ffmpeg'):
            self.codec = 'libx264'
            self.process = subprocess.Popen([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
                '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                '-threads', '2', '-preset', 'medium', '-crf', '19',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)
            ], stdin=subprocess.PIPE)
        else:
            self.codec = 'avc1'
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*self.codec),
                                          fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError('An H.264 encoder (ffmpeg or OpenCV avc1) is required.')

    def write(self, frame):
        if self.process:
            self.process.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)

    def close(self):
        if self.process:
            self.process.stdin.close()
            if self.process.wait() != 0:
                raise RuntimeError('ffmpeg failed')
        else:
            self.writer.release()


def make_gif(movie, out, width=1008, stride=4, colors=96):
    cap = cv2.VideoCapture(str(movie))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * width / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames, indices = [], []
    for i in range(total):
        ok, bgr = cap.read()
        assert ok, i
        if i % stride == 0:
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            indices.append(i)
    cap.release()
    tile_h = round(300 * height / width)
    atlas = Image.new('RGB', (1200, tile_h * 4))
    for k, i in enumerate(np.linspace(0, len(frames) - 1, 16, dtype=int)):
        atlas.paste(frames[i].resize((300, tile_h), Image.Resampling.LANCZOS),
                    ((k % 4) * 300, (k // 4) * tile_h))
    # Rare class colours (especially pedestrian orange) must survive quantization.
    reserved = [tuple(map(int, c[::-1])) for c in COLORS]
    reserved += [(239, 229, 222), (17, 19, 23), (0, 0, 0)]
    n = colors - len(reserved)
    adaptive = atlas.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    rgb = adaptive.getpalette()[:n * 3] + [v for c in reserved for v in c]
    palette = Image.new('P', (1, 1))
    palette.putpalette(rgb + [0] * (768 - len(rgb)))
    indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    boundaries = [round(i / fps * 100) * 10 for i in indices] + [round(total / fps * 100) * 10]
    durations = [b - a for a, b in zip(boundaries, boundaries[1:])]
    path = out / 'driving_zs_vs_b0.gif'
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=durations,
                    loop=0, optimize=True, disposal=1)
    with Image.open(path) as gif:
        assert gif.n_frames == len(indices) and gif.info['loop'] == 0
        actual = []
        for i in range(gif.n_frames):
            gif.seek(i)
            actual.append(gif.info['duration'])
        assert actual == durations
        gif.seek(len(indices) // 2)
        gif.convert('RGB').save(out / 'gif-preview.png')
    save(out / 'gif-check.json', dict(file=path.name, sha256=digest(path), bytes=path.stat().st_size,
         width=width, height=height, frames=len(indices), sampling_fps=fps / stride,
         duration_ms=sum(durations), loop=0, frame_stride=stride, palette_colors=colors,
         semantic_colors_reserved=True, source_sha256=digest(movie), source_frames=total,
         source_fps=fps, note='Original-speed preview of the aligned offline comparison.'))
    print('GIF_VERIFIED', path.name, path.stat().st_size, flush=True)


def render(a):
    cv2.setNumThreads(1)
    repo = Path(__file__).resolve().parents[1]
    demo = json.loads((repo / 'results/v06/driving_demo_20260928/measurement.json').read_text())
    original_video = json.loads((repo / 'results/v06/driving_demo_20260928/video-check.json').read_text())
    rows = demo['timeline']
    assert [r['frame'] for r in rows] == list(range(970, 1101))
    a.out.mkdir(parents=True, exist_ok=True)
    raw = a.root / 'data/raw/2011_09_30/2011_09_30_drive_0027_sync'
    calib_dir = a.root / 'data/raw/2011_09_30'
    cal = KittiCalib(str(calib_dir))
    indexes = {}
    assets = {r['name']: r for r in json.loads(
        (repo / 'results/v06/artifacts_20260928/artifact-manifest.json').read_text())['assets']}
    for arm in ARMS:
        name = f'v06-pred-{arm}-seq07-r1.tar.index.json'
        index_path = a.root / 'out/preservation_20260928' / name
        assert digest(index_path) == assets[name]['sha256']
        indexes[arm] = {r['path']: r for r in json.loads(index_path.read_text())}
    image_ns = np.asarray((raw / 'image_02/timestamps.txt').read_text().splitlines(),
                          dtype='datetime64[ns]').astype(np.int64)[970:1101]
    image_s = (image_ns - image_ns[0]) / 1e9
    assert np.allclose(image_s, [r['image_s'] for r in rows], atol=1e-9, rtol=0)
    paired, evidence = [], []
    for row in rows:
        f = row['frame']
        image_path = raw / f'image_02/data/{f:010d}.png'
        point_path = raw / f'velodyne_points/data/{f:010d}.bin'
        assert digest(point_path) == row['points_sha256'], ('scan mismatch', f)
        im = cv2.imread(str(image_path))
        assert im is not None and im.shape == (370, 1226, 3)
        pts = np.fromfile(point_path, np.float32).reshape(-1, 4)
        uv, depth, valid = cal.project_velo_to_cam2(pts, img_shape=im.shape, return_mask=True)
        pair, pred_records = [], {}
        for arm in ARMS:
            rel = f'out/v06_score/pred/{arm}_07_r1/f{f:06d}.npz'
            path = a.root / rel
            expected = indexes[arm][rel]
            assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256']
            with np.load(path) as z:
                lab, conf = z['lab'], z['conf']
                assert lab.shape == conf.shape == (len(pts),)
                assert lab.dtype == np.uint8 and conf.dtype == np.float16
                assert np.all(lab < 16) and np.isfinite(conf).all()
                assert np.all((conf >= 0) & (conf <= 1))
                pair.append(overlay(im, uv, depth, valid, lab, conf))
            pred_records[arm] = dict(path=rel, sha256=expected['sha256'])
        paired.append(pair)
        evidence.append(dict(frame=f, image_s=row['image_s'], points=len(pts),
            points_sha256=row['points_sha256'], image_sha256=digest(image_path), predictions=pred_records))
        if len(evidence) % 25 == 0:
            print('ALIGNED', len(evidence), '/', len(rows), flush=True)
    w, h, fps = 2520, 588, 30
    count = original_video['frames']
    assert count == 421 and fps == original_video['fps']
    movie = a.out / 'driving_zs_vs_b0_seq07.mp4'
    writer = MovieWriter(movie, w, h, fps)
    frame_map, shown, previews = [], set(), set()
    # Fixed start/mid/end-ish QA frames, chosen by time, never by either prediction.
    qa_indices = {0, 65, 108, 130}
    for v in range(count):
        t = v / fps
        k = max(0, int(np.searchsorted(image_s, t, side='right') - 1))
        f = rows[k]['frame']
        canvas = np.full((h, w, 3), (23, 19, 17), np.uint8)
        caption(canvas, 'ZS | ORIGINAL PRETRAINED', 24, 48, 1.25, 2)
        caption(canvas, 'B0 | OUR TRAINED MODEL', 1270, 48, 1.25, 2)
        caption(canvas, 'PTv3 / nuScenes weights / no KITTI fine-tuning', 24, 83, .8)
        caption(canvas, 'PTv3 / camera pseudo-label distillation + KL / seed 1', 1270, 83, .8)
        for x in [24, 1270]:
            caption(canvas, f'KITTI 07     frame {f:04d}     t = {t:05.2f} s     1x', x, 115, .8)
        canvas[130:500, 24:1250] = paired[k][0]
        canvas[130:500, 1270:2496] = paired[k][1]
        cv2.line(canvas, (1260, 20), (1260, 501), (85, 82, 80), 1)
        for offset in [24, 1270]:
            x = offset
            for label, cls, step in [('CAR', 3, 135), ('PERSON', 6, 175), ('ROAD', 10, 155),
                                      ('VEGETATION', 15, 255), ('STRUCTURE', 14, 250)]:
                cv2.rectangle(canvas, (x, 521), (x + 20, 541), tuple(map(int, COLORS[cls])), -1)
                caption(canvas, label, x + 31, 540, .74)
                x += step
            caption(canvas, 'Same scan + camera | confidence >= 0.5 | aligned offline comparison', offset, 575, .77)
        writer.write(canvas)
        frame_map.append(f)
        shown.add(f)
        if k in qa_indices and k not in previews:
            name = f'qa_{f:06d}.jpg'
            cv2.imwrite(str(a.out / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if k == 65:
                shutil.copyfile(a.out / name, a.out / 'preview.jpg')
            previews.add(k)
    writer.close()
    assert shown == set(range(970, 1101))
    cap = cv2.VideoCapture(str(movie))
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (h, w, 3)
        decoded += 1
    cap.release()
    assert decoded == count
    save(a.out / 'video-check.json', dict(file=movie.name, sha256=digest(movie), bytes=movie.stat().st_size,
        width=w, height=h, fps=fps, codec=writer.codec, frames=decoded, duration_s=decoded / fps,
        same_frame_both_panels=True, same_input_as_original_demo=True, unique_scans=len(shown),
        frame_mapping=frame_map, preview_frame=1035, qa_frames=[970, 1035, 1078, 1100]))
    checkpoint_files = ['v05-ZS-student.pth', 'v06-armB0_s1-student.pth']
    provenance = dict(schema=1, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sequence='07', first=970, last=1100, frames=131, conf_gate=.5, rate=1,
        source='Archived v0.6 scoring pass r1 for BOTH arms; no new inference or training.',
        timing='Same image/scan index in both panels, held to the next original camera timestamp. '
               'Same 421 video frames as the original demo; final frame held to its end.',
        scope='Single-sweep LiDAR predictions projected onto RGB. Offline visual comparison; '
              'no latency measurement, map fusion, SLAM, or ground-truth accuracy claim.',
        selection='Exactly the previously published contiguous driving segment; first archived '
                  'scoring pass for both arms, without frame or repeat selection by model quality.',
        preprocessing=dict(pointcept='v1.5.1', tensors=488, intensity_scale=.2, grid_m=.05,
                           fp16=True, shuffle_orders=False, tta=False,
                           voxelization='CPU numpy unique, first point per voxel, raw point order'),
        original_demo_note='B0 uses the same trained weights as the timing demo, but its archived '
                           'scoring pass rather than that fresh inference draw. PTv3 has run-to-run variation.',
        models={arm: dict(checkpoint=file, sha256=assets[file]['sha256'], url=RELEASE + file,
            prediction_archive=RELEASE + f'v06-pred-{arm}-seq07-r1.tar',
            prediction_index_sha256=assets[f'v06-pred-{arm}-seq07-r1.tar.index.json']['sha256'])
            for arm, file in zip(ARMS, checkpoint_files)},
        calibration={p.name: digest(p) for p in sorted(calib_dir.glob('calib_*.txt'))},
        palette_bgr=COLORS.tolist(), timeline=evidence)
    save(a.out / 'comparison.json', provenance)
    # A small, self-contained prediction subset; no raw dataset or checkpoint duplication.
    archive = a.out / 'comparison-predictions.tar'
    with tarfile.open(archive, 'w') as tf:
        for r in evidence:
            for pred in r['predictions'].values():
                tf.add(a.root / pred['path'], arcname=pred['path'], recursive=False)
        tf.add(a.out / 'comparison.json', arcname='comparison.json')
    with tarfile.open(archive) as tf:
        expected = {p['path']: p['sha256'] for r in evidence for p in r['predictions'].values()}
        expected['comparison.json'] = digest(a.out / 'comparison.json')
        members = tf.getmembers()
        assert len(members) == len(expected) == 263
        for member in members:
            assert member.isfile()
            assert hashlib.sha256(tf.extractfile(member).read()).hexdigest() == expected[member.name]
    save(a.out / 'archive-check.json', dict(file=archive.name, sha256=digest(archive),
         bytes=archive.stat().st_size, verified_members=263, raw_data_included=False))
    print('VIDEO_VERIFIED', movie.name, decoded, movie.stat().st_size, flush=True)
    make_gif(movie, a.out, width=a.gif_width)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gif-width', type=int, default=1008)
    render(parser.parse_args())
