"""Bounded camera-to-point association, with explicit abstention.

An instance ID is a tracker hypothesis, not a semantic correction or a guaranteed
physical identity. Raw semantic labels are never changed by this module.
"""
import base64
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import re
import zlib

import numpy as np

COCO_TO_NATIVE = {0: 6, 1: 1, 2: 3, 3: 5, 5: 2, 7: 9}
MAX_PIXELS = 2048 * 2048
MAX_INSTANCES = 128


@dataclass
class Camera:
    width: int
    height: int
    K: np.ndarray
    distortion: np.ndarray
    T_camera_lidar: np.ndarray

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=float)
        self.distortion = np.asarray(self.distortion, dtype=float)
        self.T_camera_lidar = np.asarray(self.T_camera_lidar, dtype=float)
        t = self.T_camera_lidar
        if (not 0 < self.width*self.height <= MAX_PIXELS or self.width < 1 or self.height < 1
                or self.K.shape != (3, 3) or not np.isfinite(self.K).all()
                or self.K[0, 0] <= 0 or self.K[1, 1] <= 0
                or not np.allclose(self.K[2], [0, 0, 1])
                or self.distortion.shape not in ((4,), (5,)) or not np.isfinite(self.distortion).all()
                or t.shape != (4, 4) or not np.isfinite(t).all()
                or not np.allclose(t[3], [0, 0, 0, 1])
                or not np.allclose(t[:3, :3].T@t[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-4)):
            raise ValueError('Invalid camera calibration')

    @classmethod
    def from_livo(cls, frontend, camera):
        import yaml
        ext = yaml.safe_load(Path(frontend).read_text())['/**']['ros__parameters']['extrin_calib']
        c = yaml.safe_load(Path(camera).read_text())['/**']['ros__parameters']
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = np.asarray(ext['Rcl']).reshape(3, 3), ext['Pcl']
        # Incoming camera topic is RAW full resolution, not LIVO's scaled image.
        return cls(c['cam_width'], c['cam_height'],
                   [[c['cam_fx'], 0, c['cam_cx']], [0, c['cam_fy'], c['cam_cy']], [0, 0, 1]],
                   [c[f'cam_d{i}'] for i in range(4)], t)


@dataclass
class InstanceFrame:
    stamp_ns: int
    session: str
    index: np.ndarray  # 0 = unknown/overlap/boundary; i+1 indexes metadata
    ids: np.ndarray
    labels: np.ndarray
    scores: np.ndarray

    def __post_init__(self):
        if type(self.stamp_ns) is not int or self.stamp_ns < 0 or not re.fullmatch('[a-f0-9]{32}', self.session):
            raise ValueError('Invalid timestamp/session')
        self.index = np.asarray(self.index)
        self.ids, self.labels, self.scores = map(np.asarray, (self.ids, self.labels, self.scores))
        n = len(self.ids)
        if (self.index.dtype != np.uint16 or self.index.ndim != 2 or not 0 < self.index.size <= MAX_PIXELS
                or n > MAX_INSTANCES or self.ids.shape != (n,) or self.labels.shape != (n,)
                or self.scores.shape != (n,) or np.any(self.index > n)
                or self.ids.dtype.kind not in 'ui' or np.any(self.ids <= 0)
                or np.any(self.ids > 2**32-1) or len(np.unique(self.ids)) != n
                or not np.isin(self.labels, list(COCO_TO_NATIVE.values())).all()
                or not np.isfinite(self.scores).all() or np.any((self.scores < 0) | (self.scores > 1))):
            raise ValueError('Invalid instance mask/metadata')

    def encode(self):
        return json.dumps(dict(schema=1, stamp_ns=self.stamp_ns, session=self.session,
            shape=list(self.index.shape), ids=self.ids.tolist(), labels=self.labels.tolist(),
            scores=self.scores.tolist(), index=base64.b64encode(zlib.compress(
                self.index.astype('<u2').tobytes(), 1)).decode()), separators=(',', ':'), allow_nan=False)

    @classmethod
    def decode(cls, text):
        if len(text) > 12_000_000:
            raise ValueError('Oversized instance packet')
        d = json.loads(text)
        shape = d['shape']
        if (d['schema'] != 1 or len(shape) != 2 or any(type(x) is not int or x < 1 for x in shape)
                or shape[0]*shape[1] > MAX_PIXELS):
            raise ValueError('Invalid packet shape/schema')
        size = shape[0]*shape[1]*2
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b64decode(d['index'], validate=True), size+1)
        if len(raw) != size or not decoder.eof or decoder.unused_data:
            raise ValueError('Invalid compressed mask length')
        return cls(d['stamp_ns'], d['session'], np.frombuffer(raw, '<u2').reshape(shape),
                   np.asarray(d['ids']) if d['ids'] else np.empty(0, np.int64),
                   np.asarray(d['labels']), np.asarray(d['scores']))


class FrameCache:
    def __init__(self, capacity=16, max_dt_ms=20.):
        if type(capacity) is not int or not 1 <= capacity <= 64 or not 0 <= max_dt_ms <= 100:
            raise ValueError('Invalid cache bounds')
        self.frames = deque(maxlen=capacity)
        self.max_dt_ns = int(max_dt_ms*1e6)
        self.session = None
        self.retired = deque(maxlen=16)

    def add(self, frame):
        if frame.session in self.retired:
            raise ValueError('Late packet from retired session')
        if self.session != frame.session:
            if self.session is not None:
                self.retired.append(self.session)
            self.frames.clear()
            self.session = frame.session
        if self.frames and frame.stamp_ns <= self.frames[-1].stamp_ns:
            raise ValueError('Reordered visual frame')
        self.frames.append(frame)

    def get(self, stamp_ns):
        # No future camera observations, even if already computed in replay.
        return next((f for f in reversed(self.frames) if 0 <= stamp_ns-f.stamp_ns <= self.max_dt_ns), None)


def mask_frame(masks, ids, coco_labels, scores, stamp_ns, session, shape):
    """Keep mask interiors; overlaps include low-confidence rejected objects."""
    import cv2
    masks = np.asarray(masks, dtype=bool)
    if masks.shape != (len(ids), *shape) or len(ids) > MAX_INSTANCES or len(coco_labels) != len(ids) or len(scores) != len(ids):
        raise ValueError('Mask/instance metadata mismatch')
    index = np.zeros(shape, np.uint16)
    occupied, overlap = np.zeros(shape, bool), np.zeros(shape, bool)
    kept_ids, labels, confidence = [], [], []
    for mask, tid, cls, score in zip(masks, ids, coco_labels, scores):
        overlap |= occupied & mask
        occupied |= mask
        inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if score < .35 or cls not in COCO_TO_NATIVE:
            continue
        kept_ids.append(tid); labels.append(COCO_TO_NATIVE[cls]); confidence.append(score)
        index[inner] = len(kept_ids)
    index[overlap] = 0
    return InstanceFrame(stamp_ns, session, index, np.asarray(kept_ids, np.int64),
                         np.asarray(labels, np.uint16), np.asarray(confidence, np.float32))


def camera_at_scan(camera, graph, scan_stamp, image_stamp, T_imu_lidar):
    """Ego-motion correction from received graph only; abstain without brackets.

    Nonzero time gaps use interpolated graph poses (not a motion ground truth).
    The object itself is not motion-compensated. Graph revisions are consumed
    as one consistent snapshot; no stale world-space identity state is kept.
    """
    if image_stamp > scan_stamp:
        return None
    if image_stamp == scan_stamp:
        return camera
    times, poses = graph['times_ns'], graph['poses']
    end = int(np.searchsorted(times, scan_stamp))
    if end >= len(times) or times[end] != scan_stamp:
        return None
    j = int(np.searchsorted(times, image_stamp))
    if j < len(times) and times[j] == image_stamp:
        image_pose = poses[j]
    elif j == 0 or j >= len(times) or times[j]-times[j-1] > 1_500_000_000:
        return None
    else:
        from scipy.spatial.transform import Rotation, Slerp
        alpha = float(image_stamp-times[j-1])/float(times[j]-times[j-1])
        image_pose = np.eye(4)
        image_pose[:3, 3] = (1-alpha)*poses[j-1,:3,3]+alpha*poses[j,:3,3]
        image_pose[:3,:3] = Slerp([0,1],Rotation.from_matrix(poses[j-1:j+1,:3,:3]))([alpha]).as_matrix()[0]
    image_lidar_pose = image_pose@T_imu_lidar
    scan_lidar_pose = poses[end]@T_imu_lidar
    return Camera(camera.width,camera.height,camera.K,camera.distortion,
                  camera.T_camera_lidar@np.linalg.inv(image_lidar_pose)@scan_lidar_pose)


def associate(xyz, camera, frame, *, depth_tolerance=.4, cell_pixels=3, min_points=3):
    """Project into raw distorted image; all points participate in the z-buffer.

    This sparse depth check rejects measured occluders, not unseen surfaces.
    No label-based filtering is allowed before depth testing.
    """
    import cv2
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Expected finite sensor XYZ')
    if frame.index.shape != (camera.height, camera.width):
        raise ValueError('Image dimensions differ from calibration')
    if depth_tolerance < 0 or not np.isfinite(depth_tolerance) or cell_pixels < 1 or min_points < 1:
        raise ValueError('Invalid association bounds')
    points = xyz@camera.T_camera_lidar[:3, :3].T+camera.T_camera_lidar[:3, 3]
    selected = np.flatnonzero((points[:, 2] > .1) & (points[:, 2] < 80.))
    ids = np.zeros(len(xyz), np.uint32)
    labels = np.full(len(xyz), 65535, np.uint16)
    scores = np.zeros(len(xyz), np.float32)
    info = dict(points=len(xyz), in_view=0, in_mask=0, occluded=0, associated=0, instances=0,
                session=frame.session, camera_stamp_ns=frame.stamp_ns)
    if len(selected):
        uv, _ = cv2.projectPoints(points[selected], np.zeros(3), np.zeros(3), camera.K, camera.distortion)
        uv = uv.reshape(-1, 2)
        valid = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width-.5) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height-.5)
        selected, uv = selected[valid], np.rint(uv[valid]).astype(np.int64)
        info['in_view'] = len(selected)
        w = (camera.width+cell_pixels-1)//cell_pixels
        h = (camera.height+cell_pixels-1)//cell_pixels
        cells = uv[:, 1]//cell_pixels*w+uv[:, 0]//cell_pixels
        depths = np.full(w*h, np.inf)
        np.minimum.at(depths, cells, points[selected, 2])
        slots = frame.index[uv[:, 1], uv[:, 0]]
        foreground = slots > 0
        visible = points[selected, 2] <= depths[cells]+depth_tolerance
        info['in_mask'], info['occluded'] = int(foreground.sum()), int((foreground & ~visible).sum())
        selected, slots = selected[foreground & visible], slots[foreground & visible]-1
        for slot in np.unique(slots):
            at = selected[slots == slot]
            if len(at) < min_points:
                continue
            ids[at], labels[at], scores[at] = frame.ids[slot], frame.labels[slot], frame.scores[slot]
        info['associated'] = int(np.count_nonzero(ids))
        info['instances'] = int(len(np.unique(ids[ids > 0])))
    return dict(ids=ids, labels=labels, scores=scores, info=info)
