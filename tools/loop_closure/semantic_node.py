"""Live PTv3 keyframes + reversible semantic mapping from causal graph revisions.

Dense deskewed XYZI is in backend_lidar, stamped exactly like its accepted graph
keyframe. Only graph poses that have actually arrived are used. Classification,
fusion and display cannot call or block the LIVO/pose process. This is keyframe
mapping (normally 1 Hz), not a claim of 10 Hz full semantic segmentation.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

from .online_outputs import OutputProcess
from .semantic_fusion import ReversibleSemanticMap

ROOT = Path(__file__).resolve().parents[2]


def checked_trajectory(data, previous=None, capacity=1000, frame='map'):
    if data['frame_id'] != frame or type(data['stamp_ns']) is not int or type(data['version']) is not int:
        raise ValueError('graph frame/stamp/version contract mismatch')
    if data['version'] < 0 or not 0 < len(data['times_ns']) <= capacity:
        raise ValueError('graph revision exceeds session contract')
    if any(type(v) is not int or v < 0 for v in data['times_ns']):
        raise ValueError('graph timestamps must be nonnegative integer nanoseconds')
    times = np.asarray(data['times_ns'], dtype=np.int64)
    poses = np.asarray(data['poses'], dtype=np.float64)
    if times[-1] != data['stamp_ns'] or np.any(np.diff(times) <= 0):
        raise ValueError('graph timestamps must be ordered and end at watermark')
    if poses.shape != (len(times), 4, 4) or not np.isfinite(poses).all():
        raise ValueError('invalid graph pose array')
    rotations = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-6, rtol=0)):
        raise ValueError('graph poses are not rigid transforms')
    if previous is not None:
        old = previous['times_ns']
        if (data['version'] < previous['version'] or len(times) < len(old)
                or not np.array_equal(times[:len(old)], old)):
            raise ValueError('graph reset/reordering requires a new mapping session')
        if data['version'] == previous['version'] and not np.allclose(
                poses[:len(old)], previous['poses'], atol=1e-9, rtol=0):
            raise ValueError('changed historical poses require a new graph revision')
    return dict(stamp_ns=data['stamp_ns'], version=data['version'], times_ns=times,
                poses=poses, received=time.monotonic())


def extrinsic(path):
    import yaml
    params = yaml.safe_load(Path(path).read_text())['/**']['ros__parameters']['extrin_calib']
    result = np.eye(4)
    result[:3, :3] = np.asarray(params['extrinsic_R']).reshape(3, 3)
    result[:3, 3] = params['extrinsic_T']
    checked_trajectory(dict(frame_id='map', stamp_ns=0, version=0, times_ns=[0], poses=[result.tolist()]))
    return result


def make_node(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    from tools.semantic_memory.ros_node import read_xyz_intensity

    cfg = json.loads((Path(__file__).with_name('semantic_online.json')).read_text())
    if args.config:
        cfg.update(json.loads(args.config.read_text()))
    if type(cfg['pending_clouds']) is not int or cfg['pending_clouds'] < 1 or cfg['max_queue_age_s'] <= 0:
        raise ValueError('invalid semantic queue bounds')
    transform = extrinsic(args.frontend)
    core_args = {key: cfg[key] for key in ('voxel', 'classes', 'tile_voxels', 'max_keyframes',
                 'max_points', 'max_total_points', 'excluded_classes', 'minimum_confidence')}

    class SemanticNode(Node):
        def __init__(self):
            super().__init__('online_semantic_refusion')
            args.out.mkdir(parents=True, exist_ok=False)
            (args.out/'config.json').write_text(json.dumps(dict(cfg, T_imu_lidar=transform.tolist(),
                predictions_input=args.predictions_input, frontend=str(args.frontend),
                reflectance_divisor=args.reflectance_divisor, intensity_scale=args.intensity_scale), indent=2)+'\n')
            self.core = ReversibleSemanticMap(**core_args)
            self.lock = threading.Lock()
            self.pending, self.trajectory, self.latest = OrderedDict(), None, None
            self.stop, self.wake = threading.Event(), threading.Event()
            self.client, self.last_scan = None, None
            self.last_input, self.last_graph = -1, None
            self.publication_id = 0
            self.counters = dict(received=0, processed=0, invalid=0, dropped_full=0,
                dropped_stale=0, dropped_order=0, unanchored=0, graph_invalid=0,
                revisions=0, ready=False, failed=False, error=None, capacity_reached=0)
            self.log = (args.out/'events.jsonl').open('x')
            qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            graph_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(PointCloud2, args.input_cloud, self.on_cloud, qos)
            self.create_subscription(String, '/backend/trajectory', self.on_graph, graph_qos)
            self.status_pub = self.create_publisher(String, '/backend/semantic_status', 1)
            self.output = OutputProcess(sys.executable, self.snapshot, args.map_frame,
                                         module='tools.loop_closure.semantic_outputs')
            self.thread = threading.Thread(target=self.work, name='semantic-fusion', daemon=True)
            self.thread.start()
            self.create_timer(.5, self.status)

        def on_graph(self, message):
            try:
                if len(message.data) > 2_000_000:
                    raise ValueError('oversized graph revision')
                with self.lock:
                    graph = checked_trajectory(json.loads(message.data), self.trajectory,
                                               cfg['max_keyframes'], args.map_frame)
                    if self.trajectory and graph['stamp_ns'] == self.trajectory['stamp_ns']:
                        if graph['version'] == self.trajectory['version']:
                            return
                    self.trajectory = graph
                self.wake.set()
            except (ValueError, KeyError, TypeError, OverflowError):
                self.counters['graph_invalid'] += 1

        def on_cloud(self, message):
            stamp = message.header.stamp.sec*10**9+message.header.stamp.nanosec
            if (message.header.frame_id != 'backend_lidar' or stamp < 0
                    or not 0 < message.width*message.height <= cfg['max_points']
                    or len(message.data) > cfg['max_points']*64):
                self.counters['invalid'] += 1
                return
            with self.lock:
                if stamp <= self.last_input:
                    self.counters['dropped_order'] += 1
                    return
                self.last_input = stamp
                self.pending[stamp] = (message, time.monotonic())
                self.counters['received'] += 1
                while len(self.pending) > cfg['pending_clouds']:
                    self.pending.popitem(last=False)
                    self.counters['dropped_full'] += 1
            self.wake.set()

        def snapshot(self):
            with self.lock:
                return self.latest

        def status(self):
            with self.lock:
                graph, latest, pending = self.trajectory, self.latest, len(self.pending)
            data = dict(self.counters, pending=pending, output_error=self.output.error,
                output_publications=self.output.published,
                latest_graph_stamp_ns=graph['stamp_ns'] if graph else None,
                latest_graph_version=graph['version'] if graph else None,
                map_info=latest['info'] if latest else None,
                graph_wall_age_s=time.monotonic()-graph['received'] if graph else None)
            self.status_pub.publish(String(data=json.dumps(data, allow_nan=False)))
            return data

        def event(self, row):
            self.log.write(json.dumps(row, allow_nan=False)+'\n')
            self.log.flush()

        def work(self):
            try:
                if not args.predictions_input:
                    sys.path.insert(0, str(ROOT/'src'))
                    from ptv3_client import PTv3Client
                    self.client = PTv3Client(half=True, shuffle=False, max_pts=cfg['max_points'],
                        intensity_scale=args.intensity_scale, stderr_path=str(args.out/'inference.log'),
                        startup_timeout_s=120., response_timeout_s=3.)
                    if self.client.num_classes != cfg['classes']:
                        raise ValueError('model class count does not match fusion vocabulary')
                self.counters['ready'] = True
                while not self.stop.is_set():
                    self.wake.wait(.05)
                    self.wake.clear()
                    with self.lock:
                        graph = self.trajectory
                        packet = next(iter(self.pending.items()), None)
                    if graph is None:
                        continue
                    graph_id = (graph['stamp_ns'], graph['version'])
                    graph_changed = graph_id != self.last_graph
                    started = time.monotonic()
                    revision = self.core.update_poses(graph['times_ns'], graph['poses']) if graph_changed else None
                    self.last_graph = graph_id
                    if revision and revision.get('changed_keyframes', 0):
                        self.counters['revisions'] += 1
                    added = None
                    if packet is not None:
                        stamp, (message, arrival) = packet
                        index = int(np.searchsorted(graph['times_ns'], stamp))
                        anchored = index < len(graph['times_ns']) and graph['times_ns'][index] == stamp
                        stale = time.monotonic()-arrival > cfg['max_queue_age_s']
                        if anchored or stale or stamp < graph['stamp_ns']:
                            with self.lock:
                                self.pending.pop(stamp, None)
                            if stale:
                                self.counters['dropped_stale'] += 1
                            elif not anchored:
                                self.counters['unanchored'] += 1
                            else:
                                points = read_xyz_intensity(message)
                                if not np.isfinite(points).all():
                                    raise ValueError('nonfinite semantic input')
                                infer_start = time.monotonic()
                                if args.predictions_input:
                                    from sensor_msgs_py.point_cloud2 import dtype_from_fields
                                    dt = dtype_from_fields(message.fields, point_step=message.point_step)
                                    dt = dt.newbyteorder('>' if message.is_bigendian else '<')
                                    rows = np.ndarray((message.height, message.width), dtype=dt,
                                        buffer=message.data, strides=(message.row_step, message.point_step))
                                    labels = rows['label'].reshape(-1).copy()
                                    confidence = rows['confidence'].reshape(-1).copy()
                                else:
                                    # B0's KITTI training inputs use [0,1]
                                    # reflectance. Livox exports [0,255]; the
                                    # caller explicitly declares that conversion.
                                    points[:, 3] /= args.reflectance_divisor
                                    labels, confidence = self.client.segment(points)
                                    labels, confidence = labels.copy(), confidence.copy()
                                infer_ms = (time.monotonic()-infer_start)*1000
                                body = points[:, :3].astype(np.float64) @ transform[:3, :3].T+transform[:3, 3]
                                added = self.core.add(stamp, body, labels, confidence, graph['poses'][index])
                                if added['status'] == 'capacity_reached':
                                    self.counters['capacity_reached'] += 1
                                else:
                                    self.counters['processed'] += 1
                                self.last_scan = dict(stamp_ns=stamp, points=body.astype(np.float32),
                                                      labels=labels, confidence=confidence)
                                self.event(dict(event='keyframe', stamp_ns=stamp, points=len(points),
                                    infer_ms=infer_ms, receipt_to_fusion_ms=(time.monotonic()-arrival)*1000,
                                    graph_stamp_ns=graph['stamp_ns'], graph_version=graph['version'], fusion=added))
                    # A snapshot is a coherent completed graph revision, even if
                    # a newer revision arrived while this bounded job was running.
                    if added is not None or graph_changed:
                        before_snapshot = time.monotonic()
                        rendered = self.core.snapshot(max_points=cfg['display_points'], include_scores=False)
                        info = dict(graph_stamp_ns=graph['stamp_ns'], graph_version=graph['version'],
                            graph_received_monotonic=graph['received'], ready_monotonic=time.monotonic(),
                            graph_to_ready_ms=(time.monotonic()-graph['received'])*1000,
                            snapshot_ms=(time.monotonic()-before_snapshot)*1000,
                            job_ms=(time.monotonic()-started)*1000,
                            revision=revision, summary=self.core.summary())
                        self.publication_id += 1
                        snapshot = dict(publication_id=self.publication_id, stamp_ns=graph['stamp_ns'],
                            map={k: rendered[k] for k in ('points', 'labels', 'confidence')},
                            scan=self.last_scan, info=info)
                        with self.lock:
                            self.latest = snapshot
                        self.event(dict(event='snapshot', **info))
            except Exception as exc:
                self.counters.update(failed=True, error=f'{type(exc).__name__}: {exc}')
                self.event(dict(event='error', error=self.counters['error']))
            finally:
                if self.client:
                    self.client.close()
                self.counters['ready'] = False

        def close(self):
            self.stop.set()
            self.wake.set()
            self.thread.join(timeout=10.)
            if self.thread.is_alive():
                raise TimeoutError('semantic worker did not stop')
            result = self.status()
            if args.verify_final and not self.counters['failed']:
                result['rebuild_parity'] = self.core.verify_rebuild()
            (args.out/'final.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            self.output.close()
            self.log.close()

    return SemanticNode()


def main():
    import rclpy
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frontend', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--config', type=Path)
    ap.add_argument('--input-cloud', default='/backend/input/semantic_cloud')
    ap.add_argument('--map-frame', default='map')
    ap.add_argument('--intensity-scale', type=float, default=.2)
    ap.add_argument('--reflectance-divisor', type=float, default=1.,
                    help='Input normalization before model scale: KITTI=1, Livox uint8 reflectivity=255')
    ap.add_argument('--predictions-input', action='store_true', help='Diagnostic replay only: consume explicit label/confidence fields')
    ap.add_argument('--verify-final', action='store_true', help='After stream ends, compare incremental map to fresh rebuild (not timed online work)')
    args = ap.parse_args()
    if not np.isfinite(args.reflectance_divisor) or args.reflectance_divisor <= 0:
        ap.error('reflectance divisor must be finite and positive')
    rclpy.init()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    node = None
    try:
        node = make_node(args)
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node, timeout_sec=.05)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
