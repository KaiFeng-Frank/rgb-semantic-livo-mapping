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


def claim_packet(pending, graph, now, max_age):
    """Caller holds the queue lock; owned work cannot also be evicted as pending."""
    if graph is None or not pending:
        return None
    stamp, item = next(iter(pending.items()))
    if now-item[1] > max_age or stamp <= graph['stamp_ns']:
        return stamp, pending.pop(stamp)
    return None


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
    consistency_mode = getattr(args, 'consistency', 'off')
    if consistency_mode != 'off' and args.predictions_input:
        raise ValueError('Consistency requires full model posteriors, not top-1 diagnostic inputs')
    if args.config:
        cfg.update(json.loads(args.config.read_text()))
    if type(cfg['pending_clouds']) is not int or cfg['pending_clouds'] < 1 or cfg['max_queue_age_s'] <= 0:
        raise ValueError('invalid semantic queue bounds')
    transform = extrinsic(args.frontend)
    visual_enabled = getattr(args, 'visual_instances', False)
    instance_mode = getattr(args, 'instance_semantics', 'off')
    instance_checkpoint = getattr(args, 'instance_kalman_checkpoint', None)
    if instance_checkpoint and (instance_mode == 'off' or args.predictions_input):
        raise ValueError('Learned instance semantics requires real PTv3 inference and enabled instance semantics')
    if instance_mode not in ('off', 'shadow', 'active') or instance_mode != 'off' and not visual_enabled:
        raise ValueError('Instance semantics requires visual association and a valid mode')
    if visual_enabled:
        import cv2
        cv2.setNumThreads(1)  # Import before readiness, outside the first scan timing.
        from tools.visual_instances.geometry import Camera, FrameCache, InstanceFrame, associate, camera_at_scan
        if cfg['classes'] != 16 or consistency_mode != 'off':
            raise ValueError('Visual association requires native16 and consistency=off (no double fusion)')
        camera = Camera.from_livo(args.frontend, args.camera)
        if instance_mode != 'off':
            from tools.visual_instances.semantics import InstanceSemantics
            if instance_checkpoint:
                from tools.visual_instances.kalman import KalmanInstanceSemantics
                from tools.visual_instances.semantics import instance_groups
            # Import sparse geometry before online readiness.
            from scipy.spatial import cKDTree
            from scipy.sparse.csgraph import connected_components
    core_args = {key: cfg[key] for key in ('voxel', 'classes', 'tile_voxels', 'max_keyframes',
                 'max_points', 'max_total_points', 'excluded_classes', 'minimum_confidence')}

    class SemanticNode(Node):
        def __init__(self):
            super().__init__('online_semantic_refusion')
            args.out.mkdir(parents=True, exist_ok=False)
            (args.out/'config.json').write_text(json.dumps(dict(cfg, T_imu_lidar=transform.tolist(),
                consistency=consistency_mode, visual_instances=visual_enabled, instance_semantics=instance_mode,
                map_evidence='raw_single_frame',
                visual_camera=(dict(width=camera.width, height=camera.height, K=camera.K.tolist(),
                    distortion=camera.distortion.tolist(), T_camera_lidar=camera.T_camera_lidar.tolist(),
                    max_dt_ms=getattr(args, 'visual_max_dt_ms', 20.)) if visual_enabled else None),
                predictions_input=args.predictions_input, frontend=str(args.frontend),
                reflectance_divisor=args.reflectance_divisor, intensity_scale=args.intensity_scale), indent=2)+'\n')
            self.core = ReversibleSemanticMap(**core_args)
            self.consistency = None
            self.instance_semantics = InstanceSemantics() if instance_mode != 'off' else None
            if instance_checkpoint:
                self.instance_semantics = KalmanInstanceSemantics(instance_checkpoint)
            if self.instance_semantics:
                (args.out/'instance-semantics.json').write_text(json.dumps(self.instance_semantics.configuration(),indent=2)+'\n')
            if consistency_mode != 'off':
                from src.semantic_consistency import CausalConsistency
                if cfg['classes'] != 16:
                    raise ValueError('Consistency static-class policy is native16 only')
                self.consistency = CausalConsistency()
                (args.out/'consistency.json').write_text(json.dumps(self.consistency.config_dict(), indent=2)+'\n')
            self.lock = threading.Lock()
            self.pending, self.trajectory, self.latest = OrderedDict(), None, None
            self.visual_cache = FrameCache(max_dt_ms=getattr(args, 'visual_max_dt_ms', 20.)) if visual_enabled else None
            self.visual_counts = dict(received=0, invalid=0, matched_scans=0, unmatched_scans=0, associated_points=0)
            self.stop, self.wake = threading.Event(), threading.Event()
            self.client, self.last_scan = None, None
            self.scan_sequence = 0
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
            if visual_enabled:
                self.create_subscription(String, '/backend/visual_instances', self.on_instances, qos)
            self.status_pub = self.create_publisher(String, '/backend/semantic_status', 1)
            self.output = OutputProcess(sys.executable, self.snapshot, args.map_frame,
                                         module='tools.loop_closure.semantic_outputs')
            self.scan_output = OutputProcess(sys.executable, self.scan_snapshot, args.map_frame,
                                             module='tools.loop_closure.semantic_scan_outputs')
            self.thread = threading.Thread(target=self.work, name='semantic-fusion', daemon=True)
            self.thread.start()
            self.create_timer(.5, self.status)

        def on_instances(self, message):
            try:
                frame = InstanceFrame.decode(message.data)
                if frame.index.shape != (camera.height, camera.width):
                    raise ValueError('Camera dimensions mismatch')
                with self.lock:
                    self.visual_cache.add(frame)
                    self.visual_counts['received'] += 1
            except Exception as exc:
                # Bad/absent vision must not stop semantic inference or pose output.
                self.visual_counts['invalid'] += 1
                self.visual_counts['last_error'] = f'{type(exc).__name__}: {exc}'

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

        def scan_snapshot(self):
            with self.lock:
                return self.last_scan

        def status(self):
            with self.lock:
                graph, latest, pending = self.trajectory, self.latest, len(self.pending)
            data = dict(self.counters, pending=pending, output_error=self.output.error,
                visual=dict(self.visual_counts),
                output_publications=self.output.published,
                map_publication_id=latest['publication_id'] if latest else None,
                map_completed_id=self.output.completed_id,
                map_snapshots_superseded=self.output.superseded,
                scan_output_error=self.scan_output.error,
                scan_publications=self.scan_output.published,
                scan_superseded=self.scan_output.superseded,
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
                        probability_classes=cfg['classes'] if self.consistency or self.instance_semantics else 0,
                        instance_features=bool(instance_checkpoint),
                        intensity_scale=args.intensity_scale, stderr_path=str(args.out/'inference.log'),
                        startup_timeout_s=120., response_timeout_s=3., cancel_event=self.stop)
                    if self.client.num_classes != cfg['classes']:
                        raise ValueError('model class count does not match fusion vocabulary')
                self.counters['ready'] = True
                while not self.stop.is_set():
                    self.wake.wait(.05)
                    self.wake.clear()
                    with self.lock:
                        graph = self.trajectory
                        packet = claim_packet(self.pending, graph, time.monotonic(), cfg['max_queue_age_s'])
                    if graph is None:
                        continue
                    graph_id = (graph['stamp_ns'], graph['version'])
                    graph_changed = graph_id != self.last_graph
                    started = time.monotonic()
                    revision = self.core.update_poses(graph['times_ns'], graph['poses']) if graph_changed else None
                    if self.consistency and graph_changed and self.consistency.history is not None:
                        # Same received graph revision, with the sensor extrinsic
                        # applied once. Do not query the growing semantic map.
                        hstamp = self.consistency.history['stamp_ns']
                        hi = int(np.searchsorted(graph['times_ns'], hstamp))
                        if hi < len(graph['times_ns']) and graph['times_ns'][hi] == hstamp:
                            self.consistency.revise_pose([hstamp], [graph['poses'][hi] @ transform])
                        else:
                            self.consistency.history = None
                    self.last_graph = graph_id
                    if self.instance_semantics and graph_changed:
                        self.instance_semantics.revise_poses(graph['times_ns'], graph['poses'] @ transform)
                    if revision and revision.get('changed_keyframes', 0):
                        self.counters['revisions'] += 1
                    added = None
                    if packet is not None:
                        stamp, (message, arrival) = packet
                        index = int(np.searchsorted(graph['times_ns'], stamp))
                        anchored = index < len(graph['times_ns']) and graph['times_ns'][index] == stamp
                        stale = time.monotonic()-arrival > cfg['max_queue_age_s']
                        if anchored or stale or stamp < graph['stamp_ns']:
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
                                    if self.instance_semantics:
                                        # Explicit probability fields are fixture-only; never infer
                                        # a fake full posterior from a top-1 label/confidence pair.
                                        probabilities = np.column_stack([rows[f'p{i}'].reshape(-1) for i in range(cfg['classes'])])
                                        if not np.array_equal(labels, probabilities.argmax(1)) or not np.allclose(confidence, probabilities.max(1), atol=1e-5):
                                            raise ValueError('Fixture label/confidence differs from posterior')
                                else:
                                    # B0's KITTI training inputs use [0,1]
                                    # reflectance. Livox exports [0,255]; the
                                    # caller explicitly declares that conversion.
                                    points[:, 3] /= args.reflectance_divisor
                                    if self.consistency or self.instance_semantics:
                                        labels, confidence, probabilities = self.client.segment_probabilities(points)
                                    else:
                                        labels, confidence = self.client.segment(points)
                                    labels, confidence = labels.copy(), confidence.copy()
                                infer_ms = (time.monotonic()-infer_start)*1000
                                scan_labels, scan_confidence = labels, confidence
                                consistency_status = None
                                visual_result = None
                                if self.consistency:
                                    consistent = self.consistency.step(points[:, :3], probabilities,
                                        stamp_ns=int(stamp), pose=graph['poses'][index] @ transform)
                                    consistency_status = consistent['status']
                                    if consistency_mode == 'active':
                                        scan_labels = consistent['probabilities'].argmax(1).astype(np.uint16)
                                        scan_confidence = consistent['probabilities'].max(1)
                                if visual_enabled:
                                    with self.lock:
                                        visual_frame = self.visual_cache.get(int(stamp))
                                    association_start = time.monotonic()
                                    projection_camera = (camera_at_scan(camera, graph, int(stamp),
                                        visual_frame.stamp_ns, transform) if visual_frame is not None else None)
                                    if projection_camera is not None:
                                        visual_result = associate(points[:, :3], projection_camera, visual_frame)
                                        visual_result['info'].update(dt_ms=(stamp-visual_frame.stamp_ns)/1e6,
                                            ego_motion='same_timestamp' if stamp==visual_frame.stamp_ns else 'received_graph_interpolation',
                                            elapsed_ms=(time.monotonic()-association_start)*1000)
                                        self.visual_counts['matched_scans'] += 1
                                        self.visual_counts['associated_points'] += visual_result['info']['associated']
                                    else:
                                        self.visual_counts['unmatched_scans'] += 1
                                        visual_result = dict(ids=np.zeros(len(points), np.uint32),
                                            labels=np.full(len(points), 65535, np.uint16), scores=np.zeros(len(points), np.float32),
                                            info=dict(reason='no_causal_camera_or_pose_bracket', session=None, associated=0))
                                    if instance_checkpoint:
                                        groups, tracks = instance_groups(points[:,:3],visual_result)
                                        pooled = self.client.pool_last_instances(groups,len(tracks))
                                        instance_features = dict(zip(tracks,pooled))
                                        visual_result['info']['feature_pool_ms'] = self.client.last_pool_ms
                                    if self.instance_semantics:
                                        enhanced = self.instance_semantics.step(points[:,:3], probabilities, visual_result,
                                            stamp_ns=int(stamp), pose=graph['poses'][index] @ transform,
                                            **(dict(features=instance_features) if instance_checkpoint else {}))
                                        visual_result['semantics'] = enhanced
                                        visual_result['info'].update(instance_semantics=enhanced['info'],
                                            semantic_update=instance_mode=='active', instance_mode=instance_mode)
                                        if instance_mode == 'active':
                                            scan_labels, scan_confidence = enhanced['labels'], enhanced['confidence']
                                body = points[:, :3].astype(np.float64) @ transform[:3, :3].T+transform[:3, 3]
                                self.scan_sequence += 1
                                with self.lock:
                                    self.last_scan = dict(publication_id=self.scan_sequence,
                                        stamp_ns=stamp, points=body.astype(np.float32),
                                        labels=scan_labels, confidence=scan_confidence, instances=visual_result)
                                # Preserve independent raw evidence for reversible
                                # map fusion; enhanced outputs must not vote twice.
                                added = self.core.add(stamp, body, labels, confidence, graph['poses'][index])
                                if added['status'] == 'capacity_reached':
                                    self.counters['capacity_reached'] += 1
                                else:
                                    self.counters['processed'] += 1
                                self.event(dict(event='keyframe', stamp_ns=stamp, points=len(points),
                                    infer_ms=infer_ms, receipt_to_fusion_ms=(time.monotonic()-arrival)*1000,
                                    consistency=consistency_status,
                                    visual=visual_result['info'] if visual_result else None,
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
                            info=info)
                        with self.lock:
                            self.latest = snapshot
                        self.event(dict(event='snapshot', **info))
            except InterruptedError as exc:
                if not self.stop.is_set():
                    self.counters.update(failed=True, error=f'{type(exc).__name__}: {exc}')
                    self.event(dict(event='error', error=self.counters['error']))
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
            try:
                if self.thread.is_alive():
                    if self.client:
                        self.client.close()
                    raise TimeoutError('semantic worker did not stop')
                result = self.status()
                if args.verify_final and not self.counters['failed']:
                    result['rebuild_parity'] = self.core.verify_rebuild()
                (args.out/'final.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            finally:
                self.scan_output.close()
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
    ap.add_argument('--consistency', choices=('off', 'shadow', 'active'), default='off',
                    help='Opt-in causal short-term scan consistency; map always retains raw evidence')
    ap.add_argument('--visual-instances', action='store_true', help='Publish visual instance association; label updates require --instance-semantics active')
    ap.add_argument('--camera', type=Path, help='Full-resolution raw camera calibration for visual association')
    ap.add_argument('--visual-max-dt-ms', type=float, default=20.)
    ap.add_argument('--instance-semantics',choices=('off','shadow','active'),default='off')
    ap.add_argument('--instance-kalman-checkpoint',type=Path)
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
