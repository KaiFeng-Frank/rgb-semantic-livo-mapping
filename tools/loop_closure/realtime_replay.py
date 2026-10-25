"""Paced real-sensor replay with timestamp-paired end-to-end latency evidence.

Reads the original FAST-LIVO HKU MCAP in place. No raw data or images are saved.
Livox v1/v2 CustomMsg have the same CDR field layout; the first message is
decoded and checked before forwarding the original serialized bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import yaml

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), median_ms=float(np.median(a)), p95_ms=float(np.percentile(a, 95)),
                p99_ms=float(np.percentile(a, 99)), max_ms=float(a.max())) if len(a) else None


def stamp_ns(msg):
    return msg.header.stamp.sec*10**9+msg.header.stamp.nanosec


def run(args):
    import cv2
    import rosbag2_py
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.serialization import deserialize_message
    from livox_ros_driver2.msg import CustomMsg
    from sensor_msgs.msg import CompressedImage, Image, Imu, PointCloud2
    from nav_msgs.msg import Odometry, Path as RosPath
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String

    cv2.setNumThreads(1)
    common_config = yaml.safe_load(args.frontend.read_text())['/**']['ros__parameters']['common']
    compressed_transport = common_config.get('img_compressed', False)
    args.out.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = Node('backend_realtime_observer')
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
    rows, statuses, maps, paths = [], [], [], []
    arrivals, raw, corrected = {}, {}, {}
    sensor_counts = dict(lidar=0, image=0, imu=0)
    publishers = dict(lidar=node.create_publisher(CustomMsg, '/livox/lidar', 10),
                      image=node.create_publisher(CompressedImage if compressed_transport else Image,
                                                  common_config['img_topic'], 10),
                      imu=node.create_publisher(Imu, '/livox/imu', 1000))
    def pose(kind, message):
        now, stamp = time.monotonic(), stamp_ns(message)
        target = raw if kind == 'raw' else corrected
        target[stamp] = now
        p = message.pose.pose if kind == 'raw' else message.pose
        rows.append(dict(kind=kind, stamp_ns=stamp, received=now,
                         position=[p.position.x, p.position.y, p.position.z]))
    subs = [node.create_subscription(Odometry, '/backend/input/odometry', lambda m: pose('raw', m), qos),
            node.create_subscription(PoseStamped, '/backend/pose', lambda m: pose('corrected', m), qos),
            node.create_subscription(String, '/backend/status', lambda m: statuses.append(
                dict(received=time.monotonic(), **json.loads(m.data))), qos),
            node.create_subscription(PointCloud2, '/backend/map', lambda m: maps.append(
                dict(stamp_ns=stamp_ns(m), received=time.monotonic(), points=m.width)), qos),
            node.create_subscription(RosPath, '/backend/path', lambda m: paths.append(
                dict(stamp_ns=stamp_ns(m), received=time.monotonic(), poses=len(m.poses))), qos)]
    # Subscribe in both arms so the frontend performs the same bounded export.
    subs.append(node.create_subscription(PointCloud2, '/backend/input/cloud', lambda m: None, qos))
    processes, handles = [], []
    def start(command, name):
        handle = (args.out/name).open('w'); handles.append(handle)
        proc = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)
        processes.append(proc)
        return proc
    def spin(seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=min(.002, max(0., end-time.monotonic())))
    source_hash = hashlib.sha256()
    source_rows, replay_lateness, image_preparation = [], [], []
    try:
        start([sys.executable, 'tools/loop_closure/frontend/parameter_blackboard.py',
               '--ros-args', '--params-file', str(args.camera)], 'camera.log')
        spin(1.)
        if args.backend:
            start([sys.executable, '-m', 'tools.loop_closure.online_node', '--worker-python',
                   str(args.worker), '--log', str(args.out/'worker.jsonl')], 'backend.log')
        front = start([str(args.workspace/'install/fast_livo/lib/fast_livo/fastlivo_mapping'),
                       '--ros-args', '--params-file', str(args.frontend)], 'frontend.log')
        until = time.monotonic()+45
        while (not all(p.get_subscription_count() for p in publishers.values()) or
               (args.backend and not any(s['worker_ready'] for s in statuses))):
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('pipeline exited during startup')
            if time.monotonic() > until:
                raise TimeoutError('sensor/worker discovery deadline')
            spin(.05)
        spin(1.)
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id='mcap'),
                    rosbag2_py.ConverterOptions('', ''))
        first, base, last = None, None, None
        while reader.has_next():
            topic, payload, when = reader.read_next()
            if topic not in ('/livox/lidar', '/livox/imu', '/left_camera/image/compressed'):
                continue
            if first is None:
                first, base = when, time.monotonic()+.1
            elapsed = (when-first)*1e-9
            if elapsed > args.seconds:
                break
            scheduled = base+elapsed
            # Source conversion is done before its deadline, not serially after
            # each wait. Replay lateness exposes any failure to sustain 1x.
            preparation = time.monotonic()
            if topic == '/left_camera/image/compressed':
                compressed = deserialize_message(payload, CompressedImage)
                if compressed_transport:
                    message = payload  # Byte-identical camera transport; frontend decodes.
                else:
                    pixels = cv2.imdecode(np.frombuffer(compressed.data, np.uint8), cv2.IMREAD_COLOR)
                    if pixels is None:
                        raise ValueError('cannot decode source image')
                    message = Image(header=compressed.header, height=pixels.shape[0], width=pixels.shape[1],
                                    encoding='bgr8', step=pixels.shape[1]*3, data=pixels.tobytes())
                kind, stamp = 'image', stamp_ns(compressed)
                image_preparation.append((time.monotonic()-preparation)*1000)
            elif topic == '/livox/lidar':
                if sensor_counts['lidar'] == 0:
                    check = deserialize_message(payload, CustomMsg)
                    if check.point_num != len(check.points) or not 100 < check.point_num < 200000:
                        raise ValueError('invalid Livox CDR compatibility/point count')
                    if not all(np.isfinite([p.x, p.y, p.z]).all() for p in check.points):
                        raise ValueError('nonfinite source cloud')
                kind, message, stamp = 'lidar', payload, None
            else:
                kind, message, stamp = 'imu', payload, None
            spin(max(0., scheduled-time.monotonic()))
            sent = time.monotonic()
            replay_lateness.append(max(0., sent-scheduled)*1000)
            publishers[kind].publish(message)
            sensor_counts[kind] += 1
            if stamp is not None:
                arrivals[stamp] = sent
                source_rows.append(dict(stamp_ns=stamp, sent=sent, scheduled=scheduled))
            source_hash.update(topic.encode()); source_hash.update(payload)
            last = when
            rclpy.spin_once(node, timeout_sec=0.)
            if front.poll() is not None:
                raise RuntimeError(f'frontend exited {front.returncode}')
        playback_end = time.monotonic()
        spin(3.)
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('pipeline exited during replay')
        common = sorted(set(raw)&set(corrected))
        matched = sorted(set(raw)&set(arrivals))
        lat_raw = [(raw[s]-arrivals[s])*1000 for s in matched]
        lat_corrected = [(corrected[s]-arrivals[s])*1000 for s in common if s in arrivals]
        added = [(corrected[s]-raw[s])*1000 for s in common]
        # Observer scheduling can reorder callbacks by a fraction of a ms;
        # retain signed added latency rather than silently clipping it.
        sensor_duration = (last-first)*1e-9
        verdict = dict(backend=args.backend, source=str(args.bag), source_payload_sha256=source_hash.hexdigest(),
                       duration_s=sensor_duration, replay_wall_s=playback_end-base,
                       rmw=os.environ.get('RMW_IMPLEMENTATION'), cyclone_uri=os.environ.get('CYCLONEDDS_URI'),
                       input_counts=sensor_counts, frontend_poses=len(raw), corrected_poses=len(corrected),
                       matched_sensor_stamps=len(matched), corrected_matches=len(common),
                       missing_corrected=len(set(raw)-set(corrected)) if args.backend else None,
                       frontend_end_to_end=stats(lat_raw), corrected_end_to_end=stats(lat_corrected),
                       backend_added=stats(added), replay_lateness=stats(replay_lateness),
                       image_transport='original_compressed' if compressed_transport else 'decoded_raw',
                       image_preparation=stats(image_preparation), maps=len(maps), paths=len(paths),
                       last_status=statuses[-1] if statuses else None,
                       warmup_prefix_images=sum(s < min(raw) for s in arrivals) if raw else len(arrivals),
                       missing_images_after_initialization=sum(s >= min(raw) and s not in raw for s in arrivals) if raw else None,
                       scope='1x real-sensor LIVO + geometric backend; semantic inference is not part of this measurement')
        (args.out/'assessment.json').write_text(json.dumps(verdict, indent=2)+'\n')
        for name, data in [('poses', rows), ('inputs', source_rows), ('status', statuses), ('maps', maps), ('paths', paths)]:
            (args.out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in data))
        print(json.dumps(verdict, indent=2), flush=True)
        if not raw or len(matched) != len(raw) or (args.backend and len(common) != len(raw)):
            raise AssertionError('missing or unpaired sensor-time output')
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        for handle in handles:
            handle.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'worker', 'bag', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--frontend', type=Path, default=ROOT/'tools/loop_closure/frontend/hku.yaml')
    parser.add_argument('--camera', type=Path, default=ROOT/'tools/loop_closure/frontend/camera_hku.yaml')
    parser.add_argument('--seconds', type=float, default=130.)
    parser.add_argument('--backend', action='store_true')
    run(parser.parse_args())
