"""Actual ROS + FAST-LIVO2 sensor-time/IPC smoke; synthetic sensors, no accuracy claim."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

from .online_node import make_node, arguments, stamp_ns

ROOT = Path(__file__).resolve().parents[2]


def run(workspace, worker, out):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2, Imu, Image
    from std_msgs.msg import String
    from sensor_msgs_py.point_cloud2 import read_points_numpy
    sys.path.insert(0, str(ROOT/'src'))
    from kitti_to_ros2bag import PC_DTYPE, make_pointcloud2
    from kitti_scan import synth

    out.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    args, _ = arguments(['--worker-python', str(worker), '--log', str(out/'worker.jsonl')])
    backend = make_node(args)
    observer = Node('backend_contract_fixture')
    executor = SingleThreadedExecutor()
    executor.add_node(backend)
    executor.add_node(observer)
    odoms, clouds, corrected, status, inputs = [], [], [], [], {}
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
    subscriptions = [observer.create_subscription(Odometry, '/backend/input/odometry', odoms.append, qos),
                     observer.create_subscription(PointCloud2, '/backend/input/cloud', clouds.append, qos),
                     observer.create_subscription(PoseStamped, '/backend/pose', corrected.append, 100),
                     observer.create_subscription(String, '/backend/status', lambda m: status.append(json.loads(m.data)), 10)]
    pub_cloud = observer.create_publisher(PointCloud2, '/velodyne_points', 10)
    pub_imu = observer.create_publisher(Imu, '/imu', 200)
    pub_image = observer.create_publisher(Image, '/camera/image_raw', 10)
    processes, handles = [], []
    def spin(seconds):
        until = time.monotonic()+seconds
        while time.monotonic() < until:
            executor.spin_once(timeout_sec=.002)
    def start(command, name):
        if name == 'frontend.log' and os.environ.get('BACKEND_TEST_GDB'):
            command = ['gdb', '--batch', '-ex', 'run', '-ex', 'thread apply all bt 8', '--args'] + command
        h = (out/name).open('w'); handles.append(h)
        p = subprocess.Popen(command, cwd=ROOT, stdout=h, stderr=subprocess.STDOUT)
        processes.append(p)
        return p
    def stamp(msg, seconds):
        ns = round(seconds*1e9)
        msg.header.stamp.sec, msg.header.stamp.nanosec = ns//10**9, ns % 10**9
    try:
        start([sys.executable, 'frontend/parameter_blackboard.py', '--ros-args', '--params-file',
               'config/camera_kitti_cam2.yaml'], 'camera.log')
        config = yaml.safe_load((ROOT/'config/kitti_velodyne64.yaml').read_text())
        params = config['/**']['ros__parameters']
        params['backend'] = dict(enabled=True, keyframe_interval_s=1., max_points=12000)
        params['semantic'] = dict(enabled=False)
        params['evo'] = dict(seq_name='online_backend_synthetic_contract', pose_output_en=False)
        (out/'frontend.yaml').write_text(yaml.safe_dump(config))
        spin(1.)
        front = start([str(workspace/'install/fast_livo/lib/fast_livo/fastlivo_mapping'),
                      '--ros-args', '--params-file', str(out/'frontend.yaml')], 'frontend.log')
        deadline = time.monotonic()+40
        while not backend.client.ready.is_set() or pub_cloud.get_subscription_count() == 0:
            if front.poll() is not None or backend.client.error:
                raise RuntimeError(f'startup failed: {front.poll()}, {backend.client.error}')
            if time.monotonic() > deadline:
                raise TimeoutError('startup/discovery deadline')
            spin(.05)
        spin(1.)
        rng = np.random.default_rng(631)
        points = rng.uniform(-12, 12, (18000, 4)).astype(np.float32)
        points[:6000, 0] = -8
        points[6000:12000, 1] = 9
        points[12000:, 2] = -2
        ring, relative, order = synth(points[:, :3], sweep_duration=.1)
        array = np.empty(len(points), PC_DTYPE)
        for i, key in enumerate(('x', 'y', 'z', 'intensity')):
            array[key] = points[order, i]
        array['time'], array['ring'] = relative*1e6, ring
        pixels = rng.integers(0, 255, (370, 1226, 3), dtype=np.uint8).tobytes()
        for frame in range(70):
            base = 1000.+frame*.1
            for k in range(10):
                imu = Imu(); stamp(imu, base+k*.01)
                imu.header.frame_id = 'imu'
                imu.linear_acceleration.z = 9.81
                pub_imu.publish(imu)
            msg = make_pointcloud2(round(base*1e9), array, 'velodyne')
            pub_cloud.publish(msg)
            image = Image(); stamp(image, base+.05)
            image.height, image.width, image.encoding, image.step = 370, 1226, 'bgr8', 1226*3
            image.data = pixels
            inputs[stamp_ns(image)] = time.monotonic()
            pub_image.publish(image)
            spin(.1)
            if front.poll() is not None:
                raise RuntimeError(f'frontend exited during sensor fixture: {front.returncode}')
        spin(2.)
        if front.poll() is not None or backend.client.error:
            raise RuntimeError(f'runtime exited: frontend={front.poll()}, worker={backend.client.error}')
        os_stamps = [stamp_ns(m) for m in odoms]
        cs_stamps = [stamp_ns(m) for m in corrected]
        assert len(odoms) >= 50, len(odoms)
        assert os_stamps == cs_stamps, (len(odoms), len(corrected))
        assert all(s in inputs for s in os_stamps)
        assert len(set(os_stamps)) == len(os_stamps)
        original = {stamp_ns(m): m for m in odoms}
        max_error, max_quaternion_error = 0., 0.
        for m in corrected:
            p, raw = m.pose, original[stamp_ns(m)].pose.pose
            values = [p.position.x-raw.position.x, p.position.y-raw.position.y, p.position.z-raw.position.z]
            max_error = max(max_error, max(abs(x) for x in values))
            q = np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
            qr = np.array([raw.orientation.x, raw.orientation.y, raw.orientation.z, raw.orientation.w])
            max_quaternion_error = max(max_quaternion_error, min(np.linalg.norm(q-qr), np.linalg.norm(q+qr)))
        assert max_error < 1e-9, max_error
        assert max_quaternion_error < 1e-9, max_quaternion_error
        for m in clouds:
            assert stamp_ns(m) in original and m.header.frame_id == 'backend_imu'
            assert 100 <= m.width*m.height <= 12000
            assert np.isfinite(read_points_numpy(m, field_names=['x', 'y', 'z'])).all()
        snapshot = backend.client.snapshot()
        assert snapshot and snapshot['keyframe'] >= 3, snapshot
        assert snapshot['loop_count'] == 0
        # Pause ONLY our worker to prove the independent pose path does not wait
        # for native optimization. This controlled fixture is not a field test.
        os.kill(backend.client.process.pid, signal.SIGSTOP)
        extra_odom = observer.create_publisher(Odometry, '/backend/input/odometry', 10)
        extra_cloud = observer.create_publisher(PointCloud2, '/backend/input/cloud', 10)
        spin(.5)
        before = backend.poses_out
        for i in range(12):
            m, c = deepcopy(odoms[-1]), deepcopy(clouds[-1])
            stamp(m, 1010.+i); c.header.stamp = m.header.stamp
            extra_odom.publish(m); extra_cloud.publish(c)
            spin(.03)
        spin(.1)
        during = backend.poses_out-before
        queue_full = backend.client.stats['queue_full']
        os.kill(backend.client.process.pid, signal.SIGCONT)
        spin(1.)
        assert during == 12 and queue_full > 0, (during, queue_full)
        result = dict(scope='synthetic sensor ROS integration; not accuracy or sustained realtime admission',
                      frontend_poses=len(os_stamps), corrected_poses=len(cs_stamps),
                      keyframe_clouds=len(clouds)-12, exact_sensor_timestamps=True,
                      no_loop_max_position_error_m=max_error,
                      no_loop_max_quaternion_error=max_quaternion_error,
                      backend_keyframes=snapshot['keyframe']+1,
                      pose_outputs_while_worker_paused=during, overload_dropped_keyframes=queue_full,
                      last_status=status[-1] if status else None, passed=True)
        (out/'assessment.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    finally:
        if backend.client.process.poll() is None:
            os.kill(backend.client.process.pid, signal.SIGCONT)
        for p in processes:
            if p.poll() is None:
                p.terminate()
            try: p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=3)
        for h in handles: h.close()
        backend.destroy_node(); observer.destroy_node(); executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--worker-python', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    # Resolving a venv Python symlink bypasses that venv's dependency isolation.
    run(args.workspace.resolve(), args.worker_python.absolute(), args.out.resolve())
