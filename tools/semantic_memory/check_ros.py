"""Actual ROS Jazzy/PTv3 shadow integration, using a fixed real scan read-only.

Run with /usr/bin/python3 after sourcing /opt/ros/jazzy/setup.bash. This is an
engineering fixture: identity odometry and sensor times are synthetic. Passing
--input-is-deskewed exercises the adapter CLI; it is NOT evidence of deskew,
driving accuracy, or calibrated confidence. No training is performed.

Queues are bounded: this fixture sends one pair at a time, publisher depth 1,
subscriber depth 2. The adapter has depth-2 subscriptions, 32-entry matching
caches, and a 2-job queue. Its synchronous client uses a 4-slot shared-memory
ring (client in-flight limit 2). This is not a high-rate backpressure test.
"""
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCAN = Path('/media/frank/Extreme SSD/rgb_lidar_shared_backbone/remote_snapshot/'
            'rgb_lidar_shared_backbone/datasets/nuscenes/s0_trainval/samples/'
            'LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin')
ENV = dict(PTV3_PYTHON='/home/frank/semantic_livo_runtime/venv/bin/python',
           PTV3_POINTCEPT='/home/frank/semantic_livo_runtime/src/Pointcept_v151',
           PTV3_CONFIG='/home/frank/semantic_livo_runtime/weights/nuscenes-semseg-pt-v3m1-0-base/config.py',
           PTV3_CHECKPOINT='/home/frank/semantic_livo_runtime/v06-armB0_s1-student.pth',
           RMW_IMPLEMENTATION='rmw_cyclonedds_cpp', ROS_DOMAIN_ID='65',
           PYTHONDONTWRITEBYTECODE='1')
OUTPUT_DTYPE = np.dtype(dict(
    names=['x', 'y', 'z', 'intensity', 'label', 'confidence'],
    formats=['<f4', '<f4', '<f4', '<f4', '<u2', '<f4'],
    offsets=[0, 4, 8, 12, 16, 20], itemsize=24))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def run(args):
    require(os.path.realpath(sys.executable) == os.path.realpath('/usr/bin/python3'),
            'Run this fixture with /usr/bin/python3, not the PTv3 interpreter')
    require(os.environ.get('ROS_DISTRO') == 'jazzy', 'Source /opt/ros/jazzy/setup.bash first')
    os.environ.update(ENV)
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String

    raw = np.fromfile(args.scan, dtype='<f4')
    require(raw.size > 0 and raw.size % 5 == 0, 'Expected a nonempty Nx5 float32 scan')
    points = np.ascontiguousarray(raw.reshape(-1, 5)[:, :4])
    require(len(points) <= 200000 and np.isfinite(points).all(), 'Invalid source scan')
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, scope='fixed_real_scan_identity_odometry_synthetic_times_engineering_only',
                  driving_accuracy_evaluated=False, deskew_validated=False, training=False,
                  scan=str(args.scan), source_sha256=sha256(args.scan), points=len(points),
                  intensity='Source column 3 unchanged; adapter client intensity_scale=0.2',
                  environment={name: os.environ[name] for name in ENV},
                  queue_policy=dict(harness_outstanding_pairs=1, publisher_depth=1,
                                    subscriber_depth=2, adapter_job_capacity=2,
                                    adapter_matching_cache_entries_each=32, shm_slots=4,
                                    client_max_inflight=2, adapter_max_age_s=0.3,
                                    stress_tested=False),
                  attempts=[], outputs=[], negative_checks=[])
    result['source_code_sha256'] = {
        name: sha256(ROOT / name) for name in
        ('tools/semantic_memory/ros_node.py', 'src/ptv3_client.py',
         'src/ptv3_worker.py', 'src/semantic_memory.py')}
    topic = '/memory_ros_check_20261025_' + str(os.getpid())
    command = ['/usr/bin/python3', '-B', '-u', '-m', 'tools.semantic_memory.ros_node',
               '--shadow', '--input-is-deskewed', '--epoch', '0',
               '--input-cloud', topic + '/cloud', '--input-odometry', topic + '/odom',
               '--output', topic + '/output', '--out', str(args.out / 'node')]
    result['command'] = command
    process = node = None
    node_log = None
    initialized = False
    started = time.monotonic()
    errors, pending, received = [], {}, {}
    statuses = deque(maxlen=32)
    last_status = {}
    max_queue = 0

    def on_status(message):
        nonlocal last_status, max_queue
        try:
            last_status = json.loads(message.data)
            statuses.append(dict(last_status))
            max_queue = max(max_queue, last_status['queue'])
            require(0 <= last_status['queue'] <= 2, 'Adapter job queue exceeded its declared bound')
        except Exception as exc:
            errors.append(str(exc))

    def on_output(message):
        try:
            stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            require(stamp in pending, f'Unexpected/duplicate publication at stamp {stamp}')
            require(stamp not in received, f'Duplicate output at stamp {stamp}')
            require(message.header.frame_id == pending[stamp]['frame_id'], 'Source frame_id changed')
            require(message.height == 1 and message.width == len(points), 'Output count/layout changed')
            require(not message.is_bigendian and message.point_step == 24,
                    'Output byte order/point_step differs from the wire contract')
            require(message.row_step == 24 * len(points) and len(message.data) == message.row_step,
                    'Output byte count/row_step is invalid')
            expected_fields = [(name, OUTPUT_DTYPE.fields[name][1],
                                PointField.UINT16 if name == 'label' else PointField.FLOAT32, 1)
                               for name in OUTPUT_DTYPE.names]
            actual_fields = [(field.name, field.offset, field.datatype, field.count)
                             for field in message.fields]
            require(actual_fields == expected_fields, 'Output field offsets/dtypes/counts changed')
            data = np.frombuffer(message.data, dtype=OUTPUT_DTYPE, count=len(points))
            for column, name in enumerate(('x', 'y', 'z', 'intensity')):
                require(np.array_equal(data[name], points[:, column]), name + ' changed or reordered')
            labels, confidence = data['label'], data['confidence']
            require(labels.dtype == np.dtype('<u2') and confidence.dtype == np.dtype('<f4'),
                    'Label/confidence dtype is invalid')
            require(np.isfinite(labels).all() and np.all(labels < 16), 'Labels are not finite native16 IDs')
            require(np.isfinite(confidence).all() and np.all((confidence >= 0) & (confidence <= 1)),
                    'Nonfinite/out-of-range confidence')
            record = dict(stamp_ns=stamp, frame_id=message.header.frame_id,
                          worker_frame=pending[stamp]['worker_frame'], points=len(data),
                          label_dtype=str(labels.dtype), confidence_dtype=str(confidence.dtype),
                          label_min=int(labels.min()), label_max=int(labels.max()),
                          confidence_min=float(confidence.min()), confidence_max=float(confidence.max()),
                          payload_sha256=hashlib.sha256(message.data).hexdigest(),
                          exact_source_header=True, exact_xyzi=True)
            received[stamp] = record
            result['outputs'].append(record)
        except Exception as exc:
            errors.append(str(exc))

    def wait_for(predicate, timeout, description):
        deadline = time.monotonic() + timeout
        while True:
            require(not errors, '; '.join(errors))
            require(process.poll() is None, f'ROS CLI exited early: {process.returncode}; see node.log')
            require(not last_status.get('failed'), 'Adapter/worker failed: ' + str(last_status))
            if predicate():
                return
            require(time.monotonic() < deadline, 'Timeout: ' + description)
            rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))

    def publish_pair(stamp, child_frame='memory_ros_check_lidar'):
        sec, nanosec = divmod(stamp, 10**9)
        cloud = PointCloud2()
        cloud.header.stamp.sec, cloud.header.stamp.nanosec = sec, nanosec
        cloud.header.frame_id = 'memory_ros_check_lidar'
        cloud.height, cloud.width = 1, len(points)
        cloud.fields = [PointField(name=name, offset=4 * column,
                                   datatype=PointField.FLOAT32, count=1)
                        for column, name in enumerate(('x', 'y', 'z', 'intensity'))]
        cloud.point_step, cloud.row_step = 16, 16 * len(points)
        cloud.is_bigendian, cloud.is_dense = False, True
        cloud.data = points.tobytes()
        odom = Odometry()
        odom.header.stamp.sec, odom.header.stamp.nanosec = sec, nanosec
        odom.header.frame_id = 'memory_ros_check_world'
        odom.child_frame_id = child_frame
        odom.pose.pose.orientation.w = 1.0
        odom_pub.publish(odom)
        cloud_pub.publish(cloud)

    stamp_counter = 1700000000123456789

    def positive(warm=False):
        nonlocal stamp_counter
        stamp_counter += 100000007
        stamp = stamp_counter
        before = dict(last_status)
        pending[stamp] = dict(frame_id='memory_ros_check_lidar', worker_frame=before['queued'])
        publish_pair(stamp)

        def completed():
            dropped = sum(last_status.get(k, 0) - before.get(k, 0)
                          for k in ('dropped_stale', 'dropped_late'))
            return stamp in received or dropped > 0

        wait_for(completed, args.frame_timeout, 'fixed scan output or explicit warmup drop')
        published = stamp in received
        wait_for(lambda: last_status['queued'] == before['queued'] + 1
                 and last_status['output'] == before['output'] + int(published)
                 and last_status['queue'] == 0,
                 args.frame_timeout, 'settled adapter counters')
        require(last_status['dropped_full'] == before['dropped_full'], 'Sequential fixture filled job queue')
        result['attempts'].append(dict(stamp_ns=stamp, worker_frame=pending[stamp]['worker_frame'],
                                       warmup=warm, published=published,
                                       dropped_late=last_status.get('dropped_late', 0) - before.get('dropped_late', 0),
                                       dropped_stale=last_status.get('dropped_stale', 0) - before.get('dropped_stale', 0)))
        del pending[stamp]
        require(warm or published, 'A required post-warmup scan was dropped')
        return published, stamp

    def negative(name, stamp, counter, increment, child_frame='memory_ros_check_lidar'):
        before = dict(last_status)
        count = len(received)
        publish_pair(stamp, child_frame)
        wait_for(lambda: last_status[counter] >= before[counter] + increment,
                 args.frame_timeout, name + ' rejection diagnostic')
        # A fresh subsequent status also rules out a delayed accidental job/output.
        previous_status_count = len(statuses)
        wait_for(lambda: len(statuses) > previous_status_count,
                 args.frame_timeout, name + ' quiet interval')
        require(len(received) == count, name + ' produced an output')
        for field in ('queued', 'output'):
            require(last_status[field] == before[field], name + ' changed ' + field)
        require(last_status[counter] == before[counter] + increment, name + ' unexpected rejection count')
        result['negative_checks'].append(dict(name=name, stamp_ns=stamp,
                                              diagnostic=counter, increment=increment, no_publication=True))

    try:
        rclpy.init(args=[])
        initialized = True
        node = rclpy.create_node('semantic_memory_ros_check_' + str(os.getpid()))
        input_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        cloud_pub = node.create_publisher(PointCloud2, topic + '/cloud', input_qos)
        odom_pub = node.create_publisher(Odometry, topic + '/odom', input_qos)
        node.create_subscription(PointCloud2, topic + '/output', on_output, 2)
        node.create_subscription(String, topic + '/output/status', on_status, 2)
        node_log = (args.out / 'node.log').open('xb')
        process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ),
                                   stdout=node_log, stderr=subprocess.STDOUT, start_new_session=True)
        result['node_pid'] = process.pid
        wait_for(lambda: cloud_pub.get_subscription_count() > 0
                 and odom_pub.get_subscription_count() > 0 and bool(last_status),
                 args.startup_timeout, 'PTv3 warmup and ROS discovery')
        result['startup_s'] = time.monotonic() - started
        for attempt in range(6):
            published, _ = positive(warm=True)
            if published:
                break
        else:
            raise AssertionError('No successful scan within six bounded warmup attempts')
        positive()
        _, last_stamp = positive()
        negative('cloud_odometry_frame_mismatch', last_stamp + 50000003,
                 'invalid', 1, child_frame='wrong_lidar_frame')
        negative('duplicate_source_timestamp', last_stamp, 'dropped_order', 2)
        positive()  # Rejections must not poison the next valid scan.
        require(last_status['shadow'] is True and last_status['failed'] is False, 'Invalid adapter mode/status')
        require(last_status['dropped_full'] == 0, 'Unexpected queue overflow')
        adapter_records = [json.loads(line) for line in (args.out / 'node/adapter.jsonl').read_text().splitlines()]
        require(len(adapter_records) == len(received), 'Adapter/output counts disagree')
        for record, output in zip(adapter_records, result['outputs']):
            require(record['stamp_ns'] == output['stamp_ns'] and record['frame'] == output['worker_frame'],
                    'Adapter source stamp or accepted-frame sequence changed')
            require(record['points'] == len(points) and record['shadow'] is True, 'Invalid adapter record')
            for field in ('worker_ms', 'process_ms', 'receipt_to_publish_ms'):
                require(math.isfinite(record[field]) and record[field] >= 0, 'Invalid timing diagnostic')
            require(record['receipt_to_publish_ms'] < 300.0, 'Stale scan was published')
        result['adapter_records'] = adapter_records
        worker_text = (args.out / 'node/worker.log').read_text(errors='replace')
        worker_records = [json.loads(line[len('[semantic_memory] '):])
                          for line in worker_text.splitlines() if line.startswith('[semantic_memory] {')]
        require(len(worker_records) == len(result['attempts']), 'Worker did not process exactly the accepted scans')
        for record, attempt in zip(worker_records, result['attempts']):
            require(record['frame'] == attempt['worker_frame']
                    and record['stamp'] == attempt['stamp_ns'] / 1e9,
                    'Worker temporal context changed')
            require(record['shadow'] is True and record['records'] <= 16384, 'Worker shadow/memory bound failed')
        require(any(record['history_points'] > 0 for record in worker_records[1:]),
                'Repeated scan never exercised memory retrieval')
        result['worker_memory_records'] = worker_records
        result['passed'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if process is not None:
            forced = False
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    forced = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
            result['shutdown'] = dict(returncode=process.returncode, forced=forced)
            if forced or process.returncode != 0:
                result['passed'] = False
                result.setdefault('error', 'ROS CLI did not shut down cleanly; see node.log')
        if node is not None:
            node.destroy_node()
        if initialized and rclpy.ok():
            rclpy.shutdown()
        if node_log is not None:
            node_log.close()
        result['final_status'] = last_status
        result['max_sampled_job_queue'] = max_queue
        result['recent_status'] = list(statuses)
        result['elapsed_s'] = time.monotonic() - started
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    summary = dict(passed=result['passed'], outputs=len(received),
                   negative_checks=len(result['negative_checks']),
                   warmup_drops=sum(not item['published'] for item in result['attempts']),
                   result=str(args.out / 'result.json'))
    if 'error' in result:
        summary['error'] = result['error']
    print(json.dumps(summary))
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=Path, default=SCAN)
    parser.add_argument('--out', type=Path,
                        default=Path('/home/frank/semantic_livo_runtime/memory-ros-check-20261025'))
    parser.add_argument('--startup-timeout', type=float, default=150.0)
    parser.add_argument('--frame-timeout', type=float, default=10.0)
    args = parser.parse_args()
    args.scan, args.out = args.scan.resolve(), args.out.resolve()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
