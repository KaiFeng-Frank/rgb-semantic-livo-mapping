r"""Actual ROS/native online semantic re-fusion contract fixture, without a frontend.

Source ROS externally, then run with the ROS Python (not the native worker venv)::

    ROS_DOMAIN_ID=90 /usr/bin/python3 -m tools.loop_closure.validate_semantic_ros \
        --out /tmp/semantic-contract --verify-final \
        --worker /home/frank/semantic_backend_runtime/venv/bin/python

--predictions-input is DIAGNOSTIC ONLY: deterministic synthetic predictions enter
only semantic_node. The real geometric BackendProcess receives XYZ and odometry,
never labels, confidence, GT, a dataset, or a precomputed graph. This is a bounded
functional fixture, not a frontend, segmentation accuracy, or performance study.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKER = Path('/home/frank/semantic_backend_runtime/venv/bin/python')
KEYFRAMES = ((1, 0., 0.), (11, 2., 2.1), (21, 4., 4.2), (32, 0., .3))
# Short, loop-free sensor times let three semantic keyframes share the same
# three-second, 10 Hz odometry interval. The last three exercise scan coalescing.
FAULT_KEYFRAMES = tuple((index + 1, float(index), index * 1.05) for index in range(7))
POSE_LIMIT_MS = 200.
SCAN_LIMIT_MS = 1000.
FAULT_POSES = 30


def check(condition, detail):
    if not condition:
        raise AssertionError(detail)


def stamp_ns(message):
    return message.header.stamp.sec * 10**9 + message.header.stamp.nanosec


def set_header(message, stamp, frame):
    message.header.frame_id = frame
    message.header.stamp.sec = int(stamp // 10**9)
    message.header.stamp.nanosec = int(stamp % 10**9)


def semantic_rows(message):
    """Decode the ACTUAL output schema independently of semantic_outputs.cloud.

    label is UINT16 at byte 12, confidence FLOAT32 at byte 16; there is no
    intensity field in semantic output. Respect row padding and byte order.
    """
    schema = [('x', 0, 7, 1), ('y', 4, 7, 1), ('z', 8, 7, 1),
              ('label', 12, 4, 1), ('confidence', 16, 7, 1)]
    check([(f.name, f.offset, f.datatype, f.count) for f in message.fields] == schema,
          'semantic output label/confidence schema changed')
    check(message.point_step == 20 and message.height >= 1, 'invalid semantic output stride')
    check(message.row_step >= message.width * 20, 'invalid semantic output row stride')
    check(len(message.data) == message.row_step * message.height, 'invalid semantic output length')
    endian = '>' if message.is_bigendian else '<'
    dtype = np.dtype(dict(names=['x', 'y', 'z', 'label', 'confidence'],
                         formats=[endian + 'f4'] * 3 + [endian + 'u2', endian + 'f4'],
                         offsets=[0, 4, 8, 12, 16], itemsize=20))
    rows = np.ndarray((message.height, message.width), dtype=dtype,
                      buffer=message.data, strides=(message.row_step, message.point_step))
    xyz = np.column_stack([rows[name].reshape(-1) for name in ('x', 'y', 'z')])
    labels, confidence = rows['label'].reshape(-1).copy(), rows['confidence'].reshape(-1).copy()
    check(np.isfinite(xyz).all() and np.isfinite(confidence).all(), 'nonfinite semantic output')
    check(np.all((confidence >= 0) & (confidence <= 1)), 'invalid output vote shares')
    return xyz, labels, confidence


def room_scene():
    # Identical scene and four poses to test_online's real retrieval/ICP test.
    points = np.random.default_rng(42).uniform(-8, 8, (12000, 3)).astype(np.float32)
    points[:4000, 0], points[4000:8000, 1], points[8000:, 2] = -6, 7, -2
    return points


def predictions(count):
    index = np.arange(count)
    labels = np.array([0, 7, 10, 11, 12, 13, 14, 15], np.uint16)[index % 8]
    confidence = np.array([.5, .625, .75, .875], np.float32)[index % 4]
    labels[index % 97 == 0] = 1  # Explicitly excluded from persistent evidence.
    confidence[index % 89 == 0] = .125  # Below the configured confidence gate.
    return labels, confidence


def reference_map(packets, poses, config):
    """Independent full-history reducer; no fusion-core imports or tile code."""
    points, labels, weights = [], [], []
    for packet, pose in zip(packets, poses):
        xyz, label, confidence = packet
        keep = (~np.isin(label, config['excluded_classes'])
                & (confidence >= config['minimum_confidence']) & (confidence > 0))
        body = xyz[keep].astype(np.float64)
        world = body @ pose[:3, :3].T + pose[:3, 3]
        points.append(np.floor(world / config['voxel']).astype(np.int64))
        labels.append(label[keep])
        weights.append(confidence[keep].astype(np.float64))
    keys, inverse = np.unique(np.concatenate(points), axis=0, return_inverse=True)
    scores = np.zeros((len(keys), config['classes']), np.float64)
    np.add.at(scores, (inverse, np.concatenate(labels)), np.concatenate(weights))
    winners = np.argmax(scores, axis=1).astype(np.uint16)
    confidence = (scores[np.arange(len(keys)), winners] / scores.sum(axis=1)).astype(np.float32)
    return ((keys.astype(np.float64) + .5) * config['voxel']).astype(np.float32), winners, confidence


def compare_map(message, packets, poses, config):
    actual = semantic_rows(message)
    expected = reference_map(packets, poses, config)
    # The public contract sorts by integer voxel coordinates. Comparing all
    # points also detects an old contribution left behind after a pose update.
    np.testing.assert_array_equal(actual[0], expected[0], err_msg='map geometry differs from raw-point rebuild')
    np.testing.assert_array_equal(actual[1], expected[1], err_msg='map labels differ from independent votes')
    np.testing.assert_allclose(actual[2], expected[2], atol=1e-7, rtol=0,
                               err_msg='map confidence differs from independent votes')
    return dict(voxels=len(expected[0]), points_labels_exact=True,
                max_confidence_error=float(np.max(np.abs(actual[2] - expected[2]), initial=0)))


def child_identity(parent_pid, module):
    """Find only a direct child of our Popen process; never match global pgrep."""
    matches = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            status = dict(line.split(':', 1) for line in (path / 'status').read_text().splitlines()
                          if ':' in line)
            command = (path / 'cmdline').read_bytes().split(b'\0')
            if int(status['PPid']) != parent_pid or module.encode() not in command:
                continue
            # Field 22 in /proc/pid/stat, after the possibly spaced comm field.
            start_ticks = (path / 'stat').read_text().rsplit(')', 1)[1].split()[19]
            matches.append(dict(pid=int(path.name), parent_pid=parent_pid,
                                start_ticks=start_ticks, module=module))
        except (OSError, KeyError, ValueError):
            continue
    check(len(matches) == 1, f'expected exactly one owned {module} child, got {matches}')
    return matches[0]


def signal_child(identity, sig):
    current = child_identity(identity['parent_pid'], identity['module'])
    check(current == identity, 'owned child identity changed before signal')
    os.kill(identity['pid'], sig)


class RosFixture:
    def __init__(self, out, worker, verify_final, keyframes=KEYFRAMES):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import String
        self.rclpy, self.out = rclpy, out
        out.mkdir(parents=True, exist_ok=False)
        self.processes, self.handles, self.paused = [], [], []
        self.stopped = False
        self.sent, self.corrected = {}, {}
        self.semantic_sent, self.scans_by_stamp = {}, {}
        self.keyframes = keyframes
        self.graphs, self.maps, self.infos, self.scans = [], [], [], []
        self.backend_status, self.semantic_status = [], []
        self.input_stamps, self.packets = [], []
        self.node = rclpy.create_node('semantic_contract_fixture')
        reliable = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
        best = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
        graph_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)

        def corrected(message):
            stamp = stamp_ns(message)
            check(stamp not in self.corrected, f'duplicate corrected pose {stamp}')
            self.corrected[stamp] = (time.monotonic(), message)

        def collect_graph(message):
            self.graphs.append((time.monotonic(), json.loads(message.data)))

        def collect_map(message):
            semantic_rows(message)
            check(message.header.frame_id == 'map', 'semantic map frame mismatch')
            check(stamp_ns(message) in self.input_stamps, 'map uses an unsent/future graph stamp')
            self.maps.append((time.monotonic(), message))

        def collect_scan(message):
            stamp = stamp_ns(message)
            check(message.header.frame_id == 'backend_imu', 'semantic scan frame mismatch')
            check(stamp in self.input_stamps, 'semantic scan has an unsent/future stamp')
            check(stamp not in self.scans_by_stamp, f'duplicate semantic scan {stamp}')
            check(not self.scans or stamp > stamp_ns(self.scans[-1][1]), 'semantic scans reordered')
            received = time.monotonic()
            expected = self.packets[self.input_stamps.index(stamp)]
            for actual, original in zip(semantic_rows(message), expected):
                np.testing.assert_array_equal(actual, original,
                                               err_msg='original semantic scan points/labels/confidence changed')
            self.scans.append((received, message))
            self.scans_by_stamp[stamp] = (received, message)

        self.subscriptions = [
            self.node.create_subscription(PoseStamped, '/backend/pose', corrected, reliable),
            self.node.create_subscription(String, '/backend/trajectory', collect_graph, graph_qos),
            self.node.create_subscription(PointCloud2, '/backend/semantic_map', collect_map, best),
            self.node.create_subscription(PointCloud2, '/backend/semantic_scan', collect_scan, best),
            self.node.create_subscription(String, '/backend/semantic_map_info',
                                          lambda m: self.infos.append((time.monotonic(), json.loads(m.data))), best),
            self.node.create_subscription(String, '/backend/status',
                                          lambda m: self.backend_status.append(json.loads(m.data)), reliable),
            self.node.create_subscription(String, '/backend/semantic_status',
                                          lambda m: self.semantic_status.append(json.loads(m.data)), reliable)]
        self.odom_pub = self.node.create_publisher(Odometry, '/backend/input/odometry', reliable)
        self.cloud_pub = self.node.create_publisher(PointCloud2, '/backend/input/cloud', reliable)
        self.semantic_pub = self.node.create_publisher(PointCloud2, '/backend/input/semantic_cloud', reliable)
        self.config = json.loads((ROOT / 'tools/loop_closure/semantic_online.json').read_text())
        (out / 'identity.yaml').write_text(
            '/**:\n  ros__parameters:\n    extrin_calib:\n'
            '      extrinsic_R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]\n'
            '      extrinsic_T: [0.0, 0.0, 0.0]\n')
        (out / 'backend_config.json').write_text(json.dumps(dict(
            submap_half_window=0, max_map_points=1000, map_every_keyframes=1)))
        try:
            self.backend = self.start('backend', [sys.executable, '-m', 'tools.loop_closure.online_node',
                '--worker-python', str(worker), '--semantic-trajectory', '--config',
                str(out / 'backend_config.json'), '--log', str(out / 'backend_events.jsonl')])
            command = [sys.executable, '-m', 'tools.loop_closure.semantic_node',
                       '--frontend', str(out / 'identity.yaml'), '--out', str(out / 'semantic'),
                       '--predictions-input']
            if verify_final:
                command.append('--verify-final')
            self.semantic = self.start('semantic', command)
            self.wait(lambda: self.backend_status and self.backend_status[-1]['worker_ready']
                      and self.semantic_status and self.semantic_status[-1]['ready']
                      and all(pub.get_subscription_count() == 1
                              for pub in (self.odom_pub, self.cloud_pub, self.semantic_pub))
                      and all(sub.get_publisher_count() == 1 for sub in self.subscriptions),
                      'native worker and ROS topic discovery', timeout=45.)
            self.native_worker = child_identity(self.backend.pid, 'tools.loop_closure.online_transport')
            # Keep the venv path lexical: resolving its symlink loses the venv.
            worker_command = (Path('/proc') / str(self.native_worker['pid']) / 'cmdline').read_bytes().split(b'\0')
            check(worker_command[0].decode() == str(worker), 'backend did not launch requested native venv')
            self.output_child = child_identity(self.semantic.pid, 'tools.loop_closure.semantic_outputs')
            self.scan_child = child_identity(self.semantic.pid, 'tools.loop_closure.semantic_scan_outputs')
            check(self.scan_child['pid'] != self.output_child['pid'], 'scan and map share an output process')
            check({'scan_publications', 'scan_superseded', 'scan_output_error'} <= self.semantic_status[-1].keys(),
                  'independent scan status fields are missing')
            self.spin(.2)
        except BaseException:
            self.close()
            raise

    def start(self, name, command):
        handle = (self.out / (name + '.log')).open('w')
        self.handles.append(handle)
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        self.processes.append((name, process))
        return process

    def health(self):
        for name, process in self.processes:
            check(process.poll() is None, f'{name} exited with {process.returncode}; see {self.out / (name + ".log")}')
        if self.backend_status:
            check(not self.backend_status[-1]['error'], self.backend_status[-1])
        if self.semantic_status:
            check(not self.semantic_status[-1]['failed'], self.semantic_status[-1])

    def spin(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.health()
            self.rclpy.spin_once(self.node, timeout_sec=min(.01, max(0., deadline - time.monotonic())))

    def wait(self, predicate, description, timeout=15.):
        deadline = time.monotonic() + timeout
        while not predicate():
            check(time.monotonic() < deadline, f'timeout waiting for {description}; '
                  f'backend={self.backend_status[-1:]}; semantic={self.semantic_status[-1:]}')
            self.spin(.02)

    def odometry(self, stamp, x):
        from nav_msgs.msg import Odometry
        message = Odometry()
        set_header(message, stamp, 'camera_init')
        message.child_frame_id = 'backend_imu'
        message.pose.pose.orientation.w = 1.
        message.pose.pose.position.x = float(x)
        self.sent[stamp] = time.monotonic()
        self.odom_pub.publish(message)
        return message

    def cloud(self, stamp, xyz, semantic=False, frame=None):
        from sensor_msgs.msg import PointCloud2, PointField
        dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')]
        if semantic:
            dtype += [('label', '<u2'), ('padding', '<u2'), ('confidence', '<f4')]
        rows = np.zeros(len(xyz), dtype=np.dtype(dtype))
        for column, name in enumerate(('x', 'y', 'z')):
            rows[name] = xyz[:, column]
        rows['intensity'] = np.arange(len(xyz)) % 251
        if semantic:
            rows['label'], rows['confidence'] = predictions(len(xyz))
        message = PointCloud2()
        set_header(message, stamp, frame or ('backend_lidar' if semantic else 'backend_imu'))
        message.height, message.width = 1, len(xyz)
        message.point_step, message.row_step = rows.dtype.itemsize, rows.nbytes
        message.is_dense = True
        message.fields = [PointField(name=name, offset=rows.dtype.fields[name][1],
                          datatype=PointField.UINT16 if name == 'label' else PointField.FLOAT32, count=1)
                          for name in rows.dtype.names if name != 'padding']
        message.data = rows.tobytes()
        return message

    def keyframe(self, index, wait=True):
        seconds, actual_x, odom_x = self.keyframes[index]
        stamp = seconds * 10**9
        # Float32 transport precedes the body's float64 fusion/extrinsic math.
        xyz = (room_scene().astype(np.float64) - [actual_x, 0., 0.]).astype(np.float32)
        labels, confidence = predictions(len(xyz))
        self.packets.append((xyz.copy(), labels.copy(), confidence.copy()))
        self.input_stamps.append(stamp)
        semantic = self.cloud(stamp, xyz, semantic=True)
        geometry = self.cloud(stamp, xyz)
        check(not {'label', 'confidence'} & {field.name for field in geometry.fields},
              'diagnostic labels leaked into geometric graph input')
        # Semantic input may precede its graph anchor. Alternate cloud/pose
        # arrival order; every graph pair still has exactly the same sensor stamp.
        self.semantic_sent[stamp] = time.monotonic()
        self.semantic_pub.publish(semantic)
        if index % 2:
            self.odometry(stamp, odom_x)
            self.cloud_pub.publish(geometry)
        else:
            self.cloud_pub.publish(geometry)
            self.odometry(stamp, odom_x)
        if wait:
            self.wait_keyframe(index)
        return semantic

    def wait_keyframe(self, index, require_scan=True):
        stamp = self.keyframes[index][0] * 10**9
        self.wait(lambda: self.graphs and self.graphs[-1][1]['stamp_ns'] == stamp
                  and self.semantic_status and self.semantic_status[-1]['processed'] == index + 1
                  and self.infos and self.infos[-1][1]['graph_stamp_ns'] == stamp
                  and self.infos[-1][1]['summary']['keyframes'] == index + 1
                  and self.maps and stamp_ns(self.maps[-1][1]) == stamp
                  and (not require_scan or stamp in self.scans_by_stamp),
                  f'completed semantic keyframe {index}')
        self.spin(.15)
        graph = self.graphs[-1][1]
        check(graph['times_ns'] == self.input_stamps, 'graph did not pair exact keyframe timestamps')
        check(self.infos[-1][1]['graph_version'] == graph['version'], 'map/graph revision mismatch')
        compare_map(self.maps[-1][1], self.packets, np.asarray(graph['poses']), self.config)
        if require_scan:
            check(stamp_ns(self.scans[-1][1]) == stamp, 'latest semantic scan stamp mismatch')

    def check_versions(self):
        self.spin(.2)
        check(len(self.maps) == len(self.infos), 'lost map/info publications; cannot establish consistent versions')
        ids = [info['publication_id'] for _, info in self.infos]
        check(all(b > a for a, b in zip(ids, ids[1:])), 'publication IDs did not increase')
        graphs = {(g['stamp_ns'], g['version']): (received, g) for received, g in self.graphs}
        for (arrival, message), (_, info) in zip(self.maps, self.infos):
            key = (info['graph_stamp_ns'], info['graph_version'])
            check(stamp_ns(message) == key[0] and key in graphs, 'map stamp/version has no real graph revision')
            check(info['map_sha256'] == hashlib.sha256(bytes(message.data)).hexdigest()
                  and info['map_points'] == message.width * message.height,
                  'map payload does not match its publication metadata')
            check(info['graph_received_monotonic'] <= info['ready_monotonic']
                  <= info['published_monotonic'], 'map publication timing is inconsistent')
            check(self.sent[key[0]] <= arrival, 'semantic output predates its actual input')
        return dict(publications=len(ids), graph_revisions=len(graphs), coherent=True,
                    payload_hashes_match=True)

    def close(self):
        if self.stopped:
            return
        self.stopped = True
        for identity in self.paused:
            try:
                signal_child(identity, signal.SIGCONT)
            except (AssertionError, OSError):
                pass
        # Stop semantic first so its final file verifies the completed map.
        for name, process in reversed(self.processes):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM if name == 'semantic' else signal.SIGINT)
            try:
                process.wait(timeout=12.)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3.)
            # A parent startup failure may leave its own native/output children.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for handle in self.handles:
            handle.close()
        observations = dict(trajectories=[dict(received_monotonic=received, **graph)
                                          for received, graph in self.graphs],
                            map_info=[dict(received_monotonic=received, **info)
                                      for received, info in self.infos],
                            maps=[dict(received_monotonic=received, stamp_ns=stamp_ns(message),
                                       points=message.width * message.height,
                                       sha256=hashlib.sha256(bytes(message.data)).hexdigest())
                                  for received, message in self.maps],
                            scans=[dict(received_monotonic=received, stamp_ns=stamp_ns(message),
                                        sent_monotonic=self.semantic_sent[stamp_ns(message)],
                                        frame_id=message.header.frame_id, points=message.width * message.height,
                                        sha256=hashlib.sha256(bytes(message.data)).hexdigest())
                                   for received, message in self.scans],
                            corrected_poses=[dict(stamp_ns=stamp, sent_monotonic=self.sent.get(stamp),
                                                  received_monotonic=received, frame_id=message.header.frame_id)
                                             for stamp, (received, message) in self.corrected.items()])
        (self.out / 'observations.json').write_text(json.dumps(observations, indent=2) + '\n')
        self.node.destroy_node()

    def final(self, verify):
        result = json.loads((self.out / 'semantic/final.json').read_text())
        check(self.semantic.returncode == 0 and not result['failed'], result)
        if verify:
            parity = result.get('rebuild_parity', {})
            check(parity.get('passed', parity.get('ok')) is True, f'final rebuild parity failed: {parity}')
        return result


def normal_phase(out, worker, verify):
    print('phase: real four-keyframe retrieval/ICP and independent semantic map checks', flush=True)
    fixture = RosFixture(out, worker, verify)
    try:
        for index in range(4):
            fixture.keyframe(index)
            if index == 0:
                first_map = fixture.maps[-1][0]
                check(len(fixture.input_stamps) == 1 and fixture.graphs[-1][1]['version'] == 0,
                      'first map needed a future keyframe')
            if index < 3:
                check(fixture.graphs[-1][1]['version'] == 0, 'loop accepted before 30 seconds elapsed')
        graph = fixture.graphs[-1][1]
        poses = np.asarray(graph['poses'])
        rows = [json.loads(row) for row in (out / 'backend_events.jsonl').read_text().splitlines()]
        keyframes = [row for row in rows if row['event'] == 'keyframe']
        check(len(keyframes) == 4 and keyframes[-1]['loop_count'] == 1, keyframes)
        accepted = [attempt for attempt in keyframes[-1]['attempts'] if attempt['accepted']]
        check(len(accepted) == 1 and accepted[0]['i'] == 0 and accepted[0]['j'] == 3
              and 'fitness' in accepted[0] and 'rmse_m' in accepted[0], 'missing actual retrieval/ICP loop')
        check(graph['version'] > 0 and abs(poses[-1, 0, 3]) < .3, 'accepted loop did not correct drift')
        history_motion = float(np.max(np.abs(poses[1:3, 0, 3] - [2.1, 4.2])))
        check(history_motion > .001, 'fixture did not exercise historical semantic re-fusion')
        independent = compare_map(fixture.maps[-1][1], fixture.packets, poses, fixture.config)
        summary = fixture.infos[-1][1]['summary']
        eligible = sum(int((~np.isin(label, fixture.config['excluded_classes'])
                           & (confidence >= fixture.config['minimum_confidence']) & (confidence > 0)).sum())
                       for _, label, confidence in fixture.packets)
        check(summary['input_points'] == 48000 and summary['stored_points'] == eligible,
              'retained evidence is not the original eligible point set')
        check(summary['stored_points'] > summary['map_voxels'], 'fixture failed to distinguish raw points from voxels')
        versions = fixture.check_versions()
        baseline = deepcopy(fixture.semantic_status[-1])
        map_hash = hashlib.sha256(bytes(fixture.maps[-1][1].data)).hexdigest()
        publications = len(fixture.maps)
        bad = fixture.cloud(32 * 10**9, np.full((12000, 3), 100., np.float32), semantic=True)
        fixture.semantic_pub.publish(bad)  # duplicate with adversarial geometry
        bad = deepcopy(bad)
        set_header(bad, 11 * 10**9, 'backend_lidar')
        fixture.semantic_pub.publish(bad)  # out of order
        set_header(bad, 33 * 10**9, 'unknown_frame')
        fixture.semantic_pub.publish(bad)  # invalid frame, later timestamp
        fixture.wait(lambda: fixture.semantic_status[-1]['dropped_order'] >= baseline['dropped_order'] + 2
                      and fixture.semantic_status[-1]['invalid'] >= baseline['invalid'] + 1,
                      'explicit rejection of duplicate/out-of-order/invalid-frame packets')
        fixture.spin(.3)
        check(len(fixture.maps) == publications, 'rejected semantic packet republished/changed the map')
        check(hashlib.sha256(bytes(fixture.maps[-1][1].data)).hexdigest() == map_hash, 'valid map changed')
        check(fixture.semantic_status[-1]['processed'] == 4, 'rejected packet counted as valid evidence')
        check(fixture.semantic_status[-1]['map_info']['summary'] == summary, 'rejected input mutated map state')
        check(set(fixture.corrected) == set(fixture.input_stamps), 'lost/extra corrected keyframe poses')
        for stamp, (_, message) in fixture.corrected.items():
            check(message.header.frame_id == 'map' and stamp in fixture.sent, 'corrected pose frame/stamp mismatch')
        check(first_map < fixture.sent[11 * 10**9], 'first map was not published before future input')
        check(fixture.semantic_status[-1]['scan_publications'] == 4
              and fixture.semantic_status[-1]['scan_superseded'] == 0
              and fixture.semantic_status[-1]['scan_output_error'] is None,
              'paced semantic scan output lost or superseded a view')
        result = dict(keyframes=4, input_stamps_ns=fixture.input_stamps, loop_count=1,
                      accepted_icp=accepted[0], historical_pose_change_m=history_motion,
                      first_map_before_future_input=True, versions=versions,
                      original_input_points=48000, retained_original_points=eligible,
                      scans_preserve_all_original_points_labels_confidence=True,
                      independent_final_map=independent, rejected_packets_leave_map_unchanged=True,
                      final_map_sha256=map_hash, native_worker=fixture.native_worker)
    finally:
        fixture.close()
    result['final'] = fixture.final(verify)
    return result


def scan_pause_phase(fixture):
    """Keep this in the existing fixture: three more keyframes, no new workers."""
    print('phase: pause current-scan output; verify maps continue and latest views coalesce', flush=True)
    baseline = deepcopy(fixture.semantic_status[-1])
    before_scans, before_maps = len(fixture.scans), len(fixture.maps)
    identity = child_identity(fixture.semantic.pid, 'tools.loop_closure.semantic_scan_outputs')
    signal_child(identity, signal.SIGSTOP)
    fixture.paused.append(identity)
    fixture.wait(lambda: '\nState:\tT' in (Path('/proc') / str(identity['pid']) / 'status').read_text(),
                 'owned scan output process to stop', timeout=2.)
    began = time.monotonic()
    for index in (4, 5, 6):
        fixture.keyframe(index, wait=False)
        fixture.wait_keyframe(index, require_scan=False)
        check(len(fixture.scans) == before_scans, 'scan output advanced while its process was stopped')
        check(fixture.semantic_status[-1]['output_error'] is None, 'scan pause blocked map publication')
    status = fixture.semantic_status[-1]
    check(status['processed'] == 7 and status['map_info']['summary']['keyframes'] == 7,
          'fusion did not continue for all three scan-paused keyframes')
    check(status['scan_publications'] == baseline['scan_publications'], 'stopped scan output acknowledged a view')
    check(len(fixture.maps) >= before_maps + 3, 'maps stopped while scans were paused')
    paused_s = time.monotonic() - began
    signal_child(identity, signal.SIGCONT)
    fixture.paused.remove(identity)
    fixture.wait(lambda: fixture.semantic_status[-1]['scan_output_error'] is None
                 and fixture.semantic_status[-1]['scan_publications']
                 + fixture.semantic_status[-1]['scan_superseded'] == 7
                 and 7 * 10**9 in fixture.scans_by_stamp,
                 'latest scan delivery and superseded accounting after resume')
    fixture.spin(.1)
    current = fixture.semantic_status[-1]
    publications = current['scan_publications'] - baseline['scan_publications']
    superseded = current['scan_superseded'] - baseline['scan_superseded']
    delivered = [stamp_ns(message) for _, message in fixture.scans[before_scans:]]
    check(0 < publications <= 2 and superseded > 0 and publications + superseded == 3,
          f'scan output did not retain at most one in-flight and one latest view: {current}')
    check(len(delivered) == publications and delivered[-1] == 7 * 10**9,
          'scan publication accounting differs from actual received views')
    check(current['output_error'] is None and current['map_info']['summary']['keyframes'] == 7,
          'map health changed during scan recovery')
    compare_map(fixture.maps[-1][1], fixture.packets,
                np.asarray(fixture.graphs[-1][1]['poses']), fixture.config)
    return dict(paused_child=identity, paused_s=paused_s, semantic_keyframes_processed=3,
                map_publications_while_scan_paused=len(fixture.maps) - before_maps,
                views_delivered_after_resume=publications, delivered_stamps_ns=delivered,
                views_superseded=superseded, publications_plus_superseded_equals_processed=True,
                final_scan_stamp_ns=delivered[-1], map_continued=True)


def fault_phase(out, worker, verify):
    print('phase: pause map output; three semantic keyframes and 30 poses continue independently', flush=True)
    fixture = RosFixture(out, worker, verify, keyframes=FAULT_KEYFRAMES)
    try:
        fixture.keyframe(0)
        fixture.wait(lambda: fixture.semantic_status[-1]['output_publications'] == len(fixture.maps)
                      and fixture.semantic_status[-1]['scan_publications'] == 1,
                      'first map and scan output acknowledgements')
        fixture.spin(.2)
        baseline_outputs, baseline_maps = fixture.semantic_status[-1]['output_publications'], len(fixture.maps)
        baseline_scans, baseline_status = len(fixture.scans), len(fixture.semantic_status)
        identity = child_identity(fixture.semantic.pid, 'tools.loop_closure.semantic_outputs')
        check(identity['pid'] != fixture.scan_child['pid'], 'pause target also owns current scans')
        signal_child(identity, signal.SIGSTOP)
        fixture.paused.append(identity)
        fixture.wait(lambda: '\nState:\tT' in (Path('/proc') / str(identity['pid']) / 'status').read_text(),
                      'owned semantic output process to stop', timeout=2.)
        began = time.monotonic()
        fault_stamps = []
        for index in range(FAULT_POSES):
            target = began + index / 10.
            fixture.spin(max(0., target - time.monotonic()))
            stamp = 2 * 10**9 + index * 100_000_000
            fault_stamps.append(stamp)
            if index % 10 == 0:
                fixture.keyframe(1 + index // 10, wait=False)
            else:
                fixture.odometry(stamp, 1.05 * (1 + index / 10.))
            fixture.spin(max(0., target + .1 - time.monotonic()))
        fixture.wait(lambda: all(stamp in fixture.corrected for stamp in fault_stamps),
                      'all corrected poses while semantic output remains stopped', timeout=.2)
        semantic_stamps = [2 * 10**9, 3 * 10**9, 4 * 10**9]
        fixture.wait(lambda: all(stamp in fixture.scans_by_stamp for stamp in semantic_stamps)
                      and fixture.semantic_status[-1]['processed'] == 4
                      and fixture.semantic_status[-1]['scan_publications'] == 4,
                      'all three current scans and fused keyframes while map output remains stopped', timeout=.5)
        latencies = [(fixture.corrected[stamp][0] - fixture.sent[stamp]) * 1000 for stamp in fault_stamps]
        scan_latencies = [(fixture.scans_by_stamp[stamp][0] - fixture.semantic_sent[stamp]) * 1000
                          for stamp in semantic_stamps]
        intervals = np.diff([fixture.sent[stamp] for stamp in fault_stamps])
        check(max(latencies) < POSE_LIMIT_MS, f'pose latency exceeded {POSE_LIMIT_MS} ms: {latencies}')
        check(max(scan_latencies) < SCAN_LIMIT_MS, f'current-scan latency exceeded {SCAN_LIMIT_MS} ms: {scan_latencies}')
        check(np.max(np.abs(intervals - .1)) < .05, f'fixture could not sustain its 10 Hz input cadence: {intervals}')
        check(len(fixture.maps) == baseline_maps, 'semantic output advanced while its process was stopped')
        fixture.wait(lambda: fixture.semantic_status[-1]['output_error'] is not None,
                      'semantic publication stall health signal', timeout=1.)
        stalled = fixture.semantic_status[-1]
        check('stalled' in stalled['output_error'] and stalled['processed'] == 4,
              'fault did not block publication while fusion continued')
        check(stalled['output_publications'] == baseline_outputs, 'stopped output acknowledged a new map')
        check(stalled['scan_superseded'] == 0 and len(fixture.scans) == baseline_scans + 3,
              'paced current scans were dropped or superseded by a stalled map publisher')
        check(all(row['scan_output_error'] is None for row in fixture.semantic_status[baseline_status:]),
              'map publication stall affected the independent scan process')
        check(stalled['map_info']['graph_stamp_ns'] == 4 * 10**9
              and stalled['map_info']['summary']['keyframes'] == 4,
              'map fusion did not advance to the final graph while publication was paused')
        check(stalled['map_publication_id'] > stalled['map_completed_id'], 'map publisher has no pending work')
        for first, following in zip(semantic_stamps, semantic_stamps[1:]):
            check(fixture.scans_by_stamp[first][0] < fixture.semantic_sent[following],
                  'current scan waited for a future keyframe')
        for index, stamp in enumerate(fault_stamps):
            message = fixture.corrected[stamp][1]
            check(message.header.frame_id == 'map', 'corrected pose frame changed during semantic stall')
            check(abs(message.pose.position.x - 1.05 * (1 + index / 10.)) < 1e-9,
                  'no-loop corrected odometry changed during semantic stall')
        paused_s = time.monotonic() - began
        resume_time = time.monotonic()
        check(all(fixture.scans_by_stamp[stamp][0] < resume_time for stamp in semantic_stamps),
              'some current scans arrived only after map resume')
        signal_child(identity, signal.SIGCONT)
        fixture.paused.remove(identity)
        fixture.wait_keyframe(3)
        fixture.wait(lambda: fixture.semantic_status[-1]['output_error'] is None
                      and fixture.semantic_status[-1]['output_publications'] > baseline_outputs
                      and fixture.semantic_status[-1]['map_completed_id']
                      == fixture.semantic_status[-1]['map_publication_id'],
                      'semantic display recovery after SIGCONT')
        recovery_map = compare_map(fixture.maps[-1][1], fixture.packets,
                                   np.asarray(fixture.graphs[-1][1]['poses']), fixture.config)
        check(set(fixture.corrected) == set(fixture.sent), 'some odometry did not produce exactly one corrected pose')
        result = dict(paused_child=identity, paused_s=paused_s, odometry_hz=10,
                      poses_sent_while_paused=FAULT_POSES, corrected_before_resume=len(latencies),
                      input_to_pose_p95_ms=float(np.percentile(latencies, 95)),
                      input_to_pose_max_ms=max(latencies), latency_limit_ms=POSE_LIMIT_MS,
                      input_interval_min_s=float(intervals.min()), input_interval_max_s=float(intervals.max()),
                      stall_reported=stalled['output_error'], fusion_continued=True,
                      semantic_keyframes_while_map_paused=3, scans_received_before_map_resume=3,
                      scan_stamps_ns=semantic_stamps, scan_output_child=fixture.scan_child,
                      input_to_scan_p95_ms=float(np.percentile(scan_latencies, 95)),
                      input_to_scan_max_ms=max(scan_latencies), scan_latency_limit_ms=SCAN_LIMIT_MS,
                      scan_superseded_while_map_paused=stalled['scan_superseded'],
                      semantic_output_recovered=True, recovery_map=recovery_map,
                      versions=fixture.check_versions())
        (out / 'pose_latency.json').write_text(json.dumps(dict(stamps_ns=fault_stamps,
            latency_ms=latencies, input_interval_s=intervals.tolist(), scan_stamps_ns=semantic_stamps,
            scan_latency_ms=scan_latencies), indent=2) + '\n')
        result['scan_pause'] = scan_pause_phase(fixture)
        result['final_versions'] = fixture.check_versions()
    finally:
        fixture.close()
    result['final'] = fixture.final(verify)
    return result


def run(out, worker, verify_final=False):
    import rclpy
    check(worker.is_file() and os.access(worker, os.X_OK), f'native worker Python is not executable: {worker}')
    check((ROOT / 'tools/loop_closure/semantic_fusion.py').is_file(),
          'semantic_fusion.py is not present yet; wait for the core agent')
    check((ROOT / 'tools/loop_closure/semantic_scan_outputs.py').is_file(),
          'independent semantic_scan_outputs.py is not present yet')
    # The caller sources ROS; this fixture deliberately does not source a shell
    # setup, launch a frontend, import native libraries, or access a GPU.
    os.environ.setdefault('ROS_DOMAIN_ID', '90')
    out.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, ros_domain_id=int(os.environ['ROS_DOMAIN_ID']),
                  scope='bounded synthetic ROS/native contract fixture; no accuracy or sustained realtime claim',
                  diagnostic_predictions_only=True, backend_labels_or_gt=False,
                  limits=dict(geometry_keyframes=4, points_per_scan=12000, fault_odometry_hz=10,
                              fault_poses=FAULT_POSES, pose_latency_limit_ms=POSE_LIMIT_MS,
                              semantic_keyframes_during_each_pause=3, fault_semantic_keyframes=7,
                              scan_latency_limit_ms=SCAN_LIMIT_MS,
                              input_cadence_tolerance_s=.05, discovery_deadline_s=45,
                              keyframe_deadline_s=15, cpu_threads_per_native_worker=1),
                  worker_python=str(worker), verify_final=verify_final)
    initialized = False
    try:
        rclpy.init(args=[])
        initialized = True
        result['normal'] = normal_phase(out / 'normal', worker, verify_final)
        result['fault'] = fault_phase(out / 'fault', worker, verify_final)
        result['passed'] = True
        return result
    except BaseException as error:
        result.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        if initialized and rclpy.ok():
            rclpy.shutdown()
        (out / 'assessment.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True, help='new evidence directory; never overwritten')
    parser.add_argument('--worker', '--worker-python', type=Path, default=DEFAULT_WORKER,
                        help='native GTSAM/Open3D worker venv Python; preserve the symlink path')
    parser.add_argument('--verify-final', action='store_true', help='also request offline core rebuild parity on clean shutdown')
    args = parser.parse_args()
    try:
        result = run(args.out.absolute(), args.worker.absolute(), args.verify_final)
    except Exception as error:
        print(f'FAIL: {error}\nEvidence: {args.out.absolute()}', file=sys.stderr)
        return 1
    print(json.dumps(dict(passed=result['passed'], evidence=str(args.out.absolute()),
                         loop_count=result['normal']['loop_count'],
                         corrected_while_semantic_output_paused=result['fault']['corrected_before_resume'],
                         scans_while_map_paused=result['fault']['scans_received_before_map_resume'],
                         scans_superseded_while_scan_paused=result['fault']['scan_pause']['views_superseded'],
                         pose_latency_max_ms=result['fault']['input_to_pose_max_ms']), indent=2))
    return 0


if __name__ == '__main__':
    if not __package__:
        sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
