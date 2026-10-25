"""Independent visualization process: path/map work never runs in the pose executor.

The sender polls an immutable latest snapshot; a blocked display can hold one
in-flight transfer, never the graph worker or an unbounded output queue.
"""
import argparse
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class OutputProcess:
    def __init__(self, python, snapshot, map_frame, module='tools.loop_closure.online_outputs'):
        self.stop = threading.Event()
        self.error = None
        self.health_lock = threading.Lock()
        self.waiting_since = None
        self.last_published = None
        self.published = 0
        parent, child = socket.socketpair()
        try:
            self.process = subprocess.Popen(
                [str(python), '-m', module,
                 '--fd', str(child.fileno()), '--map-frame', map_frame],
                cwd=ROOT, pass_fds=(child.fileno(),))
        except Exception:
            parent.close()
            raise
        finally:
            child.close()
        self.connection = Connection(parent.detach())
        self.thread = threading.Thread(target=self._send, args=(snapshot,), daemon=True,
                                       name='backend-display-ipc')
        self.thread.start()
        self.monitor = threading.Thread(target=self._monitor, daemon=True, name='backend-display-health')
        self.monitor.start()

    def _monitor(self):
        while not self.stop.wait(.1):
            with self.health_lock:
                waiting = self.waiting_since
                if waiting is not None and time.monotonic()-waiting > 2.:
                    self.error = 'display publication stalled for more than 2 seconds'

    def _send(self, snapshot):
        last = None
        try:
            while not self.stop.wait(.05):
                if self.process.poll() is not None:
                    raise RuntimeError(f'display exited with {self.process.returncode}')
                result = snapshot()
                publication_id = result.get('publication_id', result['stamp_ns']) if result else None
                if not result or publication_id == last:
                    continue
                # Blocking serialization/send is confined to this thread. It
                # holds no snapshot lock, and the worker replaces snapshots.
                with self.health_lock:
                    self.waiting_since = time.monotonic()
                self.connection.send(result)
                # One in-flight output plus the latest graph snapshot. The
                # independent health thread also observes a blocked send.
                while not self.stop.is_set():
                    if self.connection.poll(.05):
                        ack = self.connection.recv()
                        if ack['stamp_ns'] != result['stamp_ns']:
                            raise RuntimeError('display acknowledgement stamp mismatch')
                        with self.health_lock:
                            self.last_published = ack['published_monotonic']
                            self.published += 1
                            self.waiting_since = None
                            self.error = None
                        break
                    if self.process.poll() is not None:
                        raise RuntimeError(f'display exited with {self.process.returncode}')
                last = publication_id
        except (OSError, EOFError, RuntimeError) as exc:
            if not self.stop.is_set():
                with self.health_lock:
                    self.error = f'{type(exc).__name__}: {exc}'

    def close(self):
        self.stop.set()
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        self.thread.join(timeout=2)
        self.monitor.join(timeout=1)
        self.connection.close()


def make_output_node(connection, map_frame):
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path as RosPath
    from sensor_msgs.msg import PointCloud2, PointField
    from .online_node import set_pose

    class Outputs(Node):
        def __init__(self):
            super().__init__('online_loop_outputs')
            self.lock = threading.Lock()
            self.latest = None
            self.stop = threading.Event()
            self.last_result, self.last_map, self.path_version = -1, None, -1
            self.path_cache = RosPath()
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.path_pub = self.create_publisher(RosPath, '/backend/path', qos)
            self.map_pub = self.create_publisher(PointCloud2, '/backend/map', qos)
            self.reader = threading.Thread(target=self.receive, daemon=True)
            self.reader.start()
            self.timer = self.create_timer(.2, self.outputs)

        def receive(self):
            try:
                while not self.stop.is_set():
                    if not connection.poll(.1):
                        continue
                    result = connection.recv()
                    with self.lock:
                        self.latest = result
            except (EOFError, OSError):
                self.stop.set()

        def snapshot(self):
            with self.lock:
                return self.latest

        def outputs(self):
            snapshot = self.snapshot()
            if not snapshot or snapshot['stamp_ns'] == self.last_result:
                return
            self.last_result = snapshot['stamp_ns']
            # Append ordinary keyframes; revisit the history only on a new loop
            # revision. Do not reconstruct every old ROS pose every second.
            path = self.path_cache
            if self.path_version != snapshot['version']:
                path.poses.clear()
                self.path_version = snapshot['version']
            path.header.frame_id = map_frame
            path.header.stamp = Time(sec=self.last_result//10**9, nanosec=self.last_result % 10**9)
            first = len(path.poses)
            for stamp, matrix in zip(snapshot['times_ns'][first:], snapshot['poses'][first:]):
                p = PoseStamped()
                p.header.frame_id = map_frame
                p.header.stamp = Time(sec=int(stamp//10**9), nanosec=int(stamp % 10**9))
                set_pose(p.pose, matrix)
                path.poses.append(p)
            self.path_pub.publish(path)
            map_id = (snapshot.get('map_stamp_ns'), snapshot.get('map_version'))
            if snapshot.get('map_points') is not None and map_id != self.last_map:
                self.last_map = map_id
                points = np.asarray(snapshot['map_points'], dtype='<f4')
                msg = PointCloud2()
                msg.header.frame_id = map_frame
                stamp = snapshot['map_stamp_ns']
                msg.header.stamp = Time(sec=stamp//10**9, nanosec=stamp % 10**9)
                msg.height, msg.width, msg.point_step = 1, len(points), 12
                msg.row_step, msg.is_dense = len(points)*12, True
                msg.fields = [PointField(name=name, offset=i*4, datatype=PointField.FLOAT32, count=1)
                              for i, name in enumerate(('x', 'y', 'z'))]
                msg.data = points.tobytes()
                self.map_pub.publish(msg)
            connection.send(dict(stamp_ns=self.last_result, published_monotonic=time.monotonic()))

        def destroy_node(self):
            self.stop.set()
            self.reader.join(timeout=1)
            return super().destroy_node()

    return Outputs()


def main():
    import rclpy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--map-frame', required=True)
    args = parser.parse_args()
    os.nice(10)
    connection = Connection(args.fd)
    rclpy.init()
    node = make_output_node(connection, args.map_frame)
    try:
        while rclpy.ok() and not node.stop.is_set():
            rclpy.spin_once(node, timeout_sec=.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        connection.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
