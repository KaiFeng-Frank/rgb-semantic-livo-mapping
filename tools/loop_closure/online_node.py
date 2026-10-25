"""ROS 2 adapter for an isolated incremental geometric loop backend.

Run with system ROS Python; --worker-python may be a different Python/venv.
The frontend publishes its sensor-time contract with backend.enabled:=true.
"""
import argparse
import json
from pathlib import Path
import time
import sys

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat

from .online_transport import BackendProcess, ExactPairs, load_settings
from .online_outputs import OutputProcess


def stamp_ns(msg):
    return msg.header.stamp.sec*10**9+msg.header.stamp.nanosec


def pose_matrix(pose):
    p, q = pose.position, pose.orientation
    quat = np.array([q.w, q.x, q.y, q.z])
    if not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12:
        raise ValueError('invalid quaternion')
    result = np.eye(4)
    result[:3, :3] = quat2mat(quat)
    result[:3, 3] = [p.x, p.y, p.z]
    if not np.isfinite(result).all():
        raise ValueError('nonfinite pose')
    return result


def set_pose(msg, matrix):
    msg.position.x, msg.position.y, msg.position.z = map(float, matrix[:3, 3])
    q = mat2quat(matrix[:3, :3])
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = map(float, q)


def make_node(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs_py.point_cloud2 import read_points_numpy
    from std_msgs.msg import String
    from tf2_ros import TransformBroadcaster

    class BackendNode(Node):
        def __init__(self):
            super().__init__('online_loop_backend')
            if len({args.map_frame, args.odom_frame, args.body_frame}) != 3:
                raise ValueError('map, odom and body frames must be distinct')
            self.cfg = load_settings(args.config)
            self.client = BackendProcess(args.worker_python, self.cfg, args.log)
            self.pairs = ExactPairs(self.cfg['pair_cache_size'])
            self.last_odom, self.last_result, self.last_map = -1, -1, None
            self.rejected, self.poses_out = 0, 0
            self.callback_ms = []
            self.correction = np.eye(4)
            self.tf = TransformBroadcaster(self)
            self.pose_pub = self.create_publisher(PoseStamped, '/backend/pose', 10)
            self.status_pub = self.create_publisher(String, '/backend/status', 1)
            qos = QoSProfile(depth=8, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.odom_sub = self.create_subscription(Odometry, args.odometry, self.on_odom, qos)
            self.cloud_sub = self.create_subscription(PointCloud2, args.cloud, self.on_cloud, qos)
            self.outputs = OutputProcess(sys.executable, self.client.snapshot, args.map_frame)
            self.trajectory = None
            if args.semantic_trajectory:
                from .semantic_trajectory import compact_snapshot
                self.trajectory = OutputProcess(sys.executable,
                    lambda: compact_snapshot(self.client.snapshot), args.map_frame,
                    module='tools.loop_closure.semantic_trajectory')
            self.status_timer = self.create_timer(1., self.status)

        def on_odom(self, msg):
            start = time.monotonic()
            stamp = stamp_ns(msg)
            if msg.header.frame_id != args.odom_frame or msg.child_frame_id != args.body_frame:
                self.rejected += 1
                return
            if stamp <= self.last_odom:
                self.rejected += 1
                return
            try:
                raw = pose_matrix(msg.pose.pose)
            except ValueError:
                self.rejected += 1
                return
            self.last_odom = stamp
            snapshot = self.client.snapshot()
            # Never apply a correction from the future if streams are delayed.
            if snapshot and snapshot['stamp_ns'] <= stamp:
                self.correction = snapshot['correction']
            pose = PoseStamped()
            pose.header.stamp, pose.header.frame_id = msg.header.stamp, args.map_frame
            set_pose(pose.pose, self.correction @ raw)
            self.pose_pub.publish(pose)
            # Separate continuous odom and globally corrected map frame. Do not
            # feed corrected poses back as independent frontend measurements.
            for parent, child, matrix in ((args.map_frame, args.odom_frame, self.correction),
                                         (args.odom_frame, args.body_frame, raw)):
                tf = TransformStamped()
                tf.header.stamp, tf.header.frame_id, tf.child_frame_id = msg.header.stamp, parent, child
                tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, matrix[:3, 3])
                q = mat2quat(matrix[:3, :3])
                tf.transform.rotation.w, tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z = map(float, q)
                self.tf.sendTransform(tf)
            self.poses_out += 1
            self.pair(stamp, self.pairs.add('pose', stamp, raw))
            self.callback_ms.append((time.monotonic()-start)*1000)
            self.callback_ms = self.callback_ms[-1000:]

        def on_cloud(self, msg):
            if (msg.header.frame_id != args.body_frame or msg.width*msg.height > self.cfg['max_points_per_keyframe']
                    or len(msg.data) > self.cfg['max_points_per_keyframe']*32):
                self.rejected += 1
                return
            self.pair(stamp_ns(msg), self.pairs.add('cloud', stamp_ns(msg), msg))

        def pair(self, stamp, pair):
            if pair is None:
                return
            pose, cloud_msg = pair
            try:
                points = read_points_numpy(cloud_msg, field_names=['x', 'y', 'z'], skip_nans=True).reshape(-1, 3)
                self.client.submit(stamp, pose, points)
            except (ValueError, AssertionError) as exc:
                self.rejected += 1
                self.get_logger().warning(f'keyframe rejected: {exc}')

        def status(self):
            snapshot = self.client.snapshot()
            data = dict(output_error=self.outputs.error, worker_ready=self.client.ready.is_set(), error=self.client.error,
                        trajectory_error=self.trajectory.error if self.trajectory else None,
                        output_publications=self.outputs.published,
                        output_wall_age_s=(time.monotonic()-self.outputs.last_published) if self.outputs.last_published else None,
                        poses_published=self.poses_out, rejected=self.rejected,
                        pair_evictions=self.pairs.evicted, pending=self.client.pending.qsize(),
                        ipc=dict(self.client.stats),
                        odometry_callback_p95_ms=float(np.percentile(self.callback_ms, 95)) if self.callback_ms else None)
            if snapshot:
                data.update({k: snapshot[k] for k in ('status', 'stamp_ns', 'keyframe', 'version',
                            'loop_count', 'total_ms', 'queue_and_processing_ms')})
                data['correction_age_s'] = max(0., (self.last_odom-snapshot['stamp_ns'])*1e-9)
                data['result_wall_age_s'] = max(0., time.monotonic()-snapshot['received_monotonic'])
                data['late_result'] = snapshot.get('late', False)
            msg = String()
            msg.data = json.dumps(data, allow_nan=False)
            self.status_pub.publish(msg)

        def destroy_node(self):
            if self.trajectory:
                self.trajectory.close()
            self.outputs.close()
            self.client.close()
            return super().destroy_node()

    return BackendNode()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-python', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--odometry', default='/backend/input/odometry')
    parser.add_argument('--cloud', default='/backend/input/cloud')
    parser.add_argument('--map-frame', default='map')
    parser.add_argument('--odom-frame', default='camera_init')
    parser.add_argument('--body-frame', default='backend_imu')
    parser.add_argument('--semantic-trajectory', action='store_true',
                        help='Publish compact graph revisions in an independent process')
    return parser.parse_known_args(argv)


def main():
    import rclpy
    from rclpy.executors import ExternalShutdownException
    args, ros_args = arguments()
    rclpy.init(args=ros_args)
    node = None
    try:
        node = make_node(args)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # SIGTERM can invalidate the ROS context between the loop guard and wait-set creation.
        if rclpy.ok():
            raise
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
