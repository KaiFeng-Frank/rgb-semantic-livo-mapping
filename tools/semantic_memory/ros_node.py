"""Online semantic memory on explicitly deskewed XYZI clouds and same-time odometry.

Cloud coordinates must be in odometry.child_frame_id at the header timestamp.
This adapter does not deskew raw spinning scans or estimate a transform. Legacy
semantic_map_node remains unchanged. Shadow mode publishes single-frame labels.
"""
import argparse
from collections import OrderedDict
import copy
import json
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def read_xyz_intensity(pc):
    """Respect field offsets, point/row padding and endianness without mutation."""
    from sensor_msgs_py.point_cloud2 import dtype_from_fields
    if (pc.height < 1 or pc.width < 1 or pc.point_step < 1
            or pc.row_step < pc.width * pc.point_step
            or len(pc.data) != pc.row_step * pc.height):
        raise ValueError('Invalid PointCloud2 dimensions/row layout')
    dtype = dtype_from_fields(pc.fields, point_step=pc.point_step)
    dtype = dtype.newbyteorder('>' if pc.is_bigendian else '<')
    names = ('x', 'y', 'z', 'intensity')
    if any(k not in dtype.names or dtype.fields[k][0].shape for k in names):
        raise ValueError('Scalar x/y/z/intensity fields required')
    rows = np.ndarray((pc.height, pc.width), dtype=dtype, buffer=pc.data,
                      strides=(pc.row_step, pc.point_step))
    return np.column_stack([rows[k].reshape(-1) for k in names]).astype(np.float32)


def transform_from_odometry(msg):
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    a = np.array([q.x, q.y, q.z, q.w], dtype=float)
    if not np.isfinite(a).all() or abs(np.linalg.norm(a) - 1) > 1e-3:
        raise ValueError('Invalid odometry quaternion')
    a /= np.linalg.norm(a)
    x, y, z, w = a
    t = np.eye(4)
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3, 3] = [p.x, p.y, p.z]
    if not np.isfinite(t).all():
        raise ValueError('Nonfinite odometry')
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-cloud', default='/semantic/deskewed_points')
    ap.add_argument('--input-odometry', default='/semantic/deskewed_odometry')
    ap.add_argument('--output', default='/semantic/memory_scan')
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--shadow', action='store_true')
    ap.add_argument('--input-is-deskewed', action='store_true', required=True)
    ap.add_argument('--epoch', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not args.shadow and not args.checkpoint:
        ap.error('Active mode requires --checkpoint; untrained mode requires --shadow')
    if args.epoch < 0:
        ap.error('epoch must be nonnegative')
    args.out.mkdir(parents=True, exist_ok=False)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from ptv3_client import PTv3Client

    class SemanticNode(Node):
        def __init__(self):
            super().__init__('causal_semantic_memory')
            self.clouds, self.poses = OrderedDict(), OrderedDict()
            self.jobs = queue.Queue(maxsize=2)
            self.done = threading.Event()
            self.last_stamp, self.counter = -1, 0
            self.world = None
            self.failed = False
            self.status = dict(queued=0, output=0, dropped_full=0, dropped_stale=0,
                               dropped_order=0, dropped_late=0, unmatched=0, invalid=0, failed=False, shadow=args.shadow)
            self.log = (args.out / 'adapter.jsonl').open('x')
            self.pub = self.create_publisher(PointCloud2, args.output, 2)
            self.pub_status = self.create_publisher(String, args.output + '/status', 2)
            self.client = PTv3Client(half=True, shuffle=False,
                                     memory_checkpoint=str(args.checkpoint) if args.checkpoint else None,
                                     memory_shadow=args.shadow, stderr_path=str(args.out / 'worker.log'),
                                     response_timeout_s=.3)
            qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(PointCloud2, args.input_cloud, lambda m: self.accept(m, True), qos)
            self.create_subscription(Odometry, args.input_odometry, lambda m: self.accept(m, False), qos)
            self.thread = threading.Thread(target=self.work, daemon=True)
            self.thread.start()
            self.create_timer(1., self.publish_status)

        def publish_status(self):
            msg = String(data=json.dumps(dict(self.status, queue=self.jobs.qsize())))
            self.pub_status.publish(msg)

        def accept(self, msg, cloud):
            if self.failed:
                return
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if stamp <= self.last_stamp:
                self.status['dropped_order'] += 1
                return
            cache = self.clouds if cloud else self.poses
            if stamp in cache:
                self.status['dropped_order'] += 1
                return
            cache[stamp] = (msg, time.monotonic())
            while len(cache) > 32:
                cache.popitem(last=False)
                self.status['unmatched'] += 1
            if stamp not in self.clouds or stamp not in self.poses:
                return
            pc, arrival = self.clouds.pop(stamp)
            odom, _ = self.poses.pop(stamp)
            try:
                if not pc.header.frame_id or pc.header.frame_id != odom.child_frame_id or not odom.header.frame_id:
                    raise ValueError('Cloud and odometry frame mismatch')
                if self.world is not None and self.world != odom.header.frame_id:
                    raise ValueError('Coordinate frame changed; restart with new epoch')
                transform = transform_from_odometry(odom)
                if pc.width * pc.height > self.client.max_pts:
                    raise ValueError('Input exceeds declared point capacity')
                pts = read_xyz_intensity(pc)
                if not len(pts) or not np.isfinite(pts).all():
                    raise ValueError('Nonfinite/empty point cloud')
                self.jobs.put_nowait((pc.header, pts, transform, stamp, arrival, self.counter))
            except queue.Full:
                self.status['dropped_full'] += 1
                return
            except Exception as exc:
                self.status['invalid'] += 1
                self.get_logger().error(str(exc))
                return
            self.world, self.last_stamp = odom.header.frame_id, stamp
            self.counter += 1
            self.status['queued'] += 1

        def work(self):
            while not self.done.is_set():
                try:
                    header, pts, transform, stamp, arrival, frame = self.jobs.get(timeout=.1)
                except queue.Empty:
                    continue
                if time.monotonic() - arrival > .3:
                    self.status['dropped_stale'] += 1
                    continue
                try:
                    t = time.monotonic()
                    labels, confidence = self.client.segment(pts, memory_context=dict(
                        stamp=stamp / 1e9, frame=frame, epoch=args.epoch, transform=transform))
                    if time.monotonic() - arrival > .3:
                        self.status['dropped_late'] += 1
                        continue
                    dtype = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('intensity','<f4'),
                                      ('label','<u2'),('padding','<u2'),('confidence','<f4')])
                    data = np.zeros(len(pts), dtype=dtype)
                    for j, k in enumerate(('x','y','z','intensity')):
                        data[k] = pts[:, j]
                    data['label'], data['confidence'] = labels, confidence
                    fields = [PointField(name=k, offset=dtype.fields[k][1], datatype=(PointField.UINT16 if k=='label' else PointField.FLOAT32), count=1)
                              for k in ('x','y','z','intensity','label','confidence')]
                    cloud = PointCloud2(header=copy.deepcopy(header), height=1, width=len(pts), fields=fields,
                                        is_bigendian=False, point_step=24, row_step=24*len(pts),
                                        data=data.tobytes(), is_dense=True)
                    self.pub.publish(cloud)
                    self.status['output'] += 1
                    rec = dict(frame=frame, stamp_ns=stamp, points=len(pts), shadow=args.shadow,
                               worker_ms=self.client.last_infer_ms,
                               process_ms=(time.monotonic()-t)*1000,
                               receipt_to_publish_ms=(time.monotonic()-arrival)*1000)
                    self.log.write(json.dumps(rec)+'\n'); self.log.flush()
                except Exception as exc:
                    self.failed = self.status['failed'] = True
                    self.status['error'] = str(exc)
                    self.get_logger().error('Semantic worker failed: '+str(exc))
                    self.done.set()

        def close(self):
            self.done.set()
            self.thread.join(timeout=5)
            self.client.close()
            self.thread.join(timeout=2)
            self.log.close()

    rclpy.init()
    node = None
    try:
        node = SemanticNode()
        rclpy.spin(node)
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
