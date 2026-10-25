"""Bounded asynchronous semantic display. Never owns graph or fusion state."""
import argparse
import json
import hashlib
from multiprocessing.connection import Connection
import os
import time

import numpy as np


def cloud(points, labels, confidence, frame, stamp):
    from sensor_msgs.msg import PointCloud2, PointField
    from builtin_interfaces.msg import Time
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('label', '<u2'), ('padding', '<u2'), ('confidence', '<f4')])
    data = np.zeros(len(points), dtype=dtype)
    for j, name in enumerate(('x', 'y', 'z')):
        data[name] = points[:, j]
    data['label'], data['confidence'] = labels, confidence
    msg = PointCloud2()
    msg.header.frame_id = frame
    msg.header.stamp = Time(sec=int(stamp//10**9), nanosec=int(stamp % 10**9))
    msg.height, msg.width, msg.point_step = 1, len(points), dtype.itemsize
    msg.row_step, msg.is_dense = len(points)*dtype.itemsize, True
    msg.fields = [PointField(name=name, offset=dtype.fields[name][1],
                  datatype=PointField.UINT16 if name == 'label' else PointField.FLOAT32, count=1)
                  for name in ('x', 'y', 'z', 'label', 'confidence')]
    msg.data = data.tobytes()
    return msg


def main():
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--map-frame', required=True)
    args = parser.parse_args()
    os.nice(10)
    connection = Connection(args.fd)
    rclpy.init()
    node = rclpy.create_node('semantic_refusion_outputs')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    map_pub = node.create_publisher(PointCloud2, '/backend/semantic_map', qos)
    info_pub = node.create_publisher(String, '/backend/semantic_map_info', qos)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.01)
            if not connection.poll(.02):
                continue
            item = connection.recv()
            m = item['map']
            message = cloud(m['points'], m['labels'], m['confidence'], args.map_frame, item['stamp_ns'])
            digest = hashlib.sha256(message.data).hexdigest()
            map_pub.publish(message)
            info_pub.publish(String(data=json.dumps(dict(item['info'],
                publication_id=item['publication_id'], map_sha256=digest, map_points=message.width,
                published_monotonic=time.monotonic()), allow_nan=False)))
            connection.send(dict(stamp_ns=item['stamp_ns'], published_monotonic=time.monotonic()))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        connection.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
