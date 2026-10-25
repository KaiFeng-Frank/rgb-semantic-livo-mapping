"""Independent current-scan display; a stalled static-map publisher cannot block it."""
import argparse
from multiprocessing.connection import Connection
import os
import time

from .semantic_outputs import cloud


def main():
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--map-frame', required=True)
    args = parser.parse_args()
    os.nice(10)
    connection = Connection(args.fd)
    rclpy.init()
    node = rclpy.create_node('semantic_current_scan_outputs')
    pub = node.create_publisher(PointCloud2, '/backend/semantic_scan',
                               QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.01)
            if not connection.poll(.02):
                continue
            item = connection.recv()
            pub.publish(cloud(item['points'], item['labels'], item['confidence'], 'backend_imu', item['stamp_ns']))
            connection.send(dict(stamp_ns=item['stamp_ns'], published_monotonic=time.monotonic()))
    except (EOFError, BrokenPipeError, KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # Suppress only the expected teardown race, never errors in a live context.
        if rclpy.ok():
            raise
    finally:
        connection.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
