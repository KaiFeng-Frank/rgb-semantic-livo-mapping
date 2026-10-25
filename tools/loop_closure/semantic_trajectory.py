"""Compact graph revisions, isolated from geometric/semantic visualization."""
import argparse
from multiprocessing.connection import Connection
import json
import os
import time


def compact_snapshot(snapshot):
    result = snapshot()
    if not result:
        return None
    return {key: result[key] for key in ('stamp_ns', 'version', 'times_ns', 'poses')}


def main():
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--map-frame', required=True)
    args = parser.parse_args()
    os.nice(10)
    connection = Connection(args.fd)
    rclpy.init()
    node = rclpy.create_node('semantic_graph_revisions')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publisher = node.create_publisher(String, '/backend/trajectory', qos)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.01)
            if not connection.poll(.02):
                continue
            result = connection.recv()
            payload = dict(frame_id=args.map_frame, stamp_ns=result['stamp_ns'],
                           version=result['version'], times_ns=result['times_ns'].tolist(),
                           poses=result['poses'].tolist())
            publisher.publish(String(data=json.dumps(payload, allow_nan=False)))
            connection.send(dict(stamp_ns=result['stamp_ns'], published_monotonic=time.monotonic()))
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
