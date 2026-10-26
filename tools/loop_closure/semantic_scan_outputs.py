"""Independent current-scan display; a stalled static-map publisher cannot block it."""
import argparse
from multiprocessing.connection import Connection
import os
import time

from .semantic_outputs import cloud


def instance_cloud(item):
    import numpy as np
    from sensor_msgs.msg import PointField
    m = cloud(item['points'], item['labels'], item['confidence'], 'backend_imu', item['stamp_ns'])
    old = np.frombuffer(m.data, dtype=np.uint8).reshape(len(item['points']), m.point_step)
    result = item['instances']
    semantic = result.get('semantics')
    fields = [('instance_id', '<u4'), ('visual_label', '<u2'), ('padding', '<u2'), ('instance_score', '<f4')]
    if semantic is not None:
        fields += [('raw_label','<u2'),('instance_label','<u2'),('raw_confidence','<f4'),
                   ('semantic_epoch','<u4'),('instance_confidence','<f4'),('instance_valid','u1'),('pad2','u1',(3,))]
    extra_dtype = np.dtype(fields)
    extra = np.zeros(len(old), extra_dtype)
    extra['instance_id'], extra['visual_label'], extra['instance_score'] = result['ids'], result['labels'], result['scores']
    ros_fields = [('instance_id', PointField.UINT32), ('visual_label', PointField.UINT16), ('instance_score', PointField.FLOAT32)]
    if semantic is not None:
        for field,key,datatype in [('raw_label','raw_labels',PointField.UINT16),('instance_label','instance_labels',PointField.UINT16),
            ('raw_confidence','raw_confidence',PointField.FLOAT32),('semantic_epoch','epochs',PointField.UINT32),
            ('instance_confidence','instance_confidence',PointField.FLOAT32),('instance_valid','valid',PointField.UINT8)]:
            extra[field] = semantic[key]; ros_fields.append((field,datatype))
    for name, datatype in ros_fields:
        m.fields.append(PointField(name=name, offset=m.point_step+extra_dtype.fields[name][1], datatype=datatype, count=1))
    m.data = np.concatenate((old, extra.view(np.uint8).reshape(len(old), extra_dtype.itemsize)), axis=1).tobytes()
    m.point_step += extra_dtype.itemsize
    m.row_step = m.point_step*m.width
    return m


def main():
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    import json
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
    instance_pub = node.create_publisher(PointCloud2, '/backend/semantic_instances', 2)
    instance_info = node.create_publisher(String, '/backend/semantic_instance_info', 2)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.01)
            if not connection.poll(.02):
                continue
            item = connection.recv()
            pub.publish(cloud(item['points'], item['labels'], item['confidence'], 'backend_imu', item['stamp_ns']))
            if item.get('instances') is not None:
                instance_pub.publish(instance_cloud(item))
                instance_info.publish(String(data=json.dumps(dict(item['instances']['info'],
                    stamp_ns=item['stamp_ns'], identity_key='(session, instance_id, semantic_epoch)' if item['instances'].get('semantics') else '(session, instance_id)',
                    semantic_update=item['instances']['info'].get('semantic_update',False)))))
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
