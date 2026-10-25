"""Exercise geometry fallback through the running ROS bridge, without a GPU."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


class LiveBridge(unittest.TestCase):
    def test_once_only_fallback_preserves_sensor_points(self):
        self.exercise(1,46)

    def test_full_size_cloud_delivery(self):
        self.exercise(60000,47)

    def test_indexed_scan_reliable_delivery(self):
        self.exercise(63000,49,indexed=True)

    def exercise(self,repeats,domain,indexed=False):
        import numpy as np
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import PointCloud2, PointField
        from frontend.replay_smoke import terminate
        root = Path(__file__).resolve().parents[2]
        env = dict(os.environ, ROS_DOMAIN_ID=str(domain))
        rclpy.init(domain_id=domain)
        node = Node('semantic_bridge_contract')
        received = []
        pub = node.create_publisher(PointCloud2, '/contract/raw', 10)
        sub = node.create_subscription(PointCloud2, '/contract/annotated', received.append, 10)
        process = None
        with tempfile.TemporaryDirectory(prefix='semantic-bridge-') as temp:
            try:
                process = subprocess.Popen([sys.executable, '-m', 'frontend.input_bridge',
                    '--mode', 'geometry', '--fixed-delay', '--wait-ms', '80',
                    '--input-reliability', 'reliable' if repeats>1 else 'best_effort',
                    '--input', '/contract/raw', '--output', '/contract/annotated',
                    '--log', temp+'/input.jsonl'], cwd=root, env=env, start_new_session=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                deadline = time.monotonic()+10
                while pub.get_subscription_count()==0 or sub.get_publisher_count()==0:
                    if process.poll() is not None: self.fail('bridge exited during startup')
                    if time.monotonic()>deadline: self.fail('ROS discovery timeout')
                    rclpy.spin_once(node, timeout_sec=.05)
                # Graph discovery can precede the best-effort reader's match.
                # As with rosbag --delay, measure only after transport startup.
                ready=time.monotonic()+.5
                while time.monotonic()<ready: rclpy.spin_once(node,timeout_sec=.01)
                points = np.tile(np.array([[3., 1., -1., .5], [4., 2., -.5, .7]], dtype='<f4'),(repeats,1))
                fields = [PointField(name=k, offset=4*i, datatype=7, count=1)
                          for i,k in enumerate(('x','y','z','intensity'))]
                original=points.tobytes();step=16
                if indexed:
                    dtype=np.dtype([('xyzi','<f4',(4,)),('time','<f4'),('ring','<u2'),('raw_point_index','<u4')])
                    arr=np.zeros(len(points),dtype=dtype);arr['xyzi']=points
                    arr['time']=np.linspace(0,100000,len(points));arr['ring']=np.arange(len(points))%64
                    arr['raw_point_index']=np.arange(len(points));original=arr.tobytes();step=26
                    fields += [PointField(name='time',offset=16,datatype=7,count=1),
                               PointField(name='ring',offset=20,datatype=4,count=1),
                               PointField(name='raw_point_index',offset=22,datatype=6,count=1)]
                for i in range(5):
                    msg = PointCloud2(height=1, width=len(points), point_step=step, row_step=step*len(points),
                                     is_dense=True, fields=fields, data=original)
                    msg.header.frame_id='lidar'; msg.header.stamp.sec=100; msg.header.stamp.nanosec=i*100_000_000
                    pub.publish(msg)
                    until=time.monotonic()+.1
                    while time.monotonic()<until: rclpy.spin_once(node,timeout_sec=.005)
                until=time.monotonic()+.5
                while time.monotonic()<until: rclpy.spin_once(node,timeout_sec=.01)
                self.assertEqual(len(received),5,Path(temp+'/input.jsonl').read_text())
                self.assertEqual([m.header.stamp.nanosec for m in received],[i*100_000_000 for i in range(5)])
                for m in received:
                    self.assertEqual(m.header.frame_id,'lidar')
                    data=np.frombuffer(m.data,np.uint8).reshape(len(points),m.point_step)
                    np.testing.assert_array_equal(data[:,:step],np.frombuffer(original,np.uint8).reshape(len(points),step))
                    np.testing.assert_array_equal(data[:,step:step+2].copy().view('<u2').ravel(),np.full(len(points),65535))
                    self.assertEqual(m.point_step,step+8)
                import json
                rows=[json.loads(line) for line in Path(temp+'/input.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),5)
                self.assertTrue(all(not r['semantic'] and r['emitted']>=r['deadline'] for r in rows))
            finally:
                if process is not None: terminate(process)
                node.destroy_node(); rclpy.shutdown()


if __name__=='__main__': unittest.main()
