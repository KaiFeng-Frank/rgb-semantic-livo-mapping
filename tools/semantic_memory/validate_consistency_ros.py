"""Actual ROS input/output wiring fixture with explicit synthetic posteriors.

Not an accuracy or timing test. The real-data PTv3 replay is separate. Checks
that enhanced scans arrive at a subscriber, raw map evidence stays raw, and a
received graph revision reanchors short-term evidence before another frame.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def run(out):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header, String
    from builtin_interfaces.msg import Time
    from sensor_msgs_py.point_cloud2 import create_cloud
    from tools.loop_closure.semantic_node import make_node
    from tools.loop_closure.validate_semantic_ros import semantic_rows
    sys.path.insert(0,str(ROOT/'src'))
    import ptv3_client

    class PosteriorFixture:
        num_classes=16
        calls=0
        def __init__(self,**kw):
            assert kw['probability_classes']==16
        def segment_probabilities(self, points):
            self.calls += 1
            p=np.full((2,16),.01/15,np.float32); p[:,10]=.99
            if self.calls==2:
                p[0]=.22/14; p[0,10]=.38; p[0,12]=.4
                p[1]=.05/15; p[1,12]=.95
            if self.calls==3:
                p[:]=.05/15; p[:,12]=.95
            return p.argmax(1).astype(np.uint16),p.max(1),p
        def close(self): pass

    out.mkdir(parents=True,exist_ok=False)
    frontend=out/'identity.yaml'
    frontend.write_text('/**:\n  ros__parameters:\n    extrin_calib:\n      extrinsic_R: [1,0,0,0,1,0,0,0,1]\n      extrinsic_T: [0,0,0]\n')
    args=SimpleNamespace(frontend=frontend,out=out/'node',config=None,predictions_input=False,
        consistency='active',input_cloud='/backend/input/semantic_cloud',map_frame='map',
        reflectance_divisor=1.,intensity_scale=.2,verify_final=True)
    rclpy.init()
    observer=rclpy.create_node('consistency_fixture_observer')
    received={}
    subscription=observer.create_subscription(PointCloud2,'/backend/semantic_scan',
        lambda m:received.update({m.header.stamp.sec:semantic_rows(m)}),
        QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT))
    graph_pub=observer.create_publisher(String,'/backend/trajectory',QoSProfile(depth=1,
        reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    cloud_pub=observer.create_publisher(PointCloud2,args.input_cloud,10)
    node=None
    def until(predicate):
        end=time.monotonic()+15
        while not predicate():
            if time.monotonic()>end: raise TimeoutError('ROS fixture delivery deadline')
            rclpy.spin_once(observer,timeout_sec=.005)
            rclpy.spin_once(node,timeout_sec=.005)
            if node.counters['failed']: raise AssertionError(node.counters['error'])
    def send(frame,revision=0,shift=0.,cloud=True):
        pose=np.eye(4); pose[0,3]=shift
        graph_pub.publish(String(data=json.dumps(dict(frame_id='map',stamp_ns=frame*10**9,
            version=revision,times_ns=[i*10**9 for i in range(1,frame+1)],
            poses=[pose.tolist() for _ in range(frame)]))))
        until(lambda:node.last_graph==(frame*10**9,revision))
        if cloud:
            header=Header(stamp=Time(sec=frame),frame_id='backend_lidar')
            fields=[PointField(name=name,offset=i*4,datatype=7,count=1) for i,name in enumerate(('x','y','z','intensity'))]
            cloud_pub.publish(create_cloud(header,fields,np.array([[5,0,0,.1],[5,1,0,.2]],np.float32)))
            until(lambda:frame in received and node.counters['processed']>=frame)
    try:
        with patch.object(ptv3_client,'PTv3Client',PosteriorFixture):
            node=make_node(args)
            until(lambda:node.counters['ready'] and cloud_pub.get_subscription_count()>0
                  and graph_pub.get_subscription_count()>0)
            send(1)
            send(2)
            assert received[1][1].tolist()==[10,10]
            assert received[2][1].tolist()==[10,12]
            np.testing.assert_array_equal(node.core._frames[2*10**9].packet.labels,[12,12])
            np.testing.assert_allclose(node.core._frames[2*10**9].packet.confidence,[.4,.95],atol=1e-6)
            assert node.consistency.history['probabilities'][0].argmax()==12
            send(2,revision=1,shift=3.,cloud=False)
            assert node.consistency.history['pose'][0,3]==3.
            send(3,revision=1,shift=3.)
            assert received[3][1].tolist()==[12,12]
            assert node.consistency.last_status['candidates']==0
            parity=node.core.verify_rebuild()
            assert parity.get('passed',parity.get('ok',False))
            result=dict(passed=True,synthetic=True,real_ros_transport=True,
                scan_labels={k:v[1].tolist() for k,v in received.items()},
                enhanced_scan_received=True,map_retains_raw=True,history_retains_raw=True,
                graph_revision_reanchors=True,current_evidence_overrides_history=True,parity=parity)
            (out/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
    finally:
        if node:
            node.close(); node.destroy_node()
        observer.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
