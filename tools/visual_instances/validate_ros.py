"""Real ROS delivery test with explicitly synthetic masks/semantic predictions."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .geometry import InstanceFrame


def run(out):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header, String
    from builtin_interfaces.msg import Time
    from sensor_msgs_py.point_cloud2 import create_cloud, dtype_from_fields
    from tools.loop_closure.semantic_node import make_node
    out.mkdir(parents=True,exist_ok=False)
    frontend=out/'frontend.yaml'
    frontend.write_text('/**:\n  ros__parameters:\n    extrin_calib:\n      extrinsic_R: [1,0,0,0,1,0,0,0,1]\n      extrinsic_T: [0,0,0]\n      Rcl: [1,0,0,0,1,0,0,0,1]\n      Pcl: [0,0,0]\n')
    camera=out/'camera.yaml'
    camera.write_text('/**:\n  ros__parameters:\n    cam_width: 40\n    cam_height: 40\n    cam_fx: 10\n    cam_fy: 10\n    cam_cx: 0\n    cam_cy: 0\n    cam_d0: 0\n    cam_d1: 0\n    cam_d2: 0\n    cam_d3: 0\n')
    args=SimpleNamespace(frontend=frontend,camera=camera,out=out/'node',config=None,predictions_input=True,
        consistency='off',visual_instances=True,visual_max_dt_ms=20.,input_cloud='/backend/input/semantic_cloud',
        map_frame='map',reflectance_divisor=1.,intensity_scale=.2,verify_final=True)
    rclpy.init()
    observer=rclpy.create_node('visual_association_fixture')
    qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE)
    received, infos={},{}
    def observe(m):
        received[m.header.stamp.sec]=np.frombuffer(m.data,dtype=dtype_from_fields(m.fields,point_step=m.point_step)).copy()
    subs=[observer.create_subscription(PointCloud2,'/backend/semantic_instances',observe,qos),
          observer.create_subscription(String,'/backend/semantic_instance_info',lambda m:infos.update({json.loads(m.data)['stamp_ns']//10**9:json.loads(m.data)}),qos)]
    visual_pub=observer.create_publisher(String,'/backend/visual_instances',10)
    graph_pub=observer.create_publisher(String,'/backend/trajectory',QoSProfile(depth=1,
        reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    cloud_pub=observer.create_publisher(PointCloud2,args.input_cloud,10)
    node=make_node(args)
    def until(condition):
        end=time.monotonic()+15
        while not condition():
            if time.monotonic()>end:raise TimeoutError('ROS delivery deadline')
            rclpy.spin_once(observer,timeout_sec=.005)
            rclpy.spin_once(node,timeout_sec=.005)
            if node.counters['failed']:raise RuntimeError(node.counters['error'])
    def visual(sec,session):
        index=np.zeros((40,40),np.uint16);index[5:20,5:20]=1
        frame=InstanceFrame(sec*10**9,session,index,np.array([12]),np.array([6]),np.array([.95]))
        previous=node.visual_counts['received']
        visual_pub.publish(String(data=frame.encode()))
        until(lambda:node.visual_counts['received']>previous)
    def send(sec,revision=0):
        graph_pub.publish(String(data=json.dumps(dict(frame_id='map',stamp_ns=sec*10**9,version=revision,
            times_ns=[i*10**9 for i in range(1,sec+1)],poses=[np.eye(4).tolist() for _ in range(sec)]))))
        fields=[PointField(name=name,offset=i*4,datatype=PointField.UINT16 if name=='label' else PointField.FLOAT32,count=1)
                for i,name in enumerate(('x','y','z','intensity','label','confidence'))]
        cloud_pub.publish(create_cloud(Header(stamp=Time(sec=sec),frame_id='backend_lidar'),fields,
            [(5.,5.,5.,.1,10,.9),(5.05,5.,5.,.1,10,.9),(5.,5.05,5.,.1,10,.9),(10.,10.,10.,.1,10,.9)]))
        until(lambda:sec in received and sec in infos)
    try:
        until(lambda:node.counters['ready'] and visual_pub.get_subscription_count()>0 and cloud_pub.get_subscription_count()>0)
        visual(1,'a'*32);send(1)
        assert received[1]['instance_id'].tolist()==[12,12,12,0]
        assert received[1]['label'].tolist()==[10]*4  # Vision says person; PTv3/raw class remains road.
        visual(3,'a'*32);send(2)
        assert received[2]['instance_id'].tolist()==[0]*4  # Future image cannot leak.
        send(3,revision=1)
        assert received[3]['instance_id'].tolist()==[12,12,12,0]
        visual(4,'b'*32);send(4,revision=1)
        assert infos[1]['session']!=infos[4]['session']
        assert received[4]['instance_id'].tolist()==[12,12,12,0]
        assert all(not v['semantic_update'] for v in infos.values())
        parity=node.core.verify_rebuild()
        assert parity['ok']
        (out/'report.json').write_text(json.dumps(dict(passed=True,scope='synthetic actual-ROS contract, not accuracy',
            ids={k:v['instance_id'].tolist() for k,v in received.items()},info=infos,map_parity=parity),indent=2)+'\n')
    finally:
        node.close();node.destroy_node();observer.destroy_node();rclpy.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
