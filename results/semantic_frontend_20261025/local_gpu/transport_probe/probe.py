import os, sys, time, subprocess, signal, json
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

root=Path('/home/frank/桌面/rgb-semantic-livo-mapping')
size=int(sys.argv[1]); domain=int(sys.argv[2]); count=10
out=Path('/tmp/semantic-transport-20261025'); out.mkdir(exist_ok=True)
tag=f"{os.environ.get('RMW_IMPLEMENTATION', 'default')}-{domain}-{size}-{time.time_ns()}"
logpath=out/(tag+'.jsonl'); processlog=out/(tag+'.log')
rclpy.init(domain_id=domain); node=Node('transport_probe')
received=[]; pub=node.create_publisher(PointCloud2,'/probe/raw',10)
sub=node.create_subscription(PointCloud2,'/probe/annotated',received.append,10)
process=None
result=dict(middleware=os.environ.get('RMW_IMPLEMENTATION','default'),domain=domain,points=size,published=count,input_bytes=size*26)
try:
    with processlog.open('w') as log:
        process=subprocess.Popen([sys.executable,'-m','frontend.input_bridge','--mode','geometry','--fixed-delay','--wait-ms','80','--input','/probe/raw','--output','/probe/annotated','--log',str(logpath)],cwd=root,env=dict(os.environ,ROS_DOMAIN_ID=str(domain)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        until=time.monotonic()+12
        while not pub.get_subscription_count() or not sub.get_publisher_count():
            if process.poll() is not None: raise RuntimeError('bridge exited')
            if time.monotonic()>until:raise RuntimeError('discovery timeout')
            rclpy.spin_once(node,timeout_sec=.02)
        until=time.monotonic()+1
        while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.01)
        data=np.zeros((size,26),np.uint8).tobytes()
        fields=[PointField(name=k,offset=4*i,datatype=7,count=1) for i,k in enumerate(['x','y','z','intensity','time'])]
        fields += [PointField(name='ring',offset=20,datatype=4,count=1),PointField(name='raw_point_index',offset=22,datatype=6,count=1)]
        for i in range(count):
            msg=PointCloud2(height=1,width=size,point_step=26,row_step=size*26,is_dense=True,fields=fields,data=data)
            msg.header.stamp.sec=100;msg.header.stamp.nanosec=i*100_000_000
            pub.publish(msg)
            until=time.monotonic()+.1
            while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.001)
        until=time.monotonic()+2
        while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.01)
        rows=[json.loads(x) for x in logpath.read_text().splitlines()]
        result.update(bridge_published=len(rows),received=len(received),stamps=[m.header.stamp.nanosec for m in received],complete_ordered_coverage=[m.header.stamp.nanosec for m in received]==[i*100_000_000 for i in range(count)],original_bytes_preserved=(all(np.array_equal(np.frombuffer(m.data,np.uint8).reshape(size,m.point_step)[:,:26],np.frombuffer(data,np.uint8).reshape(size,26)) for m in received) if received else None))
except Exception as e:result['error']=repr(e)
finally:
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGINT)
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    node.destroy_node();rclpy.shutdown()
(out/(tag+'.result.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
