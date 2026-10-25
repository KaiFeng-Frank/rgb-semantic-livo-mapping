"""Real-scan GPU/ROS admission; original KITTI samples remain in RAM only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'tools'));sys.path.insert(0,str(REPO/'src'))
from restore_driving_comparison import RemoteZip

WARM=[0,20,40,60,80,100]
MEASURE=list(range(120,140))


class MemoryRangeReader(RemoteZip):
    """Retry interrupted response bodies without retaining raw files on disk."""
    def read(self,n=-1):
        remaining=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if remaining>32*1024**2:raise ValueError('unexpected large archive read')
        parts=[]
        while remaining:
            size=min(1024**2,remaining)
            for attempt in range(5):
                try:
                    part=super().read(size)
                    break
                except Exception:
                    if attempt==4:raise
                    time.sleep(.5)
            if len(part)!=size:raise ValueError('incomplete archive range')
            parts.append(part);remaining-=len(part)
        return b''.join(parts)


def recover():
    url='https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0027/2011_09_30_drive_0027_sync.zip'
    prefix='2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/'
    frames=WARM+MEASURE
    def shard(i):
        reader=MemoryRangeReader(url);reader.session.headers['Connection']='close'
        rows={};meta={}
        with zipfile.ZipFile(reader) as z:
            if i==0:
                for name in ['timestamps_start.txt','timestamps_end.txt']:
                    meta[name]=np.asarray(z.read(prefix+name).decode().splitlines(),dtype='datetime64[ns]').astype(np.int64)
            for f in frames[i::4]:
                raw=z.read(prefix+f'data/{f:010d}.bin')
                rows[f]=(np.frombuffer(raw,dtype='<f4').reshape(-1,4).copy(),hashlib.sha256(raw).hexdigest())
                print('RAM_SCAN',f,flush=True)
        return rows,meta
    scans={};meta={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for a,b in pool.map(shard,range(4)):scans.update(a);meta.update(b)
    return scans,meta


def statistics(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),max_ms=float(a.max())) if len(a) else None


def direct(scans,out):
    from ptv3_client import PTv3Client
    client=PTv3Client(half=1,shuffle=0,stderr_path=str(out/'worker.log'))
    rows=[]
    try:
        if client.num_classes!=16:raise ValueError('wrong native taxonomy')
        for f in WARM:client.segment(scans[f][0])
        for f in MEASURE:
            t=time.monotonic();labels,conf=client.segment(scans[f][0]);elapsed=(time.monotonic()-t)*1000
            if len(labels)!=len(scans[f][0]) or not np.all((labels>=0)&(labels<16)) or not np.all(np.isfinite(conf)&(conf>=0)&(conf<=1)):raise ValueError('invalid native predictions')
            rows.append(dict(frame=f,points=len(labels),client_ms=elapsed,worker_ms=client.last_infer_ms,voxels=client.last_voxels,
                             class_counts=np.bincount(labels,minlength=16).tolist()))
            print('INFERENCE',json.dumps(rows[-1]),flush=True)
    finally:client.close()
    (out/'inference.json').write_text(json.dumps(rows,indent=2)+'\n')
    return dict(client=statistics([r['client_ms'] for r in rows]),worker=statistics([r['worker_ms'] for r in rows]))


def bridge(scans,meta,out,mode):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from kitti_to_ros2bag import INDEXED_PC_DTYPE,make_pointcloud2
    from kitti_scan import synth
    from frontend.replay_smoke import terminate
    domain=180+2*(os.getpid()%20)+int(mode=='online')
    rclpy.init(domain_id=domain);node=Node('semantic_online_admission')
    received=[]
    pub=node.create_publisher(PointCloud2,'/admission/raw',10)
    sub=node.create_subscription(PointCloud2,'/admission/annotated',received.append,10)
    process=None;expected={};count=len(WARM)+len(MEASURE)
    try:
        with (out/f'{mode}.process.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-m','frontend.input_bridge','--mode',mode,'--fixed-delay','--wait-ms','80',
                '--input','/admission/raw','--output','/admission/annotated','--log',str(out/f'{mode}.jsonl')],
                cwd=REPO,env=dict(os.environ,ROS_DOMAIN_ID=str(domain)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            until=time.monotonic()+300
            while not (out/f'{mode}.jsonl').is_file() or not pub.get_subscription_count() or not sub.get_publisher_count():
                if process.poll() is not None:raise RuntimeError('bridge startup failed: '+mode)
                if time.monotonic()>until:raise RuntimeError('bridge discovery timeout: '+mode)
                rclpy.spin_once(node,timeout_sec=.05)
            until=time.monotonic()+1.
            while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.01)
            for f in WARM+MEASURE:
                pts=scans[f][0];start=int(meta['timestamps_start.txt'][f]);end=int(meta['timestamps_end.txt'][f])
                ring,times,order=synth(pts,sweep_duration=(end-start)/1e9)
                arr=np.empty(len(pts),INDEXED_PC_DTYPE)
                for i,k in enumerate(['x','y','z','intensity']):arr[k]=pts[order,i]
                arr['time']=times*1e6;arr['ring']=ring;arr['raw_point_index']=order
                msg=make_pointcloud2(start,arr,'velodyne');expected[start]=msg
                pub.publish(msg)
                until=time.monotonic()+(.5 if f in WARM else (end-start)/1e9)
                while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.001)
            until=time.monotonic()+5.
            while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.01)
    finally:
        if process is not None:terminate(process)
        node.destroy_node();rclpy.shutdown()
    stamps=[]
    for msg in received:
        stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec;stamps.append(stamp)
        src=expected[stamp]
        a=np.frombuffer(msg.data,np.uint8).reshape(msg.width,msg.point_step)[:,:src.point_step]
        b=np.frombuffer(src.data,np.uint8).reshape(src.width,src.point_step)
        if not np.array_equal(a,b):raise ValueError('original point bytes changed')
    coverage=stamps==list(expected)
    rows=[json.loads(line) for line in (out/f'{mode}.jsonl').read_text().splitlines()]
    selected={int(meta['timestamps_start.txt'][f]) for f in MEASURE}
    measured=[r for r in rows if r['stamp_ns'] in selected]
    semantic=sum(r['semantic'] for r in measured)
    return dict(mode=mode,ros_domain=domain,scans=len(MEASURE),received_measurement_scans=len(measured),
                published_total=count,received_total=len(stamps),complete_ordered_coverage=coverage,
                expected_stamp_ns=list(expected),received_stamp_ns=stamps,
                semantic_scans=semantic,semantic_fraction=semantic/len(MEASURE),
                emission=statistics([(r['emitted']-r['arrival'])*1000 for r in measured]),
                late_discarded=max((r['late_discarded'] for r in rows),default=0),point_bytes_unchanged=True,
                all_timely_semantics=all(r['inference_complete'] is None or r['inference_complete']<=r['deadline'] for r in rows))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--runtime-ready',type=Path,help='wait for a verified runtime marker while raw samples stay in RAM')
    a=ap.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    protocol=dict(scope='local GPU and ROS bridge admission, not full LIO accuracy',warm_frames=WARM,measure_frames=MEASURE,
                  deadline_ms=80,required_semantic_fraction=.95,raw_data_persisted=False,
                  source_sha256={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in
                                 ['frontend/online_admission.py','frontend/input_bridge.py','src/ptv3_client.py','src/ptv3_worker.py','src/ptv3_loader_verified.py','src/ptv3_fast.py']})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    try:
        scans,meta=recover()
        (out/'scan-hashes.json').write_text(json.dumps({f:dict(points=len(v[0]),sha256=v[1]) for f,v in scans.items()},indent=2)+'\n')
        if a.runtime_ready:
            until=time.monotonic()+3600
            print('WAITING_FOR_VERIFIED_RUNTIME',flush=True)
            while not a.runtime_ready.is_file():
                if time.monotonic()>until:raise RuntimeError('runtime readiness timeout')
                time.sleep(1)
            (out/'runtime.json').write_bytes(a.runtime_ready.read_bytes())
        inference=direct(scans,out)
        baseline=bridge(scans,meta,out,'geometry')
        (out/'geometry-result.json').write_text(json.dumps(baseline,indent=2)+'\n')
        online=bridge(scans,meta,out,'online')
        result=dict(geometry=baseline,inference=inference,online=online,
                    admitted=baseline['complete_ordered_coverage'] and online['complete_ordered_coverage']
                             and online['semantic_fraction']>=.95 and online['all_timely_semantics'],
                    full_lio_online_validated=False)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc)[:300],admitted=False),indent=2)+'\n')
        raise

if __name__=='__main__':main()
