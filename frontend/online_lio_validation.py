"""Bounded 1x LiDAR/IMU/image replay with live inference and actual FAST-LIVO2.

Raw inputs and the replay bag exist only in a task-owned tmpfs directory. Stored
evidence contains hashes, timings, configurations, logs and trajectories.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import numpy as np
import yaml

from frontend.online_admission import MemoryRangeReader, REPO, statistics
from frontend.restore_smoke import save_member, client, BASE
from frontend.replay_smoke import terminate

FIRST, END = 730, 850
DRIVE = '2011_09_30_drive_0027'


def recover(root):
    records=[];raw=root/'raw/2011_09_30'/f'{DRIVE}_sync'
    response=client().get(BASE+'raw_data/2011_09_30_calib.zip',timeout=60);response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for name in z.namelist():
            if name.endswith('.txt'):save_member(z,name,root/'raw'/name,records)
    def shard(worker):
        r=MemoryRangeReader(BASE+f'raw_data/{DRIVE}/{DRIVE}_sync.zip');rows=[]
        with zipfile.ZipFile(r) as z:
            names=[f'{folder}/data/{f:010d}.{ext}' for f in range(FIRST+worker,END,4)
                   for folder,ext in [('velodyne_points','bin'),('image_02','png')]]
            if worker==0:names+=['velodyne_points/timestamps_start.txt','velodyne_points/timestamps_end.txt','image_02/timestamps.txt']
            for name in names:save_member(z,f'2011_09_30/{DRIVE}_sync/{name}',raw/name,rows)
        return rows
    with ThreadPoolExecutor(max_workers=4) as pool:
        for rows in pool.map(shard,range(4)):records+=rows
    print('TMPFS_RAW_READY',END-FIRST,flush=True)
    times={k:np.asarray((raw/f'velodyne_points/timestamps_{k}.txt').read_text().splitlines(),dtype='datetime64[ns]').astype(np.int64)
           for k in ['start','end']}
    lo,hi=times['start'][FIRST]-200_000_000,times['end'][END-1]+200_000_000
    r=MemoryRangeReader(BASE+f'raw_data/{DRIVE}/{DRIVE}_extract.zip')
    with zipfile.ZipFile(r) as z:
        prefix=f'2011_09_30/{DRIVE}_extract/oxts/'
        lines=z.read(prefix+'timestamps.txt').decode().splitlines()
        stamps=np.asarray(lines,dtype='datetime64[ns]').astype(np.int64)
        ids=np.flatnonzero((stamps>=lo)&(stamps<=hi))
        imu=root/'oxts100';(imu/'data').mkdir(parents=True)
        (imu/'timestamps.txt').write_text('\n'.join(lines[i] for i in ids)+'\n')
        for j,i in enumerate(ids):save_member(z,prefix+f'data/{i:010d}.txt',imu/f'data/{j:010d}.txt',records)
    print('TMPFS_IMU_READY',len(ids),flush=True)
    bag=root/'bag'
    subprocess.run([sys.executable,str(REPO/'src/kitti_to_ros2bag.py'),'--drive',str(raw),'--oxts100',str(imu),
                    '--out',str(bag),'--start',str(FIRST),'--end',str(END),'--time-unit','us','--point-index-field'],check=True)
    return bag,times,records


def scan_metadata(bag):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
    expected={};warm=[]
    while reader.has_next():
        topic,data,when=reader.read_next()
        if topic!='/velodyne_points':continue
        msg=deserialize_message(data,PointCloud2);stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
        expected[stamp]=dict(points=msg.width,step=msg.point_step,sha256=hashlib.sha256(bytes(msg.data)).hexdigest())
        if len(warm)<6:warm.append(msg)
    if len(expected)!=END-FIRST:raise ValueError('incomplete source bag')
    return expected,warm


def pose_delivery_latency(trajectory, odometry, raw_arrivals, end_stamps):
    """Pair by publication order only after checking all published positions.

Upstream Odometry stamps use wall time; TUM rows carry sensor time. Validate
one-to-one count/position correspondence before associating arrival times.
"""
    if len(trajectory)!=len(odometry) or len(trajectory)==0:return None
    if not np.allclose(trajectory[:,1:4],np.array([r['xyz'] for r in odometry]),atol=2e-6,rtol=0):return None
    indices=np.searchsorted(end_stamps/1e9,trajectory[:,0]-1e-6)
    if np.any(indices>=len(raw_arrivals)):return None
    latencies=np.array([o['received']-raw_arrivals[int(i)][1] for o,i in zip(odometry,indices)])*1000
    if np.any(latencies < -5):return None
    return latencies


def run_arm(bag,times,expected,warm,workspace,out,mode,rep):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    target=out/f'{mode}_r{rep}';target.mkdir()
    domain=62+rep*2+int(mode=='online');env=dict(os.environ,ROS_DOMAIN_ID=str(domain),RMW_IMPLEMENTATION='rmw_cyclonedds_cpp')
    rclpy.init(domain_id=domain);observer=Node('semantic_online_lio_observer')
    raw_arrivals=[];annotated=[];odometry=[]
    def key(msg):return msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
    def raw_received(msg):
        if key(msg) in expected:raw_arrivals.append((key(msg),time.monotonic()))
    def annotated_received(msg):
        stamp=key(msg)
        if stamp not in expected:return
        spec=expected[stamp]
        original=np.frombuffer(msg.data,np.uint8).reshape(msg.width,msg.point_step)[:,:spec['step']].copy().tobytes()
        annotated.append(dict(stamp_ns=stamp,received=time.monotonic(),point_bytes_unchanged=hashlib.sha256(original).hexdigest()==spec['sha256']))
    def odom_received(msg):
        p=msg.pose.pose.position
        odometry.append(dict(received=time.monotonic(),xyz=[p.x,p.y,p.z]))
    pub=observer.create_publisher(PointCloud2,'/online/raw',10)
    subscriptions=[observer.create_subscription(PointCloud2,'/online/raw',raw_received,20),
                   observer.create_subscription(PointCloud2,'/online/annotated',annotated_received,20),
                   observer.create_subscription(Odometry,'/aft_mapped_to_init',odom_received,100)]
    processes=[];handles=[]
    def start(command,name):
        handle=(target/name).open('w');handles.append(handle)
        p=subprocess.Popen(command,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(p);return p
    def spin(seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:rclpy.spin_once(observer,timeout_sec=.002)
    name=f'online_verify_{mode}_r{rep}'
    try:
        bridge=start([sys.executable,'-m','frontend.input_bridge','--mode',mode,'--fixed-delay','--wait-ms','80',
                      '--input-reliability','reliable','--input','/online/raw','--output','/online/annotated',
                      '--log',str(target/'bridge.jsonl')],'bridge.process.log')
        deadline=time.monotonic()+300
        while not (target/'bridge.jsonl').exists() or pub.get_subscription_count()<2:
            if bridge.poll() is not None:raise RuntimeError('bridge startup failed')
            if time.monotonic()>deadline:raise RuntimeError('bridge discovery timeout')
            spin(.05)
        spin(1)
        # Explicit pre-roll before the estimator starts. Shift only warmup stamps;
        # measured bag stamps and every original point byte remain unchanged.
        warm_stamps=[]
        for original in warm:
            msg=copy.deepcopy(original);msg.header.stamp.sec-=120;warm_stamps.append(key(msg))
            pub.publish(msg);spin(3)
        warm_rows=[json.loads(line) for line in (target/'bridge.jsonl').read_text().splitlines()]
        if mode=='online' and (len(warm_rows)<6 or not all(r['semantic'] for r in warm_rows[-3:])):
            raise RuntimeError('fixed six-scan pre-roll did not warm online inference')
        config=yaml.safe_load((REPO/'config/kitti_velodyne64.yaml').read_text());p=config['/**']['ros__parameters']
        p['common']['lid_topic']='/online/annotated';p['evo']['seq_name']=name
        p['semantic']=yaml.safe_load((REPO/'frontend/enhanced.yaml').read_text())['/**']['ros__parameters']['semantic']
        if mode=='geometry':p['semantic']={k:False for k in p['semantic']}
        path=target/'config.yaml';path.write_text(yaml.safe_dump(config))
        start([sys.executable,str(REPO/'frontend/parameter_blackboard.py'),'--ros-args','--params-file',str(REPO/'config/camera_kitti_cam2.yaml')],'camera.log')
        spin(1)
        estimator=start([str(workspace/'install/fast_livo/lib/fast_livo/fastlivo_mapping'),'--ros-args','--params-file',str(path)],'estimator.log')
        deadline=time.monotonic()+20
        while 'intrinsic: 707.091200, 707.091200, 601.887300, 183.110400' not in (target/'estimator.log').read_text():
            if estimator.poll() is not None:raise RuntimeError('estimator startup failed')
            if time.monotonic()>deadline:raise RuntimeError('camera startup timeout')
            spin(.05)
        player=start(['ros2','bag','play',str(bag),'--rate','1','--delay','3','--remap','/velodyne_points:=/online/raw'],'bag.log')
        deadline=time.monotonic()+60
        while player.poll() is None:
            if time.monotonic()>deadline:raise RuntimeError('1x replay timeout')
            if estimator.poll() is not None:raise RuntimeError('estimator died')
            spin(.02)
        if player.returncode:raise RuntimeError('bag replay failed')
        spin(3)
        if estimator.poll() is not None:raise RuntimeError('estimator exited before drain')
    finally:
        for process in reversed(processes):terminate(process)
        for handle in handles:handle.close()
        observer.destroy_node();rclpy.shutdown()
    tum=workspace/f'src/fast_livo/Log/result/{name}.txt';saved=target/'trajectory.tum';saved.write_bytes(tum.read_bytes())
    trajectory=np.loadtxt(saved)
    if trajectory.ndim!=2 or not np.isfinite(trajectory).all():raise ValueError('invalid trajectory')
    rows=[json.loads(line) for line in (target/'bridge.jsonl').read_text().splitlines()]
    measured=[r for r in rows if r['stamp_ns'] in expected]
    selected=set(list(expected)[20:]);steady=[r for r in measured if r['stamp_ns'] in selected]
    coverage=([s for s,_ in raw_arrivals]==list(expected) and [r['stamp_ns'] for r in annotated]==list(expected)
              and [r['stamp_ns'] for r in measured]==list(expected))
    latency=pose_delivery_latency(trajectory,odometry,raw_arrivals,times['end'][FIRST:END]) if coverage else None
    # Discard only the predeclared first twenty sensor scans for steady-state timing.
    keep=trajectory[:,0]>=times['end'][FIRST+19]/1e9
    delay=latency[keep] if latency is not None else None
    drift=float(np.median(delay[-20:])-np.median(delay[:20])) if delay is not None and len(delay)>=40 else None
    log=(target/'estimator.log').read_text()
    categorical=[int(x) for x in re.findall(r'likelihood_rows=(\d+)',log)]
    masks=[int(x) for x in re.findall(r'\[ Semantic visual \] moving_pixels=(\d+)',log)]
    result=dict(mode=mode,repetition=rep,poses=len(trajectory),pose_timestamps=trajectory[:,0].tolist(),
                raw_scans=len(raw_arrivals),annotated_scans=len(annotated),complete_ordered_coverage=coverage,
                point_bytes_unchanged=all(r['point_bytes_unchanged'] for r in annotated),
                steady_scans=len(steady),semantic_fraction=sum(r['semantic'] for r in steady)/100,
                all_semantics_timely=all(r['inference_complete'] is None or r['inference_complete']<=r['deadline'] for r in measured),
                emission_ms=statistics([(r['emitted']-r['arrival'])*1000 for r in steady]),
                pose_delivery_after_required_scan_ms=statistics(delay) if delay is not None else None,
                latency_drift_last20_minus_first20_ms=drift,
                categorical_updates=sum(x>0 for x in categorical),moving_mask_updates=sum(x>0 for x in masks),
                trajectory_sha256=hashlib.sha256(saved.read_bytes()).hexdigest())
    result['passed']=(coverage and result['point_bytes_unchanged'] and len(trajectory)>=110 and
                      result['all_semantics_timely'] and (mode=='geometry' or result['semantic_fraction']>=.95) and
                      drift is not None and drift<=100 and float(np.percentile(delay,95))<=1000)
    (target/'observations.json').write_text(json.dumps(dict(raw_arrivals=raw_arrivals,annotated=annotated,odometry=odometry,warm_stamps=warm_stamps),indent=2)+'\n')
    (target/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ONLINE_LIO_RUN',json.dumps({k:v for k,v in result.items() if k!='pose_timestamps'}),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--bridge-admission',type=Path,required=True)
    a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    paths=['frontend/online_lio_validation.py','frontend/input_bridge.py','frontend/enhanced.yaml',
           'frontend/patches/fastlivo2-semantic.patch','src/kitti_to_ros2bag.py','src/kitti_scan.py',
           'config/kitti_velodyne64.yaml','config/camera_kitti_cam2.yaml']
    paths += [str(p.relative_to(REPO)) for p in sorted((REPO/'frontend/include').glob('*.hpp'))]
    protocol=dict(sequence='07',first_frame=FIRST,end_exclusive=END,development=True,repetitions=3,playback_rate=1.,
                  modes=['geometry','online'],pre_roll='six source scans with stamps shifted 120 seconds before estimator startup, 3 seconds apart; last three must have timely semantics',
                  steady_excludes_first_scans=20,deadline_ms=80,minimum_semantic_fraction=.95,minimum_poses=110,
                  maximum_latency_drift_ms=100,maximum_pose_delivery_p95_ms=1000,
                  coverage='all original scans once in order, unchanged original point bytes; equal finite pose timestamps',
                  timing='observer monotonic receipt; pair Odometry publication order to sensor-time TUM only when counts/positions agree; time from arrival of the first complete scan containing the update time',
                  caveat='short warm-cache 1x replay, includes observer overhead; not a sustained deployment or accuracy result',
                  raw_storage='task-owned tmpfs only, removed at exit',
                  source_sha256={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in paths})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    try:
        marker=json.loads((a.workspace/'.semantic-frontend-source.json').read_text())
        if marker['patch_sha256']!=protocol['source_sha256']['frontend/patches/fastlivo2-semantic.patch']:
            raise RuntimeError('workspace estimator differs from frozen patch')
        for p in (REPO/'frontend/include').glob('*.hpp'):
            if p.read_bytes()!=(a.workspace/'src/fast_livo/include'/p.name).read_bytes():
                raise RuntimeError('workspace semantic header differs: '+p.name)
        executable=a.workspace/'install/fast_livo/lib/fast_livo/fastlivo_mapping'
        (out/'runtime.json').write_text(json.dumps(dict(estimator_marker=marker,
            executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest()),indent=2)+'\n')
        if shutil.disk_usage('/dev/shm').free<2*1024**3:raise RuntimeError('less than 2 GiB free tmpfs')
        with tempfile.TemporaryDirectory(prefix='semantic-online-lio-',dir='/dev/shm') as temporary:
            bag,times,recovery=recover(Path(temporary))
            (out/'input-recovery.json').write_text(json.dumps(recovery,indent=2)+'\n')
            expected,warm=scan_metadata(bag)
            (out/'scan-hashes.json').write_text(json.dumps(expected,indent=2)+'\n')
            deadline=time.monotonic()+600
            while not a.bridge_admission.exists():
                if time.monotonic()>deadline:raise RuntimeError('bridge admission unavailable')
                time.sleep(1)
            if not json.loads(a.bridge_admission.read_text())['admitted']:raise RuntimeError('prerequisite bridge admission failed')
            records=[];reference=None
            for rep in range(1,4):
                for mode in (['geometry','online'] if rep%2 else ['online','geometry']):
                    row=run_arm(bag,times,expected,warm,a.workspace.resolve(),out,mode,rep)
                    if reference is None:reference=row['pose_timestamps']
                    row['paired_pose_timestamps']=row['pose_timestamps']==reference
                    records.append(row);(out/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
            passed=all(r['passed'] and r['paired_pose_timestamps'] for r in records)
            (out/'assessment.json').write_text(json.dumps(dict(admitted=passed,scope=protocol['caveat'],runs=records),indent=2)+'\n')
            print('ONLINE_LIO_ADMISSION',passed,flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc)[:400]),indent=2)+'\n')
        raise


if __name__=='__main__':main()
