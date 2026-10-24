"""Run the frozen cached-label frontend smoke; preserves every arm before scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import numpy as np
import yaml

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'src'))
from frontend.input_bridge import add_fields
from kitti_scan import synth


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def annotate(root, source, dest, arm):
    import rosbag2_py
    from rclpy.serialization import serialize_message,deserialize_message
    from sensor_msgs.msg import PointCloud2
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(source),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
    writer=rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(dest),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
    for topic in reader.get_all_topics_and_types():writer.create_topic(topic)
    raw=root/'raw/2011_09_30/2011_09_30_drive_0027_sync'
    starts=np.asarray((raw/'velodyne_points/timestamps_start.txt').read_text().splitlines(),dtype='datetime64[ns]').astype(np.int64)
    ends=np.asarray((raw/'velodyne_points/timestamps_end.txt').read_text().splitlines(),dtype='datetime64[ns]').astype(np.int64)
    lookup={int(t):i for i,t in enumerate(starts)}
    seen=[]
    while reader.has_next():
        topic,data,when=reader.read_next()
        if topic=='/velodyne_points':
            msg=deserialize_message(data,PointCloud2)
            stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
            f=lookup[stamp]
            points=np.fromfile(raw/f'velodyne_points/data/{f:010d}.bin',np.float32).reshape(-1,4)
            _,_,order=synth(points[:,:3],sweep_duration=(ends[f]-starts[f])/1e9)
            with np.load(root/f'pred/{arm}/f{f:06d}.npz') as z:
                labels,conf=z['lab'],z['conf']
                if len(labels)!=len(points):raise ValueError('raw prediction index mismatch')
                # Verify geometry in the actual bag before transporting reordered labels.
                bag_xyz=np.ndarray((len(points),3),dtype='<f4',buffer=msg.data,strides=(msg.point_step,4))
                if not np.array_equal(bag_xyz,points[order,:3]):raise ValueError('bag/raw ordering mismatch')
                msg=add_fields(msg,labels[order],conf[order])
            data=serialize_message(msg);seen.append(f)
        writer.write(topic,data,when)
    del writer
    if seen!=list(range(120)):raise ValueError('smoke frame coverage changed')


def terminate(process):
    if process.poll() is not None:return
    os.killpg(process.pid,signal.SIGINT)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait()


def run(args):
    root,ws,out=args.root.resolve(),args.workspace.resolve(),args.out.resolve()
    if out.exists():raise ValueError('refusing to overwrite an experiment directory')
    out.mkdir(parents=True)
    cfg=json.loads((REPO/'frontend/protocol.json').read_text())
    (out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    raw=root/'raw/2011_09_30/2011_09_30_drive_0027_sync'
    subprocess.run([sys.executable,str(REPO/'src/kitti_to_ros2bag.py'),'--drive',str(raw),
                    '--oxts100',str(root/'oxts100'),'--out',str(root/'bag_geometry'),
                    '--end','120','--time-unit','us'],check=True)
    for arm in ['armB0_s1','ZS']:annotate(root,root/'bag_geometry',root/f'bag_{arm}',arm)
    env=dict(os.environ,ROS_DOMAIN_ID='173',RCUTILS_LOGGING_BUFFERED_STREAM='0',OMP_NUM_THREADS='2')
    executable=ws/'install/fast_livo/lib/fast_livo/fastlivo_mapping'
    records=[]
    reference_times=None
    for rep in range(cfg['repetitions']):
        # Rotate arm order across repetitions without selecting parameters/results.
        arms=cfg['arms'][rep:]+cfg['arms'][:rep]
        for arm in arms:
            name=f'{arm}_r{rep+1}'
            config=yaml.safe_load((REPO/'config/kitti_velodyne64.yaml').read_text())
            p=config['/**']['ros__parameters'];p['evo']['seq_name']=name
            p['semantic']=dict(enabled=arm!='geometry',association=arm in ['association_B0','full_B0','full_ZS'],boundaries=arm in ['boundary_B0','full_B0','full_ZS'])
            path=out/f'{name}.yaml';path.write_text(yaml.safe_dump(config))
            bag='geometry' if arm=='geometry' else 'ZS' if arm.endswith('_ZS') else 'armB0_s1'
            with (out/f'{name}.camera.log').open('w') as camera_log, (out/f'{name}.log').open('w') as log:
                camera=subprocess.Popen(['ros2','run','demo_nodes_cpp','parameter_blackboard','--ros-args','--params-file',str(REPO/'config/camera_kitti_cam2.yaml')],env=env,stdout=camera_log,stderr=subprocess.STDOUT,start_new_session=True)
                node=None
                try:
                    time.sleep(1)
                    node=subprocess.Popen([str(executable),'--ros-args','--params-file',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    for _ in range(100):
                        if node.poll() is not None:raise RuntimeError('frontend startup failed: '+name)
                        if 'intrinsic: 707.091200, 707.091200, 601.887300, 183.110400' in (out/f'{name}.log').read_text():break
                        time.sleep(.1)
                    else:raise RuntimeError('camera intrinsics not confirmed')
                    with (out/f'{name}.bag.log').open('w') as bag_log:
                        subprocess.run(['ros2','bag','play',str(root/f'bag_{bag}'),'--rate','0.5','--delay','3'],env=env,stdout=bag_log,stderr=subprocess.STDOUT,check=True,timeout=90)
                    time.sleep(3)
                    if node.poll() is not None:raise RuntimeError('frontend died during replay: '+name)
                finally:
                    if node is not None:terminate(node)
                    terminate(camera)
            tum=ws/f'src/fast_livo/Log/result/{name}.txt'
            e=np.loadtxt(tum)
            if e.ndim!=2 or len(e)<100 or not np.isfinite(e).all():raise RuntimeError('invalid/short trajectory: '+name)
            if reference_times is None:reference_times=e[:,0]
            if not np.array_equal(reference_times,e[:,0]):raise RuntimeError('pose timestamp coverage differs: '+name)
            saved=out/f'{name}.tum';saved.write_bytes(tum.read_bytes())
            logtext=(out/f'{name}.log').read_text()
            rows=[tuple(map(int,m)) for m in re.findall(r'\[ Semantic \] frame=(\d+) boundaries=(\d+) constraints=(\d+) historical_points=(\d+)',logtext)]
            record=dict(arm=arm,repetition=rep+1,poses=len(e),trajectory_sha256=sha(saved),
                        semantic_updates=len(rows),updates_with_constraints=sum(r[2]>0 for r in rows),
                        total_constraints=sum(r[2] for r in rows),max_history_points=max([r[3] for r in rows] or [0]))
            records.append(record)
            (out/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
            print('FRONTEND_RUN',json.dumps(record),flush=True)
    (out/'frozen.json').write_text(json.dumps({p.name:sha(p) for p in out.glob('*.tum')},indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['root','workspace','out']:ap.add_argument('--'+name,type=Path,required=True)
    run(ap.parse_args())
