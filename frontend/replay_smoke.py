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
from frontend.protocol import load, raw_dir


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_point_order(msg, points):
    fields=[f for f in msg.fields if f.name=='raw_point_index']
    if len(fields)!=1 or fields[0].datatype!=6 or fields[0].count!=1:
        raise ValueError('replay requires the original raw_point_index uint32 field')
    if msg.height!=1 or msg.width!=len(points) or msg.is_bigendian:
        raise ValueError('raw scan shape/endianness mismatch')
    field=fields[0]
    if field.offset+4>msg.point_step:raise ValueError('truncated point index field')
    order=np.ndarray((len(points),),dtype='<u4',buffer=msg.data,
                     offset=field.offset,strides=(msg.point_step,)).copy()
    if not np.array_equal(np.sort(order),np.arange(len(points),dtype=np.uint32)):
        raise ValueError('point IDs are not a complete raw-scan permutation')
    xyzi=np.ndarray((len(points),4),dtype='<f4',buffer=msg.data,strides=(msg.point_step,4))
    if not np.array_equal(xyzi,points[order]):
        raise ValueError('carried point IDs do not match original xyz/intensity')
    return order


def annotate(root, source, dest, arm, cfg=None):
    cfg=cfg or load()
    import rosbag2_py
    from rclpy.serialization import serialize_message,deserialize_message
    from sensor_msgs.msg import PointCloud2
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(source),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
    writer=rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(dest),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
    for topic in reader.get_all_topics_and_types():writer.create_topic(topic)
    raw=raw_dir(root,cfg)
    starts=np.asarray((raw/'velodyne_points/timestamps_start.txt').read_text().splitlines(),dtype='datetime64[ns]').astype(np.int64)
    ends=np.asarray((raw/'velodyne_points/timestamps_end.txt').read_text().splitlines(),dtype='datetime64[ns]').astype(np.int64)
    lookup={int(t):i for i,t in enumerate(starts)}
    seen=[];audit=[]
    while reader.has_next():
        topic,data,when=reader.read_next()
        if topic=='/velodyne_points':
            msg=deserialize_message(data,PointCloud2)
            stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
            f=lookup[stamp]
            points=np.fromfile(raw/f'velodyne_points/data/{f:010d}.bin',np.float32).reshape(-1,4)
            order=raw_point_order(msg,points)
            _,_,recomputed=synth(points[:,:3],sweep_duration=(ends[f]-starts[f])/1e9)
            audit.append(dict(frame=f,points=len(points),
                              order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                              recomputed_disagreements=int(np.count_nonzero(order!=recomputed))))
            with np.load(root/f'pred/{arm}/f{f:06d}.npz') as z:
                labels,conf=z['lab'],z['conf']
                if len(labels)!=len(points):raise ValueError('raw prediction index mismatch')
                msg=add_fields(msg,labels[order],conf[order])
            data=serialize_message(msg);seen.append(f)
        writer.write(topic,data,when)
    del writer
    part=cfg['short_sequence']
    if seen!=list(range(part['first_frame'],part['end_exclusive'])):raise ValueError('smoke frame coverage changed')
    (root/f'annotation-{arm}.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('POINT_IDENTITY_VERIFIED',arm,len(seen),'frames; recomputed disagreements',
          sum(r['recomputed_disagreements'] for r in audit),flush=True)


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
    cfg=load(args.protocol)
    if args.diagnostic:
        if cfg['short_sequence']['sequence']!='07':raise ValueError('diagnostic only uses development sequence 07')
        # The first scored window is now development data. This rerun checks
        # activation only; score_smoke explicitly rejects this protocol.
        cfg.update(purpose='Boundary source diagnostic only; no accuracy scoring',
                   arms=['geometry','full_B0','full_ZS'],repetitions=1,
                   diagnostic=True,reference_run=40125959560,
                   change='extract from deskewed pre-voxel cloud; same gates and information caps')
    (out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    raw=raw_dir(root,cfg)
    part=cfg['short_sequence']
    subprocess.run([sys.executable,str(REPO/'src/kitti_to_ros2bag.py'),'--drive',str(raw),
                    '--oxts100',str(root/'oxts100'),'--out',str(root/'bag_geometry'),
                    '--start',str(part['first_frame']),'--end',str(part['end_exclusive']),
                    '--time-unit','us','--point-index-field'],check=True)
    for arm in ['armB0_s1','ZS']:annotate(root,root/'bag_geometry',root/f'bag_{arm}',arm,cfg)
    for path in root.glob('annotation-*.json'):(out/path.name).write_bytes(path.read_bytes())
    env=dict(os.environ,ROS_DOMAIN_ID='43',RCUTILS_LOGGING_BUFFERED_STREAM='0',OMP_NUM_THREADS='2',
             RMW_IMPLEMENTATION=cfg.get('middleware',os.environ.get('RMW_IMPLEMENTATION','rmw_fastrtps_cpp')))
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
            if 'arm_settings' in cfg:
                p['semantic']=cfg['arm_settings'][arm]
            path=out/f'{name}.yaml';path.write_text(yaml.safe_dump(config))
            bag='geometry' if arm=='geometry' else 'ZS' if arm.endswith('_ZS') else 'armB0_s1'
            with (out/f'{name}.camera.log').open('w') as camera_log, (out/f'{name}.log').open('w') as log:
                camera=subprocess.Popen([sys.executable,str(REPO/'frontend/parameter_blackboard.py'),'--ros-args','--params-file',str(REPO/'config/camera_kitti_cam2.yaml')],env=env,stdout=camera_log,stderr=subprocess.STDOUT,start_new_session=True)
                node=None
                try:
                    time.sleep(1)
                    node=subprocess.Popen([str(executable),'--ros-args','--params-file',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    for _ in range(150):
                        if node.poll() is not None:raise RuntimeError('frontend startup failed: '+name)
                        if 'intrinsic: 707.091200, 707.091200, 601.887300, 183.110400' in (out/f'{name}.log').read_text():break
                        time.sleep(.1)
                    else:raise RuntimeError('camera intrinsics not confirmed')
                    with (out/f'{name}.bag.log').open('w') as bag_log:
                        subprocess.run(['ros2','bag','play',str(root/f'bag_{bag}'),'--rate',str(cfg.get('playback_rate',.5)),'--delay','3'],env=env,stdout=bag_log,stderr=subprocess.STDOUT,check=True,timeout=120)
                    time.sleep(3)
                    if node.poll() is not None:raise RuntimeError('frontend died during replay: '+name)
                finally:
                    if node is not None:terminate(node)
                    terminate(camera)
            tum=ws/f'src/fast_livo/Log/result/{name}.txt'
            e=np.loadtxt(tum)
            if e.ndim!=2 or len(e)<cfg.get('minimum_poses',100) or not np.isfinite(e).all():raise RuntimeError('invalid/short trajectory: '+name)
            if reference_times is None:reference_times=e[:,0]
            if not np.array_equal(reference_times,e[:,0]):raise RuntimeError('pose timestamp coverage differs: '+name)
            saved=out/f'{name}.tum';saved.write_bytes(tum.read_bytes())
            logtext=(out/f'{name}.log').read_text()
            rows=[tuple(map(int,m)) for m in re.findall(r'\[ Semantic \] frame=(\d+) boundaries=(\d+) constraints=(\d+) historical_points=(\d+)',logtext)]
            record=dict(arm=arm,repetition=rep+1,poses=len(e),trajectory_sha256=sha(saved),
                        semantic_updates=len(rows),updates_with_constraints=sum(r[2]>0 for r in rows),
                        total_constraints=sum(r[2] for r in rows),max_history_points=max([r[3] for r in rows] or [0]))
            enhanced=[tuple(map(int,m)) for m in re.findall(r'likelihood_rows=(\d+) moving=(\d+) pending=(\d+) invalidated=(\d+)',logtext)]
            visual=[tuple(map(int,m)) for m in re.findall(r'\[ Semantic visual \] moving_pixels=(\d+) uncertain_pixels=(\d+) accepted_rows=(\d+)',logtext)]
            record.update(likelihood_updates=sum(r[0]>0 for r in enhanced),
                          moving_updates=sum(r[1]>0 for r in enhanced),
                          moving_points=sum(r[1] for r in enhanced),
                          uncertain_points=sum(r[2] for r in enhanced),
                          invalidated_nodes=sum(r[3] for r in enhanced),
                          visual_mask_updates=sum(r[0]>0 for r in visual),
                          visual_uncertain_updates=sum(r[1]>0 for r in visual),
                          visual_accepted_rows=sum(r[2] for r in visual))
            records.append(record)
            (out/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
            print('FRONTEND_RUN',json.dumps(record),flush=True)
    (out/'frozen.json').write_text(json.dumps({p.name:sha(p) for p in out.glob('*.tum')},indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['root','workspace','out']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--diagnostic',action='store_true')
    ap.add_argument('--protocol',type=Path)
    run(ap.parse_args())
