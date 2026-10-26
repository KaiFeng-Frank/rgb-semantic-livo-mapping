"""Small instance-only cache, selected by collection log before viewing outputs.

Raw scans/images stay at the dataset location. Inputs and evaluation targets are
stored separately. GT is read only after feature extraction and association.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from tools.semantic_memory.nuscenes_pilot_data import ROOT, DATA, records, transform, digest, LABEL_MAP, build_index
from tools.semantic_memory.pilot_metrics import confusion
from .geometry import Camera, associate
from .semantics import InstanceSemantics, instance_groups
from .kalman import inputs


def gpu_admission():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],
                          capture_output=True,text=True,check=True)
    foreign=[line for line in result.stdout.splitlines() if int(line.split(',')[0]) != os.getpid()]
    if foreign:
        raise RuntimeError('GPU occupied; no competing experiment launched: '+str(foreign))
    return dict(time=time.time(),pid=os.getpid(),compute_processes=result.stdout.splitlines())


def prepare(out):
    old=json.loads(Path('/home/frank/semantic_livo_runtime/memory-real-pilot-20261025/index.json').read_text())
    diag=json.loads(Path('/home/frank/semantic_livo_runtime/consistency-20261026-index/index.json').read_text())
    excluded=set(old['sequences'])|set(diag['sequences'])
    index=build_index(out,8,2,4,selection_salt='instance-kalman-20261027',excluded_scenes=excluded)
    index['parts']['diagnostic']=list(diag['sequences'])
    index['sequences'].update(diag['sequences'])
    index['previously_inspected_scenes_excluded_from_new_parts']=sorted(excluded)
    index['scope']='Pilot; official train/dev labels supervise heads. New log-disjoint official validation scenes are sealed test. Old six scenes diagnostic only.'
    (out/'index.json').write_text(json.dumps(index,separators=(',',':'))+'\n')


def extract(index_path,out,weights,scene_limit=0):
    admission=gpu_admission()
    import torch
    import cv2
    from .detector import Detector
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from ptv3_worker import Segmenter
    from instance_features import pool_features
    torch.set_num_threads(1); torch.manual_seed(20261027)
    index=json.loads(index_path.read_text())
    out.mkdir(parents=True,exist_ok=True)
    source={}
    for part in ('train','val'):
        for r in json.loads((ROOT/f'runs/s0_formal/{part}_manifest.json').read_text())['samples']:
            source[r['sample_data_token']]=r
    tokens={source[r['sample_data_token']]['camera_token'] for rows in index['sequences'].values() for r in rows}
    meta=DATA/'v1.0-trainval'
    cams={r['token']:r for r in records(meta/'sample_data.json') if r['token'] in tokens}
    pose_tokens={r['ego_pose_token'] for r in cams.values()}
    poses={r['token']:r for r in records(meta/'ego_pose.json') if r['token'] in pose_tokens}
    calibrations={r['token']:r for r in records(meta/'calibrated_sensor.json')}
    model=Segmenter(half=True,shuffle=False,intensity_scale=1.)
    protocol=dict(index_sha256=digest(index_path),checkpoint_sha256=digest(os.environ['PTV3_CHECKPOINT']),
        model_sha256=model.memory_model_sha256(),parts=index['parts'],admission=admission,
        supervision=index['supervision'],feature_dim=128,classes=16,
        frozen_backbones=True,visual_scores='raw independent COCO80 sigmoid scores, no forced renormalization',
        input='raw nuScenes front camera + LiDAR, supplied ego poses, 2Hz keyframes',
        sources={str(p.relative_to(Path(__file__).resolve().parents[2])):digest(p) for p in
            [Path(__file__),Path(__file__).with_name('kalman.py'),Path(__file__).with_name('semantics.py'),
             Path(__file__).with_name('evidence_predictor.py'),Path(__file__).with_name('geometry.py')]})
    pp=out/'protocol.json'
    if pp.exists():
        prior=json.loads(pp.read_text())
        for k in ('index_sha256','checkpoint_sha256','model_sha256','sources'):
            if prior[k] != protocol[k]: raise ValueError('Cannot resume incompatible cache: '+k)
    else: pp.write_text(json.dumps(protocol,indent=2)+'\n')
    for scene_i,(scene,rows) in enumerate(index['sequences'].items()):
        if scene_limit and scene_i>=scene_limit: break
        target=out/(scene+'.json.gz')
        if target.exists(): continue
        gpu_admission()
        detector=Detector(weights)
        core=InstanceSemantics(); single=InstanceSemantics(temporal=False)
        observations=[]; frames=[]
        for i,row in enumerate(rows):
            if i%10==0: gpu_admission()
            src=source[row['sample_data_token']]
            image=cv2.imread(str(DATA/src['image_path']))
            if image is None or digest(DATA/src['image_path'])!=src['image_sha256'] or digest(DATA/row['scan_path'])!=row['scan_sha256']:
                raise ValueError('Input missing or changed')
            cal=calibrations[cams[src['camera_token']]['calibrated_sensor_token']]
            wc=transform(poses[cams[src['camera_token']]['ego_pose_token']])@transform(cal)
            pose=np.asarray(row['transform'])
            camera=Camera(image.shape[1],image.shape[0],cal['camera_intrinsic'],np.zeros(4),np.linalg.inv(wc)@pose)
            scan=np.fromfile(DATA/row['scan_path'],dtype='<f4').reshape(-1,5)[:,:4].copy();scan[:,3]/=255.
            frame,visual_timing=detector.step(image,int(src['camera_timestamp_us']*1000))
            associated=associate(scan[:,:3],camera,frame)
            groups,tracks=instance_groups(scan[:,:3],associated)
            start=time.perf_counter()
            raw=model.segment(scan,return_features=True)
            with torch.inference_mode():
                voxel_prob=raw['logits'].float().softmax(-1)
                probabilities=voxel_prob[raw['inverse']].cpu().numpy()
                pooled=pool_features(raw['features'],raw['inverse'],groups,len(tracks)).cpu().numpy()
            inference_ms=(time.perf_counter()-start)*1000
            features=dict(zip(tracks,pooled))
            stamp=int(row['timestamp_us']*1000)
            baseline=core.step(scan[:,:3],probabilities,associated,stamp_ns=stamp,pose=pose,collect_observations=True)
            current=single.step(scan[:,:3],probabilities,associated,stamp_ns=stamp,pose=pose)
            # GT starts here, after all network and geometry-only processing.
            gt=LABEL_MAP[np.fromfile(DATA/row['label_path'],np.uint8)]
            raw_labels=probabilities.argmax(1)
            frame_record=dict(frame=i,stamp_ns=stamp,points=len(scan),raw_confusion=confusion(gt,raw_labels,16).tolist(),
                guarded_confusion=confusion(gt,baseline['labels'],16).tolist(),
                single_confusion=confusion(gt,current['labels'],16).tolist(),
                visual_ms=visual_timing['elapsed_ms'],ptv3_feature_ms=inference_ms,
                rescued_guarded=int(((raw_labels!=gt)&(baseline['labels']==gt)&(gt>=0)).sum()),
                harmed_guarded=int(((raw_labels==gt)&(baseline['labels']!=gt)&(gt>=0)).sum()),
                rescued_single=int(((raw_labels!=gt)&(current['labels']==gt)&(gt>=0)).sum()),
                harmed_single=int(((raw_labels==gt)&(current['labels']!=gt)&(gt>=0)).sum()))
            for o in baseline.pop('observations'):
                at=o['indices'];hist=np.bincount(gt[at][gt[at]>=0],minlength=16)
                # Inputs contain no GT identities, boxes, class labels or future observations.
                observations.append(dict(frame=i,track=o['track_id'],epoch=o['epoch'],stamp=stamp,
                    identity=o['identity'],x=inputs(o,features[o['track_id']]).tolist(),
                    raw_mean=o['raw_mean'].tolist(),raw_probabilities=probabilities[at].tolist(),
                    targets=dict(histogram=hist.tolist(),point_labels=gt[at].tolist())))
            frames.append(frame_record)
            if i%10==0: print(json.dumps(dict(scene=scene,frame=i,observations=len(observations))),flush=True)
        with gzip.open(target,'wt') as stream:
            json.dump(dict(scene=scene,frames=frames,observations=observations),stream,separators=(',',':'))
        print(json.dumps(dict(scene_done=scene,observations=len(observations),bytes=target.stat().st_size)),flush=True)
        del detector,raw
    print(json.dumps(dict(stage='extract_complete',scenes=len(list(out.glob('scene-*.json.gz'))))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','extract'))
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--index',type=Path)
    p.add_argument('--weights',type=Path)
    p.add_argument('--scene-limit',type=int,default=0)
    a=p.parse_args()
    if a.action=='prepare': prepare(a.out)
    else: extract(a.index,a.out,a.weights,a.scene_limit)
