"""Locate loss of real instance coverage before attempting more training.

Replay frozen YOLO only, then count GT support after each existing geometric
stage. Labels are used only by the observer; no model or gate receives them.
This is a post-failure development diagnosis, not an accuracy experiment.
"""
import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
from .data import read,save
from tools.visual_instances.kalman_cache import gpu_admission
from .metrics import identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA


def audit(index,cache,weights):
    import cv2
    import torch
    from tools.visual_instances.detector import Detector
    from tools.visual_instances.geometry import Camera,associate
    from tools.visual_instances.semantics import Config,component,THING
    torch.set_num_threads(1);torch.manual_seed(20261029);cv2.setNumThreads(1)
    admission=gpu_admission();cfg=Config();totals=Counter();scenes=[]
    for name in index['added']['dev']:
        detector=Detector(weights);counts=Counter();frames=[]
        targets=read(cache/(name+'.targets.json.gz'))['frames']
        cached=read(cache/(name+'.inputs.json.gz'))['frames']
        for row,target,old in zip(index['sequences'][name],targets,cached):
            image=cv2.imread(str(DATA/row['image_path']))
            scan=np.fromfile(DATA/row['scan_path'],dtype='<f4').reshape(-1,5)
            frame,_=detector.step(image,int(row['camera_timestamp_us']*1000))
            t=np.linalg.inv(row['camera_transform'])@row['transform']
            cam=Camera(image.shape[1],image.shape[0],row['camera_intrinsic'],np.zeros(4),t)
            assoc=associate(scan[:,:3],cam,frame)
            # Reproduce the all-point depth test without consulting detector/GT.
            xyz=scan[:,:3]@t[:3,:3].T+t[:3,3];sel=np.flatnonzero((xyz[:,2]>.1)&(xyz[:,2]<80.))
            uv,_=cv2.projectPoints(xyz[sel],np.zeros(3),np.zeros(3),cam.K,cam.distortion)
            uv=uv.reshape(-1,2);ok=(uv[:,0]>=0)&(uv[:,0]<cam.width-.5)&(uv[:,1]>=0)&(uv[:,1]<cam.height-.5)
            sel=sel[ok];uv=np.rint(uv[ok]).astype(int)
            inside=np.zeros(len(scan),bool);inside[sel]=True
            w=(cam.width+2)//3;cells=uv[:,1]//3*w+uv[:,0]//3
            depth=np.full(w*((cam.height+2)//3),np.inf);np.minimum.at(depth,cells,xyz[sel,2])
            visible=np.zeros(len(scan),bool);visible[sel[xyz[sel,2]<=depth[cells]+.4]]=True
            # Targets enter only the observer below.
            truth=identity_truth(target)>0;y=np.asarray(target['gt'])
            supported=truth&np.isin(y,THING)
            masks=dict(gt_all=truth,gt_supported=supported,gt_in_view=truth&inside,
                gt_supported_in_view=supported&inside,gt_supported_visible=supported&visible,
                gt_associated=truth&(assoc['ids']>0))
            stages=Counter({k:int(v.sum()) for k,v in masks.items()})
            accepted=np.zeros(len(scan),bool);fragment_diagnostics=[]
            for tid in np.unique(assoc['ids'][assoc['ids']>0]):
                at=np.flatnonzero(assoc['ids']==tid);c=component(scan[at,:3],cfg)
                reason='accepted' if c is not None else ('too_large' if len(at)>cfg.max_object_points else 'too_small' if len(at)<cfg.min_points else 'fragmented')
                stages[reason+'_objects']+=1;stages[reason+'_gt_points']+=int(truth[at].sum())
                if c is not None:accepted[at[c]]=True
                elif reason=='fragmented':
                    # Observe the existing largest component without changing
                    # any detector, live gate, cache or model input. This tests
                    # whether the 60% whole-mask veto discards useful surfaces.
                    relaxed=component(scan[at,:3],replace(cfg,component_fraction=0.))
                    if relaxed is not None:
                        chosen=at[relaxed];pan=np.asarray(target['panoptic']);known=pan[chosen]>0
                        identity=identity_truth(target)[chosen]
                        v,n=np.unique(identity[(identity>0)&known],return_counts=True)
                        dominant=int(n.max()) if len(n) else 0
                        denominator=int(known.sum());purity=dominant/max(1,denominator)
                        raw=np.asarray(target['raw_labels'])[chosen];yy=y[chosen];valid=yy>=0
                        diagnostic=dict(visual_track=int(tid),points=len(chosen),known_points=denominator,
                            gt_instance_points=int(truth[chosen].sum()),dominant_instance_points=dominant,
                            identity_purity=purity,raw_semantic_errors=int((valid&(raw!=yy)).sum()))
                        fragment_diagnostics.append(diagnostic)
                        stages['relaxed_extra_points']+=len(chosen)
                        stages['relaxed_extra_known_points']+=denominator
                        stages['relaxed_extra_gt_points']+=diagnostic['gt_instance_points']
                        stages['relaxed_extra_dominant_points']+=dominant
                        stages['relaxed_extra_raw_errors']+=diagnostic['raw_semantic_errors']
                        stages['relaxed_high_purity_groups']+=purity>=.9
                        stages['relaxed_low_purity_groups']+=purity<.9
            stages['gt_after_component']=int((truth&accepted).sum())
            cached_mask=np.zeros(len(scan),bool)
            for o in old['observations']:cached_mask[o['indices']]=True
            stages['cached_component_symmetric_difference']=int((cached_mask!=accepted).sum())
            counts.update(stages);frames.append(dict(frame=row['frame'],counts=dict(stages),fragment_diagnostics=fragment_diagnostics))
        totals.update(counts);scenes.append(dict(scene=name,counts=dict(counts),frames=frames))
        print(name,dict(counts),flush=True);del detector
    return dict(counts=dict(totals),scenes=scenes,admission=admission,
        scope='Frozen YOLO replay, added development scenes only; counts distinguish unsupported classes, camera visibility, detection and bounded component rejection. No accuracy/latency claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['index','cache','weights','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Refuse overwrite')
    result=audit(read(a.index),a.cache,a.weights);save(a.out,result);print(result['counts'])
