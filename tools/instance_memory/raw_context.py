"""One raw historical scan supplies bounded geometric context to frozen PTv3.

No semantic prediction is stored. The original current points remain first and
unchanged; history cannot replace their 5 cm voxel representatives or move the
current scan's voxel origin/bounds. This is an information-availability probe,
not a trained temporal model or a claim of calibrated object motion.
"""
from dataclasses import dataclass
import copy
import time
import numpy as np


@dataclass(frozen=True)
class Config:
    ttl_s:float=1.
    max_history_points:int=12000
    max_fraction:float=.5
    support_voxel_m:float=.4
    history_voxel_m:float=.1
    native_voxel_m:float=.05
    max_speed_mps:float=35.
    max_scan_points:int=200000


def keys(points,size):
    q=np.floor(np.asarray(points,dtype=np.float64)/size).astype(np.int64)
    if np.any(np.abs(q)>=2**20):raise ValueError('Coordinate exceeds bounded key range')
    u=(q+2**20).astype(np.uint64)
    return (u[:,0]<<np.uint64(42))|(u[:,1]<<np.uint64(21))|u[:,2]


def visual_key(o):return o['visual_session'],o['visual_track']


class RawContext:
    def __init__(self,cfg=None):self.cfg=cfg or Config();self.previous=None

    def validate(self,points,stamp,pose):
        if points.ndim!=2 or points.shape[1]!=4 or not 0<len(points)<=self.cfg.max_scan_points or not np.isfinite(points).all():
            raise ValueError('Invalid bounded raw scan')
        if type(stamp) is not int or np.asarray(pose).shape!=(4,4) or not np.isfinite(pose).all():raise ValueError('Invalid pose/time')
        if not np.allclose(pose[3],[0,0,0,1]) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-4):raise ValueError('Nonrigid pose')
        if self.previous and stamp<=self.previous['stamp']:raise ValueError('Nonincreasing sensor time')

    def prepare(self,points,stamp,pose,observations,mode):
        if mode not in ('ego','motion','unwarped'):raise ValueError('Unknown context mode')
        self.validate(points,stamp,pose);start=time.perf_counter();old=self.previous;cfg=self.cfg
        for o in observations:
            if o['stamp']!=stamp or o['camera_stamp']>stamp or np.asarray(o['center']).shape!=(3,) or not np.isfinite(o['center']).all():
                raise ValueError('Noncausal/invalid current object geometry')
        info=dict(mode=mode,history_points=0,motion_points=0,unmatched_instance_points=0)
        if old is None or (stamp-old['stamp'])*1e-9>cfg.ttl_s:
            info['prepare_ms']=(time.perf_counter()-start)*1000;return points,info
        dt=(stamp-old['stamp'])*1e-9
        history=old['points'].copy();keep=np.ones(len(history),bool)
        world=history[:,:3]@old['pose'][:3,:3].T+old['pose'][:3,3]
        current={visual_key(o):o for o in observations}
        for o in old['observations']:
            at=o['indices'];new=current.get(visual_key(o))
            if new is None:
                keep[at]=False;info['unmatched_instance_points']+=len(at);continue
            delta=np.asarray(new['center'])-o['center']
            if np.linalg.norm(delta)>cfg.max_speed_mps*dt:
                keep[at]=False;continue
            if mode=='motion':world[at]+=delta;info['motion_points']+=len(at)
        if mode!='unwarped':history[:,:3]=(world-pose[:3,3])@pose[:3,:3]
        xyz=history[:,:3];low=points[:,:3].min(0);high=points[:,:3].max(0)
        keep &= ((xyz>=low)&(xyz<=high)).all(1)
        at=np.flatnonzero(keep)
        # Coarse support excludes free-space ghosts; it is a spatial consistency
        # check, not a proof that an unmatched moving object is static.
        active=np.unique(keys(points[:,:3],cfg.support_voxel_m))
        at=at[np.isin(keys(xyz[at],cfg.support_voxel_m),active,kind='sort')]
        _,first=np.unique(keys(xyz[at],cfg.history_voxel_m),return_index=True);at=at[first]
        existing=np.unique(keys(points[:,:3],cfg.native_voxel_m))
        at=at[~np.isin(keys(xyz[at],cfg.native_voxel_m),existing,kind='sort')]
        budget=min(cfg.max_history_points,int(len(points)*cfg.max_fraction),cfg.max_scan_points-len(points))
        if len(at)>budget:
            k=keys(xyz[at],cfg.history_voxel_m)
            # Deterministic hash ranking avoids retaining only one spatial edge.
            with np.errstate(over='ignore'):
                h=(k^(k>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
                h=(h^(h>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
                h=h^(h>>np.uint64(31))
            at=at[np.argsort(h,kind='stable')[:budget]]
        augmented=np.concatenate([points,history[at]])
        info.update(history_points=len(at),prepare_ms=(time.perf_counter()-start)*1000)
        return augmented,info

    def store(self,points,stamp,pose,observations):
        self.validate(points,stamp,pose)
        retained=[]
        for o in observations:
            if o['stamp']!=stamp or o['camera_stamp']>stamp:raise ValueError('Noncausal object geometry')
            at=np.asarray(o['indices'],np.int64)
            if np.any((at<0)|(at>=len(points))):raise ValueError('Invalid point association')
            retained.append(dict(indices=at.copy(),visual_session=o['visual_session'],visual_track=o['visual_track'],
                center=np.asarray(o['center']).copy()))
        self.previous=dict(points=points.copy(),stamp=stamp,pose=np.asarray(pose).copy(),observations=retained)
