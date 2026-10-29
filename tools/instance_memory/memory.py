"""GT-free, bounded object association and raw-observation memory.

Visual IDs are hints, never identity keys. Matching is one-to-one, uses frozen
sensor features through a learned descriptor plus motion, and never gates on
semantic class. A new scene/reset starts a new identity namespace.
"""
from dataclasses import dataclass,asdict
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Config:
    max_tracks:int=128
    short_records:int=4
    long_records:int=2
    ttl_s:float=8.
    max_cost:float=.72
    ambiguity_margin:float=.04
    motion_slack_m:float=1.
    initial_speed_mps:float=10.
    acceleration_mps2:float=3.


class ObjectMemory:
    def __init__(self,cfg=None):
        self.cfg=cfg or Config()
        self.reset()

    def reset(self):
        self.tracks={};self.next_id=1;self.last_stamp=-1;self.pending=None

    def candidates(self,stamp):
        if type(stamp) is not int or stamp<=self.last_stamp or self.pending is not None:
            raise ValueError('Timestamps must increase; finish current write before next read')
        self.last_stamp=stamp
        self.tracks={k:v for k,v in self.tracks.items() if (stamp-v['last']['stamp'])*1e-9<=self.cfg.ttl_s}
        return list(self.tracks)

    def representative_inputs(self,ids):
        return [self.tracks[k]['last']['x'] for k in ids]

    def match(self,obs,embeddings,ids,old_embeddings,stamp):
        """Read first. Caller writes current RAW evidence after prediction."""
        n,m=len(obs),len(ids)
        if n>self.cfg.max_tracks or embeddings.shape!=(n,32) or old_embeddings.shape!=(m,32):
            raise ValueError('Invalid bounded descriptor inputs')
        cost=np.full((n,m),1e6,float)
        for i,o in enumerate(obs):
            if o['stamp']!=stamp or o['camera_stamp']>stamp:raise ValueError('Observation time mismatch')
            for j,tid in enumerate(ids):
                tr=self.tracks[tid];last=tr['last'];dt=(stamp-last['stamp'])*1e-9
                velocity=np.asarray(tr['velocity'])
                pred=np.asarray(last['center'])+velocity*dt
                distance=float(np.linalg.norm(np.asarray(o['center'])-pred))
                # Extent/partial-surface allowance plus finite motion uncertainty.
                shape=.25*(np.linalg.norm(o['extent'])+np.linalg.norm(last['extent']))
                radius=self.cfg.motion_slack_m+shape+(
                    .5*self.cfg.acceleration_mps2*dt*dt if tr['observations']>=2 else self.cfg.initial_speed_mps*dt)
                radius=min(radius,15.)
                similarity=float(np.clip(embeddings[i]@old_embeddings[j],-1,1))
                if distance>radius or similarity<.1:continue
                hint=(o['visual_track']==last['visual_track'] and o['visual_session']==last['visual_session'])
                cost[i,j]=.7*(1-similarity)/2+.3*distance/max(radius,1e-6)-.03*hint
        assigned={}
        if n and m:
            # Each detection can select its own new-ID slot. This avoids forcing
            # bad matches or stealing a valid identity through rectangular costs.
            augmented=np.concatenate([cost,np.full((n,n),self.cfg.max_cost)],axis=1)
            rows,cols=linear_sum_assignment(augmented)
            for i,j in zip(rows,cols):
                if j>=m or cost[i,j]>=self.cfg.max_cost:continue
                alternatives=np.delete(cost[i],j)
                competitors=np.delete(cost[:,j],i)
                if (len(alternatives) and alternatives.min()<cost[i,j]+self.cfg.ambiguity_margin
                    or len(competitors) and competitors.min()<cost[i,j]+self.cfg.ambiguity_margin):continue
                assigned[int(i)]=ids[int(j)]
        track_ids=[];histories=[]
        for i,o in enumerate(obs):
            tid=assigned.get(i)
            if tid is None:tid=self.next_id;self.next_id+=1
            tr=self.tracks.get(tid)
            history=[]
            if tr is not None:
                seen=set()
                for r in reversed(tr['short']):
                    if r['stamp']<stamp and (stamp-r['stamp'])*1e-9<=self.cfg.ttl_s:
                        history.append(r);seen.add(r['stamp'])
                for r in tr['long']:
                    if r['stamp'] not in seen and (stamp-r['stamp'])*1e-9<=self.cfg.ttl_s:
                        history.append(r);seen.add(r['stamp'])
            track_ids.append(tid);histories.append(copy.deepcopy(history[:6]))
        self.pending=(stamp,track_ids)
        return track_ids,histories,dict(matches=len(assigned),new=n-len(assigned),tracks=len(self.tracks))

    def write(self,obs,track_ids,stamp):
        if self.pending!=(stamp,track_ids):raise ValueError('Write without matching read')
        for o,tid in zip(obs,track_ids):
            # Explicit whitelist prevents enhanced output/GT accidentally becoming
            # new evidence. No feature predicted by MemoryHead is stored.
            r={k:copy.deepcopy(o[k]) for k in ('x','stamp','camera_stamp','center','extent','visual_track','visual_session')}
            if 'sensor_center' in o:r['sensor_center']=copy.deepcopy(o['sensor_center'])
            r['raw_mean']=copy.deepcopy(o['raw_mean'])
            r['quality']=float(o['visual_confidence'])
            old=self.tracks.get(tid)
            velocity=np.zeros(3)
            if old is not None:
                dt=(stamp-old['last']['stamp'])*1e-9
                observed=(np.asarray(o['center'])-np.asarray(old['last']['center']))/dt
                if np.linalg.norm(observed)<=35.:
                    velocity=.5*np.asarray(old['velocity'])+.5*observed if old['observations']>1 else observed
            r['velocity']=velocity.tolist()
            short=old['short'][:] if old else []
            long=old['long'][:] if old else []
            if short and r['camera_stamp']<=short[-1]['camera_stamp']:
                # Reused images do not become fresh multimodal observations.
                continue
            short.append(r)
            if len(short)>self.cfg.short_records:
                displaced=short.pop(0)
                pool=[p for p in long+[displaced] if (stamp-p['stamp'])*1e-9<=self.cfg.ttl_s]
                # Keep a strong old view and a temporally distinct second view;
                # quality never means correctness and is not a fusion weight.
                pool.sort(key=lambda p:(-p['quality'],p['stamp']))
                long=[]
                for p in pool if self.cfg.long_records else []:
                    if all(abs(p['stamp']-q['stamp'])>=1_000_000_000 for q in long):long.append(p)
                    if len(long)>=self.cfg.long_records:break
            self.tracks[tid]=dict(last=r,short=short,long=long,velocity=velocity.tolist(),
                                 observations=1+(old['observations'] if old else 0))
        if len(self.tracks)>self.cfg.max_tracks:
            keep=sorted(self.tracks,key=lambda k:self.tracks[k]['last']['stamp'],reverse=True)[:self.cfg.max_tracks]
            self.tracks={k:self.tracks[k] for k in keep}
        self.pending=None

    def stats(self):
        return dict(tracks=len(self.tracks),records=sum(len(t['short'])+len(t['long']) for t in self.tracks.values()))

    def revise(self,poses):
        """Reanchor retained raw records using already received pose corrections.

        poses maps sensor stamps to corrected world-from-sensor transforms. Missing anchors
        retire affected tracks. Motion is recomputed; unrelated graph versions
        are never mixed. The online adapter owns the transform/epoch contract.
        """
        if self.pending is not None:raise ValueError('Cannot revise during a prediction')
        for tid,tr in list(self.tracks.items()):
            all_records=tr['short']+tr['long']
            if any(r['stamp'] not in poses or 'sensor_center' not in r for r in all_records):
                del self.tracks[tid];continue
            centers=[]
            for r in all_records:
                pose=np.asarray(poses[r['stamp']])
                if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError('Invalid pose revision')
                centers.append(pose[:3,:3]@np.asarray(r['sensor_center'])+pose[:3,3])
            # The received trajectory also grows without changing old poses.
            # Such messages must not repeatedly reset the causal motion filter.
            if all(np.allclose(c,r['center'],rtol=0,atol=1e-7) for c,r in zip(centers,all_records)):
                continue
            for r,c in zip(all_records,centers):r['center']=c.tolist()
            velocity=np.zeros(3)
            if len(tr['short'])>=2:
                a,b=tr['short'][-2:];dt=(b['stamp']-a['stamp'])*1e-9
                velocity=(np.asarray(b['center'])-a['center'])/dt
            if np.linalg.norm(velocity)>35.:
                del self.tracks[tid];continue
            tr['velocity']=velocity.tolist();tr['last']=tr['short'][-1]
            for r in all_records:r['velocity']=velocity.tolist()
