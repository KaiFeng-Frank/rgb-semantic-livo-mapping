"""Bounded raw visual evidence; causal read-before-write, no posterior feedback."""
from collections import deque
from dataclasses import dataclass,asdict
import numpy as np


@dataclass(frozen=True)
class Config:
    max_tracks:int=128
    records:int=12
    ttl_s:float=2.
    spacing_s:float=.1
    decay_s:float=.75


class VisualMemory:
    def __init__(self,cfg=None):self.cfg=cfg or Config();self.reset()
    def reset(self):self.states={};self.session=None;self.last_stamp=-1
    def configuration(self):return dict(asdict(self.cfg),evidence='raw COCO80 sigmoid scores; history total weight <= current weight; no enhanced feedback')
    def step(self,frame):
        cfg=self.cfg;stamp=frame.stamp_ns
        if stamp<=self.last_stamp:raise ValueError('Reordered/repeated visual memory timestamp')
        if frame.session!=self.session:self.states.clear();self.session=frame.session
        self.last_stamp=stamp
        self.states={k:q for k,q in self.states.items() if stamp-q[-1][0]<=cfg.ttl_s*1e9}
        if frame.class_scores is None:
            if len(frame.ids):raise ValueError('Missing raw class evidence')
            raw=np.empty((0,80),np.float32)
        else:raw=np.asarray(frame.class_scores,np.float32)
        if raw.shape!=(len(frame.ids),80) or not np.isfinite(raw).all() or np.any((raw<0)|(raw>1)):
            raise ValueError('Invalid raw visual evidence')
        output=raw.copy();counts=np.zeros(len(raw),np.int32)
        for i,tid in enumerate(frame.ids):
            history=[(t,p) for t,p in self.states.get(int(tid),[]) if 0<stamp-t<=cfg.ttl_s*1e9]
            if history:
                w=np.exp(-np.array([(stamp-t)*1e-9 for t,_ in history])/cfg.decay_s)
                w/=max(1.,w.sum())
                output[i]=(raw[i]+sum(a*p for a,(_,p) in zip(w,history)))/(1+w.sum())
                counts[i]=len(history)
        # Current raw values only, after every read. Consecutive close frames do
        # not evict temporally distinct observations or multiply confidence.
        for i,tid in enumerate(frame.ids):
            q=self.states.setdefault(int(tid),deque(maxlen=cfg.records))
            if not q or stamp-q[-1][0]>=cfg.spacing_s*1e9:q.append((stamp,raw[i].copy()))
        if len(self.states)>cfg.max_tracks:
            keep=sorted(self.states,key=lambda k:self.states[k][-1][0],reverse=True)[:cfg.max_tracks]
            self.states={k:self.states[k] for k in keep}
        return output,counts,dict(tracks=len(self.states),raw_records=sum(len(q) for q in self.states.values()),
            observations_with_history=int(np.count_nonzero(counts)))
