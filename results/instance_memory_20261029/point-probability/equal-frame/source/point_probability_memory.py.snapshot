"""Equal-per-frame raw probability memory; no posterior feedback or GT inputs."""
from dataclasses import dataclass
import numpy as np
from .point_history_audit import neighbors,vote


def mixture(current,source,indices,valid):
    """Total historical mass is bounded by one current observation, not K."""
    total=valid.sum(1);mean=np.zeros_like(current)
    if len(source):
        for col in range(4):mean+=source[indices[:,col]]*valid[:,col,None]
    mean/=np.maximum(total[:,None],1)
    return np.where((total>0)[:,None],.5*(current+mean),current)


def classify(p,current):
    result=p.argmax(1)
    keep=p[np.arange(len(p)),current]==p.max(1)
    result[keep]=current[keep]
    return result


@dataclass
class RawProbabilityMemory:
    previous: object=None

    def step(self,xyz,p,stamp,epoch):
        xyz=np.asarray(xyz,np.float64);p=np.asarray(p,np.float32)
        if xyz.shape!=(len(p),3) or p.shape!=(len(xyz),16) or len(p)>200000:
            raise ValueError('Invalid scan shape/capacity')
        if not np.isfinite(xyz).all() or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):
            raise ValueError('Invalid coordinates/probabilities')
        if type(stamp) is not int:raise ValueError('Integer timestamp required')
        old=self.previous
        if old is not None and old['epoch']==epoch and stamp<=old['stamp']:raise ValueError('Noncausal stamp')
        if old is None or old['epoch']!=epoch or stamp-old['stamp']>750000000:old=None
        spatial=neighbors(xyz,xyz,same_scan=True)
        historical=neighbors(xyz,old['xyz'] if old is not None else np.empty((0,3)))
        prior=old['p'] if old is not None else np.empty((0,16),np.float32)
        raw=p.argmax(1)
        pred=dict(raw=raw,current_spatial_probability=classify(mixture(p,p,*spatial),raw),
                  previous_raw_probability=classify(mixture(p,prior,*historical),raw),
                  previous_hard_vote=vote(raw,prior.argmax(1),*historical))
        # No output above is written; retain the original distribution exactly.
        self.previous=dict(xyz=xyz.copy(),p=p.copy(),stamp=stamp,epoch=epoch)
        return pred,dict(historical_queries=int(historical[1].any(1).sum()),current_queries=int(spatial[1].any(1).sum()))
