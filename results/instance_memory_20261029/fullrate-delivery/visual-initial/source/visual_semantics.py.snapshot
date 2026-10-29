"""Frozen pointwise visual readout, shared by cached and eventual online paths."""
import numpy as np
from .geometry import COCO_TO_NATIVE
from .semantics import Config,component


def distribution(scores):
    scores=np.asarray(scores,np.float32)
    if scores.shape!=(80,) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError('Expected raw COCO80 evidence')
    q=np.zeros(16,np.float32)
    for coco,native in COCO_TO_NATIVE.items():q[native]=scores[coco]
    if q.sum()<=0:raise ValueError('Empty visual class evidence')
    return q/q.sum()


def fusion(probabilities,visual_scores,confidence):
    p=np.asarray(probabilities,np.float32)
    if p.ndim!=2 or p.shape[1]!=16 or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=.002):
        raise ValueError('Invalid native16 point probabilities')
    if not np.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('Invalid current visual confidence')
    # Existing visual_weight is retained; no post-result coefficient sweep.
    alpha=Config().visual_weight*confidence
    return (1-alpha)*p+alpha*distribution(visual_scores)[None]


def groups(xyz,association):
    cfg=Config();result=[]
    for tid in np.unique(association['ids'][association['ids']>0]):
        at=np.flatnonzero(association['ids']==tid);c=component(xyz[at],cfg)
        if c is not None:result.append((int(tid),at[c]))
    return result
