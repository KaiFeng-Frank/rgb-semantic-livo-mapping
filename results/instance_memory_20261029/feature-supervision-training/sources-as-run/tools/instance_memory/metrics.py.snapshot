"""Official panoptic point GT, class-independent identity diagnostics.

Point identity F1 and association quality are defined explicitly here. These are
not claimed as official nuScenes PAT or leaderboard scores. Missing coverage stays
in their denominators. No box-majority proxy is used.
"""
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


class IdentityMetrics:
    def __init__(self):
        self.gt=Counter();self.pred=Counter();self.joint=Counter()
        self.last={};self.switches=0;self.transitions=0;self.reappearance=0

    def add(self,gt,pred,frame,valid):
        gt=np.asarray(gt);pred=np.asarray(pred)
        for k,n in zip(*np.unique(gt[(gt>0)&valid],return_counts=True)):self.gt[int(k)]+=int(n)
        for k,n in zip(*np.unique(pred[(pred>0)&valid],return_counts=True)):self.pred[int(k)]+=int(n)
        both=(gt>0)&(pred>0)&valid
        pairs,counts=np.unique(np.stack([gt[both],pred[both]],axis=1),axis=0,return_counts=True)
        for (a,b),n in zip(pairs,counts):self.joint[int(a),int(b)]+=int(n)
        for g in np.unique(gt[(gt>0)&valid]):
            at=valid&(gt==g);p=pred[at];values,n=np.unique(p[p>0],return_counts=True)
            # Fixed support/coverage threshold. All misses are retained by the
            # point metrics above; IDSW alone is not used to select the model.
            if not len(n) or n.max()<5 or n.max()<.5*len(p):continue
            best=int(values[n.argmax()])
            if g in self.last:
                prev_t,prev=self.last[g]
                self.transitions+=1;self.switches+=best!=prev
                self.reappearance+=frame>prev_t+1
            self.last[g]=(frame,best)

    def result(self):
        gs,ps=list(self.gt),list(self.pred)
        overlap=np.zeros((len(gs),len(ps)),np.int64)
        gi,pi={v:i for i,v in enumerate(gs)},{v:i for i,v in enumerate(ps)}
        for (g,p),n in self.joint.items():overlap[gi[g],pi[p]]=n
        matched=0
        if overlap.size:
            r,c=linear_sum_assignment(-overlap);matched=int(overlap[r,c].sum())
        gt_total=sum(self.gt.values());pred_total=sum(self.pred.values())
        aq=0.
        for g,total in self.gt.items():
            aq+=sum(n*n/(total+self.pred[p]-n) for (gg,p),n in self.joint.items() if gg==g)/total
        return dict(point_idf1=2*matched/max(1,gt_total+pred_total),
            association_quality=aq/max(1,len(gs)),gt_points=gt_total,pred_points=pred_total,
            identity_true_positive=matched,associated_gt_points=sum(self.joint.values()),
            gt_instances=len(gs),pred_tracks=len(ps),id_switches=int(self.switches),
            supported_transitions=self.transitions,reappearance_transitions=self.reappearance)


def semantic_truth(target):
    return np.asarray(target['gt'],np.int64)


def identity_truth(target):
    pan=np.asarray(target['panoptic'],np.int64)
    return np.where((pan%1000>0)&(semantic_truth(target)>=0),pan,0)


def observation_target(o,target):
    at=np.asarray(o['indices'])
    return semantic_truth(target)[at]


def identity_target(o,target):
    at=np.asarray(o['indices']);gt=identity_truth(target)[at]
    vals,counts=np.unique(gt[gt>0],return_counts=True)
    if not len(counts) or counts.max()<6 or counts.max()<.8*len(at):return None
    return int(vals[counts.argmax()])
