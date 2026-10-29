"""Causal, raw-only point memory with existing visual ownership/evidence vetoes."""
from dataclasses import dataclass
import numpy as np
from .point_history_audit import neighbors
from .point_probability_memory import mixture, classify
from tools.visual_instances.visual_semantics import distribution


def visual_arrays(n, observations):
    """Only cached deployment inputs are accepted; GT is not an argument."""
    ids = np.zeros(n,np.int64)
    current = np.full(n,-1,np.int64)
    temporal = current.copy()
    for obs in observations:
        at = np.asarray(obs['indices'],np.int64)
        if (at.ndim != 1 or len(at) != len(np.unique(at)) or np.any(at<0) or np.any(at>=n)
                or np.any(ids[at]) or int(obs['visual_id']) <= 0):
            raise ValueError('Invalid/overlapping visual point ownership')
        c, h = (int(distribution(obs[k]).argmax()) for k in ('current_scores','memory_scores'))
        if c != obs['current_class'] or h != obs['memory_class'] or obs['history_records']<0:
            raise ValueError('Cached evidence disagrees with its own scores')
        ids[at] = int(obs['visual_id'])
        current[at], temporal[at] = c,h
    return ids,current,temporal


def owned_readout(p, source, ids, source_ids, indices, valid):
    keep = valid & (ids[:,None] == source_ids[indices]) if len(source) else valid
    return classify(mixture(p,source,indices,keep),p.argmax(1)),keep


def evidence_veto(raw, proposal, visual_class):
    """Independent evidence may reject a transfer; it never supplies a new label."""
    mask = (visual_class == raw) & (proposal != raw)
    return np.where(mask,raw,proposal),mask


@dataclass
class EvidencePointMemory:
    previous: object = None

    def step(self, xyz, p, ids, visual_class, stamp, epoch):
        xyz=np.asarray(xyz,np.float64);p=np.asarray(p,np.float32)
        ids=np.asarray(ids);visual_class=np.asarray(visual_class)
        if (xyz.shape!=(len(p),3) or p.shape!=(len(xyz),16) or not 0<len(p)<=200000
                or ids.shape!=(len(p),) or visual_class.shape!=(len(p),)
                or ids.dtype.kind not in 'ui' or visual_class.dtype.kind not in 'iu'
                or (ids<0).any() or ((visual_class < -1)|(visual_class>=16)).any()
                or not np.isfinite(xyz).all() or not np.isfinite(p).all() or (p<0).any()
                or not np.allclose(p.sum(1),1,atol=1e-5) or type(stamp) is not int or stamp<0):
            raise ValueError('Invalid causal point evidence input')
        old=self.previous
        if old is not None and old['epoch']==epoch and stamp<=old['stamp']:
            raise ValueError('Noncausal timestamp')
        if old is not None and (old['epoch']!=epoch or stamp-old['stamp']>750000000):
            old=None
        raw=p.argmax(1)
        if old is None:
            output=raw.copy();stats=dict(queries=0,retained_queries=0,rejected_neighbor_pairs=0,vetoed_points=0)
        else:
            j,v=neighbors(xyz,old['xyz'])
            proposal,keep=owned_readout(p,old['p'],ids,old['ids'],j,v)
            output,veto=evidence_veto(raw,proposal,visual_class)
            stats=dict(queries=int(v.any(1).sum()),retained_queries=int(keep.any(1).sum()),
                       rejected_neighbor_pairs=int((v&~keep).sum()),vetoed_points=int(veto.sum()))
        self.previous=dict(xyz=xyz.copy(),p=p.copy(),ids=ids.copy(),stamp=stamp,epoch=epoch)
        return output,stats
