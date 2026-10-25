"""Read-only provenance instrumentation for the frozen repair implementation.

Only original voxel indices are attached to memory. Ground-truth identity is
joined by the offline evaluator, never consumed by normal model inference.
Candidate attribution is verified against the unmodified retrieval outputs.
"""
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from semantic_memory import _keys
from semantic_repair import PreviousEvidence


def exact_source_indices(source, selected):
    """Exact float32 row identity, no nearest-neighbor or quantization guess."""
    table = {tuple(row):i for i,row in enumerate(source.detach().cpu().tolist())}
    if len(table) != len(source):
        raise ValueError('Ambiguous duplicate source coordinates')
    indices = [table[tuple(row)] for row in selected.detach().cpu().tolist()]
    return torch.tensor(indices, device=source.device, dtype=torch.long)


class TracedEvidence(PreviousEvidence):
    fields = PreviousEvidence.fields + ('raw_index',)

    def reset(self):
        super().reset()
        self.raw_index = torch.empty(0, dtype=torch.long, device=self.device)
        self.trace = {}

    @torch.no_grad()
    def write_raw(self, xyz, features, probabilities, stamp, frame):
        super().write_raw(xyz, features, probabilities, stamp, frame)
        self.raw_index = exact_source_indices(xyz, self.xyz)
        if not torch.equal(xyz[self.raw_index].float(), self.xyz):
            raise AssertionError('Stored observation provenance mismatch')

    @torch.no_grad()
    def prepare(self, stamp, transform, scan_sensor):
        self.trace = dict(written_source=self.raw_index.clone())
        status = super().prepare(stamp, transform, scan_sensor)
        self.trace.update(record_source=self.raw_index.clone(),
            readable=self.readable.clone(), record_xyz=self.xyz,
            record_features=self.features, record_probabilities=self.probabilities,
            record_stamp=self.stamp)
        return status

    @torch.no_grad()
    def retrieve(self, xyz, features, stamp):
        result = super().retrieve(xyz, features, stamp)
        c, n = self.cfg, len(xyz)
        self.trace.update(query_xyz=xyz, query_features=features)
        if not len(self.xyz):
            self.trace.update(search_index=torch.zeros((n,0),dtype=torch.long,device=xyz.device),
                search_valid=torch.zeros((n,0),dtype=torch.bool,device=xyz.device),
                search_distance2=xyz.new_zeros((n,0)),
                shortlist_index=torch.zeros_like(result['valid'],dtype=torch.long),
                shortlist_valid=torch.zeros_like(result['valid']),
                retrieval_valid=result['valid'].clone())
            return result
        # Reconstruct the exact original indices, without replacing its result.
        grid = torch.floor(xyz/c.cell_size).long()
        keys = _keys((grid[:,None]+self.offsets[None])*c.cell_size,c.cell_size,check=False)
        start = torch.searchsorted(self.key,keys)
        idx = start[...,None]+torch.arange(c.records_per_cell,device=xyz.device)
        expected = keys[...,None].expand_as(idx).reshape(n,-1)
        valid = idx.reshape(n,-1)<len(self.key)
        idx = idx.reshape(n,-1).clamp_max(len(self.key)-1)
        valid &= self.key[idx]==expected
        valid &= self.readable[idx] & (self.stamp[idx]<stamp)
        distance2 = (self.xyz[idx]-xyz[:,None]).square().sum(-1)
        valid &= distance2 <= c.radius**2
        k = min(c.neighbors,idx.shape[1])
        _,sel = distance2.masked_fill(~valid,float('inf')).topk(k,largest=False)
        short,short_valid = idx.gather(1,sel),valid.gather(1,sel)
        hist = self.features[short].to(features.dtype)
        after_cosine = short_valid & (F.cosine_similarity(features[:,None].float(),hist.float(),dim=-1)>=c.min_cosine)
        if not (torch.equal(hist,result['features']) and
                torch.equal(self.xyz[short]-xyz[:,None],result['delta']) and
                torch.equal(after_cosine,result['valid'])):
            raise AssertionError('Diagnostic retrieval differs from actual implementation')
        self.trace.update(search_index=idx,search_valid=valid,search_distance2=distance2,
            shortlist_index=short,shortlist_valid=short_valid,
            retrieval_valid=result['valid'].clone())
        return result


class FusionCapture:
    def __init__(self, fusion):
        self.value = None
        self.handle = fusion.register_forward_pre_hook(self.capture)

    def capture(self, module, args):
        f,b,candidates = args
        self.value = (f.detach().clone(), b.detach().clone(),
                      {k:v.detach().clone() for k,v in candidates.items()})

    def close(self):
        self.handle.remove()


def voxel_instances(arrays, groups, instance_ids):
    """Assign only voxels whose labeled points belong to ONE known instance.

    A voxel mixed with other objects or unassigned labeled points remains -1.
    This conservative diagnostic rule never invents an instance correspondence.
    """
    n = len(arrays['xyz'])
    valid = arrays['point_label']>=0
    inverse = arrays['inverse'][valid]
    owner = arrays['group_owner'][valid]
    low = np.full(n,np.iinfo(np.int32).max,np.int32)
    high = np.full(n,-1,np.int32)
    np.minimum.at(low,inverse,owner)
    np.maximum.at(high,inverse,owner)
    unambiguous = (low==high)&(low>=0)
    result = np.full(n,-1,np.int64)
    group_ids = np.array([instance_ids[g['instance']] for g in groups],np.int64)
    result[unambiguous] = group_ids[low[unambiguous]]
    return result


def scalar_membership(values, allowed):
    return np.isin(values,allowed) & (values>=0)


def candidate_tags(trace, previous_instances, previous_raw_prediction, previous_labels,
                   query_instances, final_candidates):
    """Join GT outside the normal model and expose loss at each retrieval stage."""
    device = query_instances.device
    nq = len(query_instances)
    keys = ('local_correct','shortlist_correct','retrieved_correct')
    if not len(trace['record_source']):
        return {**{k:torch.zeros(nq,dtype=torch.bool,device=device) for k in keys},
                'correct_slots':torch.zeros_like(final_candidates['valid']),
                'same_instance_slots':torch.zeros_like(final_candidates['valid'])}
    source = trace['record_source']
    inst = previous_instances[source]
    good = (previous_raw_prediction[source]==previous_labels[source])&(inst>=0)
    q = query_instances[:,None]
    search,short = trace['search_index'],trace['shortlist_index']
    radius2 = trace['query_radius'][:,None]**2
    local = trace['search_valid'] & (trace['search_distance2']<=radius2)
    local_good = local & (inst[search]==q)&(q>=0)&good[search]
    short_valid = trace['shortlist_valid'] & (final_candidates['delta'].square().sum(-1)<=radius2)
    same = (inst[short]==q)&(q>=0)
    short_good = short_valid & same & good[short]
    final_good = final_candidates['valid'] & same & good[short]
    return dict(local_correct=local_good.any(1), shortlist_correct=short_good.any(1),
                retrieved_correct=final_good.any(1), correct_slots=final_good,
                same_instance_slots=final_candidates['valid']&same)
