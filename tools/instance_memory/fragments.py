"""Past trusted tracks admit ambiguous current surfaces without self-confirmation.

Candidate outputs never update/create/refresh an anchor. The original trusted
stream alone drives identity and memory state. No semantic class gate or GT.
"""
from collections import Counter
from dataclasses import dataclass
import copy
import time
import numpy as np
import torch
from .runtime import Runtime
from .model import batch_inputs,point_logits


@dataclass(frozen=True)
class ConfirmationConfig:
    max_age_s:float=1.5
    min_cosine:float=.8
    motion_slack_m:float=.75
    max_radius_m:float=4.
    acceleration_mps2:float=3.
    max_speed_mps:float=35.
    max_cost:float=.5
    ambiguity_margin:float=.1


def key(o):return o['visual_session'],o['visual_track']


@torch.no_grad()
def confirm(observations,tracks,identity,now,cfg=None,appearance=True):
    cfg=cfg or ConfirmationConfig();anchors=[]
    for tid,tr in tracks.items():
        last=tr['last'];age=(now-last['stamp'])*1e-9
        if not 0<age<=cfg.max_age_s:continue
        history=[r for r in tr['short'] if key(r)==key(last)]
        if len(history)<2:continue
        a,b=history[-2:];dt=(b['stamp']-a['stamp'])*1e-9
        if not 0<dt<=cfg.max_age_s or b['camera_stamp']<=a['camera_stamp']:continue
        velocity=(np.asarray(b['center'])-np.asarray(a['center']))/dt
        if np.linalg.norm(velocity)>cfg.max_speed_mps:continue
        anchors.append((tid,tr,a,b,velocity,age))
    if not observations:return [],[],[]
    if not anchors:return [0]*len(observations),[[] for _ in observations],['no_established_anchor']*len(observations)
    device=identity.mean.device
    emb=identity.embeddings(torch.as_tensor(np.stack([o['x'] for o in observations]),dtype=torch.float32,device=device)).cpu().numpy()
    old=identity.embeddings(torch.as_tensor(np.stack([r['x'] for _,_,a,b,_,_ in anchors for r in [a,b]]),dtype=torch.float32,device=device)).cpu().numpy().reshape(len(anchors),2,32)
    costs=np.full((len(observations),len(anchors)),np.inf)
    for i,o in enumerate(observations):
        if o['stamp']!=now or o['camera_stamp']>now:raise ValueError('Noncausal fragment')
        for j,(_,_,_,b,v,age) in enumerate(anchors):
            if o['camera_stamp']<=b['camera_stamp']:continue
            cosine=float(np.min(old[j]@emb[i]))
            if appearance and cosine<cfg.min_cosine:continue
            predicted=np.asarray(b['center'])+v*age
            distance=float(np.linalg.norm(np.asarray(o['center'])-predicted))
            radius=min(cfg.max_radius_m,cfg.motion_slack_m+.5*np.linalg.norm(b['extent'])+.5*cfg.acceleration_mps2*age*age)
            if distance>radius:continue
            costs[i,j]=(.5*(1-np.clip(cosine,-1,1)) if appearance else 0.)+.5*distance/radius
    ids=[];histories=[];reasons=[]
    for i,o in enumerate(observations):
        own=[j for j,(_,_,_,b,_,_) in enumerate(anchors) if key(o)==key(b) and costs[i,j]<cfg.max_cost]
        chosen=min(own,key=lambda j:costs[i,j]) if own else None
        reason='no_matching_visual_anchor' if chosen is None else 'confirmed'
        if chosen is not None:
            alternatives=np.delete(costs[i],chosen);competitors=np.delete(costs[:,chosen],i)
            if (len(alternatives) and np.min(alternatives)<=costs[i,chosen]+cfg.ambiguity_margin
                or len(competitors) and np.min(competitors)<=costs[i,chosen]+cfg.ambiguity_margin):
                chosen=None;reason='ambiguous'
        if chosen is None:ids.append(0);histories.append([])
        else:
            tid,tr,_,_,_,_=anchors[chosen]
            records=[];seen=set()
            for r in list(reversed(tr['short']))+tr['long']:
                if key(r)==key(o) and r['stamp'] not in seen and 0<(now-r['stamp'])*1e-9<=8.:
                    records.append(copy.deepcopy(r));seen.add(r['stamp'])
            ids.append(tid);histories.append(records[:6])
        reasons.append(reason)
    return ids,histories,reasons


class FragmentRuntime:
    def __init__(self,model,identity,mode='matched_current',*,appearance=True):
        if mode not in ('matched_current','feature_memory','same_memory_head_without_history'):
            raise ValueError('Unsupported semantic readout')
        self.base=Runtime(model,identity,mode);self.model=model;self.identity=identity
        self.mode=mode;self.appearance=appearance

    def reset(self):self.base.reset()

    @torch.no_grad()
    def step(self,frame):
        if self.model.training:raise ValueError('This frozen diagnostic does not train a model')
        start=time.perf_counter();obs=frame['observations'];now=frame['stamp']
        trusted=[i for i,o in enumerate(obs) if o.get('trusted',True)]
        weak=[i for i,o in enumerate(obs) if not o.get('trusted',True)]
        candidates=[obs[i] for i in weak]
        # This read happens before the trusted stream consumes the current frame.
        candidate_ids,histories,reasons=confirm(candidates,self.base.memory.tracks,self.identity,now,appearance=self.appearance)
        outputs,ids,info=self.base.step(dict(frame,observations=[obs[i] for i in trusted]))
        all_outputs=[torch.as_tensor(o['raw_probabilities'],dtype=torch.float32).clamp_min(1e-5).log() for o in obs]
        all_ids=[0]*len(obs)
        for i,p,tid in zip(trusted,outputs,ids):all_outputs[i]=p;all_ids[i]=tid
        admitted=[];occupied=set(ids)
        for i,tid in enumerate(candidate_ids):
            if tid and tid in occupied:reasons[i]='current_identity_conflict';candidate_ids[i]=0
            elif tid:admitted.append(i);occupied.add(tid)
        if admitted:
            chosen=[candidates[i] for i in admitted];history=[histories[i] for i in admitted]
            x,m,meta,mask=batch_inputs(self.model,chosen,history,now,temporal=self.mode=='feature_memory')
            delta=self.model(x,m,meta,mask)
            for j,i in enumerate(admitted):
                all_outputs[weak[i]]=point_logits(candidates[i],delta[j]);all_ids[weak[i]]=candidate_ids[i]
        info.update(candidate_observations=len(weak),admitted_fragments=len(admitted),
            candidate_reasons=dict(Counter(reasons)),history_observations=info['history_observations']+sum(bool(histories[i]) for i in admitted),
            elapsed_ms=(time.perf_counter()-start)*1000,candidate_feedback=False)
        return all_outputs,all_ids,info
