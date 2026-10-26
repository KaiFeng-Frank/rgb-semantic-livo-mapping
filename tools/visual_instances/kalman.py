"""Causal instance semantic filtering with learned feature observations.

RKN-inspired observation/uncertainty heads and analytic diagonal updates; not a
reproduction of RKN, nor a claim of Gaussian-optimal categorical inference.
Motion/identity remains separate. State is continuous centered class evidence.
Runtime uses numpy only; neural heads are exported from the CPU trainer as JSON.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .semantics import InstanceSemantics, THING

INPUT_DIM = 232  # 64 mean + 64 log std + native16 + raw COCO80 + 8 quality
VAR_MIN, VAR_MAX, PROCESS_NOISE = .25, 30., .25


def softmax(x):
    e = np.exp(x-np.max(x, axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)


def inputs(observation, features):
    f = np.asarray(features, np.float32)
    visual = observation['visual_scores']
    if visual is None:
        raise ValueError('Kalman semantics requires full visual scores, not fabricated top-1 evidence')
    x = np.concatenate((f[:64], np.log1p(np.maximum(f[64:],0)),
                        observation['raw_mean'], visual, observation['quality'])).astype(np.float32)
    if f.shape != (128,) or x.shape != (INPUT_DIM,) or not np.isfinite(x).all():
        raise ValueError('Invalid instance observation features')
    return x


def prediction_variance(variance, dt):
    return np.clip((variance+PROCESS_NOISE*dt)*np.exp(dt/5.), VAR_MIN, VAR_MAX)


def repeat_penalty(previous, current, dt):
    cosine = float(np.dot(previous,current)/(np.linalg.norm(previous)*np.linalg.norm(current)+1e-8))
    return 1.+3.*np.exp(-dt/.5)*np.clip(cosine,0.,1.)


class ObservationModel:
    def __init__(self, path):
        path = Path(path)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        d = json.loads(path.read_text())
        if d['schema'] != 1 or d['input_dim'] != INPUT_DIM or d['classes'] != 16:
            raise ValueError('Incompatible instance observation model')
        self.meta = d.get('metadata', {})
        self.mean = np.asarray(d['mean'], np.float32)
        self.scale = np.asarray(d['scale'], np.float32)
        self.weights = {k:np.asarray(v,np.float32) for k,v in d['weights'].items()}
        if self.mean.shape != (INPUT_DIM,) or self.scale.shape != (INPUT_DIM,) or np.any(self.scale <= 0):
            raise ValueError('Invalid observation normalization')
        shapes = {'hidden.weight':(32,INPUT_DIM),'hidden.bias':(32,),
                  'delta.weight':(16,32),'delta.bias':(16,),
                  'variance.weight':(16,32),'variance.bias':(16,)}
        if set(self.weights) != set(shapes) or any(self.weights[k].shape != v for k,v in shapes.items()):
            raise ValueError('Invalid observation network')
        if not all(np.isfinite(a).all() for a in [self.mean,self.scale,*self.weights.values()]):
            raise ValueError('Nonfinite observation network')

    def __call__(self, x, raw_mean):
        w = self.weights
        normalized = np.clip((x-self.mean)/self.scale,-8,8)
        h = np.maximum(normalized@w['hidden.weight'].T+w['hidden.bias'],0)
        base = np.log(np.maximum(raw_mean,1e-4)); base -= base.mean(-1,keepdims=True)
        delta = 3.*np.tanh(h@w['delta.weight'].T+w['delta.bias'])
        z = base+delta; z -= z.mean(-1,keepdims=True)
        r = np.clip(.15+np.logaddexp(0,h@w['variance.weight'].T+w['variance.bias']),.15,VAR_MAX)
        return z, r


class EvidenceFilter:
    def __init__(self, mode='learned'):
        if mode not in ('current','fixed','weighted','learned'):
            raise ValueError('Unknown instance filter')
        self.mode, self.states = mode, {}

    def step(self, key, z, r, features, stamp_ns):
        if self.mode == 'fixed':
            r = np.ones_like(r)
        old = self.states.get(key)
        if old is not None and stamp_ns <= old['stamp']:
            raise ValueError('Repeated or reordered instance observation')
        dt = (stamp_ns-old['stamp'])*1e-9 if old is not None else 0.
        if old is None or dt > 2. or self.mode == 'current':
            h, p = z.copy(), r.copy()
            history = deque(maxlen=4)
            gain = np.ones_like(r)
        else:
            effective_r = (np.ones_like(r) if self.mode == 'fixed' else r)*repeat_penalty(old['feature'],features,dt)
            prior_p = prediction_variance(old['variance'],dt)
            gain = prior_p/(prior_p+effective_r)
            h = old['mean']+gain*(z-old['mean'])
            # Joseph form, diagonal H=I. Floor prevents permanent lock-in.
            p = np.clip((1-gain)**2*prior_p+gain**2*effective_r,VAR_MIN,VAR_MAX)
            history = old['history']
            if self.mode == 'weighted':
                evidence = [(z, r, stamp_ns), *history]
                precision = [np.exp(-(stamp_ns-t)*1e-9/1.5)/rv for _,rv,t in evidence]
                h = sum(a*v for a,(v,_,_) in zip(precision,evidence))/sum(precision)
        history.append((z.copy(),r.copy(),stamp_ns))  # raw observation, never posterior feedback
        self.states[key] = dict(mean=h.copy(),variance=p.copy(),feature=features.copy(),
                                stamp=stamp_ns,history=history)
        self.states = {k:s for k,s in self.states.items() if stamp_ns-s['stamp'] <= 2_000_000_000}
        if len(self.states)>128:
            keep=sorted(self.states,key=lambda k:self.states[k]['stamp'],reverse=True)[:128]
            self.states={k:self.states[k] for k in keep}
        return h, dict(gain_mean=float(gain.mean()),variance_mean=float(p.mean()),history=old is not None and dt<=2.)


def decision(posterior, current, raw_probabilities, cfg):
    p = softmax(posterior); now = softmax(current)
    best = int(p.argmax()); top = np.sort(p)[-2:]; current_top = np.sort(now)[-2:]
    accepted = best in THING and top[-1]>=cfg.posterior_min and top[-1]-top[-2]>=cfg.posterior_margin
    supported = now.argmax()==best and current_top[-1]>=cfg.posterior_min and current_top[-1]-current_top[-2]>=cfg.posterior_margin
    raw = raw_probabilities.argmax(1); confidence = raw_probabilities.max(1)
    # Foreground/background failure is allowed to be corrected, but temporal
    # reinforcement must not overwrite strong current competing object classes.
    contrary = np.isin(raw,THING)&(raw!=best)&(confidence>=cfg.posterior_min)&(confidence-raw_probabilities[:,best]>=cfg.posterior_margin)
    veto = bool(accepted and not supported and np.any(contrary))
    return best, float(p[best]), bool(accepted and not veto), veto


class KalmanInstanceSemantics:
    def __init__(self, checkpoint, mode='learned'):
        self.core = InstanceSemantics()
        self.model, self.filter = ObservationModel(checkpoint), EvidenceFilter(mode)
        self.cfg = self.core.cfg
        self.session = None

    def configuration(self):
        return dict(method='feature-conditioned diagonal instance Kalman',mode=self.filter.mode,
                    checkpoint_sha256=self.model.sha256,model_metadata=self.model.meta,
                    geometry=self.core.configuration(),max_tracks=128,ttl_s=2.,
                    covariance_floor=VAR_MIN,process_noise=PROCESS_NOISE,
                    observation='same-forward raw PTv3 mean/std + COCO80 scores + geometric quality',
                    output='observed credible component only; no GT or future input')

    def revise_poses(self, times, poses):
        self.core.revise_poses(times,poses)
        alive={(self.session,t,s['epoch']) for t,s in self.core.states.items()}
        self.filter.states={k:s for k,s in self.filter.states.items() if k in alive}

    def step(self, xyz, probabilities, association, *, stamp_ns, pose, features):
        start=time.perf_counter()
        session=association['info'].get('session')
        if session != self.session:
            self.filter.states.clear(); self.session=session
        result=self.core.step(xyz,probabilities,association,stamp_ns=stamp_ns,pose=pose,collect_observations=True)
        result['labels']=result['raw_labels'].copy(); result['confidence']=result['raw_confidence'].copy()
        result['valid'].fill(0); result['instance_labels'].fill(65535); result['instance_confidence'].fill(0)
        details=[]
        for o in result.pop('observations'):
            tid=o['track_id']; at=o['indices']; f=np.asarray(features[tid],np.float32)
            x=inputs(o,f); z,r=self.model(x,o['raw_mean'])
            key=(session,tid,o['epoch'])
            # Remove retired generations immediately, not just at TTL expiry.
            self.filter.states={k:s for k,s in self.filter.states.items() if k[:2]!=key[:2] or k==key}
            h,stats=self.filter.step(key,z,r,f[:64],stamp_ns)
            label,conf,ok,veto=decision(h,z,probabilities[at],self.cfg)
            if ok:
                result['labels'][at]=label; result['confidence'][at]=conf
                result['valid'][at]=1; result['instance_labels'][at]=label; result['instance_confidence'][at]=conf
            details.append(dict(track_id=tid,epoch=o['epoch'],identity=o['identity'],points=len(at),
                label=label,confidence=conf,unified=ok,semantic_conflict=veto,**stats))
        result['info']=dict(instances=details,tracks=len(self.filter.states),semantic_update=True,
            unified_points=int(result['valid'].sum()),unified_instances=sum(d['unified'] for d in details),
            changed_points=int(np.count_nonzero(result['labels']!=result['raw_labels'])),
            history_instances=sum(d['history'] for d in details),semantic_conflict=sum(d['semantic_conflict'] for d in details),
            elapsed_ms=(time.perf_counter()-start)*1000,method=self.filter.mode)
        return result
