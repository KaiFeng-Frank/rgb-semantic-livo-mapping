"""CPU-only frozen-backbone pilot with log-disjoint selection and sealed test.

Training uses official native16 labels. These checkpoints are supervised and
must not be described as no-3D-label training. All arms see identical cached
observations; no input includes GT instance IDs or future measurements.
"""
import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .kalman import INPUT_DIM, ObservationModel, EvidenceFilter, decision, VAR_MIN, VAR_MAX, PROCESS_NOISE
from .semantics import Config
from tools.semantic_memory.pilot_metrics import confusion,scores
from tools.semantic_memory.nuscenes_pilot_data import digest


class Head(nn.Module):
    def __init__(self,mean,scale):
        super().__init__()
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
        self.hidden=nn.Linear(INPUT_DIM,32)
        self.delta=nn.Linear(32,16)
        self.variance=nn.Linear(32,16)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.variance.weight); nn.init.constant_(self.variance.bias,.4)

    def forward(self,x,raw):
        h=F.relu(self.hidden(((x-self.mean)/self.scale).clamp(-8,8)))
        base=raw.clamp_min(1e-4).log();base=base-base.mean(-1,keepdim=True)
        z=base+3*self.delta(h).tanh();z=z-z.mean(-1,keepdim=True)
        r=(.15+F.softplus(self.variance(h))).clamp(.15,VAR_MAX)
        return z,r

    def export(self,path,metadata):
        data=dict(schema=1,input_dim=INPUT_DIM,classes=16,mean=self.mean.tolist(),scale=self.scale.tolist(),
                  weights={k:v.detach().tolist() for k,v in self.state_dict().items() if k not in ('mean','scale')},metadata=metadata)
        Path(path).write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')


def load_scenes(cache,names):
    scenes=[]
    for name in names:
        with gzip.open(cache/(name+'.json.gz'),'rt') as stream: scene=json.load(stream)
        for o in scene['observations']:
            for key in ('x','raw_mean','raw_probabilities'):o[key]=np.asarray(o[key],np.float32)
            o['targets']['point_labels']=np.asarray(o['targets']['point_labels'],np.int64)
            o['targets']['histogram']=np.asarray(o['targets']['histogram'],np.float32)
        scenes.append(scene)
    return scenes


def tensor_inputs(observations):
    return (torch.tensor(np.stack([o['x'] for o in observations])),
            torch.tensor(np.stack([o['raw_mean'] for o in observations])))


def train_scene(model,scene,temporal):
    obs=scene['observations']
    if not obs:return None
    x,raw=tensor_inputs(obs); z,r=model(x,raw)
    states={};outputs=[]
    for i,o in enumerate(obs):
        key=(o['track'],o['epoch']);old=states.get(key)
        dt=(o['stamp']-old[2])*1e-9 if old is not None else 0
        if temporal and old is not None and 0<dt<=2.:
            mean,var,_,previous,steps=old
            cosine=F.cosine_similarity(previous[None],x[i,:64][None]).clamp(0,1)[0]
            effective=r[i]*(1.+3*np.exp(-dt/.5)*cosine)
            prior=((var+PROCESS_NOISE*dt)*np.exp(dt/5.)).clamp(VAR_MIN,VAR_MAX)
            gain=prior/(prior+effective)
            h=mean+gain*(z[i]-mean)
            p=((1-gain).square()*prior+gain.square()*effective).clamp(VAR_MIN,VAR_MAX)
        else:h,p,steps=z[i],r[i],0
        outputs.append(h)
        states[key]=(h.detach() if steps%4==3 else h,p.detach() if steps%4==3 else p,o['stamp'],x[i,:64],steps+1)
    target=torch.tensor(np.stack([o['targets']['histogram'] for o in obs]))
    good=target.sum(1)>0
    if not good.any():return None
    target=target/target.sum(1,keepdim=True).clamp_min(1)
    ce=-(target*torch.stack(outputs).log_softmax(-1)).sum(-1)
    current=-(target*z.log_softmax(-1)).sum(-1)
    # An explicit residual likelihood trains the variance head; softmax confidence
    # is never substituted for covariance. This is a latent coding objective.
    code=(target*.9+.1/16).log();code=code-code.mean(-1,keepdim=True)
    calibration=.5*((z-code).square()/r+r.log()).mean(-1)
    return (ce+.5*current+.05*calibration)[good].mean()


def evaluate(model,scenes,modes=('current','fixed','weighted','learned')):
    totals={mode:np.zeros((16,16),np.int64) for mode in ('raw','single','guarded',*modes)}
    counts={mode:Counter() for mode in totals};per_scene=[];cases=[];times={m:[] for m in modes}
    cfg=Config()
    for scene in scenes:
        matrices={mode:np.zeros((16,16),np.int64) for mode in totals}
        for frame in scene['frames']:
            for mode in matrices:
                source=mode if mode in ('single','guarded') else 'raw'
                matrices[mode]+=np.asarray(frame[source+'_confusion'])
            for mode in ('single','guarded'):
                counts[mode].update(rescued=frame['rescued_'+mode],harmed=frame['harmed_'+mode])
        filters={m:EvidenceFilter(m) for m in modes}
        for o in scene['observations']:
            start=time.perf_counter();z,r=model(o['x'],o['raw_mean']);head_ms=(time.perf_counter()-start)*1000
            gt=o['targets']['point_labels'];raw=o['raw_probabilities'].argmax(1)
            original=confusion(gt,raw,16)
            outputs={}
            for mode,filt in filters.items():
                start=time.perf_counter()
                h,stats=filt.step((o['track'],o['epoch']),z,r,o['x'][:64],o['stamp'])
                best,conf,ok,veto=decision(h,z,o['raw_probabilities'],cfg)
                pred=np.full(len(raw),best) if ok else raw.copy()
                matrices[mode]+=confusion(gt,pred,16)-original
                good=gt>=0
                rescued=int(((raw!=gt)&(pred==gt)&good).sum());harmed=int(((raw==gt)&(pred!=gt)&good).sum())
                counts[mode].update(rescued=rescued,harmed=harmed,changed=int((pred!=raw).sum()),
                    observations=1,accepted=int(ok),history=int(stats['history']),veto=int(veto))
                times[mode].append(head_ms+(time.perf_counter()-start)*1000)
                outputs[mode]=dict(label=best,accepted=ok,rescued=rescued,harmed=harmed,**stats)
            cases.append(dict(scene=scene['scene'],frame=o['frame'],track=o['track'],epoch=o['epoch'],
                points=len(raw),gt_hist=o['targets']['histogram'].astype(int).tolist(),arms=outputs))
        for mode in totals:totals[mode]+=matrices[mode]
        per_scene.append(dict(scene=scene['scene'],frames=len(scene['frames']),observations=len(scene['observations']),
            arms={m:dict(scores(c),confusion=c.tolist()) for m,c in matrices.items()}))
    return dict(aggregate={m:dict(scores(c),counts=dict(counts[m]),confusion=c.tolist(),
        timing=dict(median_ms=float(np.median(times[m])),p95_ms=float(np.percentile(times[m],95)))
        if m in times and times[m] else None) for m,c in totals.items()},scenes=per_scene,cases=cases)


def run(cache,out,epochs=25,seed=0):
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((cache/'protocol.json').read_text());parts=protocol['parts']
    # Test and diagnostic prediction/label files are deliberately not opened here.
    train=load_scenes(cache,parts['train']);dev=load_scenes(cache,parts['dev'])
    all_x=np.stack([o['x'] for s in train for o in s['observations']])
    mean=all_x.mean(0);scale=np.maximum(all_x.std(0),.05)
    initial=Head(mean,scale)
    training_protocol=dict(seed=seed,epochs=epochs,parts=parts,cache_protocol_sha256=digest(cache/'protocol.json'),
        normalization='train observations only',supervision='Official native16 3D training labels for both heads; not label-free',
        selection='Best dev point mIoU; per-arm identical epoch count; test opened once after all selections',
        parameters=sum(p.numel() for p in initial.parameters()),device='cpu',threads=1,
        controls=['raw','single','guarded','matched_current','current_same_head','weighted','fixed','learned'],
        loss='sequence CE + .5 current CE + .05 latent-code Gaussian NLL; BPTT truncation 4 observations',
        inference='causal; predicted visual IDs + geometry epochs only; no GT identity or future observations',
        sources={str(p.name):digest(p) for p in (Path(__file__),Path(__file__).with_name('kalman.py'))})
    (out/'protocol.json').write_text(json.dumps(training_protocol,indent=2)+'\n')
    log=[]
    for arm,temporal in (('matched_current',False),('learned',True)):
        model=copy.deepcopy(initial);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        best=-float('inf')
        for epoch in range(epochs):
            model.train();losses=[]
            for scene in train:
                loss=train_scene(model,scene,temporal)
                if loss is None:continue
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                losses.append(float(loss.detach()))
            model.eval();path=out/f'{arm}-candidate.json'
            metadata=dict(training_protocol,arm=arm,epoch=epoch,backbone_sha256=protocol['checkpoint_sha256'])
            model.export(path,metadata)
            mode='learned' if temporal else 'current'
            score=evaluate(ObservationModel(path),dev,(mode,))['aggregate'][mode]['miou_fixed16']
            row=dict(arm=arm,epoch=epoch,loss=float(np.mean(losses)),dev_miou=score)
            log.append(row);print(json.dumps(row),flush=True)
            if score>best:
                best=score
                (out/f'{arm}.json').write_bytes(path.read_bytes())
        (out/f'{arm}-selection.json').write_text(json.dumps(dict(dev_miou=best),indent=2)+'\n')
    # A single sealed evaluation, including the equally trained current-only arm.
    for part in ('dev','test','diagnostic'):
        scenes=dev if part=='dev' else load_scenes(cache,parts[part])
        report=evaluate(ObservationModel(out/'learned.json'),scenes)
        current=evaluate(ObservationModel(out/'matched_current.json'),scenes,('current',))
        report['aggregate']['matched_current']=current['aggregate']['current']
        for s,c in zip(report['scenes'],current['scenes']):s['arms']['matched_current']=c['arms']['current']
        report['scope']='Small supervised pilot; supplied ego poses; full-scene metrics dilute sparse front-camera corrections'
        (out/f'{part}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(part=part,results={m:dict(miou=100*r['miou_fixed16'],**r['counts']) for m,r in report['aggregate'].items()})),flush=True)
    (out/'train-log.json').write_text(json.dumps(log,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=25);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();run(a.cache,a.out,a.epochs,a.seed)
