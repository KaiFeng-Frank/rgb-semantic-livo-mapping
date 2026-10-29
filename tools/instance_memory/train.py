"""Frozen-backbone head training. Project holdout files are never opened here.

The identity encoder is fit on official TRAIN panoptic IDs, then frozen and shared
by both semantic arms. Temporal association during semantic training uses predicted
identities, not teacher-forced GT tracks. No test-driven architectural search.
"""
import argparse
import copy
from pathlib import Path
import time
import json

import numpy as np
import torch
from torch.nn import functional as F

from .data import read,save
from .model import MemoryHead
from .runtime import Runtime
from .evaluate import load_scene,evaluate
from .metrics import identity_target,observation_target
from tools.semantic_memory.nuscenes_pilot_data import digest


def fit_identity(model,scenes,epochs=10):
    optimizer=torch.optim.Adam(list(model.encoder.parameters())+list(model.identity.parameters()),lr=1e-3)
    logs=[]
    for epoch in range(epochs):
        total=steps=pairs=0
        for data,targets in scenes:
            observations=[];labels=[];times=[]
            for frame,target in zip(data['frames'],targets['frames']):
                for o in frame['observations']:
                    label=identity_target(o,target)
                    if label is not None:
                        observations.append(o['x']);labels.append(label);times.append(frame['stamp']*1e-9)
            if len(labels)<3:continue
            x=torch.as_tensor(np.asarray(observations),dtype=torch.float32)
            y=torch.tensor(labels);t=torch.tensor(times,dtype=torch.float64)
            z=model.embeddings(x)
            similarity=z@z.T/.1
            eligible=(t[:,None]-t[None]).abs()<=8.
            eligible &= (t[:,None]!=t[None])
            positive=(y[:,None]==y[None])&eligible
            negative=(y[:,None]!=y[None])&eligible
            good=positive.any(1)&negative.any(1)
            if not good.any():continue
            # Supervised contrastive objective includes hard same-class objects;
            # no semantic class-equality shortcut is used in the association.
            logp=similarity-torch.logsumexp(similarity.masked_fill(~eligible,-1e4),1,keepdim=True)
            per=-(logp*positive).sum(1)/positive.sum(1).clamp_min(1)
            loss=per[good].mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=float(loss.detach());steps+=1;pairs+=int(positive.sum())
        logs.append(dict(epoch=epoch+1,loss=total/max(1,steps),steps=steps,positive_pairs=pairs))
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return logs


def semantic_loss(outputs,observations,target):
    loss=[]
    for logits,o in zip(outputs,observations):
        y=torch.as_tensor(observation_target(o,target));valid=y>=0
        if valid.any():loss.append(F.cross_entropy(logits[valid],y[valid],reduction='sum'))
    n=sum(int((observation_target(o,target)>=0).sum()) for o in observations)
    return sum(loss)/n if loss else None


def run(cache,out,seed=0,epochs=30):
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    out.mkdir(parents=True,exist_ok=False)
    protocol=read(cache/'protocol.json')
    # ONLY TRAIN and DEV are read here. Targets are separate from model inputs.
    train=[load_scene(cache,n) for n in protocol['parts']['train']]
    dev=[load_scene(cache,n) for n in protocol['parts']['dev']]
    all_x=np.asarray([o['x'] for s,_ in train for f in s['frames'] for o in f['observations']],np.float32)
    mean=all_x.mean(0);scale=np.maximum(all_x.std(0),.05)
    identity=MemoryHead(mean,scale)
    started=time.monotonic();idlog=fit_identity(identity,train)
    save(out/'identity-training.json',idlog)
    # Both semantic arms share the trained/frozen identity feature encoder.
    initial=copy.deepcopy(identity)
    for name,param in initial.named_parameters():
        param.requires_grad_(not name.startswith(('encoder.','identity.')))
    initial.train()
    metadata=dict(schema=1,seed=seed,epochs=epochs,cache_protocol_sha256=digest(cache/'protocol.json'),
        mean=mean.tolist(),scale=scale.tolist(),parameters=sum(p.numel() for p in initial.parameters()),
        trainable_parameters=sum(p.numel() for p in initial.parameters() if p.requires_grad),
        split=protocol['parts'],identity_epochs=10,identity_frozen_shared=True,
        supervision='Official training panoptic IDs and native16 labels; predicted online associations for semantic training',
        selection='minimum dev point CE on same observed points; epoch0 eligible',
        source={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    save(out/'protocol.json',metadata)
    history=[]
    for arm,mode in [('current','matched_current'),('temporal','feature_memory')]:
        torch.manual_seed(seed)  # Identical optimizer and randomized order schedule.
        model=copy.deepcopy(initial)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
        best=float('inf')
        for epoch in range(epochs+1):
            training_loss=0.;steps=0
            if epoch:
                model.train()
                for si in np.random.default_rng(seed+epoch).permutation(len(train)):
                    data,targets=train[si];runtime=Runtime(model,identity,mode)
                    for frame,target in zip(data['frames'],targets['frames']):
                        outputs,_,_=runtime.step(frame)
                        loss=semantic_loss(outputs,frame['observations'],target)
                        if loss is None:continue
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                        training_loss+=float(loss.detach());steps+=1
                        if time.monotonic()-started>1800:raise RuntimeError('Per-seed training budget exhausted')
            model.eval()
            # Evaluate only its own mode, with the other placeholder sharing exact
            # weights. The two arms are selected independently on the same data.
            evaluation=evaluate(model,model,identity,dev,full=False)
            selected_arm='matched_current' if arm=='current' else 'feature_memory'
            result=evaluation['aggregate'][selected_arm]
            row=dict(arm=arm,epoch=epoch,train_loss=training_loss/max(1,steps),steps=steps,
                dev_ce=result['point_ce'],dev_miou=result['miou_fixed16'],dev_observed_accuracy=result['observed_accuracy'],
                elapsed_s=time.monotonic()-started)
            history.append(row)
            if result['point_ce']<best:
                best=result['point_ce']
                torch.save(dict(**metadata,model=model.state_dict(),identity=identity.state_dict(),epoch=epoch,dev=row),out/(arm+'.pt'))
            save(out/'training.json',history)
            print(json.dumps(row),flush=True)
    save(out/'completion.json',dict(complete=True,elapsed_s=time.monotonic()-started,holdout_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=30)
    a=p.parse_args();run(a.cache,a.out,a.seed,a.epochs)
