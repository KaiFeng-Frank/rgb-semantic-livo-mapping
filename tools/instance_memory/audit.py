"""Post-selection TRAIN/DEV diagnosis; labels never enter matching or prediction.

Inspect the actual raw records read by the frozen policy. Compare its output with
the SAME head given no history, then attribute differences using evaluation-only
instance/semantic annotations. This does not tune or replace holdout scores.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .data import read, save
from .evaluate import checkpoint, load_scene
from .metrics import identity_target
from .model import batch_inputs, point_logits
from .runtime import Runtime


def majority(y):
    y=np.asarray(y);y=y[y>=0]
    return int(np.bincount(y,minlength=16).argmax()) if len(y) else None


@torch.no_grad()
def audit(cache,run,split):
    model,identity,_=checkpoint(run/'temporal.pt')
    protocol=read(cache/'protocol.json');total=Counter();scenes=[]
    for name in protocol['parts'][split]:
        data,targets=load_scene(cache,name);rt=Runtime(model,identity)
        # Evaluation sidecar only. Runtime never receives these annotations.
        truth={}
        for f,t in zip(data['frames'],targets['frames']):
            for o in f['observations']:
                y=np.asarray(t['gt'])[o['indices']]
                truth[o['stamp'],o['visual_track']]=dict(identity=identity_target(o,t),label=majority(y))
        saved={};match=rt.memory.match
        def capture(*args):
            result=match(*args);saved['histories']=result[1];return result
        rt.memory.match=capture
        counts=Counter();rows=[]
        for f,t in zip(data['frames'],targets['frames']):
            outputs,ids,info=rt.step(f);obs=f['observations']
            if not obs:continue
            x,mem,meta,mask=batch_inputs(model,obs,saved['histories'],f['stamp'],temporal=False)
            delta=model(x,mem,meta,mask)
            for i,(o,logits,hist) in enumerate(zip(obs,outputs,saved['histories'])):
                y=np.asarray(t['gt'])[o['indices']];ok=y>=0
                base=point_logits(o,delta[i]).argmax(-1).numpy();pred=logits.argmax(-1).numpy()
                current=truth[o['stamp'],o['visual_track']]
                past=[truth[r['stamp'],r['visual_track']] for r in hist]
                known=[p for p in past if p['identity'] is not None and current['identity'] is not None]
                wrong=sum(p['identity']!=current['identity'] for p in known)
                mixed=any(p['label']!=current['label'] for p in past)
                key='no_history' if not hist else ('wrong_identity' if wrong else ('unscorable_identity' if len(known)!=len(past) else 'same_identity'))
                helped=int((ok&(base!=y)&(pred==y)).sum());harmed=int((ok&(base==y)&(pred!=y)).sum())
                values=dict(observations=1,history_observations=bool(hist),records=len(hist),
                    known_records=len(known),wrong_identity_records=wrong,semantic_conflict_observations=mixed,
                    helped=helped,harmed=harmed,changed=int((ok&(base!=pred)).sum()),points=int(ok.sum()),
                    current_correct=int((ok&(base==y)).sum()),memory_correct=int((ok&(pred==y)).sum()))
                counts.update(values)
                for k in ['observations','helped','harmed']:counts[key+'_'+k]+=values[k]
                if helped or harmed:
                    rows.append(dict(frame=f['frame'],visual_track=o['visual_track'],identity=ids[i],
                        category=key,history_records=len(hist),semantic_conflict=mixed,
                        current_gt_label=current['label'],helped=helped,harmed=harmed))
        total.update(counts);scenes.append(dict(scene=name,counts=dict(counts),changed_observations=rows))
    return dict(split=split,counts=dict(total),scenes=scenes,
        scope='Frozen model TRAIN/DEV diagnostic; GT used only after predictions; comparator is same temporal head with history removed, not independently trained current head')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=['train','dev'],default='dev')
    a=p.parse_args();torch.set_num_threads(1)
    if a.out.exists():raise ValueError('Refuse to overwrite diagnostic')
    result=audit(a.cache,a.run,a.split);save(a.out,result);print(result['counts'])
