"""Privileged TRAIN/DEV evidence-availability audit, never a deployed policy."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .data import read,save
from .evaluate import checkpoint,load_scene
from .metrics import identity_target,identity_truth,IdentityMetrics
from .model import batch_inputs,point_logits
from .runtime import Runtime


@torch.no_grad()
def diagnose(cache,run,index,*,added_only=True):
    current,identity,_=checkpoint(run/'current.pt')
    counts=Counter();objects=set();logs=set();scenes=[];identity_totals=Counter()
    names=index['added']['dev'] if added_only else index['parts']['dev']
    for name in names:
        d,t=load_scene(cache,name);rt=Runtime(current,identity,'matched_current')
        sidecar={(o['stamp'],o['visual_track']):identity_target(o,gt)
            for f,gt in zip(d['frames'],t['frames']) for o in f['observations']}
        captured={};original_match=rt.memory.match
        def capture(*args):
            out=original_match(*args);captured['history']=out[1];return out
        rt.memory.match=capture
        scene=Counter();details=[];visual=IdentityMetrics();visual_ids={}
        for f,gt in zip(d['frames'],t['frames']):
            output,_,_=rt.step(f);truth=identity_truth(gt)
            visual_pred=np.zeros(len(truth),np.int64)
            for o in f['observations']:
                key=(o['visual_session'],o['visual_track'])
                visual_ids.setdefault(key,len(visual_ids)+1)
                visual_pred[o['indices']]=visual_ids[key]
            visual.add(truth,visual_pred,f['frame'],np.asarray(gt['panoptic'])>0)
            for o,logits,hist in zip(f['observations'],output,captured['history']):
                at=np.asarray(o['indices']);y=np.asarray(gt['gt'])[at];valid=(y>=0)&(y<10)
                pred=logits.argmax(-1).numpy();wrong=valid&(pred!=y)
                scene.update(observations=1,history_observations=bool(hist),foreground_points=int(valid.sum()),foreground_errors=int(wrong.sum()))
                ident=sidecar[o['stamp'],o['visual_track']]
                if not hist or ident is None:continue
                # Past record descriptors are raw. The frozen current head does
                # not receive either identity GT or current/future labels.
                x,mem,meta,mask=batch_inputs(current,hist,[[]]*len(hist),f['stamp'],temporal=False)
                delta=current(x,mem,meta,mask)
                classes=[int((torch.as_tensor(r['raw_mean']).clamp_min(1e-5).log()+delta[i]).argmax()) for i,r in enumerate(hist)]
                candidates=[c for c,r in zip(classes,hist) if sidecar[r['stamp'],r['visual_track']]==ident]
                recoverable=wrong&(truth[at]==ident)&np.isin(y,candidates)
                n=int(recoverable.sum());scene['correctable_foreground_points']+=n
                if n:
                    objects.add((name,ident));logs.add(index['scene_logs'][name])
                    details.append(dict(frame=f['frame'],identity=ident,correctable_points=n,past_classes=candidates))
        ident=visual.result()
        identity_totals.update({k:ident[k] for k in ['gt_points','pred_points','identity_true_positive','associated_gt_points']})
        counts.update(scene);scenes.append(dict(scene=name,counts=dict(scene),opportunities=details,visual_identity=ident))
    passed=counts['correctable_foreground_points']>=30 and len(objects)>=5 and len(logs)>=2
    denominator=max(1,identity_totals['gt_points']+identity_totals['pred_points'])
    baseline=2*identity_totals['identity_true_positive']/denominator
    upper=2*identity_totals['associated_gt_points']/denominator
    return dict(counts=dict(counts),distinct_gt_instances=len(objects),distinct_logs=len(logs),data_gate_passed=passed,
        identity_availability=dict(**identity_totals,visual_point_idf1=baseline,
            optimistic_fixed_coverage_upper_bound=upper,max_gain_pp=100*(upper-baseline),
            original_1pp_identity_gate_reachable=100*(upper-baseline)>=1.),
        scenes=scenes,scope='GT-filtered availability upper bound on added development only; not policy accuracy or an identity/semantic improvement')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['cache','run','index','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.out.exists():raise ValueError('Refuse to overwrite diagnostic')
    r=diagnose(a.cache,a.run,read(a.index));save(a.out,r)
    print({k:v for k,v in r.items() if k!='scenes'})
