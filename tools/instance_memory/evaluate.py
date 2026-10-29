"""Paired, same-forward causal evaluation; no labels supplied to Runtime."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import read,save
from .memory import ObjectMemory,Config
from .metrics import IdentityMetrics,identity_truth,observation_target
from .model import MemoryHead
from .runtime import Runtime
from tools.semantic_memory.pilot_metrics import confusion,scores


def load_scene(cache,name):
    return read(cache/(name+'.inputs.json.gz')),read(cache/(name+'.targets.json.gz'))


def checkpoint(path):
    data=torch.load(path,map_location='cpu',weights_only=False)
    model=MemoryHead(data['mean'],data['scale']);model.load_state_dict(data['model']);model.eval()
    identity=MemoryHead(data['mean'],data['scale']);identity.load_state_dict(data['identity']);identity.eval()
    return model,identity,data


@torch.no_grad()
def evaluate(current,temporal,identity,scenes,*,full=True):
    arm_models={'matched_current':(current,'matched_current'),
                'feature_memory':(temporal,'feature_memory')}
    if full:
        arm_models.update(uniform_history=(current,'uniform_history'),
            same_memory_head_without_history=(temporal,'same_memory_head_without_history'),
            short_identity_memory=(temporal,'feature_memory'))
    totals={m:np.zeros((16,16),np.int64) for m in ['raw','existing_instance_pool',*arm_models]}
    counts={m:Counter() for m in arm_models}
    timings={m:[] for m in arm_models}
    ce={m:[0.,0] for m in arm_models}
    scene_results=[]
    for data,targets in scenes:
        runtime={m:Runtime(model,identity,mode) for m,(model,mode) in arm_models.items()}
        if 'short_identity_memory' in runtime:
            runtime['short_identity_memory'].memory=ObjectMemory(Config(short_records=1,long_records=0,ttl_s=.75))
        identities={m:IdentityMetrics() for m in ['visual_tracker',*runtime]}
        matrices={m:np.zeros((16,16),np.int64) for m in totals}
        scene_counts={m:Counter() for m in runtime}
        visual_ids={};details=[]
        for frame,target in zip(data['frames'],targets['frames']):
            gt=np.asarray(target['gt'],np.int64);valid=gt>=0;raw=np.asarray(target['raw_labels'],np.int64)
            matrices['raw']+=np.asarray(target['raw_confusion'])
            matrices['existing_instance_pool']+=np.asarray(target['single_confusion'])
            ids_gt=identity_truth(target)
            identity_valid=valid&(np.asarray(target['panoptic'])>0)
            baseline_id=np.zeros(len(gt),np.int64)
            obs=frame['observations']
            for o in obs:
                key=(o['visual_session'],o['visual_track'])
                if key not in visual_ids:visual_ids[key]=len(visual_ids)+1
                baseline_id[o['indices']]=visual_ids[key]
            identities['visual_tracker'].add(ids_gt,baseline_id,frame['frame'],identity_valid)
            frame_details={}
            for arm,rt in runtime.items():
                outputs,ids,info=rt.step(frame)
                pred=raw.copy();pred_id=np.zeros(len(raw),np.int64)
                observed_correct=observed_raw=observed_n=foreground_correct=foreground_n=0
                for o,logits,tid in zip(obs,outputs,ids):
                    at=np.asarray(o['indices']);lab=logits.argmax(-1).cpu().numpy()
                    pred[at]=lab;pred_id[at]=tid
                    y=gt[at];ok=y>=0
                    observed_correct+=int((lab[ok]==y[ok]).sum())
                    observed_raw+=int((raw[at][ok]==y[ok]).sum());observed_n+=int(ok.sum())
                    foreground=ok&(y<10)  # native16 thing classes, including barrier/cone
                    foreground_correct+=int((lab[foreground]==y[foreground]).sum())
                    foreground_n+=int(foreground.sum())
                    if ok.any():
                        ce[arm][0]+=float(F.cross_entropy(logits[ok],torch.as_tensor(y[ok]),reduction='sum'))
                        ce[arm][1]+=int(ok.sum())
                matrices[arm]+=confusion(gt,pred)
                identities[arm].add(ids_gt,pred_id,frame['frame'],identity_valid)
                c=dict(rescued=int((valid&(raw!=gt)&(pred==gt)).sum()),
                    harmed=int((valid&(raw==gt)&(pred!=gt)).sum()),
                    observed_correct=observed_correct,observed_points=observed_n,raw_observed_correct=observed_raw,
                    foreground_correct=foreground_correct,foreground_points=foreground_n,
                    history_observations=info['history_observations'],observations=len(obs))
                scene_counts[arm].update(c);timings[arm].append(info['elapsed_ms'])
                frame_details[arm]=dict(**c,ids=ids,tracks=info['tracks'],records=info['records'])
            if full:details.append(dict(frame=frame['frame'],arms=frame_details))
        for m in totals:totals[m]+=matrices[m]
        for m in runtime:counts[m].update(scene_counts[m])
        scene_results.append(dict(scene=data['scene'],arms={m:dict(scores(v),counts=dict(scene_counts.get(m,{}))) for m,v in matrices.items()},
            identity={m:v.result() for m,v in identities.items()},frames=details))
    result={}
    for arm,mat in totals.items():
        c=counts.get(arm,{})
        result[arm]=dict(scores(mat),counts=dict(c),
            observed_accuracy=c.get('observed_correct',0)/max(1,c.get('observed_points',0)),
            foreground_accuracy=c.get('foreground_correct',0)/max(1,c.get('foreground_points',0)) if arm in counts else None,
            point_ce=ce[arm][0]/max(1,ce[arm][1]) if arm in ce else None,
            timing=dict(median_ms=float(np.median(timings[arm])),p95_ms=float(np.percentile(timings[arm],95)),
                        max_ms=float(max(timings[arm]))) if arm in timings and timings[arm] else None)
    idarms=['visual_tracker',*runtime] if scene_results else []
    identity={}
    for arm in idarms:
        rows=[s['identity'][arm] for s in scene_results]
        total={k:sum(r[k] for r in rows) for k in ['gt_points','pred_points','identity_true_positive','gt_instances','associated_gt_points','id_switches','supported_transitions','reappearance_transitions']}
        total['point_idf1']=2*total['identity_true_positive']/max(1,total['gt_points']+total['pred_points'])
        total['association_quality']=sum(r['association_quality']*r['gt_instances'] for r in rows)/max(1,total['gt_instances'])
        identity[arm]=total
    return dict(aggregate=result,identity=identity,scenes=scene_results,
                scope='Project heldout if declared by caller; native16, raw 2Hz keyframes, supplied ego poses; cached module timing is not end-to-end latency')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=['dev','test'],default='test')
    a=p.parse_args();torch.set_num_threads(1)
    if a.out.exists():raise ValueError('Refuse to overwrite scored evaluation')
    current,identity,meta=checkpoint(a.run/'current.pt')
    temporal,_,tmeta=checkpoint(a.run/'temporal.pt')
    protocol=read(a.cache/'protocol.json')
    if meta['cache_protocol_sha256']!=__import__('hashlib').sha256((a.cache/'protocol.json').read_bytes()).hexdigest():raise ValueError('Cache provenance mismatch')
    scenes=[load_scene(a.cache,n) for n in protocol['parts'][a.split]]
    result=evaluate(current,temporal,identity,scenes)
    result.update(split=a.split,seed=meta['seed'],selected_epochs=dict(current=meta['epoch'],temporal=tmeta['epoch']))
    save(a.out,result)
    print({k:dict(miou=100*v['miou_fixed16'],accuracy=v['observed_accuracy']) for k,v in result['aggregate'].items()})
