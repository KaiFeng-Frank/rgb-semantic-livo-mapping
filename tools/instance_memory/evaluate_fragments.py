"""Shared-forward, frozen-head development evaluation of evidence admission.

The candidate-union foreground domain is fixed before any policy acts. The
camera identity domain is geometric, not conditional on a successful detection.
Full-scan scores are retained. No test split or training is implemented here.
"""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from .data import read,save
from .evaluate import checkpoint,load_scene
from .fragments import FragmentRuntime
from .runtime import Runtime
from .metrics import identity_truth,IdentityMetrics
from tools.semantic_memory.pilot_metrics import confusion,scores


def configurations(current,temporal,identity):
    return dict(current_trusted=(Runtime(current,identity,'matched_current'),True),
        current_all_fragments=(Runtime(current,identity,'matched_current'),False),
        current_confirmed=(FragmentRuntime(current,identity,'matched_current'),False),
        current_geometry_confirmed=(FragmentRuntime(current,identity,'matched_current',appearance=False),False),
        temporal_trusted=(Runtime(temporal,identity,'feature_memory'),True),
        temporal_confirmed=(FragmentRuntime(temporal,identity,'feature_memory'),False),
        temporal_confirmed_without_history=(FragmentRuntime(temporal,identity,'same_memory_head_without_history'),False))


@torch.no_grad()
def evaluate(cache,run):
    current,identity,metadata=checkpoint(run/'current.pt');temporal,_,_=checkpoint(run/'temporal.pt')
    protocol=read(cache/'protocol.json')
    if not protocol['candidate_fragments']:raise ValueError('Missing candidate export')
    if any((cache/(s+'.inputs.json.gz')).exists() for s in protocol['parts']['test']):
        raise ValueError('Development-only runner refuses an unsealed test cache')
    arms=list(configurations(current,temporal,identity));matrices={a:np.zeros((16,16),np.int64) for a in ['raw',*arms]}
    totals={a:Counter() for a in ['raw',*arms]};latencies={a:[] for a in arms}
    id_totals={domain:{a:Counter() for a in ['visual_tracker_trusted','visual_tracker_all_fragments',*arms]} for domain in ['full','camera']}
    scenes=[]
    for name in protocol['parts']['dev']:
        data,targets=load_scene(cache,name);runtimes=configurations(current,temporal,identity)
        ids={domain:{a:IdentityMetrics() for a in id_totals[domain]} for domain in id_totals}
        scene_mats={a:np.zeros((16,16),np.int64) for a in matrices};scene_counts={a:Counter() for a in matrices}
        visual_ids={};frame_rows=[]
        correction_objects={a:Counter() for a in arms}
        for frame,target in zip(data['frames'],targets['frames']):
            gt=np.asarray(target['gt']);raw=np.asarray(target['raw_labels']);pan=np.asarray(target['panoptic']);truth=identity_truth(target)
            valid=gt>=0;identity_valid=valid&(pan>0);camera=np.zeros(len(gt),bool);camera[target['camera_in_view_indices']]=True
            masks=dict(full=identity_valid,camera=identity_valid&camera)
            obs=frame['observations'];union=np.zeros(len(gt),bool);weak_union=union.copy()
            visual_all=np.zeros(len(gt),np.int64);visual_trusted=visual_all.copy()
            for o in obs:
                at=np.asarray(o['indices']);union[at]=True
                key=(o['visual_session'],o['visual_track']);visual_ids.setdefault(key,len(visual_ids)+1)
                visual_all[at]=visual_ids[key]
                if o['trusted']:visual_trusted[at]=visual_ids[key]
                else:weak_union[at]=True
            domain=valid&(gt<10)&union
            for label,pred_id in [('visual_tracker_trusted',visual_trusted),('visual_tracker_all_fragments',visual_all)]:
                for scope,mask in masks.items():ids[scope][label].add(truth,pred_id,frame['frame'],mask)
            predictions={'raw':raw};details={}
            for arm,(rt,trusted_only) in runtimes.items():
                active=[o for o in obs if o['trusted']] if trusted_only else obs
                logits,track_ids,info=rt.step(dict(frame,observations=active))
                pred=raw.copy();pred_id=np.zeros(len(gt),np.int64)
                for o,p,tid in zip(active,logits,track_ids):
                    # A rejected candidate leaves both original confidence and
                    # semantics alone, matching the eventual online contract.
                    if tid<=0:continue
                    at=np.asarray(o['indices']);pred[at]=p.argmax(-1).numpy();pred_id[at]=tid
                predictions[arm]=pred
                for scope,mask in masks.items():ids[scope][arm].add(truth,pred_id,frame['frame'],mask)
                extra=dict(observations=len(active),admitted_fragments=info.get('admitted_fragments',0),
                    admitted_fragment_points=int((weak_union&(pred_id>0)).sum()),
                    admitted_fragment_gt_points=int((weak_union&(pred_id>0)&(truth>0)).sum()),
                    history_observations=info['history_observations'])
                scene_counts[arm].update(extra);latencies[arm].append(info['elapsed_ms'])
                details[arm]=dict(**extra,ids=track_ids,candidate_reasons=info.get('candidate_reasons',{}))
            reference=predictions['current_trusted']
            for arm,pred in predictions.items():
                c=dict(foreground_points=int(domain.sum()),foreground_correct=int((domain&(pred==gt)).sum()),
                    raw_repaired=int((valid&(raw!=gt)&(pred==gt)).sum()),raw_harmed=int((valid&(raw==gt)&(pred!=gt)).sum()),
                    reference_repaired=int((valid&(reference!=gt)&(pred==gt)).sum()),
                    reference_harmed=int((valid&(reference==gt)&(pred!=gt)).sum()))
                scene_counts[arm].update(c);scene_mats[arm]+=confusion(gt,pred)
                if arm in correction_objects:
                    for sign,mask in [(1,(truth>0)&(reference!=gt)&(pred==gt)),(-1,(truth>0)&(reference==gt)&(pred!=gt))]:
                        for g,n in zip(*np.unique(truth[mask],return_counts=True)):
                            correction_objects[arm][int(g)]+=sign*int(n)
            frame_rows.append(dict(frame=frame['frame'],arms=details))
        for arm in matrices:matrices[arm]+=scene_mats[arm];totals[arm].update(scene_counts[arm])
        identity_results={scope:{a:m.result() for a,m in rows.items()} for scope,rows in ids.items()}
        for scope,rows in identity_results.items():
            for arm,v in rows.items():
                id_totals[scope][arm].update({k:v[k] for k in ['gt_points','pred_points','identity_true_positive','associated_gt_points','gt_instances','id_switches','supported_transitions','reappearance_transitions']})
                id_totals[scope][arm]['aq_numerator']+=v['association_quality']*v['gt_instances']
        scenes.append(dict(scene=name,arms={a:dict(scores(m),counts=dict(scene_counts[a])) for a,m in scene_mats.items()},
            identity=identity_results,objects_with_net_corrections={a:dict(c) for a,c in correction_objects.items()},frames=frame_rows))
    aggregate={a:dict(scores(m),counts=dict(totals[a]),foreground_accuracy=totals[a]['foreground_correct']/max(1,totals[a]['foreground_points']),
        timing=dict(p95_ms=float(np.percentile(latencies[a],95)),median_ms=float(np.median(latencies[a])),max_ms=max(latencies[a])) if a in latencies else None) for a,m in matrices.items()}
    identity={}
    for scope,rows in id_totals.items():
        identity[scope]={a:dict(c,point_idf1=2*c['identity_true_positive']/max(1,c['gt_points']+c['pred_points']),
            association_quality=c['aq_numerator']/max(1,c['gt_instances'])) for a,c in rows.items()}
    return dict(seed=metadata['seed'],aggregate=aggregate,identity=identity,scenes=scenes,
        scope='Frozen-head development-only admission experiment; no new training/test access; camera domain fixed by calibration; candidate-union foreground domain identical across arms')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['cache','run','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.out.exists():raise ValueError('Refuse overwrite')
    result=evaluate(a.cache,a.run);save(a.out,result)
    print({k:dict(miou=100*v['miou_fixed16'],foreground_accuracy=100*v['foreground_accuracy'],counts=v['counts']) for k,v in result['aggregate'].items()})
