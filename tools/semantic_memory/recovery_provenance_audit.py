"""Frozen-head diagnostic: association, evidence correctness, and anchor blindness."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import gzip,os,sys,time
import numpy as np
from tools.instance_memory.data import read,save
from tools.instance_memory.metrics import identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest,LABEL_MAP
from .feature_supervision_audit import voxel_truth
from .pilot_metrics import confusion,scores,events
from .diagnose_repair_supervision_result import episodes


def provenance_masks(qgt,qid,oldgt,oldid,source,valid,hp):
    """Conservative offline identity join; unknown identities never imply a match."""
    if source is None:
        zero=np.zeros_like(valid)
        return dict(same=zero,correct=zero,unknown=zero)
    label=oldgt[source];identity=oldid[source]
    same=valid&(qgt[:,None]>=0)&(label==qgt[:,None])&((qgt[:,None]>=10)|((qid[:,None]>0)&(qid[:,None]==identity)))
    unknown=valid&((qgt[:,None]<0)|(label<0)|((qgt[:,None]<10)&(qid[:,None]==0))|((label<10)&(identity==0)))
    return dict(same=same,correct=same&(hp==qgt[:,None]),unknown=unknown)


def run(index,targets,models,anchor_path,out):
    from .repair_pilot import anchor_model,call
    from .nuscenes_pilot import current_output
    from .chain_trace import TracedEvidence,FusionCapture
    from tools.visual_instances.kalman_cache import gpu_admission
    import torch
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'src'))
    from semantic_memory import MemoryConfig
    from semantic_repair import CurrentAnchoredRepair,RepairConfig,JointRepair
    from ptv3_worker import Segmenter
    from ptv3_loader_verified import serialization_policy
    torch.set_num_threads(1);torch.manual_seed(20261029)
    idx=read(index);names=idx['independent_dev_added']
    if len(names)!=8 or sum(len(idx['sequences'][s]) for s in names)!=321:raise ValueError('Fixed development pool changed')
    if {idx['scene_logs'][s] for s in names}&{idx['scene_logs'][s] for s in idx['parts']['test']}:raise ValueError('Log leakage')
    for s in idx['parts']['test']:
        if (targets/(s+'.targets.json.gz')).exists():raise ValueError('New holdout was opened')
    admission=gpu_admission();out.mkdir(parents=True,exist_ok=False)
    ck=[torch.load(models/f'seed-{s}/balanced_temporal.pt',map_location='cpu',weights_only=True) for s in range(3)]
    cfg=MemoryConfig(**ck[0]['config']);repair=RepairConfig(**ck[0]['repair'])
    def core():
        m=CurrentAnchoredRepair(cfg,repair,device='cuda').eval();m.fusion.load_state_dict(ck[0]['fusion'],strict=True);return m
    native,traced,current=core(),core(),core()
    traced.memory=TracedEvidence(cfg,'cuda',repair.moving_classes);capture=FusionCapture(traced.fusion)
    heads=[]
    for c in ck:
        m=JointRepair(cfg,repair).cuda().eval();m.load_state_dict(c['fusion'],strict=True);heads.append(m)
    anchor=anchor_model(anchor_path);backbone=Segmenter(half=True,shuffle=False,intensity_scale=1.)
    config=root/'config/feature_repair_recovery_audit_20261029.json'
    source=[Path(__file__),root/'src/semantic_repair.py',root/'src/semantic_memory.py',root/'tools/semantic_memory/chain_trace.py',root/'tools/semantic_memory/feature_supervision_audit.py',root/'tools/semantic_memory/repair_pilot.py',root/'tools/semantic_memory/nuscenes_pilot.py',root/'src/ptv3_loader_verified.py',root/'src/ptv3_worker.py']
    save(out/'protocol.json',dict(config=read(config),config_sha256=digest(config),index_path=str(index),index_sha256=digest(index),
        targets_sha256={s:digest(targets/(s+'.targets.json.gz')) for s in names},anchor_path=str(anchor_path),anchor_sha256=digest(anchor_path),
        checkpoints_sha256={str(s):digest(models/f'seed-{s}/balanced_temporal.pt') for s in range(3)},
        backbone_sha256=digest(os.environ['PTV3_CHECKPOINT']),serialization=serialization_policy(backbone.model),admission=admission,
        sources={str(p.relative_to(root)):digest(p) for p in source}))
    modes=('temporal','no_history','oracle_association','oracle_correct')
    arms=('baseline','anchor')+tuple(f's{s}_{m}' for s in range(3) for m in modes)
    mats={a:np.zeros((16,16),np.int64) for a in arms};fg={a:Counter() for a in arms}
    counts={str(s):Counter() for s in range(3)};object_observations=[];scene_reports=[];start=time.perf_counter();parity=0
    with torch.no_grad():
        for scene in names:
            for m in (native,traced,current):m.reset()
            previous=None;rows=idx['sequences'][scene];sidecars=read(targets/(scene+'.targets.json.gz'))['frames']
            if len(rows)!=len(sidecars):raise ValueError('Incomplete target frames')
            origin=np.linalg.inv(np.asarray(rows[0]['transform']));sm={a:np.zeros((16,16),np.int64) for a in arms}
            for k,(row,target) in enumerate(zip(rows,sidecars)):
                path=DATA/row['scan_path'];lp=DATA/row['label_path']
                if digest(path)!=row['scan_sha256'] or digest(lp)!=row['label_sha256']:raise ValueError('Source changed')
                scan=np.fromfile(path,dtype='<f4').reshape(-1,5)[:,:4].copy();scan[:,3]/=255.
                r=backbone.segment(scan,return_features=True)
                raw={key:r[key] for key in ('xyz','features','logits')}
                raw.update(transform=torch.tensor(origin@np.asarray(row['transform']),device='cuda',dtype=torch.float32),stamp=row['timestamp_us']/1e6,frame=k)
                a=current_output(anchor,raw)['logits'];plain=call(native,raw,a);o=call(traced,raw,a);cur=call(current,raw,a,current_only=True)
                if not torch.equal(o['logits'],plain['logits']) or not torch.equal(o['query_indices'],plain['query_indices']):raise AssertionError('Instrumented output changed')
                for field in ('xyz','features','probabilities','key'):
                    if not torch.equal(getattr(native.memory,field),getattr(traced.memory,field)):raise AssertionError('Instrumented raw state changed')
                f,qa,c=capture.value;n=len(f)
                cc=dict(features=f[:,None],probabilities=qa.softmax(-1)[:,None],delta=f.new_zeros((n,1,3)),age=f.new_zeros((n,1)),valid=torch.zeros((n,1),device='cuda',dtype=torch.bool))
                selected={f's{s}_{mode}':head(f,qa,c if mode=='temporal' else cc) for s,head in enumerate(heads) for mode in ('temporal','no_history')}
                if not torch.equal(selected['s0_temporal'],plain['selected_logits']) or not torch.equal(selected['s0_no_history'],cur['selected_logits']):raise AssertionError('Shared inference differs from native')
                parity+=1
                # Normal inference and writes are complete. Only diagnostic copies
                # below consume labels or GT identities; they cannot change memory.
                gt=np.asarray(target['gt'],np.int64);ids=identity_truth(target);inverse=r['inverse'].cpu().numpy()
                np.testing.assert_array_equal(gt,LABEL_MAP[np.fromfile(lp,dtype=np.uint8)])
                labels,vid=voxel_truth(inverse,gt,ids,len(r['xyz']));query=o['query_indices'].cpu().numpy();qgt=labels[query]
                valid=c['valid'].cpu().numpy();hp=c['probabilities'].argmax(-1).cpu().numpy()
                source=None
                if previous is not None and len(traced.memory.trace['record_source']):source=traced.memory.trace['record_source'][traced.memory.trace['shortlist_index']].cpu().numpy()
                tags=provenance_masks(qgt,vid[query],None if previous is None else previous['labels'],None if previous is None else previous['vid'],source,valid,hp)
                for mode,tag in [('oracle_association','same'),('oracle_correct','correct')]:
                    altered=dict(c);altered['valid']=torch.tensor(tags[tag],device='cuda')
                    for s,head in enumerate(heads):selected[f's{s}_{mode}']=head(f,qa,altered)
                vpred=dict(baseline=r['logits'].argmax(1).cpu().numpy(),anchor=a.argmax(1).cpu().numpy())
                for key,value in selected.items():
                    p=vpred['anchor'].copy();p[query]=value.argmax(1).cpu().numpy();vpred[key]=p
                pred={a:p[inverse] for a,p in vpred.items()};good=gt>=0
                camera=np.zeros(len(gt),bool);camera[target['camera_in_view_indices']]=True;foreground=good&camera&(gt<10)
                for arm,p in pred.items():
                    mat=confusion(gt,p);sm[arm]+=mat;mats[arm]+=mat
                    fg[arm].update(points=int(foreground.sum()),correct=int((foreground&(p==gt)).sum()))
                def expand(mask):
                    v=np.zeros(len(labels),bool);v[query]=mask;return v[inverse]
                present=expand(valid.any(1));all_same=expand(valid.any(1)&(~valid|tags['same']).all(1))
                all_correct=expand(valid.any(1)&(~valid|tags['correct']).all(1));any_correct=expand(tags['correct'].any(1));unknown=expand(tags['unknown'].any(1))
                # A majority-voxel target must not certify evidence for an
                # individual point of a different class inside that voxel.
                agrees=labels[inverse]==gt
                all_same&=agrees;all_correct&=agrees;any_correct&=agrees
                flags={}
                for s in range(3):
                    t=pred[f's{s}_temporal'];ch=pred[f's{s}_no_history'];harm=good&(ch==gt)&(t!=gt);fix=good&(ch!=gt)&(t==gt)
                    flags[str(s)]=dict(harm=harm,fix=fix,harm_anchor_wrong=harm&(pred['anchor']!=gt),harm_no_history=harm&~present,
                        harm_all_same=harm&all_same,harm_all_correct=harm&all_correct,harm_any_correct=harm&any_correct,harm_unknown=harm&unknown,
                        harm_query_majority_mismatch=harm&~agrees,
                        harm_resolved_association=harm&(pred[f's{s}_oracle_association']==gt),harm_resolved_correct=harm&(pred[f's{s}_oracle_correct']==gt))
                    for key,mask in flags[str(s)].items():counts[str(s)][key]+=int(mask.sum())
                for obj in np.unique(ids[ids>0]):
                    at=(ids==obj)&good;number=int(at.sum())
                    if number<5:continue
                    cls=np.unique(gt[at])
                    if len(cls)!=1:raise ValueError('Mixed panoptic classes')
                    object_observations.append(dict(scene=scene,instance=int(obj),label=int(cls[0]),frame=k,stamp=raw['stamp'],points=number,
                        correct_fraction={arm:float((p[at]==gt[at]).mean()) for arm,p in pred.items()},
                        provenance={s:{key:int((mask&at).sum()) for key,mask in ff.items()} for s,ff in flags.items()}))
                previous=dict(labels=labels,vid=vid)
            sr=dict(scene=scene,frames=len(rows),metrics={a:scores(m) for a,m in sm.items()})
            save(out/(scene+'.json'),sr);scene_reports.append(sr)
            save(out/'progress.json',dict(stage='replay',scene=scene,parity_frames=parity,elapsed_s=time.perf_counter()-start));print('completed',scene,parity,flush=True)
    capture.close()
    event_metrics=events(object_observations,arms);lost={str(s):[] for s in range(3)}
    for ep in episodes(object_observations):
        for s in range(3):
            current_success=any(x['correct_fraction'][f's{s}_no_history']>=.5 for x in ep)
            temporal_success=any(x['correct_fraction'][f's{s}_temporal']>=.5 for x in ep)
            if current_success and not temporal_success:
                lost[str(s)].append(dict(scene=ep[0]['scene'],instance=ep[0]['instance'],label=ep[0]['label'],start=ep[0]['frame'],end=ep[-1]['frame'],
                    oracle_success={m:any(x['correct_fraction'][f's{s}_{m}']>=.5 for x in ep) for m in ('oracle_association','oracle_correct')},
                    anchor_success=any(x['correct_fraction']['anchor']>=.5 for x in ep),
                    current_success_frames=[dict(frame=x['frame'],points=x['points'],provenance=x['provenance'][str(s)]) for x in ep if x['correct_fraction'][f's{s}_no_history']>=.5]))
    if parity!=321:raise AssertionError('Incomplete parity')
    summary={s:dict(lost_episodes=len(v),association_resolved=sum(x['oracle_success']['oracle_association'] for x in v),
        correct_evidence_resolved=sum(x['oracle_success']['oracle_correct'] for x in v),anchor_never_succeeds=sum(not x['anchor_success'] for x in v)) for s,v in lost.items()}
    report=dict(metrics={a:dict(scores(m),camera_foreground=dict(fg[a])) for a,m in mats.items()},events=event_metrics,
        point_provenance={s:dict(v) for s,v in counts.items()},lost_recovery_episodes=lost,recovery_summary=summary,
        parity_frames=parity,elapsed_s=time.perf_counter()-start,scope='Fresh same-forward development diagnosis; GT filtering is not deployable, no model or training change.')
    save(out/'report.json',report);save(out/'progress.json',dict(stage='complete',parity_frames=parity,elapsed_s=report['elapsed_s']))
    import json
    with gzip.open(out/'object-observations.json.gz','wt') as f:json.dump(object_observations,f,separators=(',',':'))
    print('RECOVERY',summary,'POINTS',report['point_provenance'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('index','targets','models','anchor-path','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.index,a.targets,a.models,a.anchor_path,a.out)
