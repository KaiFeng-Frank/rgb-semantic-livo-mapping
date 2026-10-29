"""Streaming audit of the existing feature-repair supervision on fixed training logs."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from tools.instance_memory.data import read,save
from tools.instance_memory.metrics import identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest,LABEL_MAP
from tools.semantic_memory.pilot_metrics import confusion,scores


def voxel_truth(inverse, gt, identity, n):
    """Only voxels whose nonignored points share one nonzero ID get an ID."""
    valid=gt>=0
    counts=np.bincount(inverse[valid]*16+gt[valid],minlength=n*16).reshape(n,16)
    labels=np.where(counts.sum(1)>0,counts.argmax(1),-1)
    low=np.full(n,np.iinfo(np.int64).max,np.int64);high=np.full(n,-1,np.int64)
    np.minimum.at(low,inverse[valid],identity[valid]);np.maximum.at(high,inverse[valid],identity[valid])
    ids=np.where((low==high)&(low>0),low,0)
    return labels,ids


def object_rows(gt,identity,predictions):
    result={}
    for obj in np.unique(identity[identity>0]):
        at=np.flatnonzero((identity==obj)&(gt>=0))
        if len(at)<5:continue
        labels,count=np.unique(gt[at],return_counts=True)
        if len(labels)!=1:raise ValueError('GT panoptic object spans semantic classes')
        result[int(obj)]=dict(label=int(labels[0]),points=len(at),
            success={a:bool(np.count_nonzero(p[at]==gt[at])*2>=len(at)) for a,p in predictions.items()})
    return result


def run(index_path,sealed_index,cache,anchors,models,out):
    from .repair_pilot import anchor_model,call
    from .nuscenes_pilot import current_output
    from .chain_trace import TracedEvidence,FusionCapture
    from tools.visual_instances.kalman_cache import gpu_admission
    import torch
    from torch.nn import functional as F
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from ptv3_worker import Segmenter
    from ptv3_loader_verified import serialization_policy
    from semantic_repair import CurrentAnchoredRepair,RepairConfig
    from semantic_memory import MemoryConfig
    torch.set_num_threads(1);torch.manual_seed(20261029)
    index=read(index_path);names=index['parts']['train'];sealed=read(sealed_index)
    if len(names)!=16 or sum(len(index['sequences'][s]) for s in names)!=642:raise ValueError('Frozen training pool changed')
    train_logs={index['scene_logs'][s] for s in names}
    if train_logs&{sealed['scene_logs'][s] for s in sealed['parts']['test']}:raise ValueError('Collection-log leakage')
    if any((cache/(s+'.targets.json.gz')).exists() or (cache/(s+'.inputs.json.gz')).exists() for s in sealed['parts']['test']):
        raise ValueError('New holdout already extracted in this cache')
    if shutil.disk_usage(out.parent).free<4*1024**3:raise RuntimeError('Less than 4 GiB headroom')
    admission=gpu_admission();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2];cfg_path=root/'config/feature_supervision_audit_20261029.json'
    anchor_path=anchors/'seed-0/current_only.pt';model_path=models/'seed-0/temporal.pt'
    checkpoint=torch.load(model_path,map_location='cpu',weights_only=True)
    if checkpoint['schema']!='current-anchored-repair-v2' or checkpoint['provenance']['kind']!='temporal':raise ValueError('Unexpected repair checkpoint')
    anchor=anchor_model(anchor_path)
    def create():
        m=CurrentAnchoredRepair(MemoryConfig(**checkpoint['config']),RepairConfig(**checkpoint['repair']),device='cuda').eval()
        m.fusion.load_state_dict(checkpoint['fusion'],strict=True);return m
    temporal=create();plain=create();current=create()
    temporal.memory=TracedEvidence(temporal.cfg,'cuda',temporal.repair.moving_classes)
    capture=FusionCapture(temporal.fusion)
    backbone=Segmenter(half=True,shuffle=False,intensity_scale=1.)
    source=[Path(__file__),root/'src/semantic_repair.py',root/'src/semantic_memory.py',root/'tools/semantic_memory/chain_trace.py',root/'tools/semantic_memory/repair_pilot.py',root/'tools/semantic_memory/nuscenes_pilot.py',root/'src/ptv3_loader_verified.py',root/'src/ptv3_worker.py']
    save(out/'protocol.json',dict(config=read(cfg_path),config_sha256=digest(cfg_path),index_path=str(index_path),index_sha256=digest(index_path),
        sealed_index_path=str(sealed_index),sealed_index_sha256=digest(sealed_index),target_path=str(cache),
        targets_sha256={s:digest(cache/(s+'.targets.json.gz')) for s in names},
        anchor_path=str(anchor_path),anchor_sha256=digest(anchor_path),repair_path=str(model_path),repair_sha256=digest(model_path),
        backbone_sha256=digest(os.environ['PTV3_CHECKPOINT']),serialization=serialization_policy(backbone.model),
        sources={str(p.relative_to(root)):digest(p) for p in source},admission=admission))
    reports=[];started=time.perf_counter();global_classes=np.zeros(16,np.int64)
    arms=('raw','anchor','no_history','temporal')
    with torch.no_grad():
        for name in names:
            for m in (temporal,plain,current):m.reset()
            targets=read(cache/(name+'.targets.json.gz'))['frames'];rows=index['sequences'][name]
            if len(rows)!=len(targets):raise ValueError('Incomplete targets')
            origin=np.linalg.inv(np.asarray(rows[0]['transform']));previous=None
            totals=Counter();classes=defaultdict(Counter);instances=defaultdict(lambda:defaultdict(set))
            mats={a:np.zeros((16,16),np.int64) for a in arms};frames=[];object_events=[]
            loss_sums={g:np.zeros(16,np.float64) for g in ('repair','preserve','other')}
            for k,(row,target) in enumerate(zip(rows,targets)):
                scan_path=DATA/row['scan_path'];label_path=DATA/row['label_path']
                if digest(scan_path)!=row['scan_sha256'] or digest(label_path)!=row['label_sha256']:raise ValueError('Source changed')
                scan=np.fromfile(scan_path,dtype='<f4').reshape(-1,5)[:,:4].copy();scan[:,3]/=255.
                r=backbone.segment(scan,return_features=True)
                raw={a:r[a] for a in ('xyz','features','logits')}
                raw.update(transform=torch.tensor(origin@np.asarray(row['transform']),device='cuda',dtype=torch.float32),
                    stamp=row['timestamp_us']/1e6,frame=k)
                a=current_output(anchor,raw)['logits']
                normal=call(plain,raw,a);output=call(temporal,raw,a);nohist=call(current,raw,a,current_only=True)
                if not torch.equal(output['logits'],normal['logits']) or not torch.equal(output['query_indices'],normal['query_indices']):
                    raise AssertionError('Instrumented output changed')
                for field in ('xyz','features','probabilities','key'):
                    if not torch.equal(getattr(plain.memory,field),getattr(temporal.memory,field)):raise AssertionError('Raw memory state changed')
                f,qa,candidates=capture.value;query=output['query_indices'].cpu().numpy()
                inverse=r['inverse'].cpu().numpy();vp=r['logits'].argmax(1).cpu().numpy()
                vpred=dict(raw=vp,anchor=a.argmax(1).cpu().numpy(),no_history=nohist['logits'].argmax(1).cpu().numpy(),temporal=output['logits'].argmax(1).cpu().numpy())
                # No labels/IDs above enter any inference, memory write or query.
                gt=np.asarray(target['gt'],np.int64);identity=identity_truth(target)
                np.testing.assert_array_equal(gt,LABEL_MAP[np.fromfile(label_path,dtype=np.uint8)])
                labels,vid=voxel_truth(inverse,gt,identity,len(r['xyz']))
                if k:global_classes+=np.bincount(labels[labels>=0],minlength=16)
                # Both original learned arms skip the first scan for supervision.
                qgt=labels[query];qvalid=(qgt>=0)&(k>0);correct=vpred['anchor'][query]==qgt
                valid=candidates['valid'].cpu().numpy();hp=candidates['probabilities'].argmax(-1).cpu().numpy()
                predicted_good=(valid&(hp==qgt[:,None])&qvalid[:,None])
                strict=np.zeros_like(valid)
                if previous is not None and len(temporal.memory.trace['record_source']):
                    source=temporal.memory.trace['record_source'][temporal.memory.trace['shortlist_index']].cpu().numpy()
                    same_class=previous['labels'][source]==qgt[:,None]
                    same_instance=(vid[query,None]>0)&(vid[query,None]==previous['vid'][source])
                    known_stuff=qgt[:,None]>=10
                    strict=predicted_good&same_class&(known_stuff|same_instance)
                groups=dict(repair=qvalid&~correct&predicted_good.any(1),
                    preserve=qvalid&correct&(valid&(hp!=qgt[:,None])).any(1))
                groups['other']=qvalid&~(groups['repair']|groups['preserve'])
                events=dict(queries=qvalid,queries_with_history=qvalid&valid.any(1),anchor_wrong=qvalid&~correct,
                    repair_supported=groups['repair'],repair_strictly_supported=qvalid&~correct&strict.any(1),
                    preserve_conflict=groups['preserve'],other=groups['other'])
                frame_counts={}
                for event,mask in events.items():
                    frame_counts[event]=int(mask.sum());totals[event]+=int(mask.sum())
                    for c,n in zip(*np.unique(qgt[mask],return_counts=True)):classes[str(int(c))][event]+=int(n)
                    for obj in np.unique(vid[query][mask&(vid[query]>0)]):
                        at=np.flatnonzero(vid==obj);c=int(labels[at[0]])
                        instances[str(c)][event].add(int(obj))
                # Existing per-query objective, before train-derived class weights.
                q=torch.tensor(np.maximum(qgt,0),device='cuda',dtype=torch.long)
                logits=output['selected_logits'];mask=F.one_hot(q,16).bool()
                margin=logits.gather(1,q[:,None]).squeeze(1)-logits.masked_fill(mask,-1e4).max(1).values
                old_margin=qa.gather(1,q[:,None]).squeeze(1)-qa.masked_fill(mask,-1e4).max(1).values
                cc=torch.tensor(correct,device='cuda')
                loss=(F.cross_entropy(logits,q,reduction='none')+.5*(~cc)*F.relu(.5-margin)+.5*cc*F.relu(old_margin.clamp(max=2.)-margin)).cpu().numpy()
                for g,mask in groups.items():loss_sums[g]+=np.bincount(qgt[mask],weights=loss[mask],minlength=16)
                predictions={a:p[inverse] for a,p in vpred.items()}
                objects=object_rows(gt,identity,predictions)
                for a,pred in predictions.items():mats[a]+=confusion(gt,pred)
                supported=np.zeros(len(labels),bool);supported[query]=strict.any(1)
                for obj,obs in objects.items():
                    if previous is None or not 0<raw['stamp']-previous['stamp']<=temporal.cfg.max_age_s:continue
                    before=previous['objects'].get(obj)
                    if before is None or not before['success']['raw']:continue
                    at=(identity==obj)&(gt>=0)
                    item=dict(instance=obj,label=obs['label'],frame=k,points=obs['points'],
                        anchor_opportunity=not obs['success']['anchor'],no_history_opportunity=not obs['success']['no_history'],
                        temporal_success=obs['success']['temporal'],strict_support_wrong_points=int((at&(predictions['anchor']!=gt)&supported[inverse]).sum()))
                    if item['anchor_opportunity'] or item['no_history_opportunity']:object_events.append(item)
                frames.append(dict(frame=k,parity=True,counts=frame_counts,records=len(temporal.memory.xyz)))
                previous=dict(labels=labels,vid=vid,objects=objects,stamp=raw['stamp'])
            report=dict(scene=name,log=index['scene_logs'][name],counts=dict(totals),per_class={c:dict(v) for c,v in classes.items()},
                instances={c:{e:sorted(v) for e,v in events.items()} for c,events in instances.items()},
                loss_sums_by_class={g:v.tolist() for g,v in loss_sums.items()},object_opportunities=object_events,
                metrics={a:scores(m) for a,m in mats.items()},frames=frames)
            save(out/(name+'.json'),report);reports.append(report)
            save(out/'progress.json',dict(completed_scenes=len(reports),scene=name,elapsed_s=time.perf_counter()-started))
            print(name,dict(totals),'object opportunities',len(object_events),flush=True)
    counts=Counter();cls=defaultdict(Counter);unique=defaultdict(lambda:defaultdict(set))
    for r in reports:
        counts.update(r['counts'])
        for c,v in r['per_class'].items():cls[c].update(v)
        for c,events in r['instances'].items():
            for event,ids in events.items():unique[c][event].update((r['scene'],obj) for obj in ids)
    inv=np.sqrt(global_classes.sum()/np.maximum(global_classes,1));weights=np.clip(inv/inv.mean(),.25,4.)
    weighted={g:float(sum(np.asarray(r['loss_sums_by_class'][g])@weights for r in reports)) for g in ('repair','preserve','other')}
    result=dict(scenes=names,frames=sum(len(r['frames']) for r in reports),supervised_frames=sum(len(r['frames'])-1 for r in reports),counts=dict(counts),per_class={c:dict(v) for c,v in cls.items()},
        unique_instances={c:{e:len(v) for e,v in events.items()} for c,events in unique.items()},
        train_voxel_counts=global_classes.tolist(),class_weights=weights.tolist(),weighted_loss_mass=weighted,
        weighted_loss_fraction={g:v/sum(weighted.values()) for g,v in weighted.items()},
        object_opportunities=[dict(scene=r['scene'],**e) for r in reports for e in r['object_opportunities']],
        all_642_parity_passed=all(f['parity'] for r in reports for f in r['frames']),elapsed_s=time.perf_counter()-started,
        scope='Training-pool supervision diagnosis with frozen legacy heads under the current all-off feature policy. No optimizer, no new evaluation/holdout admission; loss mass is not gradient contribution. Instance counts use panoptic rather than legacy box grouping.')
    save(out/'report.json',result);save(out/'progress.json',dict(status='complete',frames=result['frames'],elapsed_s=result['elapsed_s']))
    capture.close();print('aggregate',dict(counts),'loss fractions',result['weighted_loss_fraction'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('index','sealed-index','cache','anchors','models','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.index,a.sealed_index,a.cache,a.anchors,a.models,a.out)
