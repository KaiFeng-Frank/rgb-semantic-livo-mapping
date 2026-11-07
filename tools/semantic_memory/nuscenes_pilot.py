"""Real-label pilot with matched-supervision current-frame control and sealed scenes.

Uses raw keyframes with rigid ego alignment, NOT the deskewed ROS contract.
All learned checkpoints remain non-deployable. The test partition is scored
only after all epochs and dev-based checkpoint selection for all fixed seeds.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
from semantic_memory import CausalSemanticMemory, MemoryConfig, training_loss, checkpoint
from .train import AugmentedFusion
from .nuscenes_pilot_data import digest
from .pilot_metrics import confusion, scores, paired_summary, events

ARMS = ('baseline','uniform_history','current_only','temporal','ungated_temporal')


def save(path, obj):
    path.write_text(json.dumps(obj, separators=(',',':'), allow_nan=False)+'\n')


def announce(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def current_output(fusion, raw):
    """Same fusion parameterization, but only one CURRENT observation token."""
    f, b = raw['features'], raw['logits']
    n = len(f)
    candidates = dict(features=f[:,None], probabilities=b.float().softmax(-1)[:,None],
        delta=f.new_zeros((n,1,3)), age=f.new_zeros((n,1)),
        valid=torch.ones((n,1),device=f.device,dtype=torch.bool))
    return fusion(f, b, candidates)


class UniformFusion(nn.Module):
    def forward(self, features, logits, candidates):
        valid = candidates['valid']
        present = valid.any(1)
        p = logits.float().softmax(-1)
        hp = (candidates['probabilities'].float()*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        fused = torch.where(present[:,None], (.5*p+.5*hp).clamp_min(1e-8).log(), logits.float())
        return dict(logits=fused, candidate_logits=fused, base_logits=logits.float(),
                    gate=present.float()*.5, has_history=present)


def load(cache, scene, row):
    path = cache/scene/f"{row['frame']:03d}.npz"
    with np.load(path,allow_pickle=False) as file:
        arrays={k:file[k] for k in file.files}
    raw={k:torch.from_numpy(arrays[k]).to('cuda') for k in ('xyz','features','logits','transform','label')}
    raw['stamp'] = row['timestamp_us']/1e6
    raw['frame'] = row['frame']
    return raw, arrays


def forward(model, raw, epoch=0):
    return model(raw['xyz'],raw['features'],raw['logits'],stamp=raw['stamp'],
                 frame=raw['frame'],transform=raw['transform'],epoch=epoch)


def train_epoch(models, optimizers, index, cache, seed, epoch, clip=4):
    report={k:dict(loss=0.,frames=0,steps=0,history_points=0) for k in models}
    for model in models.values():
        model.train()
    # Deterministically shuffle only complete scenes, never frames within a scene.
    names=list(index['parts']['train'])
    np.random.default_rng(seed*100+epoch).shuffle(names)
    for scene in names:
        for model in models.values():
            model.reset()
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        losses={k:[] for k in models}
        rows=index['sequences'][scene]
        for j,row in enumerate(rows):
            raw,_=load(cache,scene,row)
            outputs=dict(temporal=forward(models['temporal'],raw,epoch),
                         current_only=current_output(models['current_only'].fusion,raw))
            # Identical supervised frames. The first scan initializes memory only.
            if j:
                for kind in models:
                    loss,detail=training_loss(outputs[kind],raw['label'])
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('Nonfinite training loss')
                    losses[kind].append(loss)
                    report[kind]['loss']+=float(loss.detach())
                    report[kind]['frames']+=1
                    report[kind]['history_points']+=detail['history_supervised']
            if j and (len(losses['temporal']) == clip or j == len(rows)-1):
                for kind in models:
                    torch.stack(losses[kind]).mean().backward()
                    nn.utils.clip_grad_norm_(models[kind].fusion.parameters(),5.,error_if_nonfinite=True)
                    optimizers[kind].step()
                    optimizers[kind].zero_grad(set_to_none=True)
                    report[kind]['steps']+=1
                    losses[kind].clear()
        for model in models.values():
            model.reset()
    for kind in report:
        report[kind]['loss']/=max(report[kind]['frames'],1)
    return report


@torch.no_grad()
def dev_scores(models,index,cache):
    matrices={k:np.zeros((16,16),np.int64) for k in models}
    for m in models.values():
        m.eval()
    for scene in index['parts']['dev']:
        for m in models.values():
            m.reset()
        for row in index['sequences'][scene]:
            raw,arrays=load(cache,scene,row)
            outputs=dict(temporal=forward(models['temporal'],raw),
                         current_only=current_output(models['current_only'].fusion,raw))
            for kind,out in outputs.items():
                prediction=out['logits'].argmax(-1).cpu().numpy()[arrays['inverse']]
                matrices[kind]+=confusion(arrays['point_label'].astype(np.int64),prediction)
    for m in models.values():
        m.reset()
    return {kind:scores(cm) for kind,cm in matrices.items()}


def unwrap(model):
    return model.fusion.fusion if isinstance(model.fusion,AugmentedFusion) else model.fusion


def run_training(seed,index,cache,out,cfg,provenance,epochs):
    torch.manual_seed(seed)
    models={k:CausalSemanticMemory(cfg,'cuda') for k in ('temporal','current_only')}
    # Same initialization of all corresponding parameters, then separate optimizers.
    models['current_only'].fusion.load_state_dict(models['temporal'].fusion.state_dict())
    models['temporal'].fusion=AugmentedFusion(models['temporal'].fusion,.15,.15,seed)
    optimizers={k:torch.optim.Adam(m.fusion.parameters(),lr=1e-3) for k,m in models.items()}
    best={k:-1. for k in models}
    folder=out/f'seed-{seed}'
    folder.mkdir(exist_ok=False)
    reports=[]
    for epoch in range(1,epochs+1):
        train=train_epoch(models,optimizers,index,cache,seed,epoch)
        dev=dev_scores(models,index,cache)
        report=dict(seed=seed,epoch=epoch,train=train,dev=dev)
        for kind,m in models.items():
            if dev[kind]['miou_fixed16'] > best[kind]:
                best[kind]=dev[kind]['miou_fixed16']
                p=dict(provenance,seed=seed,epoch=epoch,kind=kind,selected_on='dev only',
                       dev_miou=best[kind],scope='real_pilot',deployable=False)
                torch.save(checkpoint(cfg,unwrap(m),provenance=p,trained=True),folder/f'{kind}.pt')
        reports.append(report)
        save(folder/f'epoch-{epoch}.json',report)
        announce(stage='train',seed=seed,epoch=epoch,
                 loss={k:v['loss'] for k,v in train.items()},
                 dev_miou={k:round(v['miou_fixed16']*100,3) for k,v in dev.items()})
    del models,optimizers
    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(seed,index,cache,out,cfg):
    models={k:CausalSemanticMemory(cfg,'cuda').eval() for k in ('temporal','current_only')}
    checkpoints={}
    for kind,m in models.items():
        path=out/f'seed-{seed}'/f'{kind}.pt'
        saved=torch.load(path,map_location='cpu',weights_only=True)
        m.fusion.load_state_dict(saved['fusion'],strict=True)
        checkpoints[kind]=dict(sha256=digest(path),epoch=saved['provenance']['epoch'])
    uniform=CausalSemanticMemory(cfg,'cuda').eval()
    uniform.fusion=UniformFusion()
    global_cm={k:np.zeros((16,16),np.int64) for k in ARMS}
    pair={k:defaultdict(int) for k in ARMS}
    all_events,per_scene,per_frame=[],{},[]
    timing=defaultdict(list)
    for scene in index['parts']['test']:
        for m in (*models.values(),uniform):
            m.reset()
        matrices={k:np.zeros((16,16),np.int64) for k in ARMS}
        for row in index['sequences'][scene]:
            raw,arrays=load(cache,scene,row)
            torch.cuda.synchronize(); start=time.perf_counter()
            temporal=forward(models['temporal'],raw)
            torch.cuda.synchronize(); temporal_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter()
            current=current_output(models['current_only'].fusion,raw)
            torch.cuda.synchronize(); current_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter()
            unif=forward(uniform,raw)
            torch.cuda.synchronize(); uniform_ms=(time.perf_counter()-start)*1000
            for key,value in [('temporal',temporal_ms),('current_only',current_ms),('uniform_history',uniform_ms)]:
                timing[key].append(value)
            outputs=dict(baseline=raw['logits'],temporal=temporal['logits'],
                         current_only=current['logits'],uniform_history=unif['logits'],
                         ungated_temporal=temporal['candidate_logits'])
            predictions={k:v.argmax(-1).cpu().numpy()[arrays['inverse']] for k,v in outputs.items()}
            labels=arrays['point_label'].astype(np.int64)
            for k,pred in predictions.items():
                matrices[k]+=confusion(labels,pred)
                for name,value in paired_summary(labels,predictions['baseline'],pred).items():
                    pair[k][name]+=value
            status=dict(models['temporal'].last_status)
            eligible=int(temporal['has_history'].sum())
            accepted=int(((temporal['gate']>=cfg.gate_threshold)&temporal['has_history']).sum())
            rec=dict(scene=scene,frame=row['frame'],stamp=raw['stamp'],points=len(labels),
                     eligible_voxels=eligible,accepted_voxels=accepted,memory_status=status,
                     temporal_ms=temporal_ms,current_only_ms=current_ms,uniform_history_ms=uniform_ms)
            per_frame.append(rec)
            groups=json.loads(arrays['groups'].item())
            for i,g in enumerate(groups):
                mask=arrays['group_owner']==i
                all_events.append(dict(scene=scene,frame=row['frame'],stamp=raw['stamp'],
                    instance=g['instance'],label=g['label'],points=int(mask.sum()),
                    correct_fraction={k:float((p[mask]==g['label']).mean()) for k,p in predictions.items()}))
        per_scene[scene]={k:scores(cm) for k,cm in matrices.items()}
        for k,cm in matrices.items():
            global_cm[k]+=cm
        announce(stage='test_scene',seed=seed,scene=scene,
                 miou={k:round(scores(cm)['miou_fixed16']*100,3) for k,cm in matrices.items()})
    result=dict(seed=seed,checkpoints=checkpoints,scope='small real-data pilot; official val subset, not leaderboard',
        point_metrics={k:dict(scores(cm),**dict(pair[k])) for k,cm in global_cm.items()},
        per_scene=per_scene,events=events(all_events,ARMS),
        class_events={str(c):events([r for r in all_events if r['label']==c],ARMS) for c in (3,6)},
        module_timing={k:dict(n=len(v),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),max_ms=max(v)) for k,v in timing.items()},
        history=dict(eligible_voxels=sum(r['eligible_voxels'] for r in per_frame),
                     accepted_voxels=sum(r['accepted_voxels'] for r in per_frame)))
    save(out/f'seed-{seed}'/'evaluation.json',result)
    save(out/f'seed-{seed}'/'frames.json',per_frame)
    save(out/f'seed-{seed}'/'object-observations.json',all_events)
    del models,uniform
    torch.cuda.empty_cache()
    return result


def summarize(results,index):
    summary={}
    for arm in ARMS:
        values=[r['point_metrics'][arm]['miou_fixed16'] for r in results]
        summary[arm]=dict(miou_percent_mean=float(np.mean(values)*100),
                         miou_percent_std=float(np.std(values)*100),
                         miou_percent_seeds=[v*100 for v in values])
    # Paired cluster bootstrap over scenes. Average paired seed differences,
    # never count individual points as independent statistical replicates.
    rng=np.random.default_rng(20261025)
    scenes=index['parts']['test']
    boot=[]
    for _ in range(2000):
        draw=rng.choice(scenes,len(scenes),replace=True)
        deltas=[]
        for r in results:
            cms={k:sum((np.asarray(r['per_scene'][s][k]['confusion']) for s in draw),np.zeros((16,16),np.int64))
                 for k in ('temporal','current_only')}
            deltas.append(scores(cms['temporal'])['miou_fixed16']-scores(cms['current_only'])['miou_fixed16'])
        boot.append(np.mean(deltas)*100)
    delta=summary['temporal']['miou_percent_mean']-summary['current_only']['miou_percent_mean']
    ci=np.percentile(boot,[2.5,97.5]).tolist()
    person=[]
    for r in results:
        a=r['point_metrics']['temporal']['class_iou'][6]
        b=r['point_metrics']['current_only']['class_iou'][6]
        person.append((a-b)*100 if a is not None and b is not None else None)
    medians=[r['module_timing']['temporal']['median_ms'] for r in results]
    return dict(arms=summary,temporal_minus_current_only_pp=delta,
        paired_scene_bootstrap_95_percent_interval_pp=ci,
        pedestrian_delta_pp=person,
        pilot_semantic_admission=bool(delta>0 and ci[0]>0 and all(v is not None and v>=-1 for v in person)),
        five_ms_module_budget_passed=bool(max(medians)<=5.),
        module_median_ms_by_seed=medians,real_training=True,deployable=False,
        limitations=['14 selected scenes, 240 evaluation frames; not full dataset',
                    'keyframe rigid ego alignment; no per-point deskew validation',
                    'event times quantized by ~0.5-second labeled keyframes',
                    'GT boxes/instance IDs used for scoring only',
                    'cached-feature module timing excludes backbone and transport'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    index=json.loads(args.index.read_text())
    manifest=json.loads((args.cache/'manifest.json').read_text())
    if manifest['index_sha256'] != digest(args.index):
        raise ValueError('Cache index mismatch')
    logs=[{index['scene_logs'][s] for s in index['parts'][part]} for part in ('train','dev','test')]
    if any(logs[i]&logs[j] for i in range(3) for j in range(i)):
        raise ValueError('Collection-log leakage')
    for f in manifest['files']:
        if digest(args.cache/f['path']) != f['sha256']:
            raise ValueError('Feature cache hash mismatch')
    cfg=MemoryConfig()
    protocol=dict(seeds=[0,1,2],epochs=5,clip=4,lr=.001,config=asdict(cfg),
        index_sha256=digest(args.index),feature_manifest_sha256=digest(args.cache/'manifest.json'),
        backbone_sha256=manifest['backbone_sha256'],model_config_sha256=manifest['model_config_sha256'],
        geometry=index['geometry'],label_space='native16',parts=index['parts'],
        current_control='same parameterization and initialization; single current token; identical labeled training frames/steps',
        temporal_augmentation=dict(dropout=.15,corruption=.15),current_augmentation='none',
        loss='fused CE + 0.2 utility BCE + 0.2 candidate CE, voxel-majority GT',
        selection='best dev fixed-16 point mIoU checkpoint separately for each arm/seed; threshold fixed 0.5',
        evaluation='one sealed pass per seed after all training and selection; original point GT through inverse index',
        semantic_admission='temporal minus matched-current mIoU positive, paired scene bootstrap lower 95% bound >0, pedestrian delta >= -1 pp for all seeds',
        latency_admission='module median <=5ms for every seed; no full driving realtime claim',
        source_sha256={str(x.relative_to(REPO)):digest(x) for x in [REPO/'src/semantic_memory.py',REPO/'tools/semantic_memory/train.py',Path(__file__),REPO/'tools/semantic_memory/pilot_metrics.py']})
    save(args.out/'protocol.json',protocol)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for seed in protocol['seeds']:
        run_training(seed,index,args.cache,args.out,cfg,protocol,protocol['epochs'])
    results=[evaluate(seed,index,args.cache,args.out,cfg) for seed in protocol['seeds']]
    result=summarize(results,index)
    save(args.out/'summary.json',result)
    announce(stage='complete',**result)


if __name__ == '__main__':
    main()
