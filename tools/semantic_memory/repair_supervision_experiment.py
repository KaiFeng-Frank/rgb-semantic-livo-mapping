"""Frozen four-arm supervision experiment on exact causal JointRepair inputs."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import os
from pathlib import Path
import sys,time
import numpy as np
from tools.instance_memory.data import read,save
from tools.semantic_memory.nuscenes_pilot_data import digest
from tools.semantic_memory.pilot_metrics import confusion,scores,events
from .repair_supervision import current_token_input,repair_objective,present_class_weights

ARMS=('standard_current','standard_temporal','balanced_current','balanced_temporal')
ABLATIONS=('standard_temporal_no_history','balanced_temporal_no_history')
STATIC=('baseline','anchor','current_image')
CURRENT=STATIC+('standard_current','balanced_current')+ABLATIONS


def arrays(cache,entry,training=False):
    with np.load(cache/entry['path'],allow_pickle=False) as f:
        keys=('joint','anchor','label','group') if training else f.files
        return {k:f[k] for k in keys}


def predict(model,joint,anchor):
    import torch
    return anchor+model.repair.residual_limit*torch.tanh(model.net(joint))


def evaluate(models,cache,entries,*,detailed=False):
    import torch
    names=STATIC+tuple(models)
    total={a:np.zeros((16,16),np.int64) for a in names};counts={a:Counter() for a in names}
    scene_mats={};observations=[];frames=[]
    for m in models.values():m.eval()
    with torch.no_grad():
        for row in entries:
            d=arrays(cache,row);z=torch.from_numpy(d['joint']).cuda();a=torch.from_numpy(d['anchor']).cuda();zc=current_token_input(z)
            inverse=d['inverse'];gt=d['point_label'].astype(np.int64);valid=gt>=0;fg=valid&(gt<10)&d['camera']
            prediction=dict(baseline=d['baseline'][inverse],anchor=d['anchor_labels'][inverse],current_image=d['current_image'])
            for name,model in models.items():
                inp=zc if name.endswith('_current') or name.endswith('_no_history') else z
                labels=d['anchor_labels'].copy();labels[d['query']]=predict(model,inp,a).argmax(1).cpu().numpy()
                prediction[name]=labels[inverse]
            sm=scene_mats.setdefault(row['scene'],{a:np.zeros((16,16),np.int64) for a in names})
            for name,y in prediction.items():
                mat=confusion(gt,y);total[name]+=mat;sm[name]+=mat
                counts[name].update(camera_foreground_points=int(fg.sum()),camera_foreground_correct=int((fg&(y==gt)).sum()),
                    fixes_raw=int((valid&(prediction['baseline']!=gt)&(y==gt)).sum()),harms_raw=int((valid&(prediction['baseline']==gt)&(y!=gt)).sum()))
            if detailed:
                for obj in np.unique(d['identity'][d['identity']>0]):
                    at=(d['identity']==obj)&valid;n=int(at.sum())
                    if n<5:continue
                    classes=np.unique(gt[at])
                    if len(classes)!=1:raise ValueError('Mixed GT panoptic object classes')
                    observations.append(dict(scene=row['scene'],instance=int(obj),label=int(classes[0]),frame=row['frame'],stamp=row['stamp'],points=n,
                        correct_fraction={a:float((y[at]==gt[at]).mean()) for a,y in prediction.items()},
                        fixes_raw={a:int((at&(prediction['baseline']!=gt)&(y==gt)).sum()) for a,y in prediction.items()},
                        harms_raw={a:int((at&(prediction['baseline']==gt)&(y!=gt)).sum()) for a,y in prediction.items()}))
                frames.append(dict(scene=row['scene'],frame=row['frame'],camera_foreground_points=int(fg.sum()),
                    camera_foreground_correct={a:int((fg&(y==gt)).sum()) for a,y in prediction.items()}))
    report=dict(arms={a:dict(scores(total[a]),counts=dict(counts[a]),camera_foreground_accuracy=counts[a]['camera_foreground_correct']/max(1,counts[a]['camera_foreground_points'])) for a in names},
        per_scene={s:{a:scores(m) for a,m in mm.items()} for s,mm in scene_mats.items()})
    if detailed:
        report.update(events=events(observations,names),object_observations=observations,frames=frames)
    return report


def train_seed(seed,cache,entries,out,protocol,manifest):
    import torch
    from semantic_repair import JointRepair,RepairConfig
    from semantic_memory import MemoryConfig
    torch.manual_seed(seed);cfg=MemoryConfig(**protocol['memory_config']);repair=RepairConfig(**protocol['repair_config'])
    template=JointRepair(cfg,repair).cuda();initial={k:v.detach().clone() for k,v in template.state_dict().items()}
    if sum(p.numel() for p in template.parameters())!=15760:raise AssertionError('Architecture changed')
    models={k:JointRepair(cfg,repair).cuda() for k in ARMS}
    for m in models.values():m.load_state_dict(initial,strict=True)
    del template,initial
    optcfg=protocol['config']['optimization'];opts={k:torch.optim.Adam(m.parameters(),lr=optcfg['learning_rate']) for k,m in models.items()}
    weights=torch.tensor(manifest['class_weights'],device='cuda',dtype=torch.float32)
    train=defaultdict(list);dev=[]
    for e in entries:
        if e['part']=='train':train[e['scene']].append(e)
        else:dev.append(e)
    folder=out/f'seed-{seed}';folder.mkdir(exist_ok=False)
    best={k:-1. for k in ARMS};selected={};start=time.perf_counter()
    for epoch in range(1,optcfg['epochs']+1):
        stats={k:dict(loss=0.,frames=0,steps=0,queries=0,groups=[0,0,0],unreduced_loss_mass=[0.,0.,0.]) for k in ARMS}
        for m in models.values():m.train()
        scenes=list(train);np.random.default_rng(seed*100+epoch).shuffle(scenes)
        for scene in scenes:
            for opt in opts.values():opt.zero_grad(set_to_none=True)
            pending={k:[] for k in ARMS}
            rows=train[scene]
            for j,row in enumerate(rows):
                if not j:continue
                d=arrays(cache,row,True);z=torch.from_numpy(d['joint']).cuda();zc=current_token_input(z);anchor=torch.from_numpy(d['anchor']).cuda()
                y=torch.from_numpy(d['label'].astype(np.int64)).cuda();g=torch.from_numpy(d['group'].astype(np.int64)).cuda()
                gc=np.bincount(d['group'][d['group']>=0],minlength=3);n=int((d['label']>=0).sum())
                for kind,m in models.items():
                    logits=predict(m,zc if kind.endswith('_current') else z,anchor)
                    loss,mass=repair_objective(logits,anchor,y,g,weights,kind.startswith('balanced'))
                    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training objective')
                    pending[kind].append(loss);s=stats[kind];s['loss']+=float(loss.detach());s['frames']+=1;s['queries']+=n
                    s['groups']=(np.asarray(s['groups'])+gc).tolist();s['unreduced_loss_mass']=(np.asarray(s['unreduced_loss_mass'])+mass.cpu().numpy()).tolist()
                if len(pending[ARMS[0]])==optcfg['clip_frames'] or j==len(rows)-1:
                    for kind,m in models.items():
                        torch.stack(pending[kind]).mean().backward()
                        torch.nn.utils.clip_grad_norm_(m.parameters(),optcfg['gradient_norm_max'],error_if_nonfinite=True)
                        opts[kind].step();opts[kind].zero_grad(set_to_none=True);pending[kind].clear();stats[kind]['steps']+=1
            save(out.parent/'progress.json',dict(stage='training',seed=seed,epoch=epoch,scene=scene,frames_this_epoch=stats[ARMS[0]]['frames'],elapsed_s=time.perf_counter()-start))
        for s in stats.values():
            if s['frames']!=626 or s['groups']!=manifest['train_groups']:raise AssertionError('Matched supervision budget changed')
            s['loss']/=s['frames']
        if len({s['steps'] for s in stats.values()})!=1:raise AssertionError('Unequal optimizer steps')
        validation=evaluate(models,cache,dev)
        for kind,m in models.items():
            score=validation['arms'][kind]['miou_fixed16']
            if score>best[kind]:
                best[kind]=score;dest=folder/(kind+'.pt')
                saved=dict(schema='current-anchored-repair-v2',trained=True,config=asdict(cfg),repair=asdict(repair),fusion=m.state_dict(),
                    provenance=dict(scope='feature_supervision_factorial',deployable=False,seed=seed,kind=kind,epoch=epoch,selected_on='development only',
                        anchor_sha256=protocol['anchor_sha256'],feature_manifest_sha256=digest(out.parent/'feature-manifest.json'),
                        protocol_sha256=digest(out.parent/'training-protocol.json')))
                tmp=dest.with_suffix('.pt.partial');torch.save(saved,tmp);tmp.replace(dest)
                selected[kind]=dict(epoch=epoch,sha256=digest(dest),dev_miou=score)
        save(folder/f'epoch-{epoch}.json',dict(train=stats,development=validation,selected=selected))
        print('seed',seed,'epoch',epoch,'dev miou',{k:round(100*validation['arms'][k]['miou_fixed16'],5) for k in ARMS},flush=True)
    save(folder/'selected.json',selected)
    selected_models={}
    for kind in ARMS:
        m=JointRepair(cfg,repair).cuda().eval();ck=torch.load(folder/(kind+'.pt'),map_location='cpu',weights_only=True)
        m.load_state_dict(ck['fusion'],strict=True);selected_models[kind]=m
    for kind in ABLATIONS:selected_models[kind]=selected_models[kind.removesuffix('_no_history')]
    report=evaluate(selected_models,cache,dev,detailed=True)
    save(folder/'object-observations.json',report.pop('object_observations'));save(folder/'frames.json',report.pop('frames'))
    report.update(seed=seed,checkpoints=selected,scope='Development, same cached causal inputs; no heldout or realtime admission.')
    save(folder/'evaluation.json',report)
    del models,opts,selected_models
    torch.cuda.empty_cache()
    return report


def summarize(reports):
    candidate='balanced_temporal';gains=[];fg=[];positive=[]
    for r in reports:
        a=r['arms'];gains.append(100*(a[candidate]['miou_fixed16']-max(a[k]['miou_fixed16'] for k in CURRENT)))
        fg.append(100*(a[candidate]['camera_foreground_accuracy']-max(a[k]['camera_foreground_accuracy'] for k in CURRENT)))
        positive.append(sum(s[candidate]['miou_fixed16']>max(s[k]['miou_fixed16'] for k in CURRENT) for s in r['per_scene'].values()))
    recovery={k:float(np.mean([r['events']['baseline_error_recovery'][k]['success_rate'] for r in reports])) for k in reports[0]['arms']}
    gate=dict(miou_gain_vs_best_current_pp_by_seed=gains,mean_miou_gain_pp=float(np.mean(gains)),
        foreground_gain_vs_best_current_pp_by_seed=fg,positive_logs_by_seed=positive,
        mean_recovery_success=recovery[candidate],best_current_mean_recovery_success=max(recovery[k] for k in CURRENT))
    gate['passed']=gate['mean_miou_gain_pp']>=.05 and min(gains)>0 and min(fg)>=-.1 and min(positive)>=5 and gate['mean_recovery_success']>=gate['best_current_mean_recovery_success']
    names=reports[0]['arms']
    aggregate={k:dict(miou_percent_by_seed=[100*r['arms'][k]['miou_fixed16'] for r in reports],
        mean_miou_percent=float(np.mean([100*r['arms'][k]['miou_fixed16'] for r in reports])),
        foreground_accuracy_percent_by_seed=[100*r['arms'][k]['camera_foreground_accuracy'] for r in reports],
        mean_recovery_success=recovery[k]) for k in names}
    return dict(arms=aggregate,gate=gate,current_controls=list(CURRENT),
        scope='Three fixed seeds, development only. Candidate chosen before training. Cached-head computation is not end-to-end latency. No new instance identity output or holdout score.',deployable=False)


def run(cache,out):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from tools.visual_instances.kalman_cache import gpu_admission
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'src'))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    admission=gpu_admission();protocol=read(out/'protocol.json');manifest=read(out/'feature-manifest.json')
    if digest(out/'protocol.json')!=manifest['protocol_sha256'] or read(cache/'manifest.json')!=manifest:raise ValueError('Incompatible cache metadata')
    if digest(root/'config/feature_repair_supervision_20261029.json')!=protocol['config_sha256']:raise ValueError('Frozen experiment config changed')
    for path,h in protocol['sources'].items():
        if digest(root/path)!=h:raise ValueError('Extraction source changed: '+path)
    if len(manifest['files'])!=963 or manifest['parity_frames']!=963 or manifest['real_output_checks']!=144:raise ValueError('Unverified real cache')
    np.testing.assert_array_equal(present_class_weights(manifest['train_voxel_counts']),manifest['class_weights'])
    for f in manifest['files']:
        p=(cache/f['path']).resolve()
        if not p.is_relative_to(cache.resolve()) or digest(p)!=f['sha256']:raise ValueError('Input cache changed')
    run_dir=out/'run';run_dir.mkdir(exist_ok=False)
    save(out/'training-protocol.json',dict(extraction_protocol_sha256=digest(out/'protocol.json'),
        feature_manifest_sha256=digest(out/'feature-manifest.json'),admission=admission,
        implementation_sha256=digest(Path(__file__)),objective_source_sha256=digest(root/'tools/semantic_memory/repair_supervision.py'),
        resolved_current_controls=list(CURRENT),params=protocol['config']['optimization'],
        selection='Development mIoU, no output-policy or threshold selection; same-head current-token ablations are included among current controls.'))
    reports=[]
    for seed in protocol['config']['optimization']['seeds']:
        reports.append(train_seed(seed,cache,manifest['files'],run_dir,protocol,manifest))
    result=summarize(reports);save(out/'report.json',result)
    save(out/'progress.json',dict(stage='training_complete',seeds=3,epochs_per_arm=5,gate=result['gate']))
    print('FINAL',result['gate'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.cache,a.out)
