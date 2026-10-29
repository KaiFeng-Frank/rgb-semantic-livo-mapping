"""Frozen causal ownership/evidence readout with independent same-forward controls."""
import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

from tools.instance_memory.data import read, save
from tools.instance_memory.metrics import identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA, digest
from tools.semantic_memory.pilot_metrics import confusion, scores
from tools.visual_instances.visual_semantics import fusion
from .point_probability_memory import RawProbabilityMemory
from .point_history_audit import neighbors
from .point_evidence_memory import EvidencePointMemory, visual_arrays, owned_readout, evidence_veto

CURRENT = ['raw','current_image','current_spatial_probability','current_spatial_same_id_memory_guard']
CANDIDATE = 'previous_same_id_memory_guard'


def run(index_path, targets_path, visual_path, reference, out):
    import torch
    from tools.visual_instances.kalman_cache import gpu_admission
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from ptv3_worker import Segmenter
    torch.set_num_threads(1);torch.manual_seed(20261029)
    index=read(index_path);names=index['independent_dev_added'];prior=read(reference/'report.json')
    if len(names)!=8 or names!=prior['scenes']:
        raise ValueError('Frozen selection mismatch')
    for folder in (targets_path,visual_path):
        if any((folder/(s+'.inputs.json.gz')).exists() or (folder/(s+'.targets.json.gz')).exists() for s in index['parts']['test']):
            raise ValueError('New holdout has been opened')
    if shutil.disk_usage(out.parent).free<4*1024**3:raise RuntimeError('Less than 4 GiB headroom')
    admission=gpu_admission();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    cfg_path=root/'config/point_evidence_memory_20261029.json';config=read(cfg_path);arms=config['arms']
    sources=[Path(__file__),Path(__file__).with_name('point_evidence_memory.py'),Path(__file__).with_name('point_probability_memory.py'),
             Path(__file__).with_name('point_history_audit.py'),root/'tools/visual_instances/visual_semantics.py']
    save(out/'protocol.json',dict(config=config,config_sha256=digest(cfg_path),index_path=str(index_path),index_sha256=digest(index_path),
        target_path=str(targets_path),visual_input_path=str(visual_path),reference_path=str(reference),
        reference_sha256=digest(reference/'report.json'),visual_protocol_sha256=digest(visual_path/'protocol.json'),
        input_sha256={s:digest(visual_path/(s+'.inputs.json.gz')) for s in names},target_sha256={s:digest(targets_path/(s+'.targets.json.gz')) for s in names},
        sources={str(p.relative_to(root)):digest(p) for p in sources},checkpoint_sha256=digest(os.environ['PTV3_CHECKPOINT']),admission=admission))
    model=Segmenter(half=True,shuffle=False,intensity_scale=1.)
    reports=[];cost=[];backbone=[];started=time.perf_counter()
    for name in names:
        rows=index['sequences'][name];targets=read(targets_path/(name+'.targets.json.gz'))['frames']
        visual=read(visual_path/(name+'.inputs.json.gz'))['frames']
        if len(rows)!=len(targets) or len(rows)!=len(visual):raise ValueError('Incomplete sequence')
        memory=EvidencePointMemory();control=RawProbabilityMemory()
        mats={a:np.zeros((16,16),np.int64) for a in arms};counts={a:Counter() for a in arms}
        categories={s:Counter() for s in ('all','camera_foreground')};objects=defaultdict(Counter)
        frames=[];cache_diff=0;cache_points=0
        for k,(row,target,vframe) in enumerate(zip(rows,targets,visual)):
            stamp=int(row['timestamp_us']*1000)
            if vframe['frame']!=k or vframe['stamp']!=stamp:raise ValueError('Visual cache timestamp/order mismatch')
            path=DATA/row['scan_path']
            if digest(path)!=row['scan_sha256']:raise ValueError('Scan changed')
            scan=np.fromfile(path,dtype='<f4').reshape(-1,5)[:,:4].copy();scan[:,3]/=255.
            torch.cuda.synchronize();t=time.perf_counter();raw=model.segment(scan,return_features=True)
            p=raw['logits'].float().softmax(-1)[raw['inverse']].cpu().numpy();torch.cuda.synchronize();backbone.append((time.perf_counter()-t)*1000)
            t=time.perf_counter()
            pose=np.asarray(row['transform']);xyz=scan[:,:3].astype(np.float64)@pose[:3,:3].T+pose[:3,3]
            ids,cur,mem=visual_arrays(len(scan),vframe['observations'])
            old=memory.previous
            if old is not None and stamp-old['stamp']>750000000:old=None
            candidate,detail=memory.step(xyz,p,ids,mem,stamp,name)
            cost.append((time.perf_counter()-t)*1000)
            np.testing.assert_array_equal(memory.previous['p'],p)
            # All controls below share exactly the same forward and causal inputs.
            baseline,_=control.step(xyz,p,stamp,name)
            pred={a:baseline[a] for a in ('raw','current_spatial_probability','previous_raw_probability')}
            pred['current_image']=pred['raw'].copy()
            for obs in vframe['observations']:
                at=np.asarray(obs['indices'])
                pred['current_image'][at]=fusion(p[at],obs['current_scores'],obs['confidence']).argmax(1)
                cache_diff+=int((pred['raw'][at]!=np.asarray(obs['raw_probabilities']).argmax(1)).sum());cache_points+=len(at)
            sj,sv=neighbors(xyz,xyz,same_scan=True)
            spatial,_=owned_readout(p,p,ids,ids,sj,sv)
            pred['current_spatial_same_id_memory_guard']=evidence_veto(pred['raw'],spatial,mem)[0]
            if old is not None:
                assert old['stamp']<stamp
                hj,hv=neighbors(xyz,old['xyz'])
                owned,_=owned_readout(p,old['p'],ids,old['ids'],hj,hv)
            else:owned=pred['raw'].copy()
            pred['previous_same_id']=owned
            pred['previous_same_id_current_guard'],cv=evidence_veto(pred['raw'],owned,cur)
            pred[CANDIDATE],mv=evidence_veto(pred['raw'],owned,mem)
            np.testing.assert_array_equal(pred[CANDIDATE],candidate)
            # Observer begins here. Never feed GT, corrected outputs or observer IDs back.
            gt=np.asarray(target['gt']);truth=identity_truth(target);good=gt>=0
            camera=np.zeros(len(gt),bool);camera[target['camera_in_view_indices']]=True
            fg=good&camera&(gt<10)
            original_harm=good&(pred['raw']==gt)&(pred['previous_raw_probability']!=gt)
            original_fix=good&(pred['raw']!=gt)&(pred['previous_raw_probability']==gt)
            events=dict(raw_history_harms=original_harm,raw_history_fixes=original_fix,
                raw_history_harms_on_visual_component=original_harm&(ids>0),
                raw_history_harms_current_visual_agrees_raw=original_harm&(cur==pred['raw']),
                raw_history_harms_memory_visual_agrees_raw=original_harm&(mem==pred['raw']),
                ownership_avoided_original_harms=original_harm&(owned==gt),
                ownership_lost_original_fixes=original_fix&(owned!=gt),
                current_visual_veto_prevented_harm=cv&(pred['raw']==gt),
                current_visual_veto_blocked_fix=cv&(owned==gt)&(pred['raw']!=gt),
                memory_visual_veto_prevented_harm=mv&(pred['raw']==gt),
                memory_visual_veto_blocked_fix=mv&(owned==gt)&(pred['raw']!=gt),
                visual_memory_changed_guard=pred[CANDIDATE]!=pred['previous_same_id_current_guard'],
                observed_by_visual_component=ids>0)
            frame_counts={}
            for scope,mask in (('all',good),('camera_foreground',fg)):
                frame_counts[scope]={event:int((mask&m).sum()) for event,m in events.items()}
                categories[scope].update(frame_counts[scope])
            for a,y in pred.items():
                mats[a]+=confusion(gt,y)
                counts[a].update(fixes_raw=int((good&(pred['raw']!=gt)&(y==gt)).sum()),harms_raw=int((good&(pred['raw']==gt)&(y!=gt)).sum()),
                    camera_foreground_correct=int((fg&(y==gt)).sum()),camera_foreground_points=int(fg.sum()))
            for obj in np.unique(truth[fg&(truth>0)]):
                at=fg&(truth==obj);c=objects[str(int(obj))];c['points']+=int(at.sum());c['observed_frames']+=1
                for a,y in pred.items():
                    c[a+'_fixes']+=int((at&(pred['raw']!=gt)&(y==gt)).sum())
                    c[a+'_harms']+=int((at&(pred['raw']==gt)&(y!=gt)).sum())
            frames.append(dict(frame=k,stamp_ns=stamp,prior_stamp_ns=None if old is None else old['stamp'],candidate_component_ms=cost[-1],
                backbone_ms=backbone[-1],counts=frame_counts,**detail))
        old_report=read(reference/(name+'.json'))
        parity={a:bool(np.array_equal(mats[a],old_report['arms'][a]['confusion'])) for a in ('raw','current_spatial_probability','previous_raw_probability')}
        report=dict(scene=name,log=index['scene_logs'][name],arms={a:dict(scores(mats[a]),counts=dict(counts[a]),
            camera_foreground_accuracy=counts[a]['camera_foreground_correct']/max(1,counts[a]['camera_foreground_points'])) for a in arms},
            counts={s:dict(c) for s,c in categories.items()},camera_foreground_objects={k:dict(v) for k,v in objects.items()},frames=frames,
            reference_confusion_exact=parity,cached_visual_raw_label_differences=cache_diff,cached_visual_point_observations=cache_points)
        save(out/(name+'.json'),report);reports.append(report)
        save(out/'progress.json',dict(completed_scenes=len(reports),scene=name,elapsed_s=time.perf_counter()-started))
        print(name,{a:round(100*d['miou_fixed16'],6) for a,d in report['arms'].items()},'FG counts',report['counts']['camera_foreground'],flush=True)
    aggregate={}
    for a in arms:
        c=Counter()
        for r in reports:c.update(r['arms'][a]['counts'])
        aggregate[a]=dict(scores(sum(np.asarray(r['arms'][a]['confusion']) for r in reports)),counts=dict(c),
            camera_foreground_accuracy=c['camera_foreground_correct']/max(1,c['camera_foreground_points']))
    total={}
    for scope in ('all','camera_foreground'):
        c=Counter()
        for r in reports:c.update(r['counts'][scope])
        total[scope]=dict(c)
    candidate=aggregate[CANDIDATE]
    gate=dict(miou_gain_vs_best_current_pp=100*(candidate['miou_fixed16']-max(aggregate[a]['miou_fixed16'] for a in CURRENT)),
        foreground_gain_vs_best_current_pp=100*(candidate['camera_foreground_accuracy']-max(aggregate[a]['camera_foreground_accuracy'] for a in CURRENT)),
        positive_logs_vs_every_current=sum(r['arms'][CANDIDATE]['miou_fixed16']>max(r['arms'][a]['miou_fixed16'] for a in CURRENT) for r in reports),
        candidate_component_p95_ms=float(np.percentile(cost,95)))
    gate['passed']=gate['miou_gain_vs_best_current_pp']>=.05 and gate['foreground_gain_vs_best_current_pp']>=-.1 and gate['positive_logs_vs_every_current']>=5 and gate['candidate_component_p95_ms']<=50
    result=dict(aggregate=aggregate,counts=total,gate=gate,scenes=names,frames=sum(len(r['frames']) for r in reports),
        all_reference_confusions_exact=all(all(r['reference_confusion_exact'].values()) for r in reports),
        cached_visual_raw_label_differences=sum(r['cached_visual_raw_label_differences'] for r in reports),
        timings_ms={k:dict(median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(max(v))) for k,v in [('candidate_component',cost),('backbone',backbone)]},
        elapsed_s=time.perf_counter()-started,
        scope='Fixed development-only test. Cached causal visual inputs and one PTv3 forward shared by every arm. No GT gating. Identity outputs unchanged. Candidate-only CPU component is not full online latency.')
    save(out/'report.json',result);save(out/'progress.json',dict(status='complete',frames=result['frames'],elapsed_s=result['elapsed_s']))
    print('aggregate',{a:dict(miou=100*v['miou_fixed16'],fg=100*v['camera_foreground_accuracy']) for a,v in aggregate.items()},gate,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('index','targets','visual','reference','out'):parser.add_argument('--'+k,type=Path,required=True)
    args=parser.parse_args();run(args.index,args.targets,args.visual,args.reference,args.out)
