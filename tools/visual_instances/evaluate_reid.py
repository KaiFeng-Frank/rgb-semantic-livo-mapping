"""Paired native appearance-memory evaluation; no training or holdout access."""
import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
from tools.instance_memory.data import read,save
from tools.instance_memory.metrics import IdentityMetrics,identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest


COUNT_KEYS=['gt_points','pred_points','identity_true_positive','associated_gt_points',
    'gt_instances','pred_tracks','id_switches','supported_transitions','reappearance_transitions']


def merge(rows):
    total=Counter()
    for r in rows:
        total.update({k:r[k] for k in COUNT_KEYS})
        total['aq_numerator']+=r['association_quality']*r['gt_instances']
    return dict(total,point_idf1=2*total['identity_true_positive']/max(1,total['gt_points']+total['pred_points']),
        association_quality=total['aq_numerator']/max(1,total['gt_instances']))


def run(index_path,cache,weights,out):
    import cv2,torch
    from .reid import PairedNativeTrackers
    from .geometry import Camera,associate
    from .semantics import Config,component
    from .kalman_cache import gpu_admission
    torch.set_num_threads(1);torch.manual_seed(20261029);cv2.setNumThreads(1)
    index=read(index_path);protocol=read(cache/'protocol.json');cfg=Config()
    if any((cache/(n+'.inputs.json.gz')).exists() for n in index['parts']['test']):
        raise ValueError('Development-only evaluator refuses an opened holdout cache')
    if index['parts']['dev']!=protocol['parts']['dev']:raise ValueError('Scene selection mismatch')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'protocol.json',dict(index_sha256=digest(index_path),cache_protocol_sha256=digest(cache/'protocol.json'),
        weights_sha256=digest(weights),admission=gpu_admission(),
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('reid.py'),
            Path(__file__).with_name('evidence_predictor.py'),Path(__file__).with_name('botsort.yaml'),Path(__file__).with_name('botsort_reid.yaml')]},
        preregistration=read(Path(__file__).resolve().parents[2]/'config/native_reid_20261029.json')))
    policies=['projected_mask','trusted_component','largest_component'];names=['baseline','native_reid']
    arms=[n+'/'+p for n in names for p in policies];scenes=[];timings=[];starts=[]
    for scene_name in index['parts']['dev']:
        start=time.perf_counter();paired=PairedNativeTrackers(weights);starts.append((time.perf_counter()-start)*1000)
        targets=read(cache/(scene_name+'.targets.json.gz'))['frames']
        cached=read(cache/(scene_name+'.inputs.json.gz'))['frames']
        metrics={scope:{arm:IdentityMetrics() for arm in arms} for scope in ['full','camera']}
        frames=[];cache_difference=Counter();native_feature_tracks=0
        for row,target,old in zip(index['sequences'][scene_name],targets,cached):
            if row['frame']%10==0:gpu_admission()
            image=cv2.imread(str(DATA/row['image_path']))
            scan=np.fromfile(DATA/row['scan_path'],dtype='<f4').reshape(-1,5)
            visual,detail=paired.step(image,int(row['camera_timestamp_us']*1000));timings.append(detail)
            transform=np.linalg.inv(row['camera_transform'])@row['transform']
            camera=Camera(image.shape[1],image.shape[0],row['camera_intrinsic'],np.zeros(4),transform)
            predictions={}
            for name,frame in visual.items():
                assoc=associate(scan[:,:3],camera,frame)
                # Camera time gaps reset tracker state; preserve that namespace.
                raw=assoc['ids'].astype(np.int64)
                if paired.epoch:raw=np.where(raw>0,raw+paired.epoch*2**32,0)
                predictions[name+'/projected_mask']=raw
                for policy,fraction in [('trusted_component',cfg.component_fraction),('largest_component',0.)]:
                    ids=np.zeros(len(scan),np.int64)
                    for tid in np.unique(raw[raw>0]):
                        at=np.flatnonzero(raw==tid);chosen=component(scan[at,:3],replace(cfg,component_fraction=fraction))
                        if chosen is not None:ids[at[chosen]]=tid
                    predictions[name+'/'+policy]=ids
            for policy,trusted_only in [('trusted_component',True),('largest_component',False)]:
                expected=np.zeros(len(scan),bool)
                for o in old['observations']:
                    if not trusted_only or o['trusted']:expected[o['indices']]=True
                cache_difference[policy]+=int((expected!=(predictions['baseline/'+policy]>0)).sum())
            native_feature_tracks+=sum(t.smooth_feat is not None for t in paired.trackers['native_reid'].tracked_stracks)
            # Labels are read here only after both causal tracker states update.
            gt=np.asarray(target['gt']);pan=np.asarray(target['panoptic']);truth=identity_truth(target)
            valid=(gt>=0)&(pan>0);inside=np.zeros(len(scan),bool);inside[target['camera_in_view_indices']]=True
            for scope,mask in [('full',valid),('camera',valid&inside)]:
                for arm,pred in predictions.items():metrics[scope][arm].add(truth,pred,row['frame'],mask)
            frames.append(dict(frame=row['frame'],timing=detail,covered_points={a:int((p>0).sum()) for a,p in predictions.items()}))
        paired.close();del paired
        result=dict(scene=scene_name,log=index['scene_logs'][scene_name],
            identity={scope:{arm:m.result() for arm,m in rows.items()} for scope,rows in metrics.items()},
            cached_baseline_coverage_symmetric_difference=dict(cache_difference),native_feature_track_frames=native_feature_tracks,frames=frames)
        save(out/(scene_name+'.json'),result);scenes.append(result)
        print(scene_name,{a:100*v['point_idf1'] for a,v in result['identity']['camera'].items()},dict(cache_difference),flush=True)
    identity={scope:{a:merge([s['identity'][scope][a] for s in scenes]) for a in arms} for scope in ['full','camera']}
    logs={log:{scope:{a:merge([s['identity'][scope][a] for s in scenes if s['log']==log]) for a in arms}
        for scope in identity} for log in sorted(set(s['log'] for s in scenes))}
    elapsed={key:[r[key] for r in timings] for key in ['forward_ms','gmc_ms']}
    elapsed.update({n+'_tracker_ms':[r['tracker_ms'][n] for r in timings] for n in names})
    result=dict(identity=identity,collection_logs=logs,scenes=[s['scene'] for s in scenes],
        timing={k:dict(median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),max_ms=max(v)) for k,v in elapsed.items()},
        startup_ms=starts,gmc_errors=[r['gmc_error'] for r in timings if r['gmc_error']],
        cached_baseline_coverage_symmetric_difference={p:sum(s['cached_baseline_coverage_symmetric_difference'][p] for s in scenes) for p in ['trusted_component','largest_component']},
        native_feature_track_frames=sum(s['native_feature_track_frames'] for s in scenes),
        scope='Point identity metrics only, not semantic accuracy or leaderboard PAT. Same detection and GMC shared. Timing excludes PTv3, IPC and map publication. Development only.')
    save(out/'report.json',result)
    print({a:100*v['point_idf1'] for a,v in identity['camera'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['index','cache','weights','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.index,a.cache,a.weights,a.out)
