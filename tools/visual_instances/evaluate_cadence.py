"""Frozen full-rate/keyframe-only tracking comparison on identical scored scans."""
import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
from tools.instance_memory.data import read,save
from tools.instance_memory.metrics import IdentityMetrics,identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest
from .evaluate_reid import merge


def run(index_path,frames_path,cache,images,weights,out):
    import cv2,torch
    from .cadence import PairedCadenceTrackers
    from .geometry import Camera,associate
    from .semantics import Config,component
    from .kalman_cache import gpu_admission
    torch.set_num_threads(1);torch.manual_seed(20261029);cv2.setNumThreads(1)
    index=read(index_path);dense=read(frames_path);cfg=Config()
    if list(dense['scenes'])!=index['independent_dev_added']:raise ValueError('Selection changed')
    manifest=read(images.parent/'manifest.json');hashes={r['name']:r['sha256'] for r in manifest['files']}
    if not manifest['complete']:raise ValueError('Incomplete image fetch')
    if any((cache/(s+'.inputs.json.gz')).exists() for s in index['parts']['test']):raise ValueError('Holdout must remain closed')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'protocol.json',dict(index_sha256=digest(index_path),frames_sha256=digest(frames_path),
        image_manifest_sha256=digest(images.parent/'manifest.json'),weights_sha256=digest(weights),admission=gpu_admission(),
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('cadence.py'),Path(__file__).with_name('reid.py'),Path(__file__).with_name('evidence_predictor.py')]},
        preregistration=read(frames_path.with_name('protocol.json'))))
    policies=['projected_mask','trusted_component','largest_component'];names=['keyframe','fullrate']
    arms=[n+'/'+p for n in names for p in policies];scenes=[];all_timing=[];starts=[]
    for scene_name,sequence in dense['scenes'].items():
        start=time.perf_counter();paired=PairedCadenceTrackers(weights);starts.append((time.perf_counter()-start)*1000)
        targets=read(cache/(scene_name+'.targets.json.gz'))['frames'];cached=read(cache/(scene_name+'.inputs.json.gz'))['frames']
        metrics={scope:{arm:IdentityMetrics() for arm in arms} for scope in ['full','camera']}
        differences=Counter();frames=[];scored=0
        for i,record in enumerate(sequence):
            if i%50==0:gpu_admission()
            k=record['keyframe'];path=DATA/record['path'] if k is not None else images/Path(record['path']).name
            expected=index['sequences'][scene_name][k]['image_sha256'] if k is not None else hashes[path.name]
            if digest(path)!=expected:raise ValueError('Image content changed')
            image=cv2.imread(str(path))
            if image is None or image.shape[:2]!=(record['height'],record['width']):raise ValueError('Invalid image dimensions')
            visual,timing=paired.step(image,int(record['timestamp_us']*1000),k is not None)
            all_timing.append(timing);frames.append(dict(timestamp_us=record['timestamp_us'],keyframe=k,timing=timing))
            if k is None:continue
            row=index['sequences'][scene_name][k];target=targets[k];old=cached[k];scored+=1
            scan=np.fromfile(DATA/row['scan_path'],dtype='<f4').reshape(-1,5)
            transform=np.linalg.inv(row['camera_transform'])@row['transform']
            camera=Camera(image.shape[1],image.shape[0],row['camera_intrinsic'],np.zeros(4),transform);predictions={}
            for name,frame in visual.items():
                raw=associate(scan[:,:3],camera,frame)['ids'].astype(np.int64);predictions[name+'/projected_mask']=raw
                for policy,fraction in [('trusted_component',cfg.component_fraction),('largest_component',0.)]:
                    ids=np.zeros(len(scan),np.int64)
                    for tid in np.unique(raw[raw>0]):
                        at=np.flatnonzero(raw==tid);chosen=component(scan[at,:3],replace(cfg,component_fraction=fraction))
                        if chosen is not None:ids[at[chosen]]=tid
                    predictions[name+'/'+policy]=ids
            for policy,trusted_only in [('trusted_component',True),('largest_component',False)]:
                expected_mask=np.zeros(len(scan),bool)
                for o in old['observations']:
                    if not trusted_only or o['trusted']:expected_mask[o['indices']]=True
                differences[policy]+=int((expected_mask!=(predictions['keyframe/'+policy]>0)).sum())
            # No ground-truth information was available to either tracker.
            truth=identity_truth(target);valid=(np.asarray(target['gt'])>=0)&(np.asarray(target['panoptic'])>0)
            inside=np.zeros(len(scan),bool);inside[target['camera_in_view_indices']]=True
            for scope,mask in [('full',valid),('camera',valid&inside)]:
                for arm,pred in predictions.items():metrics[scope][arm].add(truth,pred,k,mask)
        if scored!=len(targets):raise ValueError('Scored keyframe count mismatch')
        paired.close();del paired
        result=dict(scene=scene_name,log=index['scene_logs'][scene_name],scored_keyframes=scored,
            identity={scope:{arm:m.result() for arm,m in rows.items()} for scope,rows in metrics.items()},
            cached_keyframe_coverage_symmetric_difference=dict(differences),frames=frames)
        save(out/(scene_name+'.json'),result);scenes.append(result)
        print(scene_name,{a:100*v['point_idf1'] for a,v in result['identity']['camera'].items()},dict(differences),flush=True)
    identity={scope:{a:merge([s['identity'][scope][a] for s in scenes]) for a in arms} for scope in ['full','camera']}
    timing={n:[r['tracking_including_gmc_ms'][n] for r in all_timing if n in r['tracking_including_gmc_ms']] for n in names}
    timing['shared_detector']=[r['forward_ms'] for r in all_timing]
    gate={}
    for policy in policies:
        gain=100*(identity['camera']['fullrate/'+policy]['point_idf1']-identity['camera']['keyframe/'+policy]['point_idf1'])
        full_gain=identity['full']['fullrate/'+policy]['point_idf1']-identity['full']['keyframe/'+policy]['point_idf1']
        positive=sum(s['identity']['camera']['fullrate/'+policy]['point_idf1']>s['identity']['camera']['keyframe/'+policy]['point_idf1'] for s in scenes)
        gate[policy]=dict(camera_gain_pp=gain,full_scan_gain_pp=100*full_gain,positive_logs=positive,
            passed=gain>=1 and full_gain>=0 and positive>len(scenes)/2)
    report=dict(identity=identity,gates=gate,
        scenes=[s['scene'] for s in scenes],startup_ms=starts,
        timing={k:dict(frames=len(v),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),max_ms=max(v)) for k,v in timing.items()},
        cached_keyframe_coverage_symmetric_difference={p:sum(s['cached_keyframe_coverage_symmetric_difference'][p] for s in scenes) for p in ['trusted_component','largest_component']},
        scope='Extra causal image evidence for existing tracker. Identity only, no semantic/novel-network gain; shared keyframe detections; supplied poses; component timing excludes PTv3/IPC/map. Eight fixed development logs; holdout unopened.')
    save(out/'report.json',report);print(gate,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['index','frames','cache','images','weights','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.index,a.frames,a.cache,a.images,a.weights,a.out)
