"""Post-training descriptive attribution; never selects a model or new policy."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from tools.instance_memory.data import read,save

CLASSES=('barrier','bicycle','bus','car','construction_vehicle','motorcycle',
         'pedestrian','traffic_cone','trailer','truck','driveable_surface','other_flat',
         'sidewalk','terrain','manmade','vegetation')


def episodes(rows):
    tracks=defaultdict(list)
    for r in rows:tracks[r['scene'],r['instance']].append(r)
    for track in tracks.values():
        track.sort(key=lambda r:r['frame']);runs=[]
        for r in track:
            if not runs or r['frame']!=runs[-1][-1]['frame']+1:runs.append([])
            runs[-1].append(r)
        for run in runs:
            i=0
            while i<len(run):
                if run[i]['correct_fraction']['baseline']>=.5:i+=1;continue
                begin=i
                while i<len(run) and run[i]['correct_fraction']['baseline']<.5:i+=1
                yield run[begin:min(i+1,len(run))]


def diagnose(out):
    candidate='balanced_temporal';same='balanced_temporal_no_history';results=[]
    for seed in (0,1,2):
        folder=out/'run'/f'seed-{seed}';r=read(folder/'evaluation.json');rows=read(folder/'object-observations.json')
        comparisons={}
        for b in ('standard_current','balanced_current',same):
            a=r['arms'];comparisons[b]=dict(miou_gain_pp=100*(a[candidate]['miou_fixed16']-a[b]['miou_fixed16']),
                camera_foreground_gain_pp=100*(a[candidate]['camera_foreground_accuracy']-a[b]['camera_foreground_accuracy']),
                positive_logs=sum(s[candidate]['miou_fixed16']>s[b]['miou_fixed16'] for s in r['per_scene'].values()),
                class_iou_gain_pp={name:100*((x or 0)-(y or 0)) for name,x,y in zip(CLASSES,a[candidate]['class_iou'],a[b]['class_iou'])})
        by_class={};lost=[];won=[];counts=defaultdict(int)
        for ep in episodes(rows):
            first=ep[0];label=CLASSES[first['label']]
            c=any(x['correct_fraction'][candidate]>=.5 for x in ep)
            b=any(x['correct_fraction'][same]>=.5 for x in ep)
            key='both_success' if b and c else 'history_rescued' if c else 'history_harmed' if b else 'both_failed'
            counts[key]+=1
            bucket=by_class.setdefault(label,dict(episodes=0,both_success=0,history_rescued=0,history_harmed=0,both_failed=0))
            bucket['episodes']+=1;bucket[key]+=1
            if b!=c:
                event=dict(scene=first['scene'],instance=first['instance'],label=label,start_frame=first['frame'],end_frame=ep[-1]['frame'],
                    frames=[dict(frame=x['frame'],points=x['points'],history=x['correct_fraction'][candidate],
                                 same_head_current=x['correct_fraction'][same]) for x in ep])
                (lost if b else won).append(event)
        declared=r['events']['baseline_error_recovery']
        assert sum(counts.values())==declared[candidate]['episodes']
        assert counts['both_success']+counts['history_rescued']==declared[candidate]['successes']
        assert counts['both_success']+counts['history_harmed']==declared[same]['successes']
        results.append(dict(seed=seed,comparisons=comparisons,error_recovery_same_head=dict(counts),
            error_recovery_by_class=by_class,lost_recovery_events=lost,rescued_recovery_events=won))
    aggregate={b:dict(mean_miou_gain_pp=float(np.mean([r['comparisons'][b]['miou_gain_pp'] for r in results])),
        mean_camera_foreground_gain_pp=float(np.mean([r['comparisons'][b]['camera_foreground_gain_pp'] for r in results])))
        for b in results[0]['comparisons']}
    report=dict(scope='Descriptive analysis of already evaluated development outputs. No checkpoint reselection, new thresholds or deployment claim.',
        aggregate_comparisons=aggregate,seeds=results)
    save(out/'attribution.json',report)
    print(aggregate)
    for r in results:print(r['seed'],r['error_recovery_same_head'],r['error_recovery_by_class'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    diagnose(p.parse_args().out)
