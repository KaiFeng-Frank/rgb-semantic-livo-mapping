"""Same-forward full-point semantic comparison with a fixed probability readout."""
import argparse
from collections import Counter
from pathlib import Path
import sys,time
import numpy as np
from tools.instance_memory.data import read,save
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest
from tools.semantic_memory.pilot_metrics import confusion,scores
from .point_probability_memory import RawProbabilityMemory


def run(index_path,cache,out):
    import torch
    from tools.visual_instances.kalman_cache import gpu_admission
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from ptv3_worker import Segmenter
    torch.set_num_threads(1);torch.manual_seed(20261029)
    index=read(index_path);names=index['independent_dev_added']
    if len(names)!=8:raise ValueError('Frozen selection required')
    if any((cache/(s+'.inputs.json.gz')).exists() for s in index['parts']['test']):raise ValueError('Holdout opened')
    out.mkdir(parents=True,exist_ok=False);admission=gpu_admission()
    config=Path(__file__).resolve().parents[2]/'config/point_probability_memory_20261029.json'
    save(out/'protocol.json',dict(config=read(config),config_sha256=digest(config),index_sha256=digest(index_path),
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('point_probability_memory.py'),Path(__file__).with_name('point_history_audit.py')]},
        checkpoint_sha256=digest(__import__('os').environ['PTV3_CHECKPOINT']),admission=admission))
    model=Segmenter(half=True,shuffle=False,intensity_scale=1.)
    arms=['raw','current_spatial_probability','previous_raw_probability','previous_hard_vote'];reports=[];timing=[];backbone=[]
    for name in names:
        targets=read(cache/(name+'.targets.json.gz'))['frames'];rows=index['sequences'][name];memory=RawProbabilityMemory()
        mats={a:np.zeros((16,16),np.int64) for a in arms};counts={a:Counter() for a in arms};frame_rows=[];different=0
        for k,(row,target) in enumerate(zip(rows,targets)):
            path=DATA/row['scan_path']
            if digest(path)!=row['scan_sha256']:raise ValueError('Scan changed')
            scan=np.fromfile(path,dtype='<f4').reshape(-1,5)[:,:4].copy();scan[:,3]/=255.
            pose=np.asarray(row['transform']);xyz=scan[:,:3].astype(np.float64)@pose[:3,:3].T+pose[:3,3]
            torch.cuda.synchronize();start=time.perf_counter();raw=model.segment(scan,return_features=True)
            p=raw['logits'].float().softmax(-1)[raw['inverse']].cpu().numpy();torch.cuda.synchronize();backbone.append((time.perf_counter()-start)*1000)
            start=time.perf_counter();pred,details=memory.step(xyz,p,int(row['timestamp_us']*1000),name);timing.append((time.perf_counter()-start)*1000)
            # All arms are finalized before the observer reads labels/old results.
            gt=np.asarray(target['gt']);good=gt>=0;camera=np.zeros(len(gt),bool);camera[target['camera_in_view_indices']]=True;fg=good&camera&(gt<10)
            different+=int((pred['raw']!=target['raw_labels']).sum())
            for a,y in pred.items():
                mats[a]+=confusion(gt,y);counts[a].update(fixes_raw=int((good&(pred['raw']!=gt)&(y==gt)).sum()),
                    harms_raw=int((good&(pred['raw']==gt)&(y!=gt)).sum()),camera_foreground_correct=int((fg&(y==gt)).sum()),camera_foreground_points=int(fg.sum()))
            frame_rows.append(dict(frame=k,memory_ms=timing[-1],backbone_ms=backbone[-1],**details))
        report=dict(scene=name,log=index['scene_logs'][name],arms={a:dict(scores(mats[a]),counts=dict(counts[a])) for a in arms},frames=frame_rows,
            raw_label_differences_from_old_cache=different)
        save(out/(name+'.json'),report);reports.append(report);print(name,{a:100*v['miou_fixed16'] for a,v in report['arms'].items()},'raw differences',different,flush=True)
    aggregate={}
    for a in arms:
        c=Counter()
        for r in reports:c.update(r['arms'][a]['counts'])
        aggregate[a]=dict(scores(sum(np.asarray(r['arms'][a]['confusion']) for r in reports)),counts=dict(c),camera_foreground_accuracy=c['camera_foreground_correct']/max(1,c['camera_foreground_points']))
    history=aggregate['previous_raw_probability'];raw_gain=100*(history['miou_fixed16']-aggregate['raw']['miou_fixed16'])
    current_gain=100*(history['miou_fixed16']-aggregate['current_spatial_probability']['miou_fixed16'])
    fg_gain=100*(history['camera_foreground_accuracy']-max(aggregate[a]['camera_foreground_accuracy'] for a in ['raw','current_spatial_probability']))
    positive=sum(r['arms']['previous_raw_probability']['miou_fixed16']>max(r['arms'][a]['miou_fixed16'] for a in ['raw','current_spatial_probability']) for r in reports)
    gate=dict(raw_gain_pp=raw_gain,current_gain_pp=current_gain,foreground_gain_pp=fg_gain,positive_logs_vs_both=positive,
        passed=raw_gain>=.05 and current_gain>=.05 and positive>=5 and fg_gain>=-.1)
    result=dict(aggregate=aggregate,gate=gate,scenes=names,raw_label_differences_from_old_cache=sum(r['raw_label_differences_from_old_cache'] for r in reports),
        timing_ms={key:dict(median=float(np.median(v)),p95=float(np.percentile(v,95)),max=max(v)) for key,v in [('cpu_operator',timing),('backbone',backbone)]},
        scope='Development, fixed equal-frame probability baseline, one current PTv3 forward shared across controls; no GT correspondence or training. Component timing only.')
    save(out/'report.json',result);print(gate,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['index','cache','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.index,a.cache,a.out)
