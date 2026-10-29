"""Frozen raw-context probe; semantic GT is read only by the scoring observer."""
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import sys
import time
import numpy as np
from .data import read,save
from .raw_context import RawContext
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest
from tools.semantic_memory.pilot_metrics import confusion,scores
from tools.visual_instances.kalman_cache import gpu_admission


class GTTrackSemanticObserver:
    def __init__(self):self.previous={};self.counts=Counter()
    def add(self,pred,gt,pan):
        for ident in np.unique(pan[(pan%1000>0)&(gt>=0)]):
            at=pan==ident
            if at.sum()<6:continue
            label=int(np.bincount(pred[at],minlength=16).argmax())
            truth=int(np.bincount(gt[at],minlength=16).argmax());correct=label==truth
            if ident in self.previous:
                old,was_correct=self.previous[ident]
                self.counts['transitions']+=1;self.counts['changed_predicted_class']+=old!=label
                self.counts['both_correct' if correct and was_correct else 'both_wrong' if not correct and not was_correct else 'wrong_to_correct' if correct else 'correct_to_wrong']+=1
            self.previous[ident]=(label,correct)


def run(index_path,cache,out):
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from ptv3_worker import Segmenter
    torch.set_num_threads(1);torch.manual_seed(20261029)
    out.mkdir(parents=True,exist_ok=False);admission=gpu_admission();index=read(index_path)
    start=time.perf_counter();model=Segmenter(half=True,shuffle=False,intensity_scale=1.)
    startup=time.perf_counter()-start
    modes=['current_only','unwarped','ego','motion'];matrices={m:np.zeros((16,16),np.int64) for m in modes}
    totals={m:Counter() for m in modes};timings={m:[] for m in modes};log_matrices={}
    semantic_transitions={m:Counter() for m in modes};scene_rows=[];duplicate_checks=[]
    save(out/'protocol.json',dict(index_sha256=digest(index_path),cache_protocol_sha256=digest(cache/'protocol.json'),
        checkpoint_sha256=digest(__import__('os').environ['PTV3_CHECKPOINT']),config=asdict(RawContext().cfg),
        source={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('raw_context.py')]},admission=admission,
        startup_seconds=startup,scope='Development only. Supplied ego poses. Cached causal detector associations. GT never enters raw context. No predicted identity accuracy is measured here.'))
    for name in index['parts']['dev']:
        memory=RawContext();scene={m:np.zeros((16,16),np.int64) for m in modes};counts={m:Counter() for m in modes}
        observers={m:GTTrackSemanticObserver() for m in modes};frames=[]
        observations=read(cache/(name+'.inputs.json.gz'))['frames'];targets=read(cache/(name+'.targets.json.gz'))['frames']
        for row,frame,target in zip(index['sequences'][name],observations,targets):
            if row['frame']%10==0:gpu_admission()
            scan=np.fromfile(DATA/row['scan_path'],dtype='<f4').reshape(-1,5)[:,:4].copy();scan[:,3]/=255.
            stamp=int(row['timestamp_us']*1000);pose=np.asarray(row['transform']);obs=frame['observations']
            torch.cuda.synchronize();start=time.perf_counter();base,_=model.segment(scan);base=base.copy();base_ms=(time.perf_counter()-start)*1000
            predictions={'current_only':base};info={'current_only':dict(history_points=0,prepare_ms=0.,forward_ms=base_ms,total_ms=base_ms)}
            for mode in modes[1:]:
                augmented,detail=memory.prepare(scan,stamp,pose,obs,mode)
                if detail['history_points']:
                    torch.cuda.synchronize();start=time.perf_counter();pred,_=model.segment(augmented)
                    milliseconds=(time.perf_counter()-start)*1000;pred=pred[:len(scan)].copy()
                else:pred=base.copy();milliseconds=base_ms
                detail.update(forward_ms=milliseconds,total_ms=milliseconds+detail['prepare_ms'])
                predictions[mode]=pred;info[mode]=detail
            # Verify the current-first voxel representative invariant on five
            # predetermined frames. This is not a second sampling/model search.
            if name==index['parts']['dev'][0] and row['frame'] in [0,1,2,3,4]:
                dup=np.concatenate([scan,scan[:min(12000,len(scan)//2)]])
                pred,_=model.segment(dup)
                changed=int((pred[:len(scan)]!=base).sum())
                duplicate_checks.append(dict(scene=name,frame=row['frame'],changed_current_labels=changed))
            memory.store(scan,stamp,pose,obs)
            # The algorithm's state is committed before the observer opens GT.
            gt=np.asarray(target['gt']);pan=np.asarray(target['panoptic']);valid=gt>=0
            for mode,pred in predictions.items():
                scene[mode]+=confusion(gt,pred)
                counts[mode].update(repaired=int((valid&(base!=gt)&(pred==gt)).sum()),harmed=int((valid&(base==gt)&(pred!=gt)).sum()),
                    history_points=info[mode]['history_points'],frames=1)
                timings[mode].append(info[mode]['total_ms']);observers[mode].add(pred,gt,pan)
            frames.append(dict(frame=row['frame'],modes=info))
            if row['frame']%10==0:print(name,row['frame'],{m:round(v['total_ms'],2) for m,v in info.items()},flush=True)
        log=index['scene_logs'][name];log_matrices.setdefault(log,{m:np.zeros((16,16),np.int64) for m in modes})
        for m in modes:
            matrices[m]+=scene[m];totals[m].update(counts[m]);log_matrices[log][m]+=scene[m]
            semantic_transitions[m].update(observers[m].counts)
        result=dict(scene=name,log=log,arms={m:dict(scores(scene[m]),counts=dict(counts[m]),gt_track_semantic_transitions=dict(observers[m].counts)) for m in modes},frames=frames)
        scene_rows.append(result);save(out/(name+'.json'),result)
        print('completed',name,{m:100*scores(scene[m])['miou_fixed16'] for m in modes},flush=True)
    result=dict(aggregate={m:dict(scores(matrices[m]),counts=dict(totals[m]),
        timing=dict(median_ms=float(np.median(timings[m])),p95_ms=float(np.percentile(timings[m],95)),max_ms=max(timings[m])),
        gt_track_semantic_transitions=dict(semantic_transitions[m])) for m in modes},
        collection_logs={log:{m:scores(mat) for m,mat in row.items()} for log,row in log_matrices.items()},
        duplicate_current_checks=duplicate_checks,scenes=[r['scene'] for r in scene_rows],
        scope='Frozen-backbone semantic information probe only; original raw history, 12 inspected development scenes. Timings exclude live detection, IPC and map publication. GT-track semantic transitions are not predicted instance tracking metrics.')
    save(out/'report.json',result);print({m:100*v['miou_fixed16'] for m,v in result['aggregate'].items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['index','cache','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.index,a.cache,a.out)
