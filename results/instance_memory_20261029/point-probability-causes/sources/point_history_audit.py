"""Bounded raw point-history vote; labels are confined to the observer in run()."""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from tools.instance_memory.data import read,save
from tools.instance_memory.metrics import identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest
from tools.semantic_memory.pilot_metrics import confusion,scores


def neighbors(query, source, *, same_scan=False):
    """Fixed four neighbors within 25 cm; never substitute the query for a miss."""
    n=len(query)
    if not len(source):return np.zeros((n,4),np.int64),np.zeros((n,4),bool)
    d,j=cKDTree(source).query(query,k=5 if same_scan else 4,distance_upper_bound=.25,workers=1)
    valid=np.isfinite(d)&(j<len(source))
    if same_scan:
        valid&=j!=np.arange(n)[:,None]
        # Stable filtering preserves distance ordering, including duplicate XYZ.
        order=np.argsort(~valid,axis=1,kind='stable')[:,:4]
        j=np.take_along_axis(j,order,1);valid=np.take_along_axis(valid,order,1)
    return np.minimum(j,len(source)-1),valid


def vote(raw, source_labels, indices, valid):
    counts=np.eye(16,dtype=np.int16)[raw]
    if len(source_labels):
        for col in range(4):
            at=np.flatnonzero(valid[:,col]);counts[at,source_labels[indices[at,col]]]+=1
    pred=counts.argmax(1)
    preserve=counts[np.arange(len(raw)),raw]==counts.max(1)
    pred[preserve]=raw[preserve]
    return pred


class RawPointHistory:
    """One raw frame, read before write; epoch and timestamp resets are explicit."""
    def __init__(self):self.previous=None
    def step(self,xyz,raw,stamp,epoch):
        xyz=np.asarray(xyz,np.float64);raw=np.asarray(raw,np.int64)
        if xyz.shape!=(len(raw),3) or not np.isfinite(xyz).all() or np.any((raw<0)|(raw>=16)):
            raise ValueError('Invalid raw point inputs')
        if len(raw)>200000 or type(stamp) is not int:raise ValueError('Invalid scan budget/timestamp')
        old=self.previous
        if old is not None and old['epoch']==epoch and stamp<=old['stamp']:raise ValueError('Noncausal stamp')
        if old is None or old['epoch']!=epoch or stamp-old['stamp']>750000000:old=None
        spatial=neighbors(xyz,xyz,same_scan=True)
        historical=neighbors(xyz,old['xyz'] if old is not None else np.empty((0,3)))
        pred=dict(raw=raw.copy(),current_spatial=vote(raw,raw,*spatial),
                  previous_raw=vote(raw,old['raw'] if old is not None else np.empty(0,np.int64),*historical))
        self.previous=dict(xyz=xyz.copy(),raw=raw.copy(),stamp=stamp,epoch=epoch)
        return pred,spatial,historical,old


def run(index_path,cache,out):
    index=read(index_path);names=index['independent_dev_added']
    if len(names)!=8:raise ValueError('Frozen eight-log development selection required')
    if any((cache/(s+'.inputs.json.gz')).exists() for s in index['parts']['test']):raise ValueError('Holdout opened')
    out.mkdir(parents=True,exist_ok=False)
    config=Path(__file__).resolve().parents[2]/'config/point_history_audit_20261029.json'
    save(out/'protocol.json',dict(config=read(config),config_sha256=digest(config),source_sha256=digest(Path(__file__)),
        index_sha256=digest(index_path),cache_sha256={s:digest(cache/(s+'.targets.json.gz')) for s in names}))
    arms=['raw','current_spatial','previous_raw'];reports=[];timings=[]
    for name in names:
        targets=read(cache/(name+'.targets.json.gz'))['frames'];rows=index['sequences'][name]
        memory=RawPointHistory();previous_target=None
        matrices={a:np.zeros((16,16),np.int64) for a in arms};counts={a:Counter() for a in arms}
        availability=Counter();classes=Counter();instances=Counter();frame_results=[]
        for k,(row,target) in enumerate(zip(rows,targets)):
            path=DATA/row['scan_path']
            if digest(path)!=row['scan_sha256']:raise ValueError('Scan changed')
            xyz=np.fromfile(path,dtype='<f4').reshape(-1,5)[:,:3];pose=np.asarray(row['transform'])
            world=xyz.astype(np.float64)@pose[:3,:3].T+pose[:3,3]
            raw=np.asarray(target['raw_labels'],np.int64)
            start=time.perf_counter();pred,spatial,historical,old=memory.step(world,raw,int(row['timestamp_us']*1000),name)
            elapsed=(time.perf_counter()-start)*1000;timings.append(elapsed)
            # The deployment operator above has no target argument or GT gating.
            gt=np.asarray(target['gt']);truth=identity_truth(target);good=gt>=0
            for a,p in pred.items():
                matrices[a]+=confusion(gt,p)
                counts[a].update(fixes_raw=int((good&(raw!=gt)&(p==gt)).sum()),harms_raw=int((good&(raw==gt)&(p!=gt)).sum()),
                    fixes_spatial=int((good&(pred['current_spatial']!=gt)&(p==gt)).sum()),harms_spatial=int((good&(pred['current_spatial']==gt)&(p!=gt)).sum()))
            si,sv=spatial;hi,hv=historical
            current_answer=(sv&(raw[si]==gt[:,None])).any(1)
            unique=np.zeros(len(gt),bool)
            if old is not None:
                pgt=np.asarray(previous_target['gt']);pid=identity_truth(previous_target)
                same=hv&(pgt[hi]==gt[:,None])&good[:,None]
                # Stuff requires matching GT class; a thing requires exact panoptic ID.
                same&=(gt[:,None]>=10)|((truth[:,None]>0)&(pid[hi]==truth[:,None]))
                correct=same&(old['raw'][hi]==gt[:,None])
                opportunity=good&(raw!=gt)&correct.any(1)
                unique=opportunity&~current_answer
                availability.update(matched_pairs=int(hv.sum()),correct_correspondence_pairs=int(same.sum()),
                    error_points=int((good&(raw!=gt)).sum()),history_answer_points=int(opportunity.sum()),unique_history_answer_points=int(unique.sum()))
                for c,n in zip(*np.unique(gt[unique],return_counts=True)):classes[int(c)]+=int(n)
                for obj,n in zip(*np.unique(truth[unique&(truth>0)],return_counts=True)):instances[int(obj)]+=int(n)
            frame_results.append(dict(frame=k,points=len(raw),memory_ms=elapsed,historical_queries=int(hv.any(1).sum()),unique_history_answer_points=int(unique.sum())))
            previous_target=target
        report=dict(scene=name,log=index['scene_logs'][name],arms={a:dict(scores(matrices[a]),counts=dict(counts[a])) for a in arms},
            availability=dict(availability),unique_classes=dict(classes),unique_instances=dict(instances),frames=frame_results)
        reports.append(report);save(out/(name+'.json'),report)
        print(name,{a:100*d['miou_fixed16'] for a,d in report['arms'].items()},dict(availability),flush=True)
    aggregate={}
    for a in arms:
        count=Counter()
        for r in reports:count.update(r['arms'][a]['counts'])
        aggregate[a]=dict(scores(sum(np.asarray(r['arms'][a]['confusion']) for r in reports)),counts=dict(count))
    raw_gain=100*(aggregate['previous_raw']['miou_fixed16']-aggregate['raw']['miou_fixed16'])
    spatial_gain=100*(aggregate['previous_raw']['miou_fixed16']-aggregate['current_spatial']['miou_fixed16'])
    logs=sum(r['arms']['previous_raw']['miou_fixed16']>r['arms']['current_spatial']['miou_fixed16'] for r in reports)
    classes=set(c for r in reports for c in r['unique_classes']);opportunity_logs=sum(bool(r['unique_classes']) for r in reports)
    gate=dict(raw_gain_pp=raw_gain,spatial_gain_pp=spatial_gain,positive_logs=logs,opportunity_classes=len(classes),opportunity_logs=opportunity_logs,
        passed=raw_gain>=.05 and spatial_gain>=.05 and logs>=5 and len(classes)>=3 and opportunity_logs>=3)
    result=dict(aggregate=aggregate,gate=gate,scenes=names,
        cpu_operator_ms=dict(median=float(np.median(timings)),p95=float(np.percentile(timings,95)),max=max(timings)),
        scope='Fixed cached raw labels; causal raw-history-only control. GT correspondence counts are privileged diagnostics, not predictions. Not an online GPU timing.')
    save(out/'report.json',result);print(gate,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['index','cache','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.index,a.cache,a.out)
