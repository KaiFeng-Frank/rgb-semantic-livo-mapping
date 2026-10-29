"""Raw semantic evidence in baseline visual tracks, TRAIN/DEV only.

Fixed algorithms see causal raw predictions, not labels. GT identifies same-object
availability after prediction. Current-frame pooling is a separate comparator;
its benefit must not be attributed to temporal memory. This is an availability
diagnosis, not a new trained model or a deployment result.
"""
import argparse
from collections import Counter,defaultdict,deque
from pathlib import Path
import numpy as np
from .data import read,save
from .metrics import identity_target,identity_truth
from tools.semantic_memory.nuscenes_pilot_data import digest


def run(train_cache,dev_cache,index_path,out):
    index=read(index_path);old=read(train_cache/'protocol.json');dev=read(dev_cache/'protocol.json')
    if any((dev_cache/(s+'.inputs.json.gz')).exists() for s in index['parts']['test']):
        raise ValueError('New holdout must remain closed')
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(train=old['parts']['train'],dev=dev['parts']['dev'],index_sha256=digest(index_path),
        train_cache_sha256=digest(train_cache/'protocol.json'),dev_cache_sha256=digest(dev_cache/'protocol.json'),
        source_sha256=digest(Path(__file__)),
        controls=['raw_point','current_pool','history_pool','half_raw_half_history'],
        history='Same baseline visual session/ID; last 6 raw observations within 8 seconds; strictly past; trusted components only',
        opportunity='GT-sidecar same-instance past class correct while current pooled class is wrong; unique objects and logs, not frame windows',
        decision='No training from this audit alone. Report TRAIN/DEV separately and require evidence beyond same-frame pooling and repeated sightings of one object.',
        scope='Post-failure availability diagnosis. Fixed causal controls are development diagnostics, privileged opportunities are not deployable performance.')
    save(out/'protocol.json',protocol);summaries={}
    for part,cache,names in [('train',train_cache,old['parts']['train']),('dev',dev_cache,dev['parts']['dev'])]:
        total=Counter();scene_rows=[];objects=set();logs=set()
        for name in names:
            frames=read(cache/(name+'.inputs.json.gz'))['frames'];targets=read(cache/(name+'.targets.json.gz'))['frames']
            memory=defaultdict(lambda:deque(maxlen=6));sidecar={};counts=Counter();details=[]
            for frame,target in zip(frames,targets):
                observations=[o for o in frame['observations'] if o.get('trusted',True)]
                predictions=[]
                for o in observations:
                    key=(o['visual_session'],o['visual_track'])
                    history=[h for h in memory[key] if 0<o['stamp']-h['stamp']<=8_000_000_000]
                    raw=np.asarray(o['raw_probabilities']);current=raw.argmax(-1)
                    pool=np.full(len(current),np.asarray(o['raw_mean']).argmax(),np.int64)
                    hp=np.mean([h['raw_mean'] for h in history],axis=0) if history else np.asarray(o['raw_mean'])
                    pred=dict(raw_point=current,current_pool=pool,
                        history_pool=np.full(len(current),hp.argmax(),np.int64) if history else pool.copy(),
                        half_raw_half_history=(.5*raw+.5*hp).argmax(-1) if history else current.copy())
                    predictions.append((o,history,pred))
                # All current outputs precede memory commit and GT inspection.
                for o in observations:memory[(o['visual_session'],o['visual_track'])].append(o)
                truth=identity_truth(target);gt=np.asarray(target['gt'])
                for o,history,pred in predictions:
                    at=np.asarray(o['indices']);y=gt[at];valid=y>=0;fg=valid&(y<10)
                    counts.update(observations=1,history_observations=bool(history),valid_points=int(valid.sum()),foreground_points=int(fg.sum()))
                    for arm,p in pred.items():
                        counts[arm+'_correct']+=int((valid&(p==y)).sum())
                        counts[arm+'_fg_correct']+=int((fg&(p==y)).sum())
                        counts[arm+'_fix_vs_raw']+=int((valid&(pred['raw_point']!=y)&(p==y)).sum())
                        counts[arm+'_harm_vs_raw']+=int((valid&(pred['raw_point']==y)&(p!=y)).sum())
                    ident=identity_target(o,target)
                    record_key=(o['stamp'],o['visual_track'])
                    sidecar[record_key]=ident
                    if ident is None:continue
                    correct_history=[h for h in history if sidecar.get((h['stamp'],h['visual_track']))==ident]
                    classes=[int(np.asarray(h['raw_mean']).argmax()) for h in correct_history]
                    counts['verified_same_instance_records']+=len(correct_history)
                    counts['unverified_or_wrong_instance_records']+=len(history)-len(correct_history)
                    # GT only judges whether a raw past class contains missing
                    # information. It never changes the actual control outputs.
                    avail=(truth[at]==ident)&fg&np.isin(y,classes)
                    for base in ['raw_point','current_pool']:
                        n=int((avail&(pred[base]!=y)).sum());counts['available_vs_'+base]+=n
                    n=int((avail&(pred['current_pool']!=y)).sum())
                    if n:
                        objects.add((name,ident));logs.add(index['scene_logs'][name])
                        details.append(dict(frame=frame['frame'],identity=ident,points=n,
                            current_class=int(pred['current_pool'][0]),past_classes=classes))
            total.update(counts);scene_rows.append(dict(scene=name,log=index['scene_logs'][name],counts=dict(counts),opportunities=details))
            print(part,name,dict(counts),flush=True)
        summaries[part]=dict(counts=dict(total),distinct_opportunity_instances=len(objects),distinct_opportunity_logs=len(logs),scenes=scene_rows)
    result=dict(parts=summaries,scope=protocol['scope']);save(out/'report.json',result)
    print({p:{k:v for k,v in r.items() if k!='scenes'} for p,r in summaries.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['train-cache','dev-cache','index','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.train_cache,a.dev_cache,a.index,a.out)
