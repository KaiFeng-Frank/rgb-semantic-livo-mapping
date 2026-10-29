"""Privileged attribution of the frozen point-history failure; never deployment."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from tools.instance_memory.data import read,save
from tools.instance_memory.metrics import identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest
from tools.semantic_memory.pilot_metrics import confusion,scores
from .point_history_audit import RawPointHistory,vote


def truth_masks(gt,identity,old_gt,old_identity,indices,valid):
    """Evaluation-only class/identity filters; unknown thing IDs cannot match."""
    same_class=valid&(gt[:,None]>=0)&(old_gt[indices]==gt[:,None])
    same_object=same_class&((gt[:,None]>=10)|((identity[:,None]>0)&(old_identity[indices]==identity[:,None])))
    return same_class,same_object


def run(index_path,cache,reference,out):
    index=read(index_path);names=index['independent_dev_added'];baseline=read(reference/'report.json')
    if names!=baseline['scenes']:raise ValueError('Selection mismatch')
    if any((cache/(s+'.inputs.json.gz')).exists() for s in index['parts']['test']):raise ValueError('Holdout opened')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'protocol.json',dict(source_sha256=digest(Path(__file__)),index_sha256=digest(index_path),reference_sha256=digest(reference/'report.json'),
        arms=['raw','previous_raw','gt_same_class','gt_same_object'],
        scope='After reproducing the frozen raw-history control, GT filters only remove wrong correspondences; retained historical labels remain original predictions. No GT logits, training, parameter changes or deployment claims.'))
    arms=['raw','previous_raw','gt_same_class','gt_same_object'];reports=[]
    for name in names:
        rows=index['sequences'][name];targets=read(cache/(name+'.targets.json.gz'))['frames'];m=RawPointHistory();old_target=None
        mats={a:np.zeros((16,16),np.int64) for a in arms};counts=Counter();per_class={str(c):Counter() for c in range(16)}
        frame_rows=[]
        for k,(row,target) in enumerate(zip(rows,targets)):
            path=DATA/row['scan_path']
            if digest(path)!=row['scan_sha256']:raise ValueError('Scan changed')
            xyz=np.fromfile(path,dtype='<f4').reshape(-1,5)[:,:3];pose=np.asarray(row['transform']);world=xyz.astype(np.float64)@pose[:3,:3].T+pose[:3,3]
            raw=np.asarray(target['raw_labels']);pred,_,(hi,hv),old=m.step(world,raw,int(row['timestamp_us']*1000),name)
            # Everything below this line is an explicitly privileged observer.
            gt=np.asarray(target['gt']);identity=identity_truth(target);valid=gt>=0
            outputs={a:pred[a] for a in ['raw','previous_raw']}
            if old is not None:
                same_class,same_object=truth_masks(gt,identity,np.asarray(old_target['gt']),identity_truth(old_target),hi,hv)
                outputs['gt_same_class']=vote(raw,old['raw'],hi,same_class)
                outputs['gt_same_object']=vote(raw,old['raw'],hi,same_object)
                harm=valid&(raw==gt)&(pred['previous_raw']!=gt)
                fix=valid&(raw!=gt)&(pred['previous_raw']==gt)
                all_same=(~hv|same_class).all(1)&hv.any(1)
                clean_harm=valid&(raw==gt)&(outputs['gt_same_class']!=gt)
                removed_error=harm&(outputs['gt_same_class']==gt)
                made_harm=~harm&clean_harm
                category=dict(original_harms=harm,original_fixes=fix,harms_all_neighbors_same_gt_class=harm&all_same,
                    harms_removed_by_class_filter=removed_error,new_harms_after_class_filter=made_harm,
                    remaining_harms_with_class_filter=clean_harm,
                    wrong_current_clean_history_fixes=valid&(raw!=gt)&(outputs['gt_same_class']==gt),
                    original_harms_stuff=harm&(gt>=10),original_harms_things=harm&(gt<10))
                for key,mask in category.items():
                    counts[key]+=int(mask.sum())
                    for c,n in zip(*np.unique(gt[mask],return_counts=True)):per_class[str(int(c))][key]+=int(n)
                frame_rows.append(dict(frame=k,counts={a:int(mask.sum()) for a,mask in category.items()}))
            else:
                outputs['gt_same_class']=raw.copy();outputs['gt_same_object']=raw.copy()
            for a,p in outputs.items():mats[a]+=confusion(gt,p)
            old_target=target
        old_report=read(reference/(name+'.json'))
        for a in ['raw','previous_raw']:np.testing.assert_array_equal(mats[a],old_report['arms'][a]['confusion'])
        report=dict(scene=name,log=index['scene_logs'][name],arms={a:scores(mat) for a,mat in mats.items()},counts=dict(counts),per_class={k:dict(v) for k,v in per_class.items()},frames=frame_rows,reference_exact=True)
        save(out/(name+'.json'),report);reports.append(report);print(name,{a:100*d['miou_fixed16'] for a,d in report['arms'].items()},dict(counts),flush=True)
    aggregate={a:scores(sum(np.asarray(s['arms'][a]['confusion']) for s in reports)) for a in arms};counts=Counter()
    for r in reports:counts.update(r['counts'])
    result=dict(aggregate=aggregate,counts=dict(counts),scenes=names,all_original_confusions_exact=True,
        positive_logs_class_filter_vs_raw=sum(r['arms']['gt_same_class']['miou_fixed16']>r['arms']['raw']['miou_fixed16'] for r in reports),
        scope='Privileged diagnosis only. GT class/identity filtering is not an algorithm result or a general oracle upper bound.')
    save(out/'report.json',result);print('aggregate',{a:100*d['miou_fixed16'] for a,d in aggregate.items()},dict(counts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['index','cache','reference','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.index,a.cache,a.reference,a.out)
