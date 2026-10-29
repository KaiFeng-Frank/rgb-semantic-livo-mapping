"""Post-result observer of ownership coverage; no alternative semantic outputs."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from tools.instance_memory.data import read,save
from tools.instance_memory.metrics import identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest
from .point_history_audit import neighbors
from .point_evidence_memory import visual_arrays


def run(result_path,output):
    proto=read(result_path/'protocol.json');result=read(result_path/'report.json')
    index_path=Path(proto['index_path']);index=read(index_path)
    if digest(index_path)!=proto['index_sha256']:raise ValueError('Index changed')
    visual=Path(proto['visual_input_path']);targets=Path(proto['target_path'])
    reports=[]
    for scene in result['scenes']:
        if digest(visual/(scene+'.inputs.json.gz'))!=proto['input_sha256'][scene] or digest(targets/(scene+'.targets.json.gz'))!=proto['target_sha256'][scene]:
            raise ValueError('Input changed')
        vv=read(visual/(scene+'.inputs.json.gz'))['frames'];tt=read(targets/(scene+'.targets.json.gz'))['frames']
        count=Counter();old=None;objects={}
        for row,v,t in zip(index['sequences'][scene],vv,tt):
            path=DATA/row['scan_path']
            if digest(path)!=row['scan_sha256']:raise ValueError('Scan changed')
            scan=np.fromfile(path,dtype='<f4').reshape(-1,5)[:,:3]
            pose=np.asarray(row['transform']);xyz=scan.astype(np.float64)@pose[:3,:3].T+pose[:3,3]
            ids,cur,mem=visual_arrays(len(scan),v['observations']);stamp=v['stamp']
            valid_old=old is not None and 0<stamp-old['stamp']<=750000000
            if valid_old:j,valid=neighbors(xyz,old['xyz'])
            # GT below is only for coverage/error attribution.
            gt=np.asarray(t['gt']);truth=identity_truth(t);camera=np.zeros(len(gt),bool);camera[t['camera_in_view_indices']]=True
            fg=camera&(gt>=0)&(gt<10)
            count.update(foreground_queries=int(fg.sum()),foreground_with_visual_component=int((fg&(ids>0)).sum()))
            if valid_old:
                eq=(ids[:,None]>0)&(ids[:,None]==old['ids'][j])&valid
                truth_same=(truth[:,None]>0)&(truth[:,None]==old['truth'][j])&valid
                rejected=valid&(ids[:,None]!=old['ids'][j])
                masks=dict(foreground_with_any_history=fg&valid.any(1),
                    foreground_with_visual_and_any_history=fg&(ids>0)&valid.any(1),
                    foreground_with_same_positive_visual_id_history=fg&eq.any(1))
                count.update({k:int(m.sum()) for k,m in masks.items()})
                pair=dict(history_pairs=valid,positive_same_id_pairs=eq,
                    positive_same_id_and_same_gt_instance_pairs=eq&truth_same,
                    positive_same_id_with_known_gt_pairs=eq&(truth[:,None]>0)&(old['truth'][j]>0),
                    rejected_pairs=rejected,rejected_but_same_gt_instance_pairs=rejected&truth_same,
                    rejected_current_only_known=rejected&(ids[:,None]>0)&(old['ids'][j]==0),
                    rejected_history_only_known=rejected&(ids[:,None]==0)&(old['ids'][j]>0),
                    rejected_both_known_different=rejected&(ids[:,None]>0)&(old['ids'][j]>0))
                count.update({k:int((fg[:,None]&m).sum()) for k,m in pair.items()})
            for obj in np.unique(truth[fg&(truth>0)]):
                at=fg&(truth==obj);d=objects.setdefault(str(int(obj)),dict(points=0,current_visual_classes=Counter(),memory_visual_classes=Counter()))
                d['points']+=int(at.sum())
                for key,labels in [('current_visual_classes',cur),('memory_visual_classes',mem)]:
                    d[key].update({str(int(k)):int(n) for k,n in zip(*np.unique(labels[at],return_counts=True))})
            old=dict(xyz=xyz,ids=ids,truth=truth,stamp=stamp)
        reports.append(dict(scene=scene,counts=dict(count),objects=objects))
        print(scene,dict(count),flush=True)
    total=Counter()
    for r in reports:total.update(r['counts'])
    save(output,dict(scope='Post-result privileged correspondence coverage only, no new semantic policy or accuracy arm.',
        source_sha256=digest(Path(__file__)),protocol_sha256=digest(result_path/'protocol.json'),
        report_sha256=digest(result_path/'report.json'),counts=dict(total),scenes=reports))
    print('aggregate',dict(total),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.result,a.out)
