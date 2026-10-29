"""Same-forward semantic evaluation of continuous raw visual memory."""
import argparse
from collections import Counter
from pathlib import Path
import sys,time
import numpy as np
from tools.instance_memory.data import read,save
from tools.instance_memory.metrics import IdentityMetrics,identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest
from tools.semantic_memory.pilot_metrics import confusion,scores
from .evaluate_reid import merge
from .visual_semantics import fusion,groups,distribution


def run(index_path,frames_path,cache,images,weights,reference,out):
    import cv2,torch
    from .fullrate import FullRateDetector
    from .geometry import Camera,associate
    from .kalman_cache import gpu_admission
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from ptv3_worker import Segmenter
    torch.set_num_threads(1);torch.manual_seed(20261029);cv2.setNumThreads(1)
    index=read(index_path);dense=read(frames_path);expected=read(reference/'report.json')
    if list(dense['scenes'])!=index['independent_dev_added']:raise ValueError('Selection mismatch')
    if any((cache/(s+'.inputs.json.gz')).exists() for s in index['parts']['test']):raise ValueError('Opened holdout refused')
    manifest=read(images.parent/'manifest.json');hashes={r['name']:r['sha256'] for r in manifest['files']}
    out.mkdir(parents=True,exist_ok=False);admission=gpu_admission()
    model=Segmenter(half=True,shuffle=False,intensity_scale=1.)
    save(out/'protocol.json',dict(preregistration=read(Path(__file__).resolve().parents[2]/'config/fullrate_visual_semantics_20261029.json'),
        index_sha256=digest(index_path),frames_sha256=digest(frames_path),image_manifest_sha256=digest(images.parent/'manifest.json'),
        ptv3_sha256=digest(__import__('os').environ['PTV3_CHECKPOINT']),yolo_sha256=digest(weights),admission=admission,
        source={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('fullrate.py'),Path(__file__).with_name('visual_memory.py'),Path(__file__).with_name('visual_semantics.py'),Path(__file__).with_name('geometry.py')]}))
    arms=['raw','current_image','visual_memory'];scenes=[];timings=[]
    for name,sequence in dense['scenes'].items():
        visual=FullRateDetector(weights,visual_history=True);targets=read(cache/(name+'.targets.json.gz'))['frames']
        matrices={a:np.zeros((16,16),np.int64) for a in arms};counts={a:Counter() for a in arms}
        ids={scope:IdentityMetrics() for scope in ['full','camera']};objects=Counter();availability=Counter();avail_objects=set()
        inputs=[];sidecars=[];frame_rows=[]
        for i,record in enumerate(sequence):
            if i%50==0:gpu_admission()
            k=record['keyframe'];path=DATA/record['path'] if k is not None else images/Path(record['path']).name
            wanted=index['sequences'][name][k]['image_sha256'] if k is not None else hashes[path.name]
            if digest(path)!=wanted:raise ValueError('Image hash mismatch')
            image=cv2.imread(str(path));frame,detail=visual.step(image,int(record['timestamp_us']*1000));timings.append(detail)
            if k is None:continue
            row=index['sequences'][name][k];scan=np.fromfile(DATA/row['scan_path'],dtype='<f4').reshape(-1,5)[:,:4].copy();scan[:,3]/=255.
            cam=Camera(image.shape[1],image.shape[0],row['camera_intrinsic'],np.zeros(4),np.linalg.inv(row['camera_transform'])@row['transform'])
            association=associate(scan[:,:3],cam,frame)
            raw=model.segment(scan,return_features=True)
            probabilities=raw['logits'].float().softmax(-1)[raw['inverse']].cpu().numpy()
            labels=probabilities.argmax(-1);pred={a:labels.copy() for a in arms};observations=[]
            start=time.perf_counter()
            for tid,at in groups(scan[:,:3],association):
                cur=association['class_scores'][tid];hist=association['temporal_class_scores'][tid]
                confidence=float(association['scores'][at[0]])
                pred['current_image'][at]=fusion(probabilities[at],cur,confidence).argmax(-1)
                pred['visual_memory'][at]=fusion(probabilities[at],hist,confidence).argmax(-1)
                observations.append(dict(indices=at.tolist(),raw_probabilities=probabilities[at].tolist(),
                    current_scores=cur.tolist(),memory_scores=hist.tolist(),confidence=confidence,
                    history_records=association['temporal_observations'][tid],visual_id=tid,
                    current_class=int(distribution(cur).argmax()),memory_class=int(distribution(hist).argmax())))
            semantic_ms=(time.perf_counter()-start)*1000
            # The observer opens targets after both candidate semantic outputs.
            target=targets[k];gt=np.asarray(target['gt']);pan=np.asarray(target['panoptic']);truth=identity_truth(target)
            valid=gt>=0;inside=np.zeros(len(scan),bool);inside[target['camera_in_view_indices']]=True
            camera_fg=inside&valid&(gt<10)
            for scope,mask in [('full',valid&(pan>0)),('camera',valid&(pan>0)&inside)]:ids[scope].add(truth,association['ids'],k,mask)
            for arm,p in pred.items():
                matrices[arm]+=confusion(gt,p)
                counts[arm].update(camera_foreground_points=int(camera_fg.sum()),camera_foreground_correct=int((camera_fg&(p==gt)).sum()),
                    raw_fixes=int((valid&(labels!=gt)&(p==gt)).sum()),raw_harms=int((valid&(labels==gt)&(p!=gt)).sum()),
                    current_fixes=int((valid&(pred['current_image']!=gt)&(p==gt)).sum()),current_harms=int((valid&(pred['current_image']==gt)&(p!=gt)).sum()))
            for sign,mask in [(1,(truth>0)&(pred['current_image']!=gt)&(pred['visual_memory']==gt)),(-1,(truth>0)&(pred['current_image']==gt)&(pred['visual_memory']!=gt))]:
                for obj,n in zip(*np.unique(truth[mask],return_counts=True)):objects[int(obj)]+=sign*int(n)
            target_rows=[]
            for o in observations:
                at=np.asarray(o['indices']);y=gt[at];t=truth[at];fg=(y>=0)&(y<10)
                # Privileged potential is reported separately from actual fusion.
                opportunity=fg&(t>0)&(labels[at]!=y)&(o['current_class']!=y)&(o['memory_class']==y)
                availability['unique_history_answer_points']+=int(opportunity.sum())
                for obj in np.unique(t[opportunity]):avail_objects.add(int(obj))
                target_rows.append(dict(gt=y.tolist(),identity=t.tolist()))
            inputs.append(dict(frame=k,stamp=int(row['timestamp_us']*1000),observations=observations))
            sidecars.append(dict(frame=k,observations=target_rows))
            frame_rows.append(dict(frame=k,visual=detail,semantic_ms=semantic_ms,observations=len(observations)))
        metrics={scope:m.result() for scope,m in ids.items()};old=read(reference/(name+'.json'))
        parity={scope:{key:(metrics[scope][key],old['identity'][scope]['fullrate/projected_mask'][key])
            for key in ['point_idf1','association_quality','gt_points','pred_points','identity_true_positive','id_switches'] if metrics[scope][key]!=old['identity'][scope]['fullrate/projected_mask'][key]} for scope in ids}
        report=dict(scene=name,log=index['scene_logs'][name],arms={a:dict(scores(matrices[a]),counts=dict(counts[a])) for a in arms},
            identity=metrics,reference_parity_differences=parity,net_corrected_objects=dict(objects),
            availability=dict(availability,instances=len(avail_objects)),frames=frame_rows)
        save(out/(name+'.json'),report);save(out/(name+'.inputs.json.gz'),dict(scene=name,frames=inputs));save(out/(name+'.targets.json.gz'),dict(scene=name,frames=sidecars));scenes.append(report)
        print(name,{a:100*v['miou_fixed16'] for a,v in report['arms'].items()},'parity',parity,flush=True)
        del visual
    aggregate={}
    for a in arms:
        mat=sum(np.asarray(s['arms'][a]['confusion']) for s in scenes) if 'confusion' in scenes[0]['arms'][a] else None
        # pilot_metrics.scores exposes the confusion matrix under its actual key.
        if mat is None:raise ValueError('Missing confusion matrix in semantic score')
        c=Counter()
        for s in scenes:c.update(s['arms'][a]['counts'])
        aggregate[a]=dict(scores(mat),counts=dict(c),camera_foreground_accuracy=c['camera_foreground_correct']/max(1,c['camera_foreground_points']))
    net_objects=sum(sum(v>0 for v in s['net_corrected_objects'].values()) for s in scenes)
    positive_logs=sum(s['arms']['visual_memory']['miou_fixed16']>s['arms']['current_image']['miou_fixed16'] for s in scenes)
    object_logs=sum(any(v>0 for v in s['net_corrected_objects'].values()) for s in scenes)
    gain=100*(aggregate['visual_memory']['miou_fixed16']-aggregate['current_image']['miou_fixed16'])
    raw_gain=100*(aggregate['visual_memory']['miou_fixed16']-aggregate['raw']['miou_fixed16'])
    fg_gain=100*(aggregate['visual_memory']['camera_foreground_accuracy']-aggregate['current_image']['camera_foreground_accuracy'])
    gate=dict(miou_gain_vs_current_pp=gain,miou_gain_vs_raw_pp=raw_gain,camera_foreground_gain_pp=fg_gain,
        net_positive_objects=net_objects,object_logs=object_logs,positive_logs=positive_logs,
        passed=gain>=.05 and raw_gain>=.05 and fg_gain>=.5 and net_objects>=5 and object_logs>=2 and positive_logs>4)
    result=dict(aggregate=aggregate,semantic_gate=gate,identity={scope:merge([s['identity'][scope] for s in scenes]) for scope in ids},
        reference_identity_parity=all(not d for s in scenes for d in s['reference_parity_differences'].values()),scenes=[s['scene'] for s in scenes],
        visual_timing=dict(median_ms=float(np.median([t['elapsed_ms'] for t in timings])),p95_ms=float(np.percentile([t['elapsed_ms'] for t in timings],95)),max_ms=max(t['elapsed_ms'] for t in timings)),
        scope='Development-only fixed semantic memory readout. Current/temporal share full-rate IDs, same components and PTv3 forward. Identity adapter parity is separately checked. Offline timings are not actual ROS delivery.')
    save(out/'report.json',result);print(gate,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['index','frames','cache','images','weights','reference','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.index,a.frames,a.cache,a.images,a.weights,a.reference,a.out)
