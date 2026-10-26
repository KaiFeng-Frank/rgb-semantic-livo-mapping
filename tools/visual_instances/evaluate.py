"""Real-image + LiDAR pilot with evaluation-only nuScenes labels/boxes.

Uses the six already inspected consistency scenes; this is a development
diagnostic, not a new held-out claim. No raw images/scans/feature caches saved.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .detector import Detector
from .geometry import Camera, associate
from tools.semantic_memory.nuscenes_pilot_data import ROOT, DATA, records, transform, digest, LABEL_MAP


def timing(v):
    return dict(n=len(v), median_ms=float(np.median(v)), p95_ms=float(np.percentile(v,95)), max_ms=float(max(v))) if v else None


def run(args):
    import cv2
    from pyquaternion import Quaternion
    args.out.mkdir(parents=True, exist_ok=False)
    index = json.loads(args.index.read_text())
    source = json.loads((ROOT/'runs/s0_formal/val_manifest.json').read_text())
    manifests = {r['sample_data_token']:r for r in source['samples']}
    tokens = {manifests[r['sample_data_token']]['camera_token'] for rows in index['sequences'].values() for r in rows}
    meta = DATA/'v1.0-trainval'
    cams = {r['token']:r for r in records(meta/'sample_data.json') if r['token'] in tokens}
    pose_tokens = {r['ego_pose_token'] for r in cams.values()}
    poses = {r['token']:r for r in records(meta/'ego_pose.json') if r['token'] in pose_tokens}
    calibrations = {r['token']:r for r in records(meta/'calibrated_sensor.json')}
    protocol = dict(index_sha256=digest(args.index), scenes=list(index['sequences']),
        supervision='No new training. COCO-pretrained YOLO11n-seg + BoT-SORT without ReID.',
        association='Camera masks eroded 2 pixels; overlap abstention; all-point 3-pixel z-buffer, 0.4m tolerance, >=3 points.',
        input='raw nuScenes scans, provided ego poses at camera/LiDAR time, front camera only, 2Hz keyframes',
        task=('Same-forward instance semantic fusion ablation; persistent map unchanged.' if getattr(args,'semantic_fusion',False)
              else 'Association only; PTv3 semantics and persistent map unchanged.'),
        limits=['Development scenes previously used for consistency; not new held-out evidence.',
                'Boxes + lidarseg provide an instance proxy; not official panoptic masks or LSTQ.',
                'Camera/LiDAR object motion between acquisition times is not compensated.',
                'Depth gates cannot reject all unmeasured occlusions; tracker IDs may switch.'],
        evaluation_only='GT labels/boxes never enter Detector or associate.',
        source={str(p):digest(p) for p in [Path(__file__), Path(__file__).with_name('geometry.py'),
                                         Path(__file__).with_name('detector.py'), Path(__file__).with_name('botsort.yaml')]})
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    segmenter = None
    semantic_scenes = []
    if getattr(args, 'semantic_fusion', False):
        import sys
        import torch
        from .semantics import InstanceSemantics
        from tools.semantic_memory.pilot_metrics import confusion, scores
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
        from ptv3_worker import Segmenter
        torch.set_num_threads(1); torch.manual_seed(20261026)
        segmenter = Segmenter(half=True, shuffle=False, intensity_scale=1.)
        (args.out/'semantic-protocol.json').write_text(json.dumps(dict(
            config=InstanceSemantics().configuration(),
            controls=['raw', 'single_frame_instance', 'track_only', 'guarded_without_conflict', 'guarded'],
            model_sha256=segmenter.memory_model_sha256(),
            scope='Same-forward native16 development diagnostic. GT only after inference. Rules revised on these same scenes; no numeric parameter sweep; not held-out.',
            source={str(p):digest(p) for p in [Path(__file__).with_name('semantics.py'), Path(__file__).with_name('geometry.py'), Path(__file__).with_name('detector.py')]}),indent=2)+'\n')
    audit = None
    if getattr(args, 'audit_instances', False):
        from .audit import AssociationAudit
        audit = AssociationAudit(args.out)
    summaries, allrows = [], []
    for scene, rows in index['sequences'].items():
        if segmenter is not None:
            cores = dict(single_frame_instance=InstanceSemantics(temporal=False),
                         track_only=InstanceSemantics(guard=False),
                         guarded_without_conflict=InstanceSemantics(conflict_guard=False), guarded=InstanceSemantics())
            matrices = {a:np.zeros((16,16),np.int64) for a in ['raw',*cores]}
            semantic_counts = {a:Counter() for a in matrices}
            semantic_times = {a:[] for a in cores}
            semantic_frames = []
        detector = Detector(args.weights)
        (args.out/'model.json').write_text(json.dumps(detector.meta,indent=2)+'\n')
        track_history = {}
        last_gt_track = {}
        totals = Counter()
        by_class = defaultdict(Counter)
        infer_times, projection_times = [], []
        for i, row in enumerate(rows):
            src = manifests[row['sample_data_token']]
            im_path = DATA/src['image_path']
            if digest(im_path) != src['image_sha256'] or digest(DATA/row['scan_path']) != row['scan_sha256']:
                raise ValueError('Input image/scan hash differs from source manifest')
            image = cv2.imread(str(im_path))
            if image is None:
                raise ValueError('Missing source image')
            camera_row = cams[src['camera_token']]
            cal = calibrations[camera_row['calibrated_sensor_token']]
            t_world_camera = transform(poses[camera_row['ego_pose_token']])@transform(cal)
            t_world_lidar = np.asarray(row['transform'])
            camera = Camera(image.shape[1],image.shape[0],cal['camera_intrinsic'],np.zeros(4),
                            np.linalg.inv(t_world_camera)@t_world_lidar)
            xyz = np.fromfile(DATA/row['scan_path'],dtype='<f4').reshape(-1,5)[:,:3]
            f, info = detector.step(image,int(src['camera_timestamp_us']*1000))
            start = time.perf_counter()
            result = associate(xyz,camera,f)
            elapsed = (time.perf_counter()-start)*1000
            infer_times.append(info['elapsed_ms']); projection_times.append(elapsed)
            if segmenter is not None:
                points = np.fromfile(DATA/row['scan_path'],dtype='<f4').reshape(-1,5)[:,:4].copy()
                points[:,3] /= 255.
                raw_labels, raw_confidence, probabilities = segmenter.segment(points, return_probabilities=True)
                predictions = dict(raw=raw_labels)
                enhanced = {a:core.step(xyz, probabilities, result, stamp_ns=int(row['timestamp_us']*1000),
                                        pose=t_world_lidar) for a,core in cores.items()}
                predictions.update({a:r['labels'] for a,r in enhanced.items()})
            # All GT access below is evaluation-only, after producing predictions.
            gt = LABEL_MAP[np.fromfile(DATA/row['label_path'],np.uint8)]
            if segmenter is not None:
                valid_gt = gt >= 0
                paired = {}
                for arm, labels in predictions.items():
                    matrices[arm] += confusion(gt, labels, 16)
                    c = dict(rescued=int((valid_gt & (raw_labels!=gt) & (labels==gt)).sum()),
                             harmed=int((valid_gt & (raw_labels==gt) & (labels!=gt)).sum()),
                             changed=int((labels!=raw_labels).sum()))
                    if arm in enhanced:
                        c.update({k:v for k,v in enhanced[arm]['info'].items() if isinstance(v,int)})
                        semantic_times[arm].append(enhanced[arm]['info']['elapsed_ms'])
                    semantic_counts[arm].update(c); paired[arm]=c
                evidence=[]
                from .semantics import component
                for item in enhanced['guarded']['info']['instances']:
                    tid=item['track_id'];at=np.flatnonzero(result['ids']==tid)
                    at=at[component(xyz[at],cores['guarded'].cfg)]
                    evidence.append(dict(track_id=tid,indices=at.tolist(),
                        raw_labels=raw_labels[at].tolist(),gt=gt[at].tolist(),
                        raw_confidence=raw_confidence[at].tolist(),
                        raw_mean=probabilities[at].mean(0).tolist(),
                        visual_label=int(result['labels'][at[0]]),visual_score=float(result['scores'][at[0]]),
                        predictions={a:labels[at].tolist() for a,labels in predictions.items()}))
                semantic_frames.append(dict(frame=i, paired=paired, guarded=enhanced['guarded']['info'],evidence=evidence,
                    posterior_sha256=__import__('hashlib').sha256(probabilities.tobytes()).hexdigest()))
            cp = xyz@camera.T_camera_lidar[:3,:3].T+camera.T_camera_lidar[:3,3]
            uv = cp@camera.K.T
            uv = uv[:,:2]/np.maximum(cp[:,2:3],1e-6)
            visible_things = (cp[:,2]>.1)&(cp[:,2]<80)&(uv[:,0]>=0)&(uv[:,0]<image.shape[1])&(uv[:,1]>=0)&(uv[:,1]<image.shape[0])&np.isin(gt,[1,2,3,5,6,9])
            world = xyz@t_world_lidar[:3,:3].T+t_world_lidar[:3,3]
            object_ids = np.full(len(xyz),-1,np.int32)
            overlap = np.zeros(len(xyz),bool)
            for oi, obj in enumerate(row['objects']):
                r = Quaternion(obj['rotation']).rotation_matrix
                local = (world-np.asarray(obj['translation']))@r
                half = np.asarray(obj['size'])[[1,0,2]]*.5
                mask = (np.abs(local)<=half+1e-5).all(1) & (gt==obj['label'])
                overlap |= mask & (object_ids>=0)
                object_ids[mask] = oi
            object_ids[overlap] = -1
            if audit is not None:
                boxes = detector.model.predictor.results[0].boxes
                tracked_boxes = ({int(t): b for t,b in zip(boxes.id.cpu().tolist(), boxes.xyxy.cpu().tolist())}
                                 if boxes.id is not None else {})
                audit.step(scene, i, row, src, f, result, gt, object_ids, tracked_boxes)
            associated = result['ids']>0
            valid = associated & (gt>=0)
            correct = valid & (result['labels']==gt)
            counts = Counter(points=len(xyz),associated=int(associated.sum()),semantic_labeled=int(valid.sum()),
                             gt_thing_points_in_frustum=int(visible_things.sum()), associated_gt_thing_points=int((associated&visible_things).sum()),
                             semantic_correct=int(correct.sum()),instance_box_supported=int((associated&(object_ids>=0)).sum()))
            current_gt_tracks = {}
            for tid in np.unique(result['ids'][associated]):
                pts = np.flatnonzero(result['ids']==tid)
                supported = object_ids[pts]; supported = supported[supported>=0]
                if len(supported)<3:
                    counts['unsupported_instances'] += 1
                    continue
                oi, votes = Counter(supported.tolist()).most_common(1)[0]
                counts['within_mask_dominant_points'] += votes
                counts['within_mask_box_points'] += len(supported)
                token = row['objects'][oi]['instance']
                key = (f.session,int(tid))
                if key in track_history and track_history[key][0]==i-1:
                    counts['adjacent_track_pairs'] += 1
                    counts['wrong_identity_pairs'] += int(track_history[key][1]!=token)
                track_history[key]=(i,token)
                if token not in current_gt_tracks or votes>current_gt_tracks[token][1]:
                    current_gt_tracks[token]=(key,votes)
            for token,(key,votes) in current_gt_tracks.items():
                if token in last_gt_track and last_gt_track[token][0]==i-1:
                    counts['adjacent_gt_pairs']+=1
                    counts['id_switches']+=int(last_gt_track[token][1]!=key)
                last_gt_track[token]=(i,key)
            for cls in np.unique(result['labels'][associated]):
                at=valid&(result['labels']==cls)
                by_class[int(cls)].update(associated=int(at.sum()),correct=int((at&(gt==cls)).sum()))
            totals.update(counts)
            rowout=dict(scene=scene,frame=i,stamp_ns=f.stamp_ns,camera_lidar_dt_ms=(row['timestamp_us']-src['camera_timestamp_us'])/1000,
                        counts=dict(counts),association=result['info'],infer_ms=info['elapsed_ms'],projection_ms=elapsed)
            allrows.append(rowout)
            if i%10==0:
                print(json.dumps(dict(scene=scene,frame=i,associated=counts['associated'],instances=info['instances'])),flush=True)
        summary=dict(scene=scene,frames=len(rows),counts=dict(totals),by_visual_class=dict(by_class),
                     inference=timing(infer_times),projection=timing(projection_times))
        summaries.append(summary)
        if segmenter is not None:
            semantic_scenes.append(dict(scene=scene,frames=len(rows),arms={a:dict(scores(matrices[a]),
                counts=dict(semantic_counts[a]),timing=timing(semantic_times[a]) if a in cores else None) for a in matrices}))
            (args.out/f'semantic-{scene}.json').write_text(json.dumps(semantic_scenes[-1],indent=2)+'\n')
            (args.out/f'semantic-{scene}-frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in semantic_frames))
            print(json.dumps(dict(scene=scene,semantic_miou={a:100*scores(matrices[a])['miou_fixed16'] for a in matrices})),flush=True)
        (args.out/'scenes.json').write_text(json.dumps(summaries,indent=2)+'\n')
        del detector
    total=Counter()
    for s in summaries:total.update(s['counts'])
    report=dict(frames=len(allrows),counts=dict(total),
        thing_frustum_coverage=total['associated_gt_thing_points']/max(1,total['gt_thing_points_in_frustum']),
        semantic_precision=total['semantic_correct']/max(1,total['semantic_labeled']),
        within_mask_box_purity=total['within_mask_dominant_points']/max(1,total['within_mask_box_points']),
        adjacent_identity_agreement=1-total['wrong_identity_pairs']/max(1,total['adjacent_track_pairs']),
        inference=timing([r['infer_ms'] for r in allrows]), projection=timing([r['projection_ms'] for r in allrows]),
        scope='Association pilot only; no claim of semantic mIoU gain, official tracking score or full-view coverage.')
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.out/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allrows))
    if audit is not None:
        audited = audit.finish()
        assert audited['adjacent_track_pairs'] == total['adjacent_track_pairs']
        assert audited['wrong_identity_pairs'] == total['wrong_identity_pairs']
    if segmenter is not None:
        semantic_summary=dict(scenes=semantic_scenes,aggregate={a:dict(
            scores(sum(np.asarray(s['arms'][a]['confusion']) for s in semantic_scenes)),
            counts=dict(sum((Counter(s['arms'][a]['counts']) for s in semantic_scenes),Counter()))) for a in matrices})
        (args.out/'semantic-summary.json').write_text(json.dumps(semantic_summary,indent=2)+'\n')
        print(json.dumps(dict(semantic_miou={a:100*s['miou_fixed16'] for a,s in semantic_summary['aggregate'].items()})),flush=True)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--audit-instances',action='store_true',help='Evaluation-only per-instance identity/label evidence')
    p.add_argument('--semantic-fusion',action='store_true',help='Same-forward PTv3 and instance semantic controls')
    run(p.parse_args())
