"""Offline point/object scoring for the frozen ST-Occ history intervention.

Panoptic identity is evaluation-only. This model does not predict instance IDs.
An observation gap is not automatically an occlusion. Recovery is sampled at
keyframes, distinct from neural-network inference time.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from pyquaternion import Quaternion

ROOT = Path('/home/wuyou/stocc_reference')
WORK = ROOT/'history-diagnostic'
ARMS = ('cold_history', 'native_history')
CLASSES = ['others','barrier','bicycle','bus','car','construction_vehicle',
           'motorcycle','pedestrian','traffic_cone','trailer','truck',
           'driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation','free']
# Official general (32) -> challenge (16) IDs, plus one for the Occ3D order.
CLASS_MAP = np.array([0,0,7,7,7,0,7,0,0,1,0,0,8,0,2,3,3,4,5,0,0,6,9,10,11,12,13,14,15,0,16,0])
LOW = np.array([-40., -40., -1.])
SHAPE = np.array([200, 200, 16])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def point_scores(hist):
    diag = np.diag(hist)
    union = hist.sum(0) + hist.sum(1) - diag
    iou = np.divide(diag, union, out=np.full(18, np.nan), where=union > 0)
    valid = iou[1:17]
    return dict(mIoU16_present_percent=float(np.nanmean(valid)*100),
                mIoU16_zero_for_absent_percent=float(np.nan_to_num(valid).mean()*100),
                represented_classes=int(np.isfinite(valid).sum()),
                accuracy_percent=float(diag.sum()/hist.sum()*100),
                foreground_accuracy_percent=float(diag[1:11].sum()/max(1,hist[1:11].sum())*100),
                class_iou_percent=[float(v*100) if np.isfinite(v) else None for v in iou],
                confusion=hist.tolist())


def prepare():
    protocol = json.loads((WORK/'protocol.json').read_text())
    data = ROOT/'data/nuscenes'
    with (WORK/'subset.pkl').open('rb') as stream:
        infos = {r['token']:r for r in pickle.load(stream)['infos']}
    lidarseg = {r['token']:r for r in json.loads((data/'v1.0-trainval/lidarseg.json').read_text())}
    pan_files = json.loads((WORK/'panoptic-files.json').read_text())
    dest = WORK/'targets/points';dest.mkdir(exist_ok=True)
    manifest = [];counts = defaultdict(int)
    for number, frame in enumerate(protocol['frames'], 1):
        token, sd_token = frame['token'], frame['lidar_token']
        info = infos[token]
        scan_file = data/frame['lidar_path'];scan_bytes = scan_file.read_bytes()
        xyz = np.frombuffer(scan_bytes, dtype='<f4').reshape(-1,5)[:,:3].astype(np.float64)
        semantic_file = data/lidarseg[sd_token]['filename'];semantic_bytes = semantic_file.read_bytes()
        raw = np.frombuffer(semantic_bytes, dtype=np.uint8)
        pan_file = WORK/'targets/panoptic'/f'{sd_token}_panoptic.npz'
        assert sha(pan_file) == pan_files[pan_file.name]['sha256']
        with np.load(pan_file) as archive:
            assert archive.files == ['data'];pan = archive['data']
        assert len(raw) == len(xyz) == len(pan)
        assert raw.max() < 32 and pan.dtype == np.uint16
        known_pan = pan != 0
        assert np.array_equal(pan[known_pan]//1000, raw[known_pan]), 'Panoptic/lidarseg semantic disagreement'
        labels = CLASS_MAP[raw]
        rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        translation = np.asarray(info['lidar2ego_translation'])
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        ego = xyz @ rotation.T + translation
        assert np.max(np.abs((ego-translation) @ rotation - xyz)) < 1e-8
        ijk = np.floor((ego-LOW)/.4).astype(np.int64)
        inside = ((ijk >= 0) & (ijk < SHAPE)).all(1)
        eligible = np.flatnonzero(inside & (labels > 0))
        occupancy_path = data/'gts'/frame['scene']/token/'labels.npz'
        with np.load(occupancy_path) as archive:
            camera_mask = archive['mask_camera'].astype(bool)
            occ_gt = archive['semantics']
        eligible = eligible[camera_mask[tuple(ijk[eligible].T)]]
        assert len(eligible)
        sampled_occ_gt = occ_gt[tuple(ijk[eligible].T)]
        counts['raw_points'] += len(xyz)
        counts['semantic_valid_points'] += int((labels>0).sum())
        counts['semantic_valid_in_grid_points'] += int((inside & (labels>0)).sum())
        counts['eligible_points'] += len(eligible)
        counts['voxel_gt_agrees_with_point_gt'] += int((sampled_occ_gt==labels[eligible]).sum())
        counts['panoptic_semantic_checked_points'] += int(known_pan.sum())
        counts['panoptic_zero_points'] += int((~known_pan).sum())
        point_file = dest/(token+'.npz')
        np.savez_compressed(point_file, indices=eligible.astype(np.int32),
                            ijk=ijk[eligible].astype(np.int16), labels=labels[eligible].astype(np.uint8),
                            panoptic=pan[eligible])
        manifest.append(dict(token=token, lidar_token=sd_token, eligible_points=len(eligible),
                             scan_sha256=hashlib.sha256(scan_bytes).hexdigest(),
                             lidarseg_sha256=hashlib.sha256(semantic_bytes).hexdigest(),
                             panoptic_sha256=pan_files[pan_file.name]['sha256'], target_sha256=sha(point_file)))
        if number % 100 == 0:print('prepared_point_targets',number,flush=True)
    save(WORK/'point-target-manifest.json',manifest)
    save(WORK/'target-audit.json',dict(frames=len(manifest), counts=dict(counts),
        protocol_sha256=sha(WORK/'protocol.json'), subset_metadata_sha256=sha(WORK/'subset.pkl'),
        manifest_sha256=sha(WORK/'point-target-manifest.json'),
        projection='LiDAR-to-ego calibration; floor((ego_xyz - [-40,-40,-1])/0.4); half-open [200,200,16] grid, then official camera mask.',
        panoptic_lidarseg_semantics_match=True, point_counts_match=True,
        voxel_point_gt_agreement=counts['voxel_gt_agrees_with_point_gt']/counts['eligible_points'],
        no_filter_on_voxel_point_gt_agreement=True, completed=time.time()))
    print('target_audit',json.dumps(dict(counts)),flush=True)


def build_events(rows, horizon=2.):
    tracks = defaultdict(list)
    for row in rows:tracks[(row['scene'],row['instance'])].append(row)
    episodes=[]
    for track in tracks.values():
        track.sort(key=lambda row:row['frame'])
        runs=[]
        for row in track:
            if not runs or row['frame'] != runs[-1][-1]['frame']+1:runs.append([])
            runs[-1].append(row)
        for run_index,run in enumerate(runs):
            starts=[]
            if run_index > 0:starts.append(('reappearance_after_observation_gap',0))
            elif run[0]['frame'] > 0:starts.append(('first_observed',0))
            last_error_deadline=-np.inf
            for i,row in enumerate(run):
                if row['correct_fraction']['cold_history'] >= .5:continue
                if i and run[i-1]['correct_fraction']['cold_history'] < .5:continue
                if row['stamp'] <= last_error_deadline:continue
                starts.append(('cold_history_error_onset',i));last_error_deadline=row['stamp']+horizon
            for kind,index in starts:
                first=run[index];window=[r for r in run[index:] if r['stamp']<=first['stamp']+horizon+1e-6]
                outcomes={}
                for arm in ARMS:
                    hit=next((r for r in window if r['correct_fraction'][arm]>=.5),None)
                    outcomes[arm]=dict(success=hit is not None,
                        delay_s=float(hit['stamp']-first['stamp']) if hit is not None else None,
                        delay_capped_s=float(hit['stamp']-first['stamp']) if hit is not None else horizon)
                episodes.append(dict(kind=kind,scene=first['scene'],log=first['log'],
                    instance=first['instance'],label=first['label'],start_frame=first['frame'],
                    end_frame=window[-1]['frame'],available_observation_span_s=float(window[-1]['stamp']-first['stamp']),
                    followup_truncated=run[-1]['stamp']<first['stamp']+horizon-.1,
                    outcomes=outcomes))
    return episodes


def event_summary(episodes):
    out={}
    for kind in ['first_observed','reappearance_after_observation_gap','cold_history_error_onset']:
        selected=[e for e in episodes if e['kind']==kind];n=len(selected)
        table=dict(episodes=n,collection_logs=len({e['log'] for e in selected}),
                   followup_truncated=sum(e['followup_truncated'] for e in selected),arms={})
        for arm in ARMS:
            delays=[e['outcomes'][arm]['delay_s'] for e in selected if e['outcomes'][arm]['success']]
            capped=[e['outcomes'][arm]['delay_capped_s'] for e in selected]
            table['arms'][arm]=dict(successes=len(delays),not_observed_recovered=n-len(delays),
                success_rate=len(delays)/n if n else None,
                conditional_mean_delay_s=float(np.mean(delays)) if delays else None,
                mean_delay_capped_at_2s=float(np.mean(capped)) if capped else None)
        table['gained']=sum(not e['outcomes']['cold_history']['success'] and e['outcomes']['native_history']['success'] for e in selected)
        table['lost']=sum(e['outcomes']['cold_history']['success'] and not e['outcomes']['native_history']['success'] for e in selected)
        table['net_successes']=table['gained']-table['lost'];out[kind]=table
    return out


def class_stability(rows):
    last={};total=0;flips=dict.fromkeys(ARMS,0);correct=dict.fromkeys(ARMS,0)
    for row in sorted(rows,key=lambda r:(r['scene'],r['instance'],r['frame'])):
        key=(row['scene'],row['instance']);previous=last.get(key)
        for arm in ARMS:correct[arm]+=row['correct_fraction'][arm]>=.5
        if previous is not None and previous['frame']+1==row['frame']:
            total+=1
            for arm in ARMS:flips[arm]+=previous['majority_prediction'][arm]!=row['majority_prediction'][arm]
        last[key]=row
    return dict(object_frames=len(rows),consecutive_observation_transitions=total,
                majority_class_changes=flips,semantic_successful_object_frames=correct,
                caveat='GT-associated semantic stability only; no predicted instance IDs or ID-switch measurement.')


def evaluate():
    protocol=json.loads((WORK/'protocol.json').read_text());frame_rows=protocol['frames']
    target_manifest={r['token']:r for r in json.loads((WORK/'point-target-manifest.json').read_text())}
    logs=sorted(set(protocol['scene_logs'].values()))
    histories={};runs={arm:WORK/arm for arm in ARMS};summaries={}
    for arm,run in runs.items():
        summaries[arm]=json.loads((run/'summary.json').read_text());assert summaries[arm]['status']=='complete'
        histories[arm]={r['token']:r for r in map(json.loads,(run/'frames.jsonl').read_text().splitlines())}
        assert set(histories[arm])==set(target_manifest)
    prior={r['token']:r for r in map(json.loads,(ROOT/'runs/full-validation-workers8/frames.jsonl').read_text().splitlines())}
    assert all(histories['native_history'][token]['prediction_sha256']==prior[token]['prediction_sha256'] for token in target_manifest)
    first=[r for r in frame_rows if r['frame']==0]
    assert len(first)==16 and all(histories[ARMS[0]][r['token']]['prediction_sha256']==histories[ARMS[1]][r['token']]['prediction_sha256'] for r in first)
    assert all(all(row['history_control'].values()) for row in histories['cold_history'].values())
    hist={arm:np.zeros((18,18),np.int64) for arm in ARMS}
    per_log={log:{arm:np.zeros((18,18),np.int64) for arm in ARMS} for log in logs}
    paired=defaultdict(int);class_pairs={i:defaultdict(int) for i in range(1,17)};objects=[];frame_metrics=[]
    for count,frame in enumerate(frame_rows,1):
        token=frame['token'];target_path=WORK/'targets/points'/(token+'.npz')
        assert sha(target_path)==target_manifest[token]['target_sha256']
        with np.load(target_path) as target:
            ijk=target['ijk'];gt=target['labels'].astype(np.int64);pan=target['panoptic']
        predictions={};log=protocol['scene_logs'][frame['scene']]
        for arm,run in runs.items():
            with np.load(run/'predictions'/(token+'.npz')) as record:grid=record['prediction']
            assert grid.shape==(200,200,16) and grid.dtype==np.uint8 and grid.max()<18
            assert hashlib.sha256(grid.tobytes()).hexdigest()==histories[arm][token]['prediction_sha256']
            predictions[arm]=grid[tuple(ijk.T)].astype(np.int64)
            matrix=np.bincount(18*gt+predictions[arm],minlength=324).reshape(18,18)
            hist[arm]+=matrix;per_log[log][arm]+=matrix
        cold=predictions['cold_history']==gt;native=predictions['native_history']==gt
        gain=~cold&native;harm=cold&~native
        for key,value in [('points',len(gt)),('corrected',gain.sum()),('harmed',harm.sum()),('cold_correct',cold.sum()),('cold_wrong',(~cold).sum())]:paired[key]+=int(value)
        for label in range(1,17):
            at=gt==label
            for key,value in [('points',at.sum()),('corrected',(gain&at).sum()),('harmed',(harm&at).sum())]:class_pairs[label][key]+=int(value)
        frame_metrics.append(dict(token=token,scene=frame['scene'],log=log,points=len(gt),corrected=int(gain.sum()),harmed=int(harm.sum())))
        instance_ids=np.unique(pan[(pan%1000>0)&(gt<=10)])
        for identity in instance_ids:
            at=pan==identity;n=int(at.sum())
            if n<5:continue
            labels=np.unique(gt[at]);assert len(labels)==1
            objects.append(dict(scene=frame['scene'],log=log,frame=frame['frame'],stamp=frame['timestamp_us']*1e-6,
                instance=int(identity),label=int(labels[0]),points=n,
                correct_fraction={arm:float((predictions[arm][at]==gt[at]).mean()) for arm in ARMS},
                majority_prediction={arm:int(np.bincount(predictions[arm][at],minlength=18).argmax()) for arm in ARMS}))
        if count%100==0:print('evaluated_pairs',count,flush=True)
    episodes=build_events(objects,protocol['evaluation']['event_horizon_seconds'])
    event_totals=event_summary(episodes);rng=np.random.default_rng(20261031);delta_iou=[];recovery_deltas=[];reappearance_deltas=[]
    arrays={arm:np.stack([per_log[log][arm] for log in logs]) for arm in ARMS}
    event_by_log={log:event_summary([e for e in episodes if e['log']==log]) for log in logs}
    for indices in rng.integers(0,len(logs),size=(2000,len(logs))):
        # A resampled log set may omit rare classes. Keep the same 16-class
        # denominator in both arms even when their predicted support differs.
        delta_iou.append(point_scores(arrays['native_history'][indices].sum(0))['mIoU16_zero_for_absent_percent']-point_scores(arrays['cold_history'][indices].sum(0))['mIoU16_zero_for_absent_percent'])
        for kind,output in [('cold_history_error_onset',recovery_deltas),('reappearance_after_observation_gap',reappearance_deltas)]:
            total=sum(event_by_log[logs[i]][kind]['episodes'] for i in indices)
            if total:output.append(100*sum(event_by_log[logs[i]][kind]['net_successes'] for i in indices)/total)
    ci=lambda values:[float(v) for v in np.percentile(values,[2.5,97.5])] if values else None
    recovery=event_totals['cold_history_error_onset'];reappearance=event_totals['reappearance_after_observation_gap']
    gates=dict(positive_net_point_corrections=paired['corrected']>paired['harmed'],
               positive_net_error_recovery=recovery['net_successes']>0,
               no_reappearance_success_loss=reappearance['net_successes']>=0,
               recovery_episodes_at_least_30=recovery['episodes']>=30,
               recovery_collection_logs_at_least_8=recovery['collection_logs']>=8)
    per_log_scores={log:{arm:point_scores(per_log[log][arm]) for arm in ARMS} for log in logs}
    report=dict(protocol_sha256=sha(WORK/'protocol.json'),analysis_source_sha256=sha(Path(__file__)),
        completed=time.time(),frames=len(frame_rows),scenes=16,collection_logs=len(logs),
        native_matches_full_reference_frames=len(frame_rows),matching_scene_first_frames=len(first),
        cold_history_actual_attention_verified_frames=len(histories['cold_history']),
        point_metrics={arm:point_scores(hist[arm]) for arm in ARMS},paired=dict(paired),
        per_class_paired={CLASSES[i]:dict(values) for i,values in class_pairs.items()},
        events=event_totals,object_semantic_stability=class_stability(objects),
        subgroups={CLASSES[label]:dict(events=event_summary([e for e in episodes if e['label']==label]),
                                    stability=class_stability([r for r in objects if r['label']==label])) for label in [4,7]},
        per_log_point_metrics=per_log_scores,
        positive_log_point_mIoU_count=sum(per_log_scores[log]['native_history']['mIoU16_zero_for_absent_percent']>per_log_scores[log]['cold_history']['mIoU16_zero_for_absent_percent'] for log in logs),
        log_comparison_metric='mIoU16_zero_for_absent_percent',
        paired_collection_log_bootstrap=dict(draws=2000,seed=20261031,
             class_support_policy='Fixed 16 classes in both arms; zero IoU for zero-union classes.',
             mIoU_delta_pp_95percent=ci(delta_iou),recovery_success_delta_pp_95percent=ci(recovery_deltas),
             reappearance_success_delta_pp_95percent=ci(reappearance_deltas)),
        mechanism_criteria=gates,all_recorded_criteria_met=all(gates.values()),
        occupancy_metrics={arm:summaries[arm]['metrics'] for arm in ARMS},
        timing={arm:{k:summaries[arm][k] for k in ['model_ms_p50','model_ms_p95','elapsed_seconds','peak_allocated_bytes','peak_reserved_bytes']} for arm in ARMS},
        timing_caveat='Cold arm includes state-reset and attention-check overhead. Both include metric bookkeeping. One pass per arm; not isolated deployment or a 5 ms plugin claim.',
        limitations=['Same weights, no independently trained current-only baseline.',
                     'GT identities used only for scoring, no predicted instance tracking.',
                     'Official ego poses and full-scene memory bounds retained.',
                     'No PTv3 adaptation/training or deployment claim.'])
    save(WORK/'report.json',report)
    with gzip.open(WORK/'object-frames.json.gz','wt') as stream:json.dump(objects,stream,separators=(',',':'))
    save(WORK/'events.json',episodes);save(WORK/'frame-point-effects.json',frame_metrics)
    print(json.dumps({k:v for k,v in report.items() if k not in ['per_log_point_metrics','point_metrics','occupancy_metrics','subgroups']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','evaluate']);args=parser.parse_args()
    prepare() if args.action=='prepare' else evaluate()
