"""One bounded, metadata-only development extension after coverage diagnosis."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import numpy as np
from .data import read,save
from tools.semantic_memory.nuscenes_pilot_data import DATA,ROOT,records,transform,digest


def prepare(previous,out):
    old=read(previous);out.mkdir(parents=True,exist_ok=False)
    scene={s['name']:s for s in read(DATA/'v1.0-trainval/scene.json')}
    source_path=ROOT/'runs/s0_formal/train_manifest.json';source=read(source_path)
    rows=defaultdict(list)
    for r in source['samples']:rows[r['scene_name']].append(r)
    for v in rows.values():v.sort(key=lambda r:r['timestamp_us'])
    logs=defaultdict(list)
    for n in rows:logs[scene[n]['log_token']].append(n)
    excluded={scene[n]['log_token'] for n in old['excluded_scenes']}|set(old['scene_logs'].values())
    remaining=sorted(set(logs)-excluded,key=lambda x:hashlib.sha256(('independent-raw-evidence-20261029'+x).encode()).hexdigest())
    if len(remaining)<8:raise ValueError('Need eight unused collection logs')
    added=[min(logs[k]) for k in remaining[:8]]
    new={n:rows[n] for n in added};tokens={r[k] for v in new.values() for r in v for k in ['sample_data_token','camera_token']}
    meta=DATA/'v1.0-trainval';sd={r['token']:r for r in records(meta/'sample_data.json') if r['token'] in tokens}
    pose_tokens={r['ego_pose_token'] for r in sd.values()}
    poses={r['token']:r for r in records(meta/'ego_pose.json') if r['token'] in pose_tokens}
    cal={r['token']:r for r in read(meta/'calibrated_sensor.json')}
    for v in new.values():
        for i,r in enumerate(v):
            l,c=sd[r['sample_data_token']],sd[r['camera_token']]
            r.update(frame=i,transform=(transform(poses[l['ego_pose_token']])@transform(cal[l['calibrated_sensor_token']])).tolist(),
                camera_transform=(transform(poses[c['ego_pose_token']])@transform(cal[c['calibrated_sensor_token']])).tolist(),
                camera_intrinsic=cal[c['calibrated_sensor_token']]['camera_intrinsic'])
            if r['camera_timestamp_us']>r['timestamp_us']:raise ValueError('Future image')
    parts=dict(old['parts'],dev=old['parts']['dev']+added)
    index=dict(old,parts=parts,sequences=dict(old['sequences'],**new),
        scene_logs=dict(old['scene_logs'],**{n:scene[n]['log_token'] for n in added}),
        independent_dev_added=added,parent_index_sha256=digest(previous),source_sha256=digest(source_path),
        scope='Bounded metadata-only eight-log development extension. Old train/dev/test frozen; new test remains closed. Backbone-pretraining exposure applies.')
    assert not set(index['scene_logs'][n] for n in added)&excluded
    assert index['parts']['test']==old['parts']['test']
    save(out/'index.json',index)
    save(out/'panoptic-tokens.json',sorted(r['sample_data_token'] for v in new.values() for r in v))
    save(out/'selection.json',dict(added=added,logs={n:index['scene_logs'][n] for n in added},
        frames=sum(map(len,new.values())),remaining_unselected_logs=len(remaining)-8,
        parent_index_sha256=digest(previous),index_sha256=digest(out/'index.json'),
        selection='Fixed salt SHA256 ordering of unused logs, lexicographically first scene; no label, description, prediction or score used',
        exclusions='All previous/excluded collection logs and both old/new holdout logs',
        next_step='Same frozen extractor and raw-evidence audit only. Report all eight logs separately from original dev. No further scene reselection after scores.',
        training_admission='At least 30 GT-sidecar recoverable points beyond current pooling on at least five instances and two new dev logs, plus positive net corrections of the frozen 50/50 control in a majority of new logs with any predictions. This admits only a separately frozen training protocol, not an improvement claim.'))
    print(dict(added=added,frames=sum(map(len,new.values()))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.previous,a.out)
