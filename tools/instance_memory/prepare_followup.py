"""Metadata-only coverage extension after the failed first memory pilot.

Keep all old train/dev logs and add temporally separated scenes within them.
The already inspected test logs are excluded from all subsequent selection.
Preselect new test logs, but do not extract/read their annotations here.
"""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np

from .data import read,save
from tools.semantic_memory.nuscenes_pilot_data import DATA,ROOT,records,transform,digest


def prepare(previous,out):
    old=read(previous);out.mkdir(parents=True,exist_ok=False)
    source_path=ROOT/'runs/s0_formal/train_manifest.json'
    source=read(source_path);scene={s['name']:s for s in read(DATA/'v1.0-trainval/scene.json')}
    rows=defaultdict(list)
    for r in source['samples']:rows[r['scene_name']].append(r)
    for v in rows.values():v.sort(key=lambda r:r['timestamp_us'])
    logs=defaultdict(list)
    for n in rows:logs[scene[n]['log_token']].append(n)
    added={};parts={}
    for part,number in [('train',1),('dev',2)]:
        added[part]=[]
        for initial in old['parts'][part]:
            chosen=[initial];options=set(logs[scene[initial]['log_token']])-{initial}
            for _ in range(number):
                # Farthest from every already chosen clip, determined solely by
                # recorded timestamps; deterministic name tie-break.
                n=min(options,key=lambda n:(-min(abs(rows[n][0]['timestamp_us']-rows[c][0]['timestamp_us']) for c in chosen),n))
                chosen.append(n);options.remove(n);added[part].append(n)
        parts[part]=old['parts'][part]+added[part]
    excluded={scene[n]['log_token'] for n in old['excluded_scenes']}|set(old['scene_logs'].values())
    remaining=sorted(set(logs)-excluded,key=lambda x:hashlib.sha256(('instance-memory-followup-20261029'+x).encode()).hexdigest())
    if len(remaining)<4:raise ValueError('Not enough uninspected logs')
    parts['test']=[min(logs[k]) for k in remaining[:4]]
    selected=sum(parts.values(),[]);sequences={n:rows[n] for n in selected}
    tokens={r[k] for v in sequences.values() for r in v for k in ('sample_data_token','camera_token')}
    meta=DATA/'v1.0-trainval'
    sd={r['token']:r for r in records(meta/'sample_data.json') if r['token'] in tokens}
    pose_tokens={r['ego_pose_token'] for r in sd.values()}
    poses={r['token']:r for r in records(meta/'ego_pose.json') if r['token'] in pose_tokens}
    cal={r['token']:r for r in read(meta/'calibrated_sensor.json')}
    for v in sequences.values():
        for i,r in enumerate(v):
            l,c=sd[r['sample_data_token']],sd[r['camera_token']]
            r.update(frame=i,transform=(transform(poses[l['ego_pose_token']])@transform(cal[l['calibrated_sensor_token']])).tolist(),
                camera_transform=(transform(poses[c['ego_pose_token']])@transform(cal[c['calibrated_sensor_token']])).tolist(),
                camera_intrinsic=cal[c['calibrated_sensor_token']]['camera_intrinsic'])
            if r['camera_timestamp_us']>r['timestamp_us']:raise ValueError('Future camera')
    # Old raw inputs can be reused only with exactly identical metadata.
    for n in old['parts']['train']+old['parts']['dev']:
        if sequences[n]!=old['sequences'][n]:raise ValueError('Parent scene metadata changed: '+n)
    index=dict(old,parts=parts,sequences=sequences,scene_logs={n:scene[n]['log_token'] for n in selected},
        excluded_scenes=sorted(set(old['excluded_scenes'])|set(old['parts']['test'])),
        parent_index_sha256=digest(previous),source_sha256=digest(source_path),
        added=added,scope='Coverage extension within old train/dev logs; new test logs preselected, unopened; backbone-pretraining exposure still applies')
    save(out/'index.json',index)
    save(out/'dev-panoptic-tokens.json',sorted(r['sample_data_token'] for n in added['dev'] for r in sequences[n]))
    save(out/'selection.json',dict(parts=parts,added=added,scene_logs=index['scene_logs'],
        parent_index_sha256=digest(previous),index_sha256=digest(out/'index.json'),
        frames={k:sum(len(sequences[n]) for n in v) for k,v in parts.items()}))
    print(dict(added=added,frames={k:sum(len(sequences[n]) for n in v) for k,v in parts.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.previous,a.out)
