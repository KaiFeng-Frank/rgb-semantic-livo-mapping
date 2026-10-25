"""Frozen, streaming real-data evaluation of causal inference-time consistency.

No training, raw-data copies, feature cache, GT gating, or future frames. Select
six additional official-val collection logs before inference, excluding all 20
previously inspected memory scenes. Raw nuScenes poses are NOT online SLAM poses.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from src.semantic_consistency import CausalConsistency, ConsistencyConfig, fuse
from .nuscenes_pilot_data import build_index, digest, LABEL_MAP
from .pilot_metrics import confusion, scores, paired_summary, object_groups, events

REPO = Path(__file__).resolve().parents[2]
ARMS = ('baseline', 'uniform', 'conditional', 'wrong_history')
STATIC = np.array([0, 7, 10, 11, 12, 13, 14, 15])


def prepare(old_index, out):
    old = json.loads(old_index.read_text())
    index = build_index(out, test_scenes=18)
    old_logs = set(old['scene_logs'].values())
    chosen = [s for s in index['parts']['test'] if index['scene_logs'][s] not in old_logs]
    if len(chosen) != 6:
        raise ValueError('Expected exactly six previously uninspected logs')
    selected = dict(index, parts={'confirmation': chosen},
                    sequences={s:index['sequences'][s] for s in chosen},
                    scene_logs={s:index['scene_logs'][s] for s in chosen},
                    supervision='No new training; GT labels are evaluation-only',
                    excluded_prior_index_sha256=digest(old_index))
    (out/'index.json').write_text(json.dumps(selected, separators=(',',':'))+'\n')
    print(json.dumps(dict(stage='confirmation_selected', scenes=chosen)), flush=True)


def timing(a):
    return dict(n=len(a), median_ms=float(np.median(a)), p95_ms=float(np.percentile(a,95)),
                max_ms=float(np.max(a))) if a else None


def evaluate(index_path, out):
    import torch
    sys.path.insert(0,str(REPO/'src'))
    from ptv3_worker import Segmenter
    from ptv3_loader_verified import CKPT_PATH
    out.mkdir(parents=True, exist_ok=False)
    index = json.loads(index_path.read_text())
    cfg = ConsistencyConfig()
    code = ['src/semantic_consistency.py','src/ptv3_worker.py','src/ptv3_loader_verified.py',
            'tools/semantic_memory/consistency_eval.py','tools/semantic_memory/pilot_metrics.py']
    protocol = dict(schema=1, config=asdict(cfg), arms=ARMS, index_sha256=digest(index_path),
        backbone_sha256=digest(CKPT_PATH), code={p:digest(REPO/p) for p in code},
        geometry=index['geometry'], intensity=index['intensity'], new_training=False,
        frozen_before_inference=True, shared_forward_per_frame=True,
        primary='native16 point-weighted mIoU; paired scene bootstrap 5000 draws, seed 20261026',
        success='positive conditional-minus-baseline mIoU and net fixes; no unconditional deployment from this pilot',
        timing='CPU module including validation/search/copy/argmax; excludes inference, dataset IO and evaluation',
        stability='same geometry-only reciprocal ray-compatible pairs, equal valid GT label; identical mask across arms',
        wrong_history='rotate static-class posterior columns; geometry unchanged, stress arm only; no GT access',
        map='This evaluates current-scan output; reversible map continues to fuse raw evidence',
        limitations=['raw scans with supplied ego poses, not deskewed online poses',
                     'native16 source-domain diagnostic, not target KITTI19 benchmark',
                     'moving predicted classes excluded; no moving-object identity tracking'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.set_num_threads(1)
    torch.manual_seed(20261026)
    segmenter = Segmenter(half=True,shuffle=False,intensity_scale=1.)
    (out/'model.json').write_text(json.dumps(dict(model_config_sha256=segmenter.memory_model_sha256(),
        gpu=torch.cuda.get_device_name(0)),indent=2)+'\n')
    scene_results, object_rows, latencies, inference_times = [], [], [], []
    with (out/'frames.jsonl').open('x') as log:
        for scene, rows in index['sequences'].items():
            core = CausalConsistency(cfg)
            matrices = {a:np.zeros((16,16),np.int64) for a in ARMS}
            counts = {a:dict(rescued=0,harmed=0,base_correct=0,base_wrong=0,
                            stable_pairs=0,flips=0,stable_wrong=0) for a in ARMS}
            origin = np.linalg.inv(np.asarray(rows[0]['transform']))
            previous_labels, previous_predictions = None, None
            totals = dict(points=0,pairs=0,used=0,changed=0)
            for row in rows:
                scan = Path(index['data_root'])/row['scan_path']
                label_path = Path(index['data_root'])/row['label_path']
                if digest(scan)!=row['scan_sha256'] or digest(label_path)!=row['label_sha256']:
                    raise ValueError('Input hash mismatch')
                points = np.fromfile(scan,dtype='<f4').reshape(-1,5)[:,:4].copy()
                points[:,3] /= 255.
                start = time.perf_counter()
                labels, confidence, p = segmenter.segment(points,return_probabilities=True)
                infer_ms = (time.perf_counter()-start)*1000
                if not np.array_equal(labels,p.argmax(1)) or not np.allclose(confidence,p.max(1),atol=1e-6):
                    raise ValueError('Export differs from same-forward raw predictions')
                pose = np.asarray(row['transform'])
                r = core.step(points[:,:3],p,stamp_ns=int(row['timestamp_us']*1000),
                              pose=origin@pose,diagnostics=True)
                ci, hi, hp = r['current_indices'], r['history_indices'], r['history_probabilities']
                predictions = dict(baseline=labels,conditional=r['probabilities'].argmax(1))
                for arm, conditional, history in (('uniform',False,hp),('wrong_history',True,hp.copy())):
                    if arm=='wrong_history':
                        history[:,STATIC] = history[:,np.roll(STATIC,1)]
                    pred = labels.copy()
                    if len(ci):
                        fused,_ = fuse(p[ci],history,r['status']['age_s'],cfg,conditional=conditional)
                        pred[ci] = fused.argmax(1)
                    predictions[arm] = pred
                # First GT read for metrics, after every inference decision.
                gt = LABEL_MAP[np.fromfile(label_path,dtype=np.uint8)]
                if len(gt)!=len(points):
                    raise ValueError('Label count mismatch')
                pair_valid = ((gt[ci]>=0)&(gt[ci]==previous_labels[hi])) if previous_labels is not None else np.zeros(len(ci),bool)
                frame_counts = {}
                for arm,pred in predictions.items():
                    matrices[arm] += confusion(gt,pred)
                    v = paired_summary(gt,labels,pred)
                    v.update(stable_pairs=int(pair_valid.sum()),flips=0,stable_wrong=0)
                    if previous_predictions is not None:
                        now, before = pred[ci][pair_valid], previous_predictions[arm][hi][pair_valid]
                        v['flips'] = int((now!=before).sum())
                        v['stable_wrong'] = int(((now==before)&(now!=gt[ci][pair_valid])).sum())
                    for k,x in v.items(): counts[arm][k] += x
                    frame_counts[arm] = v
                world = points[:,:3]@pose[:3,:3].T+pose[:3,3]
                for obj in object_groups(world,gt,row['objects']):
                    ids = obj['indices']
                    object_rows.append(dict(scene=scene,frame=row['frame'],stamp=row['timestamp_us']/1e6,
                        instance=obj['instance'],label=obj['label'],points=len(ids),
                        correct_fraction={a:float((predictions[a][ids]==gt[ids]).mean()) for a in ARMS}))
                for k in totals: totals[k] += r['status'][k]
                latencies.append(r['status']['elapsed_ms'])
                inference_times.append(infer_ms)
                log.write(json.dumps(dict(scene=scene,frame=row['frame'],scan_sha256=row['scan_sha256'],
                    posterior_sha256=hashlib.sha256(p.tobytes()).hexdigest(),inference_ms=infer_ms,
                    consistency=r['status'],paired=frame_counts))+'\n')
                log.flush()
                previous_labels, previous_predictions = gt,predictions
            result = dict(scene=scene,log=index['scene_logs'][scene],frames=len(rows),totals=totals,
                          arms={a:dict(**scores(matrices[a]),**counts[a]) for a in ARMS})
            scene_results.append(result)
            (out/f'{scene}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(scene=scene,miou={a:100*result['arms'][a]['miou_fixed16'] for a in ARMS},
                                  changed=totals['changed'])),flush=True)
    aggregate = {}
    for a in ARMS:
        matrix = sum(np.asarray(s['arms'][a]['confusion']) for s in scene_results)
        aggregate[a] = dict(**scores(matrix), **{k:sum(s['arms'][a][k] for s in scene_results) for k in counts[a]})
        aggregate[a]['flip_rate'] = aggregate[a]['flips']/aggregate[a]['stable_pairs'] if aggregate[a]['stable_pairs'] else None
    rng = np.random.default_rng(20261026)
    boot = {a:[] for a in ARMS if a!='baseline'}
    for _ in range(5000):
        chosen = rng.integers(len(scene_results),size=len(scene_results))
        matrix = {a:sum(np.asarray(scene_results[i]['arms'][a]['confusion']) for i in chosen) for a in ARMS}
        base = scores(matrix['baseline'])['miou_fixed16']
        for a in boot: boot[a].append(100*(scores(matrix[a])['miou_fixed16']-base))
    result = dict(scenes=scene_results,aggregate=aggregate,
        paired_scene_bootstrap_pp={a:np.percentile(v,[2.5,50,97.5]).tolist() for a,v in boot.items()},
        module_timing=timing(latencies),inference_timing=timing(inference_times),
        object_events=events(object_rows,ARMS),object_observations=len(object_rows),
        totals={k:sum(s['totals'][k] for s in scene_results) for k in totals},
        deployable=False)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('scenes','aggregate','object_events')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--old-index',type=Path,required=True); a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('evaluate'); a.add_argument('--index',type=Path,required=True); a.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare': prepare(args.old_index,args.out)
    else: evaluate(args.index,args.out)
