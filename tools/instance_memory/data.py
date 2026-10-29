"""Frozen same-forward observation export; labels live in a separate target file.

The annotation extension is downloaded on the remote. Only selected panoptic
files are recovered locally. No raw images/scans are copied into this cache.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from tools.semantic_memory.nuscenes_pilot_data import DATA, ROOT, records, transform, digest, LABEL_MAP

RUNTIME = Path('/home/frank/semantic_livo_runtime/instance-memory-20261029')
REPO = Path(__file__).resolve().parents[2]


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.partial')
    if path.suffix == '.gz':
        with gzip.open(temp, 'wt') as f:
            json.dump(value, f, separators=(',', ':'), allow_nan=False)
    else:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as f:
        return json.load(f)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    scenes = {r['name']: r for r in read(DATA/'v1.0-trainval/scene.json')}
    previous = ['memory-real-pilot-20261025', 'memory-repair-confirmation-20261025',
                'consistency-20261026-index']
    excluded = set()
    for name in previous:
        excluded.update(read(RUNTIME.parent/name/'index.json')['sequences'])
    excluded_logs = {scenes[n]['log_token'] for n in excluded}
    source_path = ROOT/'runs/s0_formal/train_manifest.json'
    source = read(source_path)
    by_scene = defaultdict(list)
    for r in source['samples']:
        if r['scene_name'] not in source['official_split_scene_names']:
            raise ValueError('Foreign official split')
        by_scene[r['scene_name']].append(r)
    logs = defaultdict(list)
    for name in by_scene:
        log = scenes[name]['log_token']
        if log not in excluded_logs:
            logs[log].append(name)
    ranked = sorted(logs, key=lambda x: hashlib.sha256(('instance-memory-20261029'+x).encode()).hexdigest())
    if len(ranked) < 24:
        raise ValueError('Insufficient independent uninspected logs')
    selected = [sorted(logs[k])[0] for k in ranked[:24]]
    parts = dict(train=selected[:16], dev=selected[16:20], test=selected[20:24])
    sequences = {s: sorted(by_scene[s], key=lambda r: r['timestamp_us']) for s in selected}
    # Metadata only; no GT labels, boxes or model predictions used for selection.
    tokens = {r[k] for rows in sequences.values() for r in rows for k in ('sample_data_token', 'camera_token')}
    meta = DATA/'v1.0-trainval'
    sd = {r['token']: r for r in records(meta/'sample_data.json') if r['token'] in tokens}
    pose_tokens = {r['ego_pose_token'] for r in sd.values()}
    poses = {r['token']: r for r in records(meta/'ego_pose.json') if r['token'] in pose_tokens}
    cal = {r['token']: r for r in read(meta/'calibrated_sensor.json')}
    for rows in sequences.values():
        for i, r in enumerate(rows):
            lidar, camera = sd[r['sample_data_token']], sd[r['camera_token']]
            lc, cc = cal[lidar['calibrated_sensor_token']], cal[camera['calibrated_sensor_token']]
            r['frame'] = i
            r['transform'] = (transform(poses[lidar['ego_pose_token']])@transform(lc)).tolist()
            r['camera_transform'] = (transform(poses[camera['ego_pose_token']])@transform(cc)).tolist()
            r['camera_intrinsic'] = cc['camera_intrinsic']
            if r['camera_timestamp_us'] > r['timestamp_us']:
                raise ValueError('Manifest camera is in the future')
    index = dict(schema=1, parts=parts, sequences=sequences, scene_logs={n:scenes[n]['log_token'] for n in selected},
        source_sha256=digest(source_path), excluded_scenes=sorted(excluded),
        scope='Project holdout from official training pool; no uninspected official-validation collection logs remain',
        data_root=str(DATA), raw_geometry=True, label_map=LABEL_MAP.tolist())
    save(out/'index.json', index)
    save(out/'panoptic-tokens.json', sorted(r['sample_data_token'] for rows in sequences.values() for r in rows))
    print(json.dumps(dict(parts=parts, frames={k:sum(len(sequences[n]) for n in v) for k,v in parts.items()})), flush=True)


class VisualFeatures:
    """Pool frozen YOLO P3 encoder features from the SAME image forward.

    Explicitly reproduce the existing rect=False square letterbox geometry.
    Hook and pooling never change detector outputs or tracking state.
    """
    def __init__(self, detector):
        self.last = None
        self.size = detector.imgsz
        self.handle = detector.model.model.model[-1].register_forward_pre_hook(self.capture)

    def capture(self, module, args):
        self.last = args[0][0].detach()

    def pool(self, frame):
        import torch
        import torch.nn.functional as F
        if self.last is None or self.last.shape[0] != 1:
            raise ValueError('Missing current visual feature map')
        f = self.last[0]
        if f.shape[0] != 64:
            raise ValueError('Frozen YOLO11n P3 contract must be 64 channels')
        h, w = frame.index.shape
        scale = min(self.size/h, self.size/w)
        nh, nw = round(h*scale), round(w*scale)
        top, left = round((self.size-nh)/2-.1), round((self.size-nw)/2-.1)
        mask = torch.as_tensor(frame.index.astype(np.int32), device=f.device)[None,None].float()
        mask = F.interpolate(mask, (nh,nw), mode='nearest')
        mask = F.pad(mask, (left,self.size-nw-left,top,self.size-nh-top))
        mask = F.interpolate(mask, f.shape[-2:], mode='nearest')[0,0].long()
        result = {}
        for i, tid in enumerate(frame.ids):
            at = mask == i+1
            # Tiny masks can have no P3 cells. Explicit missing feature, not an
            # unrelated background vector masquerading as an observation.
            value = f[:,at].float().mean(1) if at.any() else f.new_zeros(64).float()
            result[int(tid)] = (value.cpu().numpy(), bool(at.any()))
        return result


def extract(index_path, out, annotation_root, parts, scene_limit=0):
    from tools.visual_instances.kalman_cache import gpu_admission
    from tools.visual_instances.detector import Detector
    from tools.visual_instances.geometry import Camera, associate
    from tools.visual_instances.semantics import InstanceSemantics, component, Config
    import torch
    import cv2
    sys.path.insert(0, str(REPO/'src'))
    from ptv3_worker import Segmenter
    from instance_features import pool_features
    admission = gpu_admission()
    torch.set_num_threads(1)
    torch.manual_seed(20261029)
    out.mkdir(parents=True, exist_ok=True)
    index = read(index_path)
    source = {str(p.relative_to(REPO)):digest(p) for p in [Path(__file__), REPO/'src/instance_features.py',
        REPO/'src/ptv3_worker.py', REPO/'tools/visual_instances/detector.py',
        REPO/'tools/visual_instances/evidence_predictor.py',REPO/'tools/visual_instances/geometry.py']}
    protocol = dict(schema=1, index_sha256=digest(index_path), parts=index['parts'], source=source,
        checkpoint_sha256=digest(os.environ['PTV3_CHECKPOINT']), raw_only=True,
        geometry='raw keyframes with supplied ego poses; not deskewed/live LIVO', admission=admission)
    pp = out/'protocol.json'
    if pp.exists():
        old = read(pp)
        for key in ('index_sha256','source','checkpoint_sha256'):
            if old[key] != protocol[key]: raise ValueError('Incompatible extraction resume: '+key)
    else:
        save(pp, protocol)
    model = Segmenter(half=True, shuffle=False, intensity_scale=1.)
    count = 0
    for part in parts:
        for name in index['parts'][part]:
            if scene_limit and count >= scene_limit: return
            count += 1
            path = out/(name+'.inputs.json.gz')
            target_path = out/(name+'.targets.json.gz')
            if path.exists() and target_path.exists(): continue
            if shutil.disk_usage(out).free < 4_000_000_000:
                raise RuntimeError('Less than 4GB free; stop cache growth')
            if sum(p.stat().st_size for p in out.glob('*.gz')) > 1_000_000_000:
                raise RuntimeError('1GB cache cap reached')
            gpu_admission()
            detector = Detector(RUNTIME.parent/'visual-weights/yolo11n-seg.pt')
            visual = VisualFeatures(detector)
            frames, targets = [], []
            cfg = Config()
            for i, row in enumerate(index['sequences'][name]):
                if i%10 == 0: gpu_admission()
                for field, hashfield in [('scan_path','scan_sha256'),('image_path','image_sha256')]:
                    if digest(DATA/row[field]) != row[hashfield]: raise ValueError('Input hash mismatch')
                start = time.perf_counter()
                image = cv2.imread(str(DATA/row['image_path']))
                scan = np.fromfile(DATA/row['scan_path'], dtype='<f4').reshape(-1,5)[:,:4].copy()
                scan[:,3] /= 255.
                vf, vt = detector.step(image, int(row['camera_timestamp_us']*1000))
                vision = visual.pool(vf)
                pose = np.asarray(row['transform'])
                camera = Camera(image.shape[1], image.shape[0], row['camera_intrinsic'], np.zeros(4),
                                np.linalg.inv(row['camera_transform'])@pose)
                assoc = associate(scan[:,:3], camera, vf)
                groups = np.zeros(len(scan), np.int32)
                components = []
                for tid in np.unique(assoc['ids'][assoc['ids']>0]):
                    at = np.flatnonzero(assoc['ids']==tid)
                    c = component(scan[at,:3], cfg)
                    if c is None: continue
                    at = at[c]
                    groups[at] = len(components)+1
                    components.append((int(tid), at))
                raw = model.segment(scan, return_features=True)
                with torch.inference_mode():
                    probs = raw['logits'].float().softmax(-1)[raw['inverse']].cpu().numpy()
                    feats = pool_features(raw['features'],raw['inverse'],groups,len(components)).cpu().numpy()
                single = InstanceSemantics(temporal=False).step(scan[:,:3], probs, assoc,
                                    stamp_ns=int(row['timestamp_us']*1000),pose=pose)
                observations = []
                for j,(tid,at) in enumerate(components):
                    xyz = scan[at,:3]
                    center = np.median(xyz,axis=0)
                    extent = np.quantile(xyz,.9,axis=0)-np.quantile(xyz,.1,axis=0)
                    p = probs[at];mean = p.mean(0)
                    v, available = vision[tid]
                    quality = np.array([np.log1p(len(at))/8., len(at)/max(1,np.count_nonzero(assoc['ids']==tid)),
                        np.linalg.norm(center)/80., np.linalg.norm(extent)/10.,
                        -(p*np.log(np.maximum(p,1e-8))).sum(1).mean()/np.log(16),p.std(0).mean(),
                        (row['timestamp_us']-row['camera_timestamp_us'])*1e-6,float(available)],np.float32)
                    x = np.concatenate([feats[j],v,mean,assoc['class_scores'][tid],assoc['appearance'][tid],quality])
                    if len(x) != 320: raise ValueError('Feature contract changed')
                    observations.append(dict(x=x.tolist(), raw_mean=mean.tolist(),raw_probabilities=p.tolist(),
                        indices=at.tolist(), visual_track=tid, visual_session=vf.session,
                        center=(pose[:3,:3]@center+pose[:3,3]).tolist(), extent=extent.tolist(),
                        sensor_center=center.tolist(), pose=pose.tolist(),
                        stamp=int(row['timestamp_us']*1000),camera_stamp=int(row['camera_timestamp_us']*1000),
                        visual_label=int(assoc['labels'][at[0]]), visual_confidence=float(assoc['scores'][at[0]])))
                elapsed = (time.perf_counter()-start)*1000
                # All GT access below; no targets are supplied to feature/geometry code.
                gt = LABEL_MAP[np.fromfile(DATA/row['label_path'], np.uint8)]
                panpath = annotation_root/(row['sample_data_token']+'_panoptic.npz')
                with np.load(panpath) as z: pan = z['data'].reshape(-1).astype(np.int64)
                if pan.shape != gt.shape: raise ValueError('Panoptic point order/length mismatch')
                semantic = pan//1000
                # Official devkit panoptic_to_lidarseg divides by 1000. The
                # result uses the original lidarseg ontology, then LABEL_MAP.
                if np.any(semantic >= len(LABEL_MAP)): raise ValueError('Unknown panoptic ontology')
                mapped = LABEL_MAP[semantic]
                # Panoptic annotations mark some otherwise lidarseg-labelled
                # points as noise when no valid instance can be assigned.
                # Keep them in semantic scoring, exclude them from ID scoring.
                if not np.array_equal(mapped[semantic>0], gt[semantic>0]):
                    raise ValueError('Panoptic/lidarseg semantic alignment mismatch')
                from tools.semantic_memory.pilot_metrics import confusion
                raw_lab = probs.argmax(1)
                target = dict(gt=gt.tolist(), instance=(pan%1000).tolist(),panoptic=pan.tolist(),
                    panoptic_ignored_lidarseg_valid=int(((semantic==0)&(gt>=0)).sum()),
                    raw_confusion=confusion(gt,raw_lab).tolist(),
                    single_confusion=confusion(gt,single['labels']).tolist(),
                    raw_labels=raw_lab.tolist(),panoptic_sha256=digest(panpath),
                    lidarseg_sha256=digest(DATA/row['label_path']))
                frames.append(dict(frame=i,stamp=int(row['timestamp_us']*1000),observations=observations,
                    points=len(scan), timing_ms=elapsed,visual_ms=vt['elapsed_ms'],
                    scan_sha256=row['scan_sha256'],image_sha256=row['image_sha256']))
                targets.append(target)
                if i%10 == 0: print(json.dumps(dict(scene=name,frame=i,observations=len(observations),elapsed_ms=elapsed)),flush=True)
            save(path, dict(scene=name,frames=frames))
            save(target_path, dict(scene=name,frames=targets))
            visual.handle.remove()
            del detector, visual, raw
            print(json.dumps(dict(scene_done=name,part=part,frames=len(frames),observations=sum(len(f['observations']) for f in frames))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','extract'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--index',type=Path)
    p.add_argument('--annotations',type=Path,default=RUNTIME/'annotations')
    p.add_argument('--parts',nargs='+',default=['train','dev'])
    p.add_argument('--scene-limit',type=int,default=0)
    a=p.parse_args()
    if a.action=='prepare': prepare(a.out)
    else: extract(a.index,a.out,a.annotations,a.parts,a.scene_limit)
