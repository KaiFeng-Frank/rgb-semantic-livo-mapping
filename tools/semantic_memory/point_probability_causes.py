"""Observer-only attribution of probability-memory harms, with shared-forward controls."""
import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from tools.instance_memory.data import read, save
from tools.instance_memory.metrics import identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA, digest
from tools.semantic_memory.pilot_metrics import confusion, scores
from .point_history_audit import neighbors
from .point_history_causes import truth_masks
from .point_probability_memory import RawProbabilityMemory, classify, mixture

CONTROLS = ['raw', 'current_spatial_probability', 'previous_raw_probability', 'previous_hard_vote']
PRIVILEGED = ['gt_same_class', 'gt_same_object']
ARMS = CONTROLS + PRIVILEGED


def diagnose(p, old_p, gt, identity, old_gt, old_identity, indices, valid, history):
    """GT may REMOVE neighbors only; no predicted probability is changed here."""
    raw = p.argmax(1)
    cls, obj = truth_masks(gt, identity, old_gt, old_identity, indices, valid)
    predictions = {
        name: classify(mixture(p, old_p, indices, mask), raw)
        for name, mask in zip(PRIVILEGED, (cls, obj))
    }
    good = gt >= 0
    harm = good & (raw == gt) & (history != gt)
    fix = good & (raw != gt) & (history == gt)
    all_cls = valid.any(1) & (~valid | cls).all(1)
    all_obj = valid.any(1) & (~valid | obj).all(1)
    categories = dict(
        original_harms=harm,
        original_fixes=fix,
        harms_all_neighbors_same_gt_class=harm & all_cls,
        harms_all_neighbors_same_gt_object=harm & all_obj,
        fixes_all_neighbors_same_gt_class=fix & all_cls,
        fixes_all_neighbors_same_gt_object=fix & all_obj,
        harms_without_same_class_neighbor=harm & ~cls.any(1),
        harms_with_mixed_classes=harm & ~all_cls & cls.any(1),
    )
    for name, y in predictions.items():
        clean_harm = good & (raw == gt) & (y != gt)
        categories.update({
            name + '_harms_removed': harm & (y == gt),
            name + '_new_harms': ~harm & clean_harm,
            name + '_remaining_harms': clean_harm,
            name + '_fixes_lost': fix & (y != gt),
            name + '_fixes': good & (raw != gt) & (y == gt),
        })
    return predictions, categories


def online_signals(p, old_p, xyz, old_xyz, indices, valid):
    """No GT, future data or privileged masks enter these quantities."""
    count = valid.sum(1)
    hp = np.zeros_like(p)
    distance = np.zeros(len(p), np.float64)
    for col in range(4):
        hp += old_p[indices[:, col]] * valid[:, col, None]
        distance += np.linalg.norm(xyz - old_xyz[indices[:, col]], axis=1) * valid[:, col]
    hp /= np.maximum(count[:, None], 1)
    distance /= np.maximum(count, 1)
    pc, hc = p.max(1), hp.max(1)
    entropy = lambda q: -(q * np.log(np.maximum(q, 1e-12))).sum(1) / np.log(16)
    return dict(current_confidence=pc, history_confidence=hc,
                history_minus_current_confidence=hc-pc,
                current_entropy=entropy(p), history_entropy=entropy(hp),
                mean_distance_m=distance), hp.argmax(1)


def histogram_update(storage, signals, mask, bins):
    for name, x in signals.items():
        vals = x[mask].astype(np.float64)
        key = storage.setdefault(name, dict(count=0, sum=0., sum_square=0., histogram=[0]*(len(bins[name])-1)))
        hist = np.histogram(vals, bins=bins[name])[0]
        if int(hist.sum()) != len(vals):
            raise ValueError('Signal outside preregistered bins: ' + name)
        key['count'] += len(vals)
        key['sum'] += float(vals.sum())
        key['sum_square'] += float(vals @ vals)
        key['histogram'] = (np.asarray(key['histogram'])+hist).tolist()


def merge_signals(reports):
    total = {}
    for report in reports:
        for scope, groups in report['signals'].items():
            for category, signals in groups.items():
                target = total.setdefault(scope, {}).setdefault(category, {})
                for name, s in signals.items():
                    d = target.setdefault(name, dict(count=0, sum=0., sum_square=0., histogram=[0]*len(s['histogram'])))
                    for k in ('count', 'sum', 'sum_square'):
                        d[k] += s[k]
                    d['histogram'] = (np.asarray(d['histogram'])+s['histogram']).tolist()
    return total


def run(index_path, cache, reference, out):
    import torch
    from tools.visual_instances.kalman_cache import gpu_admission
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
    from ptv3_worker import Segmenter
    torch.set_num_threads(1)
    torch.manual_seed(20261029)
    index = read(index_path)
    names = index['independent_dev_added']
    baseline = read(reference/'report.json')
    if len(names) != 8 or names != baseline['scenes']:
        raise ValueError('Frozen development selection changed')
    if any((cache/(s+'.inputs.json.gz')).exists() or (cache/(s+'.targets.json.gz')).exists()
           for s in index['parts']['test']):
        raise ValueError('Holdout opened')
    if shutil.disk_usage(out.parent).free < 4*1024**3:
        raise RuntimeError('Less than 4 GiB disk headroom')
    admission = gpu_admission()
    out.mkdir(parents=True, exist_ok=False)
    config_path = Path(__file__).resolve().parents[2]/'config/point_probability_causes_20261029.json'
    config = read(config_path)
    source_paths = [Path(__file__), Path(__file__).with_name('point_probability_memory.py'),
                    Path(__file__).with_name('point_history_audit.py'), Path(__file__).with_name('point_history_causes.py')]
    save(out/'protocol.json', dict(config=config, config_sha256=digest(config_path),
        index_path=str(index_path), cache_path=str(cache), reference_path=str(reference),
        index_sha256=digest(index_path), reference_sha256=digest(reference/'report.json'),
        target_sha256={s:digest(cache/(s+'.targets.json.gz')) for s in names},
        sources={p.name:digest(p) for p in source_paths},
        checkpoint_sha256=digest(os.environ['PTV3_CHECKPOINT']), admission=admission))
    model = Segmenter(half=True, shuffle=False, intensity_scale=1.)
    reports = []
    start = time.perf_counter()
    for name in names:
        rows = index['sequences'][name]
        targets = read(cache/(name+'.targets.json.gz'))['frames']
        if len(rows) != len(targets):
            raise ValueError('Incomplete targets')
        memory = RawProbabilityMemory()
        old_target = None
        mats = {a:np.zeros((16,16),np.int64) for a in ARMS}
        counts = {a:Counter() for a in ARMS}
        attribution = {s:Counter() for s in ('all', 'camera_foreground')}
        classes = {s:defaultdict(Counter) for s in attribution}
        objects = defaultdict(Counter)
        signals = {}
        frames = []
        for k, (row, target) in enumerate(zip(rows, targets)):
            path = DATA/row['scan_path']
            if digest(path) != row['scan_sha256']:
                raise ValueError('Scan changed')
            scan = np.fromfile(path,dtype='<f4').reshape(-1,5)[:,:4].copy()
            scan[:,3] /= 255.
            pose = np.asarray(row['transform'])
            xyz = scan[:,:3].astype(np.float64) @ pose[:3,:3].T + pose[:3,3]
            raw = model.segment(scan,return_features=True)
            p = raw['logits'].float().softmax(-1)[raw['inverse']].cpu().numpy()
            stamp = int(row['timestamp_us']*1000)
            old = memory.previous
            if old is not None and (old['epoch'] != name or stamp-old['stamp'] > 750000000):
                old = None
            pred, details = memory.step(xyz,p,stamp,name)
            np.testing.assert_array_equal(memory.previous['p'],p)
            # Finish the unchanged controls and online signal extraction first.
            if old is not None:
                assert old['stamp'] < stamp
                hi, hv = neighbors(xyz,old['xyz'])
                np.testing.assert_array_equal(classify(mixture(p,old['p'],hi,hv),pred['raw']),pred['previous_raw_probability'])
                features, history_label = online_signals(p,old['p'],xyz,old['xyz'],hi,hv)
            # Everything below this line is a privileged observer, never fed back.
            gt = np.asarray(target['gt'])
            identity = identity_truth(target)
            camera = np.zeros(len(gt),bool)
            camera[target['camera_in_view_indices']] = True
            fg = (gt >= 0) & (gt < 10) & camera
            scope_masks = dict(all=gt>=0, camera_foreground=fg)
            if old is not None:
                privileged, categories = diagnose(p,old['p'],gt,identity,np.asarray(old_target['gt']),
                    identity_truth(old_target),hi,hv,pred['previous_raw_probability'])
                pred.update(privileged)
                categories['queries_with_history'] = hv.any(1)
                categories['history_argmax_wrong_on_harms'] = categories['original_harms'] & (history_label != gt)
                categories['history_argmax_correct_on_fixes'] = categories['original_fixes'] & (history_label == gt)
                frame_counts = {}
                for scope, mask in scope_masks.items():
                    frame_counts[scope] = {}
                    for event, event_mask in categories.items():
                        at = mask & event_mask
                        number = int(at.sum())
                        attribution[scope][event] += number
                        frame_counts[scope][event] = number
                        for c, n in zip(*np.unique(gt[at],return_counts=True)):
                            classes[scope][str(int(c))][event] += int(n)
                    for event in ('original_fixes','original_harms','harms_all_neighbors_same_gt_class'):
                        histogram_update(signals.setdefault(scope,{}).setdefault(event,{}),features,
                            mask & categories[event],config['signal_bins'])
                frames.append(dict(frame=k, prior_stamp_ns=old['stamp'], stamp_ns=stamp, counts=frame_counts, **details))
            else:
                for a in PRIVILEGED:
                    pred[a] = pred['raw'].copy()
                frames.append(dict(frame=k,prior_stamp_ns=None,stamp_ns=stamp,counts={},**details))
            for a, y in pred.items():
                mats[a] += confusion(gt,y)
                counts[a].update(fixes_raw=int(((gt>=0)&(pred['raw']!=gt)&(y==gt)).sum()),
                    harms_raw=int(((gt>=0)&(pred['raw']==gt)&(y!=gt)).sum()),
                    camera_foreground_correct=int((fg&(y==gt)).sum()),camera_foreground_points=int(fg.sum()))
            for obj in np.unique(identity[fg & (identity>0)]):
                at = fg & (identity==obj)
                c = objects[str(int(obj))]
                c['observed_frames'] += 1
                c['points'] += int(at.sum())
                for a, y in pred.items():
                    c[a+'_correct'] += int((at & (y==gt)).sum())
                    c[a+'_fixes'] += int((at & (pred['raw']!=gt) & (y==gt)).sum())
                    c[a+'_harms'] += int((at & (pred['raw']==gt) & (y!=gt)).sum())
            old_target = target
            if (k+1)%10 == 0:
                save(out/'progress.json',dict(scene=name,frame=k+1,total=len(rows),completed_scenes=len(reports),elapsed_s=time.perf_counter()-start))
        previous = read(reference/(name+'.json'))
        exact = {a:bool(np.array_equal(mats[a],previous['arms'][a]['confusion'])) for a in CONTROLS}
        report = dict(scene=name,log=index['scene_logs'][name],frames=frames,
            arms={a:dict(scores(mats[a]),counts=dict(counts[a]),camera_foreground_accuracy=
                counts[a]['camera_foreground_correct']/max(1,counts[a]['camera_foreground_points'])) for a in ARMS},
            counts={s:dict(c) for s,c in attribution.items()},
            per_class={s:{k:dict(v) for k,v in cc.items()} for s,cc in classes.items()},
            camera_foreground_objects={k:dict(v) for k,v in objects.items()},
            signals=signals,reference_confusion_exact=exact)
        save(out/(name+'.json'),report)
        reports.append(report)
        print(name,{a:round(100*report['arms'][a]['miou_fixed16'],6) for a in ARMS},
              'reference_exact',exact,'camera_foreground',report['counts']['camera_foreground'],flush=True)
    aggregate = {}
    for a in ARMS:
        c = Counter()
        for r in reports:
            c.update(r['arms'][a]['counts'])
        aggregate[a] = dict(scores(sum(np.asarray(r['arms'][a]['confusion']) for r in reports)),counts=dict(c),
            camera_foreground_accuracy=c['camera_foreground_correct']/max(1,c['camera_foreground_points']))
    totals = {}
    per_class = {}
    for scope in ('all','camera_foreground'):
        c = Counter()
        cls = defaultdict(Counter)
        for r in reports:
            c.update(r['counts'][scope])
            for k,v in r['per_class'][scope].items():
                cls[k].update(v)
        totals[scope] = dict(c)
        per_class[scope] = {k:dict(v) for k,v in cls.items()}
    result = dict(aggregate=aggregate,counts=totals,per_class=per_class,signals=merge_signals(reports),
        scenes=names,frames=sum(len(r['frames']) for r in reports),
        all_reference_confusions_exact=all(all(r['reference_confusion_exact'].values()) for r in reports),
        reference_confusion_exact={r['scene']:r['reference_confusion_exact'] for r in reports},
        elapsed_s=time.perf_counter()-start,
        scope='Development-only privileged diagnosis; GT filters are not deployable scores or a general oracle upper bound. Fresh same-forward controls are authoritative. Observer cost is not inference latency.')
    save(out/'report.json',result)
    save(out/'progress.json',dict(status='complete',scenes=names,frames=result['frames'],elapsed_s=result['elapsed_s']))
    print('aggregate',{a:dict(miou=100*r['miou_fixed16'],camera_fg=100*r['camera_foreground_accuracy']) for a,r in aggregate.items()},
          'reference_exact',result['all_reference_confusions_exact'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('index','cache','reference','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    run(args.index,args.cache,args.reference,args.out)
