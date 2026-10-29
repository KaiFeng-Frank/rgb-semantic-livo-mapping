"""Matched-budget real-label experiment for current-anchored semantic repair.

The earlier six evaluation scenes are a REGRESSION set: they informed this
revision, so their results cannot be described as a new untouched holdout.
All hyperparameters/three seeds are frozen before this run; selection uses dev.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
from semantic_memory import HistoryFusion, MemoryConfig
from semantic_repair import CurrentAnchoredRepair, RepairConfig, repair_memory_config, repair_loss
from .nuscenes_pilot import load, save, announce, current_output
from .nuscenes_pilot_data import digest
from .pilot_metrics import confusion, scores, events, paired_summary
from .repair_supervision import present_class_weights

ARMS = ('baseline', 'anchor', 'current_extra', 'temporal', 'no_history', 'shuffled_history', 'narrow_history')


def anchor_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['provenance']['kind'] != 'current_only':
        raise ValueError('Anchor must be independently trained current-only control')
    model = HistoryFusion(MemoryConfig(**saved['config'])).cuda().eval()
    model.load_state_dict(saved['fusion'], strict=True)
    model.requires_grad_(False)
    return model


def call(model, raw, anchor, **kwargs):
    return model(raw['xyz'], raw['features'], raw['logits'], anchor_logits=anchor,
        stamp=raw['stamp'], frame=raw['frame'], transform=raw['transform'], **kwargs)


def class_weights(index, cache):
    counts = np.zeros(16, dtype=np.int64)
    for scene in index['parts']['train']:
        for row in index['sequences'][scene][1:]:
            with np.load(cache/scene/f"{row['frame']:03d}.npz") as f:
                labels = f['label']
            counts += np.bincount(labels[labels >= 0], minlength=16)
    # Counts from training voxels only. No heldout priors/class frequencies.
    # Missing classes have no training loss; they must not dominate the mean
    # and force observed minority classes down into the clipping floor.
    weights = present_class_weights(counts)
    return counts.tolist(), torch.tensor(weights, dtype=torch.float32, device='cuda')


@torch.no_grad()
def dev(models, anchor, index, cache):
    matrices = {k:np.zeros((16,16), np.int64) for k in models}
    for m in models.values():
        m.eval()
    for scene in index['parts']['dev']:
        for m in models.values():
            m.reset()
        for row in index['sequences'][scene]:
            raw, arrays = load(cache, scene, row)
            logits = current_output(anchor, raw)['logits']
            for kind, model in models.items():
                out = call(model, raw, logits, current_only=kind=='current_extra')
                pred = out['logits'].argmax(-1).cpu().numpy()[arrays['inverse']]
                matrices[kind] += confusion(arrays['point_label'].astype(np.int64), pred)
    return {k:scores(cm) for k,cm in matrices.items()}


def train(seed, index, cache, out, anchors, protocol, weights):
    torch.manual_seed(seed)
    folder = out/f'seed-{seed}'
    folder.mkdir(exist_ok=False)
    anchor = anchor_model(anchors/f'seed-{seed}'/'current_only.pt')
    models = {k:CurrentAnchoredRepair(device='cuda') for k in ('temporal','current_extra')}
    models['current_extra'].fusion.load_state_dict(models['temporal'].fusion.state_dict())
    opts = {k:torch.optim.Adam(m.fusion.parameters(), lr=protocol['lr']) for k,m in models.items()}
    best = {k:-1. for k in models}
    for epoch in range(1, protocol['epochs']+1):
        report = {k:dict(loss=0., frames=0, steps=0, queries=0, usable=0) for k in models}
        for m in models.values():
            m.train()
        scenes = list(index['parts']['train'])
        np.random.default_rng(seed*100+epoch).shuffle(scenes)
        for scene in scenes:
            for m in models.values():
                m.reset()
            for opt in opts.values():
                opt.zero_grad(set_to_none=True)
            losses = {k:[] for k in models}
            rows = index['sequences'][scene]
            for j,row in enumerate(rows):
                raw,_ = load(cache, scene, row)
                with torch.no_grad():
                    logits = current_output(anchor, raw)['logits']
                for kind, model in models.items():
                    output = call(model, raw, logits, epoch=epoch, current_only=kind=='current_extra')
                    if j:
                        loss = repair_loss(output, raw['label'], weights)
                        if not bool(torch.isfinite(loss)):
                            raise ValueError('Nonfinite repair loss')
                        losses[kind].append(loss)
                        report[kind]['loss'] += float(loss.detach())
                        report[kind]['frames'] += 1
                        for key in ('queries','usable'):
                            report[kind][key] += model.last_status[key]
                if j and (len(losses['temporal']) == protocol['clip'] or j == len(rows)-1):
                    for kind, model in models.items():
                        torch.stack(losses[kind]).mean().backward()
                        torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), 5., error_if_nonfinite=True)
                        opts[kind].step()
                        opts[kind].zero_grad(set_to_none=True)
                        losses[kind].clear()
                        report[kind]['steps'] += 1
        validation = dev(models, anchor, index, cache)
        for kind, model in models.items():
            report[kind]['loss'] /= report[kind]['frames']
            if validation[kind]['miou_fixed16'] > best[kind]:
                best[kind] = validation[kind]['miou_fixed16']
                provenance = dict(scope='real_pilot_revision', deployable=False,
                    seed=seed, kind=kind, epoch=epoch, selected_on='dev only',
                    protocol_sha256=digest(out/'protocol.json'),
                    anchor_sha256=digest(anchors/f'seed-{seed}'/'current_only.pt'))
                torch.save(dict(schema='current-anchored-repair-v2', trained=True,
                    config=asdict(model.cfg), repair=asdict(model.repair),
                    fusion=model.fusion.state_dict(), provenance=provenance), folder/f'{kind}.pt')
        save(folder/f'epoch-{epoch}.json', dict(train=report, dev=validation))
        announce(stage='repair_train', seed=seed, epoch=epoch,
                 loss={k:v['loss'] for k,v in report.items()},
                 dev_miou={k:100*v['miou_fixed16'] for k,v in validation.items()})


@torch.no_grad()
def evaluate(seed, index, cache, out, anchors):
    anchor = anchor_model(anchors/f'seed-{seed}'/'current_only.pt')
    models = {k:CurrentAnchoredRepair(device='cuda').eval() for k in ARMS if k not in ('baseline','anchor')}
    checkpoints = {}
    for kind, m in models.items():
        path = out/f'seed-{seed}'/('current_extra.pt' if kind=='current_extra' else 'temporal.pt')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        m.fusion.load_state_dict(saved['fusion'], strict=True)
        checkpoints[kind] = dict(sha256=digest(path), epoch=saved['provenance']['epoch'])
    total = {k:np.zeros((16,16), np.int64) for k in ARMS}
    per_scene, observations, frames, timing = {}, [], [], defaultdict(list)
    pairs = {k:defaultdict(int) for k in ARMS}
    torch.cuda.reset_peak_memory_stats()
    for scene in index['parts']['test']:
        for m in models.values():
            m.reset()
        matrices = {k:np.zeros((16,16), np.int64) for k in ARMS}
        for row in index['sequences'][scene]:
            raw, arrays = load(cache, scene, row)
            torch.cuda.synchronize(); start = time.perf_counter()
            logits = current_output(anchor, raw)['logits']
            torch.cuda.synchronize(); anchor_ms = (time.perf_counter()-start)*1000
            timing['anchor'].append(anchor_ms)
            outputs = dict(baseline=raw['logits'], anchor=logits)
            for kind, model in models.items():
                torch.cuda.synchronize(); start = time.perf_counter()
                result = call(model, raw, logits, current_only=kind=='current_extra',
                    disable_history=kind=='no_history', shuffle_history=kind=='shuffled_history',
                    narrow_history=kind=='narrow_history')
                torch.cuda.synchronize(); elapsed = (time.perf_counter()-start)*1000
                timing[kind].append(elapsed)
                if kind in ('temporal','current_extra'):
                    timing[kind+'_including_anchor'].append(elapsed+anchor_ms)
                outputs[kind] = result['logits']
            pred = {k:v.argmax(-1).cpu().numpy()[arrays['inverse']] for k,v in outputs.items()}
            labels = arrays['point_label'].astype(np.int64)
            for k in ARMS:
                matrices[k] += confusion(labels, pred[k])
                for name, value in paired_summary(labels, pred['anchor'], pred[k]).items():
                    pairs[k][name] += value
            for i,group in enumerate(json.loads(arrays['groups'].item())):
                mask = arrays['group_owner'] == i
                observations.append(dict(scene=scene, frame=row['frame'], stamp=raw['stamp'],
                    instance=group['instance'], label=group['label'], points=int(mask.sum()),
                    correct_fraction={k:float((p[mask]==group['label']).mean()) for k,p in pred.items()}))
            frames.append(dict(scene=scene, frame=row['frame'], **models['temporal'].last_status))
        per_scene[scene] = {k:scores(cm) for k,cm in matrices.items()}
        for k,cm in matrices.items():
            total[k] += cm
        announce(stage='repair_regression_scene', seed=seed, scene=scene,
                 miou={k:round(scores(cm)['miou_fixed16']*100,3) for k,cm in matrices.items()})
    result = dict(seed=seed, checkpoints=checkpoints,
        point_metrics={k:dict(scores(cm), relative_to_anchor=dict(pairs[k])) for k,cm in total.items()},
        per_scene=per_scene, events=events(observations, ARMS),
        class_events={str(c):events([r for r in observations if r['label']==c], ARMS) for c in (3,6,9)},
        module_timing={k:dict(n=len(v), median_ms=float(np.median(v)), p95_ms=float(np.percentile(v,95))) for k,v in timing.items()},
        query_totals={k:sum(r[k] for r in frames) for k in ('queries','usable')},
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    folder = out/f'seed-{seed}'
    save(folder/'evaluation.json', result)
    save(folder/'object-observations.json', observations)
    save(folder/'frames.json', frames)
    return result


def summarize(results, index):
    arms = {}
    for k in ARMS:
        values = [100*r['point_metrics'][k]['miou_fixed16'] for r in results]
        arms[k] = dict(miou_percent_mean=float(np.mean(values)), miou_percent_std=float(np.std(values)), seeds=values)
    rng = np.random.default_rng(20261025)
    boot = []
    for _ in range(2000):
        draw = rng.choice(index['parts']['test'], len(index['parts']['test']), replace=True)
        differences = []
        for r in results:
            cm = {k:sum((np.asarray(r['per_scene'][s][k]['confusion']) for s in draw), np.zeros((16,16),np.int64))
                  for k in ('temporal','current_extra')}
            differences.append(100*(scores(cm['temporal'])['miou_fixed16']-scores(cm['current_extra'])['miou_fixed16']))
        boot.append(np.mean(differences))
    aggregate_events = {e:{k:float(np.mean([r['events'][e][k]['success_rate'] for r in results])) for k in ARMS}
                        for e in results[0]['events']}
    medians = [r['module_timing']['temporal_including_anchor']['median_ms'] for r in results]
    return dict(arms=arms, events=aggregate_events,
        temporal_minus_matched_current_pp=arms['temporal']['miou_percent_mean']-arms['current_extra']['miou_percent_mean'],
        regression_paired_scene_bootstrap_95_interval_pp=np.percentile(boot,[2.5,97.5]).tolist(),
        total_module_median_ms_by_seed=medians, five_ms_module_budget_passed=max(medians)<=5.,
        evaluation_role='previously inspected regression scenes, not a fresh heldout claim',
        deployable=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('index','cache','out','anchors'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    index = json.loads(args.index.read_text())
    manifest = json.loads((args.cache/'manifest.json').read_text())
    if manifest['index_sha256'] != digest(args.index):
        raise ValueError('Cache index mismatch')
    for entry in manifest['files']:
        if digest(args.cache/entry['path']) != entry['sha256']:
            raise ValueError('Feature cache hash mismatch')
    logs = [{index['scene_logs'][s] for s in index['parts'][part]} for part in ('train','dev','test')]
    if any(logs[i]&logs[j] for i in range(3) for j in range(i)):
        raise ValueError('Collection-log leakage')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    counts, weights = class_weights(index, args.cache)
    protocol = dict(seeds=[0,1,2], epochs=5, clip=4, lr=.001,
        config=asdict(repair_memory_config()), repair=asdict(RepairConfig()),
        index_sha256=digest(args.index), feature_manifest_sha256=digest(args.cache/'manifest.json'),
        backbone_sha256=manifest['backbone_sha256'], geometry=index['geometry'], parts=index['parts'],
        training_voxel_counts=counts, class_weights=weights.cpu().tolist(),
        anchors={str(s):digest(args.anchors/f'seed-{s}'/'current_only.pt') for s in range(3)},
        control='same frozen 5-epoch current anchor, same new joint-head parameters/init/query set/loss/5 epochs/steps; missing history uses the current token and still trains',
        objective='train-class weighted CE + .5 wrong-class margin repair + .5 correct-class margin retention',
        selection='best dev fixed16 point mIoU per arm/seed; no threshold search',
        regression='old test scenes were inspected and motivated this revision; not a fresh holdout',
        ablations='same selected temporal weights: current token only, permuted past content, narrow .5m radius',
        timing='synchronized GPU module, report anchor+repair together; excludes backbone/transport',
        source_sha256={str(f.relative_to(REPO)):digest(f) for f in
            (REPO/'src/semantic_repair.py', REPO/'src/semantic_memory.py', Path(__file__),
             REPO/'tools/semantic_memory/nuscenes_pilot.py', REPO/'tools/semantic_memory/pilot_metrics.py',
             REPO/'tools/semantic_memory/repair_supervision.py')})
    save(args.out/'protocol.json', protocol)
    for seed in protocol['seeds']:
        train(seed, index, args.cache, args.out, args.anchors, protocol, weights)
    result = summarize([evaluate(s, index, args.cache, args.out, args.anchors) for s in protocol['seeds']], index)
    save(args.out/'summary.json', result)
    announce(stage='repair_complete', **result)


if __name__ == '__main__':
    main()
