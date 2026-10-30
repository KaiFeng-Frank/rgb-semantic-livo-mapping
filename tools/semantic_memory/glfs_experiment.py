"""Fit only official GLFS parameters and evaluate the fixed, paired development set."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from tools.instance_memory.data import read, save
from .nuscenes_pilot_data import digest
from .pilot_metrics import confusion, scores, events
from .glfs_baseline import GLFS, probability_average, official_mdece, verify_sources


def read_frames(root, entries):
    frames = []
    for row in entries:
        path = root/'cache'/row['path']
        if digest(path) != row['sha256']:
            raise ValueError('Shared input cache changed')
        with np.load(path, allow_pickle=False) as f:
            arrays = {k:f[k] for k in f.files}
        frames.append((row, arrays))
    return frames


def observations(d, seed):
    return (torch.from_numpy(np.concatenate((d['current'][seed, :, None], d['history']), axis=1)),
            torch.from_numpy(d['depth']), torch.from_numpy(d['angle']), torch.from_numpy(d['valid']))


class Calibration:
    def __init__(self):
        self.n = 0
        self.nll = self.brier = 0.
        self.bin_n = np.zeros(10)
        self.bin_correct = np.zeros(10)
        self.bin_confidence = np.zeros(10)

    def add(self, p, y):
        valid = y >= 0
        p, y = p[valid], y[valid].astype(np.int64)
        self.n += len(y)
        self.nll -= np.log(p[np.arange(len(y)), y].clip(1e-30)).sum(dtype=np.float64)
        self.brier += (np.square(p).sum(1)-2*p[np.arange(len(y)), y]+1).sum(dtype=np.float64)
        conf = p.max(1)
        bins = np.minimum((conf*10).astype(int), 9)
        self.bin_n += np.bincount(bins, minlength=10)
        self.bin_correct += np.bincount(bins, weights=p.argmax(1)==y, minlength=10)
        self.bin_confidence += np.bincount(bins, weights=conf, minlength=10)

    def report(self):
        return dict(points=self.n, nll=self.nll/self.n, brier=self.brier/self.n,
                    ece=float(np.abs(self.bin_correct-self.bin_confidence).sum()/self.n))


def prepare_selection(frames, seed):
    # Only predictions outside the query set are cached. Every GLFS prediction
    # and every original point is still scored on each selection check.
    return [(observations(d, seed), torch.from_numpy(d['anchor']).softmax(-1).argmax(-1).numpy(),
             d['query'], d['inverse'], d['point_label'].astype(np.int64)) for _, d in frames]


@torch.no_grad()
def selection_confusion(model, prepared):
    model.eval()
    total = np.zeros((16, 16), np.int64)
    for args, fixed, query, inverse, target in prepared:
        labels = fixed.copy()
        labels[query] = model(*args).argmax(-1).numpy()
        total += confusion(target, labels[inverse])
    return total


@torch.no_grad()
def evaluate(model, frames, seed, detailed=False):
    names = ('baseline', 'current', 'probability_average', 'glfs', 'glfs_no_history')
    matrices = {k:np.zeros((16, 16), np.int64) for k in names}
    per_scene, counts = {}, {k:Counter() for k in names}
    cals = {k:Calibration() for k in names if k != 'baseline'}
    rows, costs = [], []
    model.eval()
    for row, d in frames:
        args = observations(d, seed)
        t = time.perf_counter()
        p = model(*args)
        costs.append((time.perf_counter()-t)*1000)
        query_probs = dict(current=args[0][:, 0].softmax(-1), probability_average=probability_average(args[0], args[3]),
                           glfs=p, glfs_no_history=model(*(x[:, :1] for x in args)))
        inverse, y = d['inverse'], d['point_label'].astype(np.int64)
        anchor = torch.from_numpy(d['anchor']).softmax(-1).numpy()
        predictions = dict(baseline=d['baseline'][inverse])
        for name, probability in query_probs.items():
            full = anchor.copy()
            full[d['query']] = probability.numpy()
            point = full[inverse]
            predictions[name] = point.argmax(-1)
            if detailed:
                cals[name].add(point, y)
        sm = per_scene.setdefault(row['scene'], {k:np.zeros((16, 16), np.int64) for k in names})
        valid = y >= 0
        foreground = valid & (y < 10) & d['camera']
        for name, pred in predictions.items():
            cm = confusion(y, pred)
            matrices[name] += cm
            sm[name] += cm
            counts[name].update(rescued=int((valid & (pred == y) & (predictions['current'] != y)).sum()),
                                harmed=int((valid & (pred != y) & (predictions['current'] == y)).sum()),
                                changed=int((valid & (pred != predictions['current'])).sum()),
                                foreground_correct=int((foreground & (pred == y)).sum()), foreground_points=int(foreground.sum()))
        if detailed:
            for obj in np.unique(d['identity'][d['identity'] > 0]):
                at = (d['identity'] == obj) & valid
                if at.sum() < 5:
                    continue
                classes = np.unique(y[at])
                if len(classes) != 1:
                    raise ValueError('Ambiguous evaluation identity')
                rows.append(dict(scene=row['scene'], instance=int(obj), label=int(classes[0]), frame=row['frame'],
                    stamp=row['stamp'], points=int(at.sum()), correct_fraction={k:float((v[at]==y[at]).mean()) for k,v in predictions.items()}))
    report = dict(arms={k:dict(scores(v), counts=dict(counts[k]),
        camera_foreground_accuracy=counts[k]['foreground_correct']/max(1, counts[k]['foreground_points'])) for k,v in matrices.items()},
        per_scene={s:{k:scores(v) for k,v in arms.items()} for s,arms in per_scene.items()})
    if detailed:
        report.update(calibration={k:v.report() for k,v in cals.items()}, object_observations=rows,
            raw_baseline_events=events(rows, names),
            current_baseline_events=events([{**r, 'correct_fraction':{**r['correct_fraction'], 'baseline':r['correct_fraction']['current']}} for r in rows], names),
            fusion_cpu_ms=dict(median=float(np.median(costs)), p95=float(np.percentile(costs, 95)),
                               scope='CPU2 threads, cached observations; includes validation checks, excludes geometry/retrieval/backbone; not end-to-end online admission'))
    return report


def train_seed(root, fit_frames, selection_frames, seed, cfg, out):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GLFS()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    calibration_loss = official_mdece(root/'dece.py')
    columns = [[] for _ in range(4)]
    targets = []
    for row, d in fit_frames:
        if row['frame'] == 0:
            continue
        mask = d['label'] >= 0
        for column, a in zip(columns, observations(d, seed)):
            column.append(a[mask])
        targets.append(torch.from_numpy(d['label'][mask].astype(np.int64)))
    inputs = [torch.cat(c) for c in columns]
    y = torch.cat(targets)
    frequencies = torch.bincount(y, minlength=16).float()/len(y)
    # Released log weights equal 2*log(released sqrt-inverse-frequency weights).
    # Re-estimate that same -log(frequency) rule on the fit subset only.
    weights = -torch.log(frequencies.clamp_min(1e-30))
    weights[frequencies == 0] = 0
    classification_loss = nn.NLLLoss(weight=weights)
    folder = out/f'seed-{seed}'/f"lr-{cfg['lr']:.0e}"
    folder.mkdir(parents=True, exist_ok=False)
    initial_selection = evaluate(model, selection_frames, seed)
    save(folder/'initial-selection.json', initial_selection)
    selection = prepare_selection(selection_frames, seed)
    np.testing.assert_array_equal(selection_confusion(model, selection), initial_selection['arms']['glfs']['confusion'])
    save(folder/'selection-parity.json', dict(passed=True, frames=len(selection_frames),
        check='Exact confusion matrix vs complete generic evaluator on all selection frames, before fitting'))
    save(folder/'fit-statistics.json', dict(queries=len(y), class_counts=torch.bincount(y, minlength=16).tolist(),
        class_weights=weights.tolist(), parameters=sum(p.numel() for p in model.parameters())))
    best, stale, step, check, started = -float('inf'), 0, 0, 0, time.perf_counter()
    steps_per_epoch = int(np.ceil(len(y)/cfg['batch_size']))
    check_steps = {int(np.ceil(steps_per_epoch*x/3)) for x in (1, 2, 3)}
    history = []
    stop = False
    for epoch in range(1, cfg['max_epochs']+1):
        order = rng.permutation(len(y))
        epoch_losses = []
        for batch, begin in enumerate(range(0, len(y), cfg['batch_size']), 1):
            model.train()
            at = torch.from_numpy(order[begin:begin+cfg['batch_size']])
            p = model(*(a[at] for a in inputs))
            loss = classification_loss(p.log(), y[at]) + cfg['mDECE_weight']*calibration_loss(p, y[at])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite official objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            step += 1
            if batch in check_steps:
                check += 1
                score = scores(selection_confusion(model, selection))['miou_fixed16']
                improved = score > best
                if improved:
                    best, stale = score, 0
                    torch.save(dict(state_dict=model.state_dict(), seed=seed, learning_rate=cfg['lr'], epoch=epoch, step=step, check=check,
                                    selection_miou=score, protocol_sha256=digest(root/'protocol.json'), deployable=False), folder/'best.pt')
                else:
                    stale += 1
                row = dict(epoch=epoch, step=step, check=check, loss=float(np.mean(epoch_losses)),
                    selection_miou=score, best=best, stale=stale, elapsed_s=time.perf_counter()-started,
                    temperature_min=float(model.log_temperature.exp().detach().min()),
                    temperature_max=float(model.log_temperature.exp().detach().max()),
                    max_lookup_change=float((model.weights.detach()-1).abs().max()))
                history.append(row)
                save(folder/'training.json', history)
                print('glfs', seed, row, flush=True)
                if stale >= cfg['patience_validation_checks']:
                    stop = True
                    break
        if stop:
            break
    selected = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
    result = dict(seed=seed, selected={k:v for k,v in selected.items() if k != 'state_dict'},
                  stopped_for='official early stopping' if stop else '300 epoch cap', elapsed_s=time.perf_counter()-started)
    save(folder/'fit-result.json', result)
    return result


def run(root, only_seed=None, learning_rate=None):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    protocol = read(root/'protocol.json')
    verify_sources(root, protocol)
    manifest = read(root/'cache/manifest.json')
    if manifest['protocol_sha256'] != digest(root/'protocol.json') or len(manifest['files']) != 963:
        raise ValueError('Incomplete or different experiment')
    out = root/'run'
    if only_seed is not None:
        if only_seed not in protocol['seeds'] or learning_rate not in protocol['training']['learning_rates'] or not out.is_dir():
            raise ValueError('Unregistered worker')
        all_frames = read_frames(root, [r for r in manifest['files'] if r['part'] != 'dev'])
        parts = {part:[x for x in all_frames if x[0]['part'] == part] for part in protocol['parts']}
        train_seed(root, parts['fit'], parts['selection'], only_seed, {**protocol['training'], 'lr':learning_rate}, out)
        return
    out.mkdir(exist_ok=False)
    # Independent CPU processes, each with two threads. No shared optimizer,
    # checkpoint or RNG; the existing remote GPU training continues separately.
    jobs = {}
    for seed in protocol['seeds']:
        for lr in protocol['training']['learning_rates']:
            name = f'{seed}-lr-{lr:.0e}'
            with (out/f'worker-{name}.log').open('w') as log:
                jobs[name] = subprocess.Popen([sys.executable, '-m', 'tools.semantic_memory.glfs_experiment',
                    '--root', str(root.resolve()), '--seed', str(seed), '--learning-rate', str(lr)], stdout=log, stderr=subprocess.STDOUT)
    while any(job.poll() is None for job in jobs.values()):
        save(out/'progress.json', {str(seed):dict(pid=job.pid, returncode=job.poll()) for seed,job in jobs.items()})
        time.sleep(5)
    if any(job.returncode != 0 for job in jobs.values()):
        raise RuntimeError('A GLFS seed failed; inspect the preserved worker logs')
    # Only now open development arrays. Every candidate/seed completed selection
    # independently; a tie favors the released default LR, first in the list.
    dev_frames = read_frames(root, [r for r in manifest['files'] if r['part'] == 'dev'])
    reports = []
    for seed in protocol['seeds']:
        folder = out/f'seed-{seed}'
        trials = [read(folder/f'lr-{lr:.0e}/fit-result.json') for lr in protocol['training']['learning_rates']]
        chosen = max(trials, key=lambda t:t['selected']['selection_miou'])
        lr = chosen['selected']['learning_rate']
        selected = torch.load(folder/f'lr-{lr:.0e}/best.pt', map_location='cpu', weights_only=True)
        torch.save(selected, folder/'best.pt')
        model = GLFS()
        model.load_state_dict(selected['state_dict'], strict=True)
        result = evaluate(model, dev_frames, seed, detailed=True)
        initial_dev = evaluate(GLFS(), dev_frames, seed)
        default = torch.load(folder/'lr-1e-05/best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(default['state_dict'], strict=True)
        default_dev = evaluate(model, dev_frames, seed)
        save(folder/'object-observations.json', result.pop('object_observations'))
        result.update(seed=seed, selected=chosen['selected'], fit_trials=trials,
            initial_glfs_development=initial_dev['arms']['glfs'],
            official_default_lr_development=default_dev['arms']['glfs'])
        save(folder/'evaluation.json', result)
        reports.append(result)
        print('development', seed, 'selected lr', lr, {k:100*v['miou_fixed16'] for k,v in result['arms'].items()}, flush=True)
    names = ('current', 'probability_average', 'glfs', 'glfs_no_history')
    summary = dict(arms={k:dict(miou_percent_by_seed=[100*r['arms'][k]['miou_fixed16'] for r in reports],
        mean_miou_percent=float(np.mean([100*r['arms'][k]['miou_fixed16'] for r in reports])),
        rescues_by_seed=[r['arms'][k]['counts']['rescued'] for r in reports],
        harms_by_seed=[r['arms'][k]['counts']['harmed'] for r in reports],
        raw_baseline_recovery_by_seed=[r['raw_baseline_events']['baseline_error_recovery'][k]['success_rate'] for r in reports],
        current_baseline_recovery_by_seed=[r['current_baseline_events']['baseline_error_recovery'][k]['success_rate'] for r in reports],
        calibration_by_seed=[r['calibration'][k] for r in reports]) for k in names},
        scope='Eight previously inspected development logs; GLFS adapted to frozen retrieval and frozen current heads. No new architecture, holdout result, instance tracker or online realtime admission.', deployable=False)
    gain = [100*(r['arms']['glfs']['miou_fixed16']-r['arms']['current']['miou_fixed16']) for r in reports]
    foreground = [100*(r['arms']['glfs']['camera_foreground_accuracy']-r['arms']['current']['camera_foreground_accuracy']) for r in reports]
    positive = [sum(s['glfs']['miou_fixed16'] > s['current']['miou_fixed16'] for s in r['per_scene'].values()) for r in reports]
    recovery = float(np.mean(summary['arms']['glfs']['raw_baseline_recovery_by_seed'])-np.mean(summary['arms']['current']['raw_baseline_recovery_by_seed']))
    summary['gate'] = dict(gain_pp=gain, mean_gain_pp=float(np.mean(gain)), foreground_pp=foreground,
        positive_logs=positive, mean_recovery_difference=recovery,
        passed=bool(np.mean(gain)>=.05 and min(gain)>0 and min(foreground)>=-.1 and min(positive)>=5 and recovery>=0))
    summary['attribution'] = dict(
        history_gain_vs_same_parameters_current_only_pp=[100*(r['arms']['glfs']['miou_fixed16']-r['arms']['glfs_no_history']['miou_fixed16']) for r in reports],
        learning_gain_vs_official_initialization_pp=[100*(r['arms']['glfs']['miou_fixed16']-r['initial_glfs_development']['miou_fixed16']) for r in reports])
    summary['optimization_check'] = dict(selected_learning_rates=[r['selected']['learning_rate'] for r in reports],
        official_default_lr_miou_percent=[100*r['official_default_lr_development']['miou_fixed16'] for r in reports])
    save(out/'summary.json', summary)
    print('GLFS complete', summary['gate'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--learning-rate', type=float)
    a = p.parse_args()
    run(a.root, a.seed, a.learning_rate)
