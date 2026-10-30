"""Post-selection error attribution on the already-scored shared GLFS cache.

GT is joined here for diagnostics only. No parameters, retrieval or selection
are changed. Query-level attribution excludes semantically mixed voxels.
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from tools.instance_memory.data import read, save
from .glfs_baseline import GLFS, probability_average, verify_sources
from .glfs_experiment import observations
from .nuscenes_pilot_data import digest


def truth(d):
    n = len(d['anchor'])
    valid = d['point_label'] >= 0
    inverse = d['inverse'][valid]
    counts = np.bincount(inverse * 16 + d['point_label'][valid], minlength=n*16).reshape(n, 16)
    total = counts.sum(1)
    label = np.where(total > 0, counts.argmax(1), -1)
    pure = (total > 0) & (counts.max(1) == total)
    lo = np.full(n, np.iinfo(np.int32).max, np.int32)
    hi = np.zeros(n, np.int32)
    np.minimum.at(lo, inverse, d['identity'][valid])
    np.maximum.at(hi, inverse, d['identity'][valid])
    identity = np.where((lo == hi) & (lo > 0), lo, 0)
    return label, pure, total, identity


@torch.no_grad()
def run(root):
    torch.set_num_threads(2)
    verify_sources(root, read(root/'protocol.json'))
    models = []
    for seed in range(3):
        model = GLFS().eval()
        model.load_state_dict(torch.load(root/f'run/seed-{seed}/best.pt', map_location='cpu', weights_only=True)['state_dict'])
        models.append(model)
    totals = [dict(queries=Counter(), points=Counter()) for _ in models]
    previous, scene = None, None
    for row in read(root/'cache/manifest.json')['files']:
        if row['part'] != 'dev':
            continue
        path = root/'cache'/row['path']
        assert digest(path) == row['sha256']
        with np.load(path, allow_pickle=False) as f:
            d = {key:f[key] for key in f.files}
        if row['scene'] != scene:
            previous, scene = None, row['scene']
        y, pure, count, ids = truth(d)
        qi = d['query']
        target, pure, count, query_ids = y[qi], pure[qi], count[qi], ids[qi]
        valid = d['valid'][:, 1:]
        has = valid.any(1)
        hist_correct = d['history'].argmax(-1) == target[:, None]
        same_correct = np.zeros_like(valid)
        if previous is not None:
            source = d['source_index'].clip(0)
            same_correct = valid & hist_correct & (query_ids[:, None] > 0) & (previous[source] == query_ids[:, None])
        categories = dict(all_queries=np.ones(len(qi), bool), with_history=has,
            all_raw_history_correct=has & (hist_correct | ~valid).all(1),
            some_raw_history_correct=(hist_correct & valid).any(1),
            no_raw_history_correct=has & ~(hist_correct & valid).any(1),
            some_same_instance_correct_history=same_correct.any(1),
            all_same_instance_correct_history=has & (same_correct | ~valid).all(1))
        for seed, model in enumerate(models):
            args = observations(d, seed)
            current = args[0][:, 0].argmax(-1).numpy()
            avg = probability_average(args[0], args[3]).argmax(-1).numpy()
            full = model(*args).argmax(-1).numpy()
            solo = model(*(x[:, :1] for x in args)).argmax(-1).numpy()
            calibrated = (args[0] / model.log_temperature.exp()).argmax(-1).numpy()
            consensus = ((calibrated == target[:, None]) | ~d['valid']).all(1)
            assert not np.any(pure & consensus & (full != target))
            assert not np.any(pure & (current == target) & categories['all_raw_history_correct'] & (avg != target))
            masks = dict(current_correct=current == target, current_wrong=current != target,
                glfs_harm=(current == target) & (full != target),
                average_harm=(current == target) & (avg != target),
                glfs_rescue=(current != target) & (full == target),
                history_harm_vs_same_parameters=(solo == target) & (full != target),
                history_rescue_vs_same_parameters=(solo != target) & (full == target),
                calibration_harm=(current == target) & (solo != target))
            for category, within in categories.items():
                at = pure & within
                totals[seed]['queries'][category] += int(at.sum())
                totals[seed]['points'][category] += int(count[at].sum())
                for name, mask in masks.items():
                    key = category+'/'+name
                    totals[seed]['queries'][key] += int((at & mask).sum())
                    totals[seed]['points'][key] += int(count[at & mask].sum())
        previous = ids
    result = dict(scope='Post hoc, same 321 development frames and frozen selected GLFS; semantically pure queried voxels only. GT identity is diagnostic, never an input.',
        invariants=dict(weighted_fusion_preserves_calibrated_consensus=True, uniform_average_preserves_raw_consensus=True),
        seeds=totals)
    save(root/'run/posthoc-attribution.json', result)
    for seed, record in enumerate(totals):
        c = record['queries']
        print(seed, {k:c[k] for k in ('all_queries/glfs_harm','all_queries/average_harm',
            'all_queries/history_harm_vs_same_parameters','all_raw_history_correct/glfs_harm',
            'all_same_instance_correct_history/glfs_harm','some_same_instance_correct_history/glfs_harm')}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    run(p.parse_args().root)
