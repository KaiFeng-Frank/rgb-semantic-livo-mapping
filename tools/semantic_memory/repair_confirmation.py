"""Choose fresh collection logs without metrics, then score frozen repair weights."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .nuscenes_pilot_data import build_index, digest


def prepare(old_index, out):
    out.mkdir(parents=True, exist_ok=False)
    build_index(out/'selection', test_scenes=12)
    selected = json.loads((out/'selection'/'index.json').read_text())
    old = json.loads(old_index.read_text())
    used_logs = set(old['scene_logs'].values())
    fresh = [s for s in selected['parts']['test'] if selected['scene_logs'][s] not in used_logs]
    if len(fresh) != 6:
        raise ValueError('Expected six fresh logs after excluding all prior train/dev/test logs')
    selected['parts'] = dict(train=old['parts']['train'], dev=old['parts']['dev'], test=fresh)
    selected['sequences'] = {s:selected['sequences'][s] for s in fresh}
    selected['scene_logs'] = {**old['scene_logs'], **{s:selected['scene_logs'][s] for s in fresh}}
    selected['confirmation'] = dict(selected_utc=datetime.now(timezone.utc).isoformat(),
        prior_index_sha256=digest(old_index), excluded_logs=sorted(used_logs),
        role='fresh inference-only confirmation; previous train/dev names are provenance, not cached here',
        rule='next six log-hash-ranked official validation logs; no labels or metrics used to choose')
    (out/'index.json').write_text(json.dumps(selected,indent=2)+'\n')
    print(json.dumps(dict(fresh_scenes=fresh, frames=sum(len(v) for v in selected['sequences'].values()),
                          index_sha256=digest(out/'index.json'))),flush=True)


def score(index_path, cache, models, anchors, out):
    import torch
    from .repair_pilot import evaluate, summarize, save, announce, REPO
    index = json.loads(index_path.read_text())
    manifest = json.loads((cache/'manifest.json').read_text())
    protocol = json.loads((models/'protocol.json').read_text())
    if manifest['index_sha256'] != digest(index_path):
        raise ValueError('Cache index mismatch')
    if manifest['backbone_sha256'] != protocol['backbone_sha256']:
        raise ValueError('Different backbone')
    for name, expected in protocol['source_sha256'].items():
        if digest(REPO/name) != expected:
            raise ValueError('Frozen model source changed: '+name)
    for entry in manifest['files']:
        if digest(cache/entry['path']) != entry['sha256']:
            raise ValueError('Feature cache hash mismatch')
    logs = {index['scene_logs'][s] for s in index['parts']['test']}
    if logs & set(index['confirmation']['excluded_logs']):
        raise ValueError('Confirmation logs were previously inspected')
    out.mkdir(parents=True,exist_ok=False)
    frozen = dict(scope='inference_only_fresh_log_confirmation', index_sha256=digest(index_path),
                  training_protocol_sha256=digest(models/'protocol.json'), checkpoints={})
    for seed in protocol['seeds']:
        folder = out/f'seed-{seed}'
        folder.mkdir()
        for kind in ('temporal','current_extra'):
            path = models/f'seed-{seed}'/f'{kind}.pt'
            frozen['checkpoints'][f'{seed}/{kind}'] = digest(path)
            shutil.copyfile(path,folder/path.name)
    save(out/'protocol.json',frozen)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = [evaluate(seed,index,cache,out,anchors) for seed in protocol['seeds']]
    result = summarize(results,index)
    result['evaluation_role'] = 'fresh collection logs; inference-only, fixed weights and all three seeds'
    result['paired_scene_bootstrap_95_interval_pp'] = result.pop('regression_paired_scene_bootstrap_95_interval_pp')
    ci = result['paired_scene_bootstrap_95_interval_pp']
    event_keys = ('first_observed','reappearance_after_observation_gap','baseline_error_recovery')
    result['admission'] = dict(
        miou_superiority=ci[0]>0,
        events_noninferior=all(result['events'][e]['temporal'] >= result['events'][e]['current_extra'] for e in event_keys),
        five_ms=result['five_ms_module_budget_passed'])
    result['admission']['all_passed'] = all(result['admission'].values())
    save(out/'summary.json',result)
    announce(stage='fresh_confirmation_complete',**result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--old-index', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    run = sub.add_parser('score')
    for key in ('index','cache','models','anchors','out'):
        run.add_argument('--'+key,type=Path,required=True)
    args = vars(p.parse_args())
    if args.pop('command') == 'prepare':
        prepare(**args)
    else:
        args['index_path'] = args.pop('index')
        score(**args)
