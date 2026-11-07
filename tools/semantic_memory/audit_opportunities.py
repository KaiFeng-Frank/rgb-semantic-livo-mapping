"""Post-hoc object-level temporal complementarity from archived predictions.

No training or feature-cache reconstruction. A previous correctly classified
GT instance is evidence of observation-level opportunity, NOT proof that the
deployed retrieval selected its features, nor an oracle bound on joint inference.
Repeated object-frames and seeds must not be counted as independent objects.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


FIELDS = ('object_observations', 'current_wrong', 'same_object_previous_frame',
          'wrong_with_previous_observation', 'previous_correct_current_wrong',
          'both_wrong', 'previous_wrong_current_correct', 'opportunity_fixed',
          'risk_harmed')


def audit(rows, reference='no_history', max_age=.75):
    index = {(r['scene'], r['instance'], r['frame']): r for r in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate object/frame observations')
    total = Counter()
    by_scene, by_class = defaultdict(Counter), defaultdict(Counter)
    for r in rows:
        prev = index.get((r['scene'], r['instance'], r['frame']-1))
        eligible = prev is not None and 0 < r['stamp'] - prev['stamp'] <= max_age
        if eligible and prev['label'] != r['label']:
            raise ValueError('Instance class changed')
        wrong = r['correct_fraction'][reference] < .5
        historical_correct = eligible and prev['correct_fraction']['baseline'] >= .5
        opportunity = wrong and historical_correct
        risk = not wrong and eligible and not historical_correct
        temporal_correct = r['correct_fraction']['temporal'] >= .5
        counts = dict(object_observations=1, current_wrong=wrong,
            same_object_previous_frame=eligible,
            wrong_with_previous_observation=wrong and eligible,
            previous_correct_current_wrong=opportunity,
            both_wrong=wrong and eligible and not historical_correct,
            previous_wrong_current_correct=risk,
            opportunity_fixed=opportunity and temporal_correct,
            risk_harmed=risk and not temporal_correct)
        for counter in (total, by_scene[r['scene']], by_class[str(r['label'])]):
            counter.update({k:int(v) for k,v in counts.items()})
    def report(c):
        values = {k:c[k] for k in FIELDS}
        n = c['previous_correct_current_wrong']
        values['opportunity_fraction_of_current_errors'] = n/c['current_wrong'] if c['current_wrong'] else None
        values['opportunity_fixed_fraction'] = c['opportunity_fixed']/n if n else None
        return values
    return dict(totals=report(total), by_scene={k:report(v) for k,v in sorted(by_scene.items())},
                by_class={k:report(v) for k,v in sorted(by_class.items(), key=lambda kv:int(kv[0]))})


def run(root, out):
    result = dict(scope='post-hoc diagnostic of archived object predictions; no optimization or scene reselection',
        definition=dict(unit='GT instance observation at one frame, not a unique object or independent trial',
            current='same temporal weights with history disabled (primary); separately trained matched current (secondary)',
            history='raw B0 prediction on same GT instance at frame t-1, 0 < age <= 0.75 s',
            recognition='at least 50% correct among at least 5 uniquely assigned GT semantic points',
            opportunity='current recognition fails and previous recognition succeeds',
            repair='temporal recognition succeeds at the same current frame'),
        limits=['Does not measure whether the relevant historical feature survived storage or was retrieved',
                'Does not bound feature-level joint inference when both standalone predictions are wrong',
                'GT identity used for analysis only; model has no instance tracker',
                'Object-frames and seeds are correlated; no independence or significance claim'],
        input_sha256={}, partitions={})
    for partition in ('run', 'confirmation'):
        seeds = []
        for seed in range(3):
            path = root/partition/f'seed-{seed}'/'object-observations.json'
            with path.open('rb') as stream:
                result['input_sha256'][str(path.relative_to(root))] = hashlib.file_digest(stream, 'sha256').hexdigest()
            rows = json.loads(path.read_text())
            seeds.append(dict(seed=seed, references={ref:audit(rows, ref) for ref in ('no_history','current_extra')}))
        result['partitions'][partition] = seeds
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = run(args.root, args.out)
    print(json.dumps({part:[dict(seed=r['seed'], **r['references']['no_history']['totals'])
                            for r in rows] for part,rows in result['partitions'].items()}))
