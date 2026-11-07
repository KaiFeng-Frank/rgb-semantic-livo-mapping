"""Apply preregistered short-window decision rules; never select or tune an arm."""
import argparse
import json
import math
from pathlib import Path
import statistics

from frontend.protocol import load

METRICS = ('ate_rmse_m', 'rpe_1s_translation_rmse_m', 'rpe_1s_rotation_rmse_deg')


def assess(cfg, scores):
    if cfg.get('version') != 3 or cfg.get('diagnostic'):
        raise ValueError('requires the frozen enhanced accuracy protocol')
    expected = {(arm, rep) for arm in cfg['arms'] for rep in range(1, cfg['repetitions'] + 1)}
    rows = scores['runs']
    keyed = {(r['arm'], r['repetition']): r for r in rows}
    if set(keyed) != expected or len(rows) != len(expected):
        raise ValueError('missing or duplicate paired runs')
    if any(not math.isfinite(r[m]) or r[m] < 0 for r in rows for m in METRICS):
        raise ValueError('invalid metric')
    if len({r['poses'] for r in rows}) != 1 or min(r['poses'] for r in rows) < cfg['minimum_poses']:
        raise ValueError('unequal or insufficient pose coverage')
    summary = {}
    for arm in cfg['arms']:
        summary[arm] = {m: dict(mean=statistics.mean(keyed[arm, rep][m] for rep in range(1, cfg['repetitions'] + 1)),
                               minimum=min(keyed[arm, rep][m] for rep in range(1, cfg['repetitions'] + 1)),
                               maximum=max(keyed[arm, rep][m] for rep in range(1, cfg['repetitions'] + 1))) for m in METRICS}

    def comparison(candidate, control):
        differences = [keyed[candidate, rep]['ate_rmse_m'] - keyed[control, rep]['ate_rmse_m']
                       for rep in range(1, cfg['repetitions'] + 1)]
        checks = dict(all_paired_ate_lower=all(d < 0 for d in differences),
                      ate_ranges_nonoverlapping=summary[candidate][METRICS[0]]['maximum'] < summary[control][METRICS[0]]['minimum'],
                      mean_translation_rpe_no_worse=summary[candidate][METRICS[1]]['mean'] <= summary[control][METRICS[1]]['mean'],
                      mean_rotation_rpe_no_worse=summary[candidate][METRICS[2]]['mean'] <= summary[control][METRICS[2]]['mean'])
        return dict(candidate=candidate, control=control, paired_ate_differences_m=differences,
                    checks=checks, passed=all(checks.values()))

    mechanism = dict(boundary_rows_disabled=all(r['total_constraints'] == 0 for r in rows),
                     categorical_support=all(keyed[arm, rep]['semantic_updates'] > 0 and
                         keyed[arm, rep]['likelihood_updates'] / keyed[arm, rep]['semantic_updates'] >= .8
                         for arm in ['likelihood_B0', 'enhanced_B0'] for rep in range(1, cfg['repetitions'] + 1)))
    comparisons = [comparison(arm, 'geometry') for arm in cfg['arms'] if arm != 'geometry']
    increment = comparison('enhanced_B0', 'association_B0')
    primary = next(c for c in comparisons if c['candidate'] == 'enhanced_B0')
    return dict(status='passed' if primary['passed'] and all(mechanism.values()) else 'not_passed',
                scope='single frozen 240-frame validation window, cached labels, three repeats; no full-sequence or online claim',
                mechanism=mechanism, summary=summary, comparisons_to_geometry=comparisons,
                increment_over_association=increment, automatic_expansion=False,
                interpretation='A failed screen constrains this implementation on this window; it does not establish that semantic tight coupling is ineffective.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = assess(load(args.out / 'protocol.json'), json.loads((args.out / 'scores.json').read_text()))
    (args.out / 'assessment.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
