"""Compare archived policies without pooling repeated seeds as new samples."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def summarize(root):
    report = dict(unit='object-frame observations; seeds reuse the same scans and are not independent samples',
        policies={})
    for policy in ('legacy','all_off'):
        run = root/policy/'run'
        summary = json.loads((run/'summary.json').read_text())
        result = {}
        for part in ('train','dev','regression','confirmation'):
            results = [s['partitions'][part] for s in summary['seeds']]
            totals = [r['totals'] for r in results]
            result[part] = dict(
                counts={k:[r.get(k,0) for r in totals] for k in sorted(set().union(*(r.keys() for r in totals)))},
                funnel={k:[r['funnel'][k] for r in results] for k in results[0]['funnel']},
                mean_miou={k:float(np.mean([100*r['point_metrics'][k]['miou_fixed16'] for r in results]))
                           for k in results[0]['point_metrics']},
                seed_details=[])
            for seed in range(3):
                rows=[json.loads(line) for line in gzip.decompress((run/f'seed-{seed}'/part/'object-frames.jsonl.gz').read_bytes()).splitlines()]
                opportunities=[r for r in rows if r['correct_fraction']['no_history']<.5 and
                               r['previous_raw_correct_fraction'] is not None and r['previous_raw_correct_fraction']>=.5]
                instances=lambda rr:len({(r['scene'],r['instance']) for r in rr})
                by_class={str(label):dict(object_frames=sum(r['label']==label for r in opportunities),
                                         distinct_instances=instances([r for r in opportunities if r['label']==label]))
                          for label in sorted({r['label'] for r in opportunities})}
                mass=[r['correct_history_attention_mass_mean'] for r in opportunities
                      if r['correct_history_attention_mass_mean'] is not None]
                examples=[]
                # Fixed selection: first failing example per class with sufficient
                # correctly retrieved point coverage, in scene/frame/instance order.
                for label in sorted({r['label'] for r in opportunities}):
                    candidates=[r for r in opportunities if r['label']==label and
                                r['retrieved_good_error_points']>=r['points_needed'] and
                                r['correct_fraction']['temporal']<.5]
                    if candidates:
                        examples.append(sorted(candidates,key=lambda r:(r['scene'],r['frame'],r['instance']))[0])
                result[part]['seed_details'].append(dict(seed=seed,distinct_opportunity_instances=instances(opportunities),
                    by_class=by_class,examples=examples,
                    attention_mass_mean_over_opportunity_objects_with_positive_hits=float(np.mean(mass)) if mass else None))
        report['policies'][policy] = result
    (root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    summarize(p.parse_args().root)
