"""Summarize saved same-forward semantic and identity evidence, without inference."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np


def rows(path):
    opener = open
    if not path.exists():
        path = path.with_suffix(path.suffix+'.gz'); opener = gzip.open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def run(root):
    summary=json.loads((root/'semantic-summary.json').read_text())
    frames={(s['scene'],r['frame']):r for s in summary['scenes']
            for r in rows(root/f"semantic-{s['scene']}-frames.jsonl")}
    identity=Counter(); wrong=[]; versus_single=Counter(); by_gt=Counter()
    for pair in rows(root/'identity-pairs.jsonl'):
        record=frames[pair['scene'],pair['frame']]
        match=next((d for d in record['guarded']['instances'] if d['track_id']==pair['track_id']),None)
        reason=match['identity'] if match else 'no_credible_component'
        relation='same_gt_majority' if pair['same_gt_object'] else 'changed_gt_majority'
        identity[relation+':'+reason]+=1
        if not pair['same_gt_object']:
            wrong.append(dict(scene=pair['scene'],frame=pair['frame'],track_id=pair['track_id'],
                reason=reason,unified=bool(match and match['unified']),tied_vote=pair['ambiguous_vote']))
    times=[]; peaks=Counter()
    for frame in frames.values():
        times.append(frame['guarded']['elapsed_ms'])
        for key in ('tracks','history_observations'):
            peaks[key]=max(peaks[key],frame['guarded'][key])
        for e in frame['evidence']:
            gt=np.asarray(e['gt']);raw=np.asarray(e['raw_labels'])
            current=np.asarray(e['predictions']['single_frame_instance'])
            final=np.asarray(e['predictions']['guarded'])
            versus_single.update(changed=int((current!=final).sum()),
                rescued=int(((gt>=0)&(current!=gt)&(final==gt)).sum()),
                harmed=int(((gt>=0)&(current==gt)&(final!=gt)).sum()))
            by_gt.update(map(int,gt[(gt>=0)&(raw!=gt)&(final==gt)]))
    return dict(scope='Development diagnostic; GT-majority association proxy, not official identity accuracy.',
        arms={k:dict(miou_percent=v['miou_fixed16']*100,
                     **{n:v['counts'].get(n,0) for n in ('changed','rescued','harmed','unified_points')})
              for k,v in summary['aggregate'].items()},
        guarded_vs_single_frame=dict(versus_single),corrected_points_by_gt_class=dict(by_gt),
        identity_pair_counts=dict(identity),changed_gt_majority_cases=wrong,
        observed_peak_state=dict(peaks),
        processing_ms=dict(median=float(np.median(times)),p95=float(np.percentile(times,95)),max=max(times)),
        timing_scope='Offline diagnostic only; not an end-to-end online timing claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(run(p.parse_args().root),indent=2))
