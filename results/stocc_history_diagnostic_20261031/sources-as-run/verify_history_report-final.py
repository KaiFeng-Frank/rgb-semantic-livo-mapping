"""Check exported counts, paired denominators and reported metrics on CPU."""
from collections import Counter
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/wuyou/stocc_reference/history-diagnostic')
ARMS=('cold_history','native_history')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    root=args.root
    report=json.loads((root/'report.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    frames=json.loads((root/'frame-point-effects.json').read_text())
    if (root/'events.json').exists():
        episodes=json.loads((root/'events.json').read_text())
    else:
        with gzip.open(root/'events.json.gz','rt') as stream:episodes=json.load(stream)
    with gzip.open(root/'object-frames.json.gz','rt') as stream:objects=json.load(stream)
    assert len(frames)==protocol['frame_count']==report['frames']==642
    assert {r['token'] for r in frames}=={r['token'] for r in protocol['frames']}
    for key in ['points','corrected','harmed']:
        assert sum(r[key] for r in frames)==report['paired'][key]
        assert sum(r[key] for r in report['per_class_paired'].values())==report['paired'][key]
    traces={}
    for arm in ARMS:
        m=np.array(report['point_metrics'][arm]['confusion'],np.int64)
        assert m.sum()==report['paired']['points']
        parts=sum(np.array(v[arm]['confusion'],np.int64) for v in report['per_log_point_metrics'].values())
        assert np.array_equal(m,parts)
        tp=np.diag(m);union=m.sum(0)+m.sum(1)-tp;present=union[1:17]>0
        expected=100*(tp[1:17][present]/union[1:17][present]).mean()
        assert abs(expected-report['point_metrics'][arm]['mIoU16_present_percent'])<1e-10
        fixed=100*np.divide(tp[1:17],union[1:17],out=np.zeros(16),where=present).mean()
        assert abs(fixed-report['point_metrics'][arm]['mIoU16_zero_for_absent_percent'])<1e-10
        traces[arm]=int(tp.sum())
    assert report['log_comparison_metric']=='mIoU16_zero_for_absent_percent'
    assert report['positive_log_point_mIoU_count']==sum(
        row['native_history'][report['log_comparison_metric']]>row['cold_history'][report['log_comparison_metric']]
        for row in report['per_log_point_metrics'].values())
    assert traces['native_history']-traces['cold_history']==report['paired']['corrected']-report['paired']['harmed']
    for name,subset,summary in [('all',episodes,report['events'])]+[
            (name,[e for e in episodes if e['label']==label],report['subgroups'][name]['events'])
            for name,label in [('car',4),('pedestrian',7)]]:
        for kind,table in summary.items():
            events=[e for e in subset if e['kind']==kind]
            assert table['episodes']==len(events)
            counts=Counter((e['outcomes']['cold_history']['success'],e['outcomes']['native_history']['success']) for e in events)
            assert table['gained']==counts[False,True] and table['lost']==counts[True,False]
            for arm in ARMS:
                outcomes=[e['outcomes'][arm] for e in events]
                successes=sum(v['success'] for v in outcomes)
                assert successes==table['arms'][arm]['successes']
                assert successes+table['arms'][arm]['not_observed_recovered']==len(events)
                for outcome in outcomes:
                    assert 0<=outcome['delay_capped_s']<=2+1e-6
    assert len(objects)==report['object_semantic_stability']['object_frames']
    result=dict(passed=True,frames=len(frames),object_frames=len(objects),events=len(episodes),
                checks=['query coverage','confusion matrices and mIoU','point correction/harm accounting',
                        'log aggregation','event and subgroup paired denominators','fixed recovery horizon'],
                report_sha256=hashlib.sha256((root/'report.json').read_bytes()).hexdigest())
    (root/'report-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
