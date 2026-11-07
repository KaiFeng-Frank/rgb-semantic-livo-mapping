"""Export repair ablations, recovery and inclusive module timing from saved results."""
import argparse
import json
from pathlib import Path

import numpy as np


def plot(run, old_run, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads((run/f'seed-{s}'/'evaluation.json').read_text()) for s in range(3)]
    old = [json.loads((old_run/f'seed-{s}'/'evaluation.json').read_text()) for s in range(3)]
    summary = json.loads((run/'summary.json').read_text())
    fresh = 'fresh collection logs' in summary['evaluation_role']
    arms = ['anchor','current_extra','temporal','shuffled_history','narrow_history']
    labels = ['Current anchor','Matched current\nextra training','Joint temporal\nrepair','Wrong past\ncontent','Narrow history\n0.5 m']
    colors = ['#64748b','#2563eb','#0f766e','#d97706','#9333ea']
    fig, axes = plt.subplots(2,2,figsize=(12,9),layout='constrained')
    ax = axes[0,0]
    values = [summary['arms'][k]['miou_percent_mean'] for k in arms]
    ax.bar(range(len(arms)), values, yerr=[summary['arms'][k]['miou_percent_std'] for k in arms], color=colors, capsize=3)
    ax.set_xticks(range(len(arms)),labels,fontsize=8)
    ax.set_ylim(0,60); ax.set_ylabel('Native-16 point mIoU (%)')
    ax.set_title('A. Same labels, backbone and 3 training seeds')
    for i,v in enumerate(values):
        ax.text(i,v+1.,f'{v:.2f}',ha='center',fontsize=9)
    ax = axes[0,1]
    classes = ['barrier','bicycle','bus','car','construction','motorcycle','pedestrian','cone',
               'trailer','truck','road','other flat','sidewalk','terrain','manmade','vegetation']
    ci = {k:np.mean([[0 if v is None else v*100 for v in r['point_metrics'][k]['class_iou']] for r in rows],axis=0) for k in arms}
    delta = ci['temporal']-ci['current_extra']
    order = np.argsort(delta)
    ax.barh(range(16), delta[order], color=['#dc2626' if d<0 else '#0f766e' for d in delta[order]])
    ax.set_yticks(range(16),np.array(classes)[order],fontsize=8)
    ax.axvline(0,color='#475569',lw=.8)
    ax.set_xlabel('Temporal - matched current (IoU pp)')
    ax.set_title('B. Gains and losses by class')
    ax = axes[1,0]
    event_names = ['first_observed','reappearance_after_observation_gap','baseline_error_recovery']
    for j,k in enumerate(arms[:3]):
        v = [100*summary['events'][e][k] for e in event_names]
        ax.bar(np.arange(3)+(j-1)*.24,v,.24,label=labels[j].replace('\n',' '),color=colors[j])
        for i,n in enumerate(v):
            ax.text(i+(j-1)*.24,n+1.,f'{n:.1f}',ha='center',fontsize=8)
    ax.set_xticks(range(3),['First observed','Reappearance','Error recovery'],fontsize=9)
    ax.set_ylim(0,100); ax.set_ylabel('Episode success (%)')
    ax.legend(frameon=False,fontsize=8)
    ax.set_title('C. Fixed episodes; failures stay in denominator')
    ax = axes[1,1]
    times = [np.mean([r['module_timing']['anchor']['median_ms'] for r in rows]) if fresh else
             np.mean([r['module_timing']['temporal']['median_ms'] for r in old]),
             np.mean([r['module_timing']['current_extra_including_anchor']['median_ms'] for r in rows]),
             np.mean(summary['total_module_median_ms_by_seed'])]
    ax.bar(range(3),times,color=['#dc2626','#2563eb','#0f766e'])
    ax.set_xticks(range(3),['Current anchor' if fresh else 'Previous memory','Matched current','Revised memory'])
    ax.axhline(5.,color='#475569',ls='--',label='5 ms budget')
    for i,v in enumerate(times):
        ax.text(i,v+.15,f'{v:.2f} ms',ha='center')
    ax.set_ylim(0,max(10,max(times)+2)); ax.set_ylabel('Module median (ms)')
    ax.set_title('D. New totals INCLUDE the current anchor')
    ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Current-anchored semantic repair: '+('fresh-log confirmation' if fresh else 'regression and ablation results'),fontsize=14)
    fig.supxlabel(('Fresh 6 scenes / 240 frames. Fixed weights; inference only.' if fresh else
                  'Previously inspected 6 scenes / 240 frames. Not a fresh holdout.')+
                 ' Raw rigid alignment; backbone and transport excluded.',fontsize=9)
    out.mkdir(exist_ok=True,parents=True)
    fig.savefig(out/'comparison.png',dpi=180)
    fig.savefig(out/'comparison.svg')
    svg = out/'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    aggregate = dict(class_names=classes, class_iou_mean_percent={k:v.tolist() for k,v in ci.items()},
        summary=summary, reference_median_ms=times[0],
        paired_anchor_rescue_harm={k:{m:float(np.mean([r['point_metrics'][k]['relative_to_anchor'][m] for r in rows])) for m in ('rescued','harmed')} for k in arms})
    (out/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('run','old-run','out'):
        p.add_argument('--'+k,type=Path,required=True)
    args = p.parse_args()
    plot(args.run,args.old_run,args.out)
