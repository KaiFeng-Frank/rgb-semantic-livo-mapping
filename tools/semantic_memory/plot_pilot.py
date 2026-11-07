"""Summarize frozen real-pilot results and draw an exportable comparison figure."""
import argparse
import json
from pathlib import Path
import numpy as np


def plot(run,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads((run/f'seed-{s}/evaluation.json').read_text()) for s in range(3)]
    summary=json.loads((run/'summary.json').read_text())
    arms=['baseline','uniform_history','current_only','temporal','ungated_temporal']
    labels=['Frozen B0','Uniform history','Current only','Gated memory','Ungated memory']
    colors=['#64748b','#0e7490','#2563eb','#dc2626','#f59e0b']
    classes=['barrier','bicycle','bus','car','construction','motorcycle','pedestrian','cone',
             'trailer','truck','road','other flat','sidewalk','terrain','manmade','vegetation']
    class_means={a:np.mean([[v*100 if v is not None else 0 for v in r['point_metrics'][a]['class_iou']] for r in rows],axis=0) for a in arms}
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    ax=axes[0,0]
    mean=[summary['arms'][a]['miou_percent_mean'] for a in arms]
    std=[summary['arms'][a]['miou_percent_std'] for a in arms]
    ax.bar(np.arange(5),mean,yerr=std,color=colors,capsize=3)
    ax.set_xticks(np.arange(5),labels,rotation=20,ha='right')
    ax.set_ylim(0,57); ax.set_ylabel('Native-16 point mIoU (%)')
    ax.set_title('A. Same frozen features, 3 training seeds')
    for i,v in enumerate(mean): ax.text(i,v+1.1,f'{v:.2f}',ha='center',fontsize=9)
    ax=axes[0,1]
    delta=class_means['temporal']-class_means['current_only']
    order=np.argsort(delta)
    ax.barh(np.arange(16),delta[order],color=['#dc2626' if d<0 else '#0e7490' for d in delta[order]])
    ax.set_yticks(np.arange(16),np.asarray(classes)[order],fontsize=8)
    ax.axvline(0,color='#475569',lw=.8); ax.set_xlabel('Gated memory - current only (IoU pp)')
    ax.set_title('B. Per-class effect of adding history')
    ax=axes[1,0]
    event_keys=['first_observed','reappearance_after_observation_gap','baseline_error_recovery']
    event_labels=['First observed\n179 episodes','Reappearance\n195 episodes','Baseline error recovery\n277 episodes']
    for j,arm in enumerate(['baseline','current_only','temporal']):
        v=[np.mean([r['events'][e][arm]['success_rate'] for r in rows])*100 for e in event_keys]
        ax.bar(np.arange(3)+(j-1)*.24,v,.24,label=labels[arms.index(arm)],color=colors[arms.index(arm)])
        for i,value in enumerate(v): ax.text(i+(j-1)*.24,value+1.3,f'{value:.1f}',ha='center',fontsize=8)
    ax.set_xticks(np.arange(3),event_labels,fontsize=8)
    ax.set_ylim(0,100);ax.set_ylabel('Success within observed episode (%)')
    ax.legend(frameon=False,fontsize=8,loc='upper right');ax.set_title('C. Shared GT-defined episodes; censored cases retained')
    ax=axes[1,1]
    measured=['uniform_history','current_only','temporal']
    v=[np.mean([r['module_timing'][a]['median_ms'] for r in rows]) for a in measured]
    ax.bar(np.arange(3),v,color=[colors[arms.index(a)] for a in measured])
    ax.axhline(5,ls='--',color='#334155',label='5 ms module budget')
    ax.set_xticks(np.arange(3),[labels[arms.index(a)] for a in measured])
    for i,value in enumerate(v): ax.text(i,value+.15,f'{value:.2f} ms',ha='center',fontsize=9)
    ax.set_ylim(0,10);ax.set_ylabel('Additional module median latency (ms)')
    ax.legend(frameon=False,fontsize=8);ax.set_title('D. Cached-feature timing; backbone / IPC excluded')
    fig.suptitle('Real-label pilot: memory has not beaten the matched-supervision control',fontsize=14)
    fig.supxlabel('6 held-out collection logs / 240 keyframes / 5.785M scored points. Raw rigid alignment; event sampling ~0.5 s.',fontsize=9)
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'comparison.png',dpi=180)
    fig.savefig(out/'comparison.svg')
    plt.close(fig)
    detail=dict(class_names=classes,class_iou_mean_percent={a:v.tolist() for a,v in class_means.items()},
        events={e:{a:dict(success_rate_percent=float(np.mean([r['events'][e][a]['success_rate'] for r in rows])*100),
                          successes_by_seed=[r['events'][e][a]['successes'] for r in rows],
                          episodes=rows[0]['events'][e][a]['episodes'],
                          mean_delay_with_censored_at_common_horizon_s=float(np.mean([r['events'][e][a]['mean_delay_with_censored_at_common_horizon_s'] for r in rows])))
                   for a in arms} for e in event_keys},
        accuracy_mean_percent={a:float(np.mean([r['point_metrics'][a]['accuracy'] for r in rows])*100) for a in arms},
        rescue_mean={a:float(np.mean([r['point_metrics'][a]['rescued'] for r in rows])) for a in arms},
        harm_mean={a:float(np.mean([r['point_metrics'][a]['harmed'] for r in rows])) for a in arms},
        labeled_points=rows[0]['point_metrics']['baseline']['labeled_points'])
    (out/'aggregate.json').write_text(json.dumps(detail,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plot(a.run,a.out)
