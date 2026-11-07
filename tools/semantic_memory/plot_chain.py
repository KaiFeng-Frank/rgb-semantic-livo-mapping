"""Export frozen evidence-chain diagnostics; privileged arms are visibly labeled."""
import argparse
import json
from pathlib import Path

import numpy as np


def plot(run,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((run/'summary.json').read_text())
    seeds=data['seeds']
    parts=['train','dev','regression','confirmation']
    labels=['Train\n239 frames','Dev\n81 frames','Prior evaluation\n240 frames','New-log evaluation\n240 frames']
    results={p:[s['partitions'][p] for s in seeds] for p in parts}
    def percent(p,key):
        return np.array([100*r['totals'].get(key,0)/r['totals']['opportunities'] for r in results[p]])
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    ax=axes[0,0]
    for j,(key,name,color) in enumerate([('anchor_opportunities','Frozen current anchor','#64748b'),
                                        ('opportunities','Final weights, history off','#2563eb')]):
        v=[np.mean([r['totals'][key] for r in results[p]]) for p in parts]
        ax.bar(np.arange(4)+(j-.5)*.32,v,.32,label=name,color=color)
        for i,n in enumerate(v):ax.text(i+(j-.5)*.32,n+3,f'{n:.0f}',ha='center',fontsize=8)
    ax.set_xticks(range(4),labels,fontsize=8);ax.set_ylabel('Opportunity object-frames, mean over seeds')
    ax.set_title('A. Does complementary evidence exist?');ax.legend(frameon=False,fontsize=8)
    ax=axes[0,1]
    stages=['raw_identity','stored','readable','queried','local','shortlist','retrieved','repaired']
    names=['Raw same object','Stored','Readable','Error point queried','Within radius','In top-4 shortlist','Passes cosine','Object repaired']
    values=[np.mean([100*r['funnel'][k]/r['totals']['opportunities'] for r in results['confirmation']]) for k in stages]
    ax.barh(range(8),values,color=['#64748b']*3+['#2563eb']*3+['#0f766e']*2)
    ax.set_yticks(range(8),names,fontsize=9);ax.invert_yaxis();ax.set_xlim(0,113)
    for i,n in enumerate(values):ax.text(n+1,i,f'{n:.1f}%',va='center',fontsize=8)
    ax.set_xlabel('Cumulative ANY-hit fraction of opportunities')
    ax.set_title('B. New-log set: where evidence is lost')
    ax=axes[1,0]
    arms=[('temporal_repairs','Normal model','#2563eb'),
          ('clean_existing_gt_repairs','GT filter: existing tokens','#d97706'),
          ('supply_raw_gt_repairs','GT supply: raw tokens','#9333ea')]
    for j,(key,name,color) in enumerate(arms):
        v=[percent(p,key).mean() for p in parts]
        ax.bar(np.arange(4)+(j-1)*.24,v,.24,label=name,color=color)
        for i,n in enumerate(v):ax.text(i+(j-1)*.24,n+.8,f'{n:.1f}',ha='center',fontsize=8)
    ax.set_xticks(range(4),labels,fontsize=8);ax.set_ylim(0,100)
    ax.set_ylabel('Opportunity object-frames actually repaired (%)')
    ax.set_title('C. Fixed head; orange/purple use privileged GT')
    ax.legend(frameon=False,fontsize=8,loc='upper right')
    ax=axes[1,1]
    keys=['confidence_error_points','queried_error_points','bounded_reachable_error_points',
          'retrieved_good_error_points','supplied_good_error_points']
    names=['Confidence\nallows enough','Query budget\nallows enough','Bounded logits\nallow enough','Correct retrieved\ntokens enough','GT-supplied\ntokens enough']
    v=[percent('confirmation',k+'_enough_for_recognition').mean() for k in keys]
    ax.bar(range(5),v,color=['#64748b','#2563eb','#dc2626','#0f766e','#9333ea'])
    ax.set_xticks(range(5),names,fontsize=8);ax.set_ylim(0,110)
    for i,n in enumerate(v):ax.text(i,n+1.5,f'{n:.1f}%',ha='center',fontsize=9)
    ax.set_ylabel('Opportunity observations with sufficient point coverage (%)')
    ax.set_title('D. Separate constraints; ANY hit is insufficient')
    fig.suptitle('Frozen semantic-repair audit: data, storage, retrieval and correction',fontsize=14)
    fig.supxlabel('3 seeds / 800 frames each / unchanged weights. GT-guided diagnostics are not deployable methods or general fusion upper bounds.',fontsize=9)
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'diagnostic.png',dpi=180)
    fig.savefig(out/'diagnostic.svg')
    svg=out/'diagnostic.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    compact={p:dict(
        opportunities=[r['totals']['opportunities'] for r in results[p]],
        anchor_opportunities=[r['totals']['anchor_opportunities'] for r in results[p]],
        funnel={k:[r['funnel'][k] for r in results[p]] for k in stages},
        totals={k:[r['totals'].get(k,0) for r in results[p]] for k in sorted(set().union(*(r['totals'] for r in results[p])))},
        per_class={k:[r['by_class'].get(k) for r in results[p]] for k in sorted(set().union(*(r['by_class'] for r in results[p])),key=int)}) for p in parts}
    (out/'aggregate.json').write_text(json.dumps(compact,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plot(a.run,a.out)
