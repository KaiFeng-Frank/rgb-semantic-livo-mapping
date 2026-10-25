"""Post-hoc gate diagnosis of frozen seed-0 weights; no training or retuning."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .nuscenes_pilot import CausalSemanticMemory, MemoryConfig, load, forward, save
from .nuscenes_pilot_data import digest


@torch.no_grad()
def run(index_path,cache,checkpoint_path,out):
    index=json.loads(index_path.read_text())
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    cfg=MemoryConfig(**saved['config'])
    model=CausalSemanticMemory(cfg,'cuda').eval()
    model.fusion.load_state_dict(saved['fusion'],strict=True)
    fields=('points','eligible','accepted','candidate_repairs','candidate_harms',
            'applied_repairs','applied_harms','blocked_repairs','blocked_harms')
    counts={k:np.zeros(16,np.int64) for k in fields}
    selection=np.zeros((2,2),np.int64)
    for scene in index['parts']['test']:
        model.reset()
        for row in index['sequences'][scene]:
            raw,arrays=load(cache,scene,row)
            result=forward(model,raw)
            inv=arrays['inverse']
            y=arrays['point_label'].astype(np.int64)
            b=raw['logits'].argmax(-1).cpu().numpy()[inv]
            c=result['candidate_logits'].argmax(-1).cpu().numpy()[inv]
            f=result['logits'].argmax(-1).cpu().numpy()[inv]
            eligible=result['has_history'].cpu().numpy()[inv]
            accepted=(result['gate']>=cfg.gate_threshold).cpu().numpy()[inv]&eligible
            repair=(b!=y)&(c==y); harm=(b==y)&(c!=y)
            masks=dict(points=np.ones(len(y),bool),eligible=eligible,accepted=accepted,
                candidate_repairs=repair,candidate_harms=harm,
                applied_repairs=(b!=y)&(f==y),applied_harms=(b==y)&(f!=y),
                blocked_repairs=repair&~accepted,blocked_harms=harm&~accepted)
            for field,mask in masks.items():
                counts[field]+=np.bincount(y[mask&(y>=0)],minlength=16)
            valid=(raw['label']>=0)&result['has_history']
            target=raw['label'][valid].long()
            base=F.cross_entropy(result['base_logits'][valid],target,reduction='none')
            candidate=F.cross_entropy(result['candidate_logits'][valid],target,reduction='none')
            truth=(base-candidate>.02).cpu().numpy().astype(int)
            decision=(result['gate'][valid]>=cfg.gate_threshold).cpu().numpy().astype(int)
            np.add.at(selection,(truth,decision),1)
    result=dict(scope='post-hoc diagnostic; frozen seed 0, no new optimization or threshold selection',
        checkpoint_sha256=digest(checkpoint_path),per_class={k:v.tolist() for k,v in counts.items()},
        totals={k:int(v.sum()) for k,v in counts.items()},
        utility_selector_confusion=selection.tolist(),
        utility_definition='candidate voxel CE improves raw CE by >0.02; rows truth false/true, columns reject/accept')
    out.parent.mkdir(parents=True,exist_ok=True)
    save(out,result)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','cache','checkpoint','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    run(a.index,a.cache,a.checkpoint,a.out)
