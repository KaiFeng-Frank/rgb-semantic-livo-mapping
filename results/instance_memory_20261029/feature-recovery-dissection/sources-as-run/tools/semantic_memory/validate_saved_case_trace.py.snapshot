"""Reconstruct saved case inputs and frozen-network responses independently on CPU."""
import argparse
from pathlib import Path
import numpy as np
from tools.instance_memory.data import read,save


def validate(traces,models):
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1);results=[]
    for case in read(traces/'report.json')['cases']:
        row=case['case'];name=row['scene']+'-'+str(row['instance'])+'-'+str(row['frame'])
        with np.load(traces/(name+'.npz'),allow_pickle=False) as f:d={k:f[k] for k in f.files}
        feature=torch.tensor(d['features']).float();history=torch.tensor(d['candidate_features']).float()
        probabilities=torch.tensor(d['candidate_probabilities']).float();a=torch.tensor(d['anchor']).float();p=a.softmax(-1)
        cosine=F.cosine_similarity(feature[:,None],history,dim=-1)
        def encoded(mask):
            valid=torch.tensor(mask);w=F.softmax((4*cosine).masked_fill(~valid,-1e4),-1)*valid
            w=w/w.sum(1,keepdim=True).clamp_min(1e-8);present=valid.any(1)
            h=torch.where(present[:,None],(history*w[:,:,None]).sum(1),feature)
            hp=torch.where(present[:,None],(probabilities*w[:,:,None]).sum(1),p)
            sim=torch.where(present[:,None],(cosine*w).sum(1,keepdim=True),torch.ones((len(feature),1)))
            delta=(torch.tensor(d['candidate_delta'])*w[:,:,None]).sum(1)/1.5
            age=(torch.tensor(d['candidate_age'])*w).sum(1,keepdim=True)/.75
            return torch.cat((feature,h,h-feature,p,hp-p,delta,age,sim),1)
        joint=torch.tensor(d['joint']);current=torch.tensor(d['current_joint']);correct=torch.tensor(d['correct_joint'])
        max_pool_error=0.
        for z,mask in [(joint,d['candidate_valid']),(current,np.zeros_like(d['candidate_valid'])),(correct,d['candidate_correct'])]:
            reconstructed=encoded(mask);torch.testing.assert_close(reconstructed,z,rtol=1e-5,atol=1e-5)
            max_pool_error=max(max_pool_error,float((reconstructed-z).abs().max()))
        variants=dict(native=joint,current=current,correct_candidates=correct)
        for name2,sl in [('features_only',slice(64,192)),('probabilities_only',slice(208,224)),('geometry_only',slice(224,229))]:
            z=current.clone();z[:,sl]=joint[:,sl];variants[name2]=z
        seed_reports=[]
        for seed in row['seeds']:
            ck=torch.load(models/f'seed-{seed}/balanced_temporal.pt',map_location='cpu',weights_only=True);state=ck['fusion'];max_output_error=0.
            def logits(z):return a+2*torch.tanh(F.linear(F.relu(F.linear(z,state['net.0.weight'],state['net.0.bias'])),state['net.2.weight'],state['net.2.bias']))
            margins={}
            for mode,z in variants.items():
                y=logits(z);actual=torch.tensor(d[f's{seed}_{mode}_logits']);torch.testing.assert_close(y,actual,rtol=1e-5,atol=1e-4)
                assert torch.equal(y.argmax(1),actual.argmax(1));max_output_error=max(max_output_error,float((y-actual).abs().max()))
                target=int(d['gt']);alt=actual.clone();alt[:,target]=-torch.inf;margin=actual[:,target]-alt.max(1).values
                margins[mode]=dict(min=float(margin.min()),median=float(margin.median()),max=float(margin.max()))
            for j in range(d['candidate_valid'].shape[1]):
                mask=np.zeros_like(d['candidate_valid']);mask[:,j]=d['candidate_valid'][:,j]
                z=torch.tensor(d[f'singleton_joint_{j}']);torch.testing.assert_close(encoded(mask),z,rtol=1e-5,atol=1e-5)
                actual=torch.tensor(d[f's{seed}_singleton_logits'])[:,j];y=logits(z)
                torch.testing.assert_close(y,actual,rtol=1e-5,atol=1e-4);assert torch.equal(y.argmax(1),actual.argmax(1))
                max_output_error=max(max_output_error,float((y-actual).abs().max()))
            w=state['net.0.weight'];dh=joint[:,64:128]-current[:,64:128]
            explicit=F.linear(dh,w[:,64:128]+w[:,128:192])
            direct=F.linear(variants['features_only'],w,state['net.0.bias'])-F.linear(current,w,state['net.0.bias'])
            torch.testing.assert_close(explicit,direct,rtol=1e-5,atol=1e-4)
            seed_reports.append(dict(seed=seed,max_cpu_gpu_logit_error=max_output_error,
                max_feature_effect_formula_error=float((explicit-direct).abs().max()),target_margin=margins))
        results.append(dict(case=row,pooled_input_max_error=max_pool_error,correct_candidates_per_query=d['candidate_correct'].sum(1).tolist(),seeds=seed_reports))
    report=dict(passed=True,cases=results,scope='CPU reconstruction of saved candidate pooling, singleton interventions, logits and feature-channel preactivation identity. No new data or trained model.')
    save(traces/'validation.json',report);print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--traces',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    a=p.parse_args();validate(a.traces,a.models)
