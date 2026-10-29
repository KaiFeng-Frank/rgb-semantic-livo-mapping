"""Trace two saved failure cases, with frozen weights and no new model search."""
import argparse,json,sys,os
from pathlib import Path
import numpy as np
from tools.instance_memory.data import read,save
from tools.instance_memory.metrics import identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest,LABEL_MAP
from .feature_supervision_audit import voxel_truth
from .recovery_provenance_audit import provenance_masks
from .repair_supervision import current_token_input

CASES=(dict(scene='scene-0149',frame=17,instance=17030,seeds=[0]),
       dict(scene='scene-0218',frame=39,instance=17019,seeds=[1,2]))


def run(index,targets,models,anchor_path,saved_cases,out):
    from .repair_pilot import anchor_model,call
    from .nuscenes_pilot import current_output
    from .chain_trace import TracedEvidence,FusionCapture
    from tools.visual_instances.kalman_cache import gpu_admission
    import torch
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'src'))
    from semantic_repair import CurrentAnchoredRepair,RepairConfig,JointRepair
    from semantic_memory import MemoryConfig
    from ptv3_worker import Segmenter
    torch.set_num_threads(1);torch.manual_seed(20261029)
    idx=read(index);saved=read(saved_cases)
    for case in CASES:
        assert case['scene'] in idx['independent_dev_added']
        for seed in case['seeds']:
            assert any(c['seed']==seed and c['scene']==case['scene'] and c['instance']==case['instance'] and any(f['frame']==case['frame'] for f in c['decisive_frames']) for c in saved)
    admission=gpu_admission();out.mkdir(exist_ok=False)
    checkpoints=[torch.load(models/f'seed-{s}/balanced_temporal.pt',map_location='cpu',weights_only=True) for s in range(3)]
    cfg=MemoryConfig(**checkpoints[0]['config']);repair=RepairConfig(**checkpoints[0]['repair'])
    anchor=anchor_model(anchor_path);backbone=Segmenter(half=True,shuffle=False,intensity_scale=1.)
    save(out/'protocol.json',dict(cases=CASES,index_sha256=digest(index),saved_cases_sha256=digest(saved_cases),
        source_sha256=digest(Path(__file__)),anchor_sha256=digest(anchor_path),checkpoints_sha256={str(s):digest(models/f'seed-{s}/balanced_temporal.pt') for s in range(3)},
        backbone_sha256=digest(os.environ['PTV3_CHECKPOINT']),admission=admission,
        scope='Two archived object/frame failures only. Each uses its immediately previous raw scan, because PreviousEvidence replaces its entire contents at each write independently of prior predictions. Original scene origin and timestamps retained. No optimizer or output selection.',
        variants='Original and current-token input; GT-correct candidate subset; each candidate alone; native feature/probability/geometry chunks inserted individually into the current-token input. Diagnostic substitutions are not deployment policies.'))
    reports=[]
    with torch.no_grad():
        for case in CASES:
            scene,k,obj=case['scene'],case['frame'],case['instance'];rows=idx['sequences'][scene]
            ts=read(targets/(scene+'.targets.json.gz'))['frames'];origin=np.linalg.inv(np.asarray(rows[0]['transform']))
            native=CurrentAnchoredRepair(cfg,repair,device='cuda').eval();native.fusion.load_state_dict(checkpoints[case['seeds'][0]]['fusion'],strict=True)
            traced=CurrentAnchoredRepair(cfg,repair,device='cuda').eval();traced.fusion.load_state_dict(checkpoints[case['seeds'][0]]['fusion'],strict=True)
            traced.memory=TracedEvidence(cfg,'cuda',repair.moving_classes);cap=FusionCapture(traced.fusion)
            z={};handle=traced.fusion.net.register_forward_pre_hook(lambda m,args:z.update(value=args[0].detach().clone()))
            previous=None
            for frame in (k-1,k):
                row=rows[frame];p=DATA/row['scan_path'];lp=DATA/row['label_path']
                assert digest(p)==row['scan_sha256'] and digest(lp)==row['label_sha256']
                scan=np.fromfile(p,dtype='<f4').reshape(-1,5)[:,:4].copy();scan[:,3]/=255.
                r=backbone.segment(scan,return_features=True);raw={key:r[key] for key in ('xyz','features','logits')}
                raw.update(transform=torch.tensor(origin@np.asarray(row['transform']),device='cuda',dtype=torch.float32),stamp=row['timestamp_us']/1e6,frame=frame)
                anchor_logits=current_output(anchor,raw)['logits'];normal=call(native,raw,anchor_logits);o=call(traced,raw,anchor_logits)
                assert torch.equal(normal['logits'],o['logits']) and torch.equal(normal['query_indices'],o['query_indices'])
                for field in ('xyz','features','probabilities','key'):assert torch.equal(getattr(native.memory,field),getattr(traced.memory,field))
                gt=np.asarray(ts[frame]['gt'],np.int64);ids=identity_truth(ts[frame]);inv=r['inverse'].cpu().numpy()
                np.testing.assert_array_equal(gt,LABEL_MAP[np.fromfile(lp,dtype=np.uint8)])
                labels,vid=voxel_truth(inv,gt,ids,len(r['xyz']))
                if frame<k:previous=dict(labels=labels,vid=vid);continue
                f,qa,c=cap.value;joint=z['value'].clone();query=o['query_indices'].cpu().numpy()
                point_at=np.flatnonzero((ids==obj)&(gt>=0));assert len(point_at)>=5
                classes=np.unique(gt[point_at]);assert len(classes)==1;target=int(classes[0])
                selected=np.flatnonzero(np.isin(query,inv[point_at]));sel=torch.tensor(selected,device='cuda')
                lookup={int(v):i for i,v in enumerate(query[selected])}
                weights=np.array([np.count_nonzero(inv[point_at]==q) for q in query[selected]],np.int64)
                valid=c['valid'].cpu().numpy();hp=c['probabilities'].argmax(-1).cpu().numpy()
                source=traced.memory.trace['record_source'][traced.memory.trace['shortlist_index']].cpu().numpy()
                tags=provenance_masks(labels[query],vid[query],previous['labels'],previous['vid'],source,valid,hp)
                corrected=dict(c);corrected['valid']=torch.tensor(tags['correct'],device='cuda')
                traced.fusion(f,qa,corrected);correct_joint=z['value'].clone()
                current_joint=current_token_input(joint)
                variants=dict(native=joint,current=current_joint,correct_candidates=correct_joint)
                for name,sl in [('features_only',slice(64,192)),('probabilities_only',slice(208,224)),('geometry_only',slice(224,229))]:
                    value=current_joint.clone();value[:,sl]=joint[:,sl];variants[name]=value
                singletons=[]
                for j in range(c['valid'].shape[1]):
                    one=dict(c);one['valid']=torch.zeros_like(c['valid']);one['valid'][:,j]=c['valid'][:,j]
                    traced.fusion(f,qa,one);singletons.append(z['value'].clone())
                archive=dict(query=query[selected],point_indices=point_at,inverse=inv[point_at],weights=weights,
                    features=f[sel].cpu().numpy(),anchor=qa[sel].cpu().numpy(),joint=joint[sel].cpu().numpy(),current_joint=current_joint[sel].cpu().numpy(),correct_joint=correct_joint[sel].cpu().numpy(),
                    candidate_features=c['features'][sel].cpu().numpy(),candidate_probabilities=c['probabilities'][sel].cpu().numpy(),candidate_delta=c['delta'][sel].cpu().numpy(),
                    candidate_age=c['age'][sel].cpu().numpy(),candidate_valid=valid[selected],candidate_same=tags['same'][selected],candidate_correct=tags['correct'][selected],
                    source_voxel=source[selected],source_gt=previous['labels'][source[selected]],source_identity=previous['vid'][source[selected]],gt=np.array(target))
                per_seed=[]
                for seed in case['seeds']:
                    head=JointRepair(cfg,repair).cuda().eval();head.load_state_dict(checkpoints[seed]['fusion'],strict=True)
                    def logits(j):return qa[sel]+repair.residual_limit*torch.tanh(head.net(j[sel]))
                    output={name:logits(j) for name,j in variants.items()};single=torch.stack([logits(j) for j in singletons],1)
                    at=(ids==obj)&(gt>=0);outside=int(np.count_nonzero((anchor_logits.argmax(1).cpu().numpy()[inv]==gt)&at&~np.isin(inv,query)))
                    success={name:int(outside+np.sum(weights*(v.argmax(1).cpu().numpy()==target))) for name,v in output.items()}
                    original=next(fr for x in saved if x['seed']==seed and x['scene']==scene and x['instance']==obj for fr in x['decisive_frames'] if fr['frame']==k)
                    if success['native']/len(point_at)!=original['original_temporal'] or success['current']/len(point_at)!=original['original_current']:raise AssertionError('Saved failure did not reproduce exactly at the object level')
                    current_ok=output['current'].argmax(1).cpu().numpy()==target;correct_bad=output['correct_candidates'].argmax(1).cpu().numpy()!=target
                    good=tags['correct'][selected];one_ok=single.argmax(2).cpu().numpy()==target
                    nonempty=good.any(1);all_singles_fail=~(good&one_ok).any(1)
                    per_seed.append(dict(seed=seed,object_points=len(point_at),correct_points=success,
                        correct_pool_bad_with_good_singleton_points=int(np.sum(weights*(current_ok&correct_bad&nonempty&~all_singles_fail))),
                        correct_pool_bad_all_correct_singletons_fail_points=int(np.sum(weights*(current_ok&correct_bad&nonempty&all_singles_fail)))))
                    for name,value in output.items():archive[f's{seed}_{name}_logits']=value.cpu().numpy()
                    archive[f's{seed}_singleton_logits']=single.cpu().numpy()
                name=scene+'-'+str(obj)+'-'+str(k)
                for j,one in enumerate(singletons):archive[f'singleton_joint_{j}']=one[sel].cpu().numpy()
                np.savez_compressed(out/(name+'.npz'),**archive)
                report=dict(case=case,query_count=len(selected),queried_object_points=int(weights.sum()),object_points=len(point_at),target_class=target,
                    seeds=per_seed,trace_sha256=digest(out/(name+'.npz')))
                save(out/(name+'.json'),report);reports.append(report);print(report,flush=True)
            cap.close();handle.remove()
    save(out/'report.json',dict(cases=reports,scope='Frozen two-case mechanism trace; diagnostic substitutions and same-object reproduction only. Not a new dataset result or trained fix.')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('index','targets','models','anchor-path','saved-cases','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.index,a.targets,a.models,a.anchor_path,a.saved_cases,a.out)
