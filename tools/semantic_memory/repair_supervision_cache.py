"""Exact immutable inputs for a frozen-backbone, fixed-retrieval repair experiment."""
import argparse
from pathlib import Path
import os,shutil,sys,time
import numpy as np
from tools.instance_memory.data import read,save
from tools.instance_memory.metrics import identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest,LABEL_MAP
from .feature_supervision_audit import voxel_truth


def extract(train_index,dev_index,train_targets,dev_targets,visual,anchors,models,cache,out):
    from .repair_pilot import anchor_model,call
    from .nuscenes_pilot import current_output
    from .chain_trace import FusionCapture
    from .repair_supervision import current_token_input,supervision_groups,present_class_weights
    from tools.visual_instances.kalman_cache import gpu_admission
    from tools.visual_instances.visual_semantics import fusion
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from semantic_repair import CurrentAnchoredRepair,RepairConfig
    from semantic_memory import MemoryConfig
    from ptv3_worker import Segmenter
    from ptv3_loader_verified import serialization_policy
    torch.set_num_threads(1);torch.manual_seed(20261029)
    ti,di=read(train_index),read(dev_index)
    parts=dict(train=ti['parts']['train'],dev=di['independent_dev_added'])
    if len(parts['train'])!=16 or len(parts['dev'])!=8:raise ValueError('Fixed split changed')
    logs={part:{(ti if part=='train' else di)['scene_logs'][s] for s in names} for part,names in parts.items()}
    if logs['train']&logs['dev'] or (logs['train']|logs['dev'])&{di['scene_logs'][s] for s in di['parts']['test']}:raise ValueError('Collection-log leakage')
    for folder in (train_targets,dev_targets,visual):
        if any((folder/(s+'.'+kind+'.json.gz')).exists() for s in di['parts']['test'] for kind in ('inputs','targets')):raise ValueError('New holdout opened')
    if cache.exists() or out.exists():raise ValueError('Extraction outputs must be new, never overwrite a running experiment')
    if shutil.disk_usage(out.parent).free<4*1024**3:raise RuntimeError('Low persistent disk space')
    admission=gpu_admission();cache.mkdir();out.mkdir()
    root=Path(__file__).resolve().parents[2];cfg=root/'config/feature_repair_supervision_20261029.json'
    ap=anchors/'seed-0/current_only.pt';mp=models/'seed-0/temporal.pt'
    ck=torch.load(mp,map_location='cpu',weights_only=True);anchor=anchor_model(ap)
    def make():
        m=CurrentAnchoredRepair(MemoryConfig(**ck['config']),RepairConfig(**ck['repair']),device='cuda').eval()
        m.fusion.load_state_dict(ck['fusion'],strict=True);return m
    temporal,current=make(),make();capture=FusionCapture(temporal.fusion)
    zt,zc={},{}
    ht=temporal.fusion.net.register_forward_pre_hook(lambda m,a:zt.update(value=a[0].detach()))
    hc=current.fusion.net.register_forward_pre_hook(lambda m,a:zc.update(value=a[0].detach()))
    backbone=Segmenter(half=True,shuffle=False,intensity_scale=1.)
    protocol=dict(config=read(cfg),config_sha256=digest(cfg),parts=parts,
        train_index=str(train_index),dev_index=str(dev_index),index_sha256={str(p):digest(p) for p in (train_index,dev_index)},
        target_paths=dict(train=str(train_targets),dev=str(dev_targets)),target_sha256={s:digest(folder/(s+'.targets.json.gz')) for part,folder in [('train',train_targets),('dev',dev_targets)] for s in parts[part]},
        visual_path=str(visual),visual_sha256={s:digest(visual/(s+'.inputs.json.gz')) for s in parts['dev']},
        anchor_path=str(ap.resolve()),anchor_sha256=digest(ap),repair_reference_sha256=digest(mp),
        backbone_sha256=digest(os.environ['PTV3_CHECKPOINT']),serialization=serialization_policy(backbone.model),
        repair_config=ck['repair'],memory_config=ck['config'],cache_path=str(cache),admission=admission,
        sources={str(p.relative_to(root)):digest(p) for p in [Path(__file__),root/'tools/semantic_memory/repair_supervision.py',root/'tools/semantic_memory/repair_pilot.py',root/'src/semantic_repair.py',root/'src/semantic_memory.py',root/'src/ptv3_worker.py',root/'src/ptv3_loader_verified.py']})
    save(out/'protocol.json',protocol)
    manifest=dict(protocol_sha256=digest(out/'protocol.json'),files=[],total_bytes=0,parity_frames=0,real_output_checks=0)
    voxel_counts=np.zeros(16,np.int64);groups_count=np.zeros(3,np.int64);start=time.perf_counter()
    with torch.no_grad():
        for part,names in parts.items():
            idx=ti if part=='train' else di;folder=train_targets if part=='train' else dev_targets
            for scene in names:
                temporal.reset();current.reset();rows=idx['sequences'][scene];targets=read(folder/(scene+'.targets.json.gz'))['frames']
                observations=read(visual/(scene+'.inputs.json.gz'))['frames'] if part=='dev' else None
                if len(rows)!=len(targets):raise ValueError('Incomplete target frames')
                (cache/scene).mkdir();origin=np.linalg.inv(np.asarray(rows[0]['transform']))
                for k,(row,target) in enumerate(zip(rows,targets)):
                    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
                    if available<6*1024**3 or shutil.disk_usage(cache).free<1024**3:raise RuntimeError('Memory headroom below frozen cache limits')
                    path=DATA/row['scan_path'];lp=DATA/row['label_path']
                    if digest(path)!=row['scan_sha256'] or digest(lp)!=row['label_sha256']:raise ValueError('Source changed')
                    scan=np.fromfile(path,dtype='<f4').reshape(-1,5)[:,:4].copy();scan[:,3]/=255.
                    r=backbone.segment(scan,return_features=True)
                    raw={key:r[key] for key in ('xyz','features','logits')}
                    raw.update(transform=torch.tensor(origin@np.asarray(row['transform']),device='cuda',dtype=torch.float32),stamp=row['timestamp_us']/1e6,frame=k)
                    a=current_output(anchor,raw)['logits'];o=call(temporal,raw,a);co=call(current,raw,a,current_only=True)
                    joint=zt['value'];derived=current_token_input(joint,temporal.cfg.feature_dim,temporal.cfg.classes)
                    if not torch.equal(derived,zc['value']) or not torch.equal(o['query_indices'],co['query_indices']):raise AssertionError('Current-token cache encoding differs from native path')
                    manifest['parity_frames']+=1
                    query=o['query_indices'];qa=a[query];inverse=r['inverse'].cpu().numpy().astype(np.int32)
                    z=joint.cpu().numpy().copy();anc=qa.cpu().numpy().copy()
                    if k in (0,len(rows)//2,len(rows)-1):
                        rt=torch.from_numpy(z).cuda();ra=torch.from_numpy(anc).cuda()
                        for inp,reference in [(rt,o['selected_logits']),(current_token_input(rt),co['selected_logits'])]:
                            pred=ra+temporal.repair.residual_limit*torch.tanh(temporal.fusion.net(inp))
                            if not torch.equal(pred,reference):raise AssertionError('Float32 cache roundtrip changed real network output')
                        manifest['real_output_checks']+=2
                    # GT is read after all encodings and native output assertions.
                    gt=np.asarray(target['gt'],np.int64);ids=identity_truth(target)
                    np.testing.assert_array_equal(gt,LABEL_MAP[np.fromfile(lp,dtype=np.uint8)])
                    labels,_=voxel_truth(inverse,gt,ids,len(r['xyz']))
                    if part=='train' and k:voxel_counts+=np.bincount(labels[labels>=0],minlength=16)
                    y=torch.tensor(labels[query.cpu().numpy()],device='cuda',dtype=torch.long)
                    candidate=capture.value[2]
                    group=supervision_groups(qa,y,candidate['probabilities'].argmax(-1),candidate['valid'])
                    if part=='train' and k:groups_count+=np.bincount(group[group>=0].cpu().numpy(),minlength=3)
                    base=r['logits'].argmax(1).cpu().numpy().astype(np.int8);anchor_labels=a.argmax(1).cpu().numpy().astype(np.int8)
                    camera=np.zeros(len(gt),bool);camera[target['camera_in_view_indices']]=True
                    image=base[inverse].copy()
                    if observations is not None:
                        vf=observations[k]
                        if vf['frame']!=k or vf['stamp']!=int(row['timestamp_us']*1000):raise ValueError('Visual input timestamps differ')
                        probabilities=r['logits'].float().softmax(-1)[r['inverse']].cpu().numpy()
                        for obs in vf['observations']:
                            at=np.asarray(obs['indices']);image[at]=fusion(probabilities[at],obs['current_scores'],obs['confidence']).argmax(1)
                    arrays=dict(joint=z,anchor=anc,query=query.cpu().numpy().astype(np.int32),label=y.cpu().numpy().astype(np.int8),group=group.cpu().numpy().astype(np.int8),
                        inverse=inverse,point_label=gt.astype(np.int8),identity=ids.astype(np.int32),camera=camera,
                        baseline=base,anchor_labels=anchor_labels,current_image=image)
                    size=sum(v.nbytes for v in arrays.values())
                    if manifest['total_bytes']+size+65536>4*1024**3:raise RuntimeError('Frozen 4 GiB cache cap would be exceeded')
                    dest=cache/scene/f'{k:03d}.npz';np.savez(dest,**arrays)
                    manifest['total_bytes']+=dest.stat().st_size
                    manifest['files'].append(dict(part=part,scene=scene,log=idx['scene_logs'][scene],frame=k,stamp=raw['stamp'],path=str(dest.relative_to(cache)),sha256=digest(dest),bytes=dest.stat().st_size,queries=len(query)))
                save(out/'progress.json',dict(stage='extraction',completed_frames=len(manifest['files']),scene=scene,total_bytes=manifest['total_bytes'],elapsed_s=time.perf_counter()-start))
                print('extracted',scene,len(rows),'frames',manifest['total_bytes'],'bytes',flush=True)
    ht.remove();hc.remove();capture.close()
    manifest.update(train_voxel_counts=voxel_counts.tolist(),class_weights=present_class_weights(voxel_counts).tolist(),train_groups=groups_count.tolist(),elapsed_s=time.perf_counter()-start)
    if len(manifest['files'])!=963 or manifest['parity_frames']!=963 or manifest['real_output_checks']!=144:raise AssertionError('Incomplete real-cache parity checks')
    save(cache/'manifest.json',manifest);save(out/'feature-manifest.json',manifest)
    save(out/'progress.json',dict(stage='extraction_complete',frames=963,bytes=manifest['total_bytes'],elapsed_s=manifest['elapsed_s']))
    print('extraction complete',manifest['total_bytes'],'bytes',manifest['train_groups'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('train-index','dev-index','train-targets','dev-targets','visual','anchors','models','cache','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();extract(a.train_index,a.dev_index,a.train_targets,a.dev_targets,a.visual,a.anchors,a.models,a.cache,a.out)
