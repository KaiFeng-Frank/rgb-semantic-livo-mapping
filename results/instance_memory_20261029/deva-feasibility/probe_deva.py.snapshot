"""Fixed 32-image native-online DEVA runtime/causality probe, without GT."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from tools.instance_memory.data import read,save
from tools.semantic_memory.nuscenes_pilot_data import digest

PROTOCOL=Path(__file__).resolve().parents[2]/'config/deva_memory_probe_20261029.json'


def label_image(frame):
    lookup=np.concatenate([np.asarray([-1],np.int16),frame.labels.astype(np.int16)])
    return lookup[frame.index]


def run(cache,source,weights,out,pulp_source):
    import cv2,torch
    from .geometry import InstanceFrame
    from .kalman_cache import gpu_admission
    from .deva_memory import DEVAPair
    protocol=read(PROTOCOL)
    for path,pin in [(source,protocol['source_commit']),(pulp_source,protocol['pulp_source_commit'])]:
        if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=pin:
            raise ValueError('Dependency/source commit mismatch')
        if subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip():
            raise ValueError('Dependency/source has modifications')
    sys.path.insert(0,str(pulp_source))
    if digest(weights)!=protocol['checkpoint_sha256'] or weights.stat().st_size!=protocol['checkpoint_bytes']:
        raise ValueError('Unpinned checkpoint')
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    if commit!=protocol['source_commit'] or dirty:raise ValueError('Source pin mismatch')
    if out.exists():raise ValueError('Refuse overwrite')
    manifest=read(cache/'manifest.json')
    if len(manifest['entries'])!=32:raise ValueError('Expected frozen32 inputs')
    admission=gpu_admission();torch.set_num_threads(1);cv2.setNumThreads(1)
    out.mkdir(parents=True)
    start=time.perf_counter();model=DEVAPair(source,weights,protocol['model']);init=time.perf_counter()-start
    save(out/'protocol.json',dict(protocol=protocol,protocol_sha256=digest(PROTOCOL),metadata=model.meta,
        input_manifest_sha256=digest(cache/'manifest.json'),admission=admission,init_seconds=init,
        source_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('deva_memory.py')]}))
    torch.cuda.reset_peak_memory_stats();passes=[]
    for repeat in range(1+protocol['repeats']):
        np.random.seed(20261029);torch.manual_seed(20261029);model.reset();rows=[]
        for e in manifest['entries']:
            if shutil.disk_usage(out).free<protocol['min_free_disk_bytes']:raise RuntimeError('Disk reserve reached')
            if digest(e['image'])!=e['image_sha256'] or digest(cache/e['cache'])!=e['cache_sha256']:
                raise ValueError('Cached input changed')
            image=cv2.imread(e['image'])
            with np.load(cache/e['cache'],allow_pickle=False) as a:
                frame=InstanceFrame(e['stamp_ns'],e['session'],a['index'],a['ids'],a['labels'],a['scores'])
            outputs,timing=model.step(image,frame)
            current=label_image(outputs['current']);temporal=label_image(outputs['temporal'])
            if e['frame']==0 and not np.array_equal(current,temporal):raise AssertionError('No-history controls differ')
            if e['frame']>0 and timing['memory_reads']!=1:raise AssertionError('History path was not read exactly once')
            if timing['buffered_future_frames']:raise AssertionError('Future-frame buffer used')
            record=dict(frame=e['frame'],repeat=repeat,stamp_ns=e['stamp_ns'],**timing,
                        changed_semantic_pixels=int((current!=temporal).sum()))
            payload={}
            for arm,f in outputs.items():
                payload.update({arm+'_'+k:getattr(f,k) for k in ['index','ids','labels','scores']})
            path=out/f"{e['frame']:03d}.npz"
            if repeat==0:np.savez_compressed(path,**payload)
            else:
                with np.load(path,allow_pickle=False) as a:
                    for k,v in payload.items():
                        if not np.array_equal(a[k],v):raise AssertionError(f'Repeat changed output: {repeat}/{e["frame"]}/{k}')
            rows.append(record);save(out/'progress.json',dict(completed_passes=passes,repeat=repeat,frames=rows))
            print(record,flush=True)
        v=[r['deploy_component_ms'] for r in rows[1:]]
        passes.append(dict(repeat=repeat,frames=rows,timing_ms=dict(median=float(np.median(v)),p95=float(np.percentile(v,95)),max=max(v))))
    passed=all(p['timing_ms']['p95']<=protocol['runtime_gate_ms'] for p in passes)
    save(out/'report.json',dict(passes=passes,runtime_gate=dict(passed=passed,budget_ms=protocol['runtime_gate_ms']),
        repeated_outputs_exact=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='Native DEVA causal incorporation every frame, current-only control shared encoding; no GT, no accuracy or end-to-end real-time claim.'))
    print('runtime_gate',passed,[p['timing_ms'] for p in passes],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('cache','source','weights','out','pulp-source'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache,a.source,a.weights,a.out,a.pulp_source)
