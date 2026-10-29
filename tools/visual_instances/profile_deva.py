"""Disjoint-stage timing of the failed fixed DEVA configuration, without GT."""
from contextlib import contextmanager
import argparse
from collections import defaultdict
from pathlib import Path
import sys,time
import numpy as np
import torch
from tools.instance_memory.data import read,save
from tools.semantic_memory.nuscenes_pilot_data import digest
from .geometry import InstanceFrame
from .probe_deva import PROTOCOL


@contextmanager
def stage_timers(model,events,phase):
    import deva.inference.inference_core as native
    from deva.inference.memory_manager import MemoryManager
    old=[]
    def patch(obj,name,value):
        old.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    def timer(name,fn):
        def run(*args,**kwargs):
            torch.cuda.synchronize();start=time.perf_counter()
            value=fn(*args,**kwargs)
            torch.cuda.synchronize()
            events.append(dict(stage=name,arm=phase['value'],ms=(time.perf_counter()-start)*1000))
            return value
        return run
    original=native.DEVAInferenceCore.incorporate_detection
    def incorporate(core,*args,**kwargs):
        before=phase['value'];phase['value']='temporal' if core is model.core else 'current'
        try:return original(core,*args,**kwargs)
        finally:phase['value']=before
    try:
        patch(native.DEVAInferenceCore,'incorporate_detection',incorporate)
        for name in ['encode_image','transform_key','encode_mask','segment']:
            patch(model.model,name,timer(name,getattr(model.model,name)))
        patch(native,'match_and_merge',timer('match_and_merge',native.match_and_merge))
        patch(MemoryManager,'match_memory',timer('match_memory',MemoryManager.match_memory))
        patch(MemoryManager,'add_memory',timer('memory_write_compress',MemoryManager.add_memory))
        yield
    finally:
        for obj,name,fn in reversed(old):setattr(obj,name,fn)


def run(cache,source,pulp_source,weights,reference,out):
    import cv2,subprocess
    from .deva_memory import DEVAPair
    from .kalman_cache import gpu_admission
    p=read(PROTOCOL)
    for path,key in [(source,'source_commit'),(pulp_source,'pulp_source_commit')]:
        if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=p[key]:raise ValueError('Source mismatch')
        if subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip():raise ValueError('Dirty source')
    if digest(weights)!=p['checkpoint_sha256']:raise ValueError('Weights mismatch')
    if out.exists():raise ValueError('Refuse overwrite')
    sys.path.insert(0,str(pulp_source));torch.set_num_threads(1);cv2.setNumThreads(1)
    admission=gpu_admission();model=DEVAPair(source,weights,p['model'])
    np.random.seed(20261029);torch.manual_seed(20261029);model.reset()
    events=[];phase=dict(value='shared');rows=[]
    with stage_timers(model,events,phase):
        for e in read(cache/'manifest.json')['entries']:
            if digest(e['image'])!=e['image_sha256'] or digest(cache/e['cache'])!=e['cache_sha256']:raise ValueError('Input changed')
            image=cv2.imread(e['image'])
            with np.load(cache/e['cache']) as a:
                f=InstanceFrame(e['stamp_ns'],e['session'],a['index'],a['ids'],a['labels'],a['scores'])
            events.clear();outputs,timing=model.step(image,f)
            with np.load(reference/f"{e['frame']:03d}.npz") as a:
                for arm,frame in outputs.items():
                    for key in ['index','ids','labels','scores']:
                        if not np.array_equal(a[arm+'_'+key],getattr(frame,key)):raise AssertionError('Profiler changed output')
            totals=defaultdict(float)
            for event in events:
                if event['arm']!='current':totals[event['stage']]+=event['ms']
            row=dict(frame=e['frame'],events=list(events),temporal_stages_ms=dict(totals),timing=timing)
            rows.append(row);print(e['frame'],dict(totals),flush=True)
    names=sorted({k for r in rows for k in r['temporal_stages_ms']})
    summary={k:dict(median_ms=float(np.median([r['temporal_stages_ms'].get(k,0.) for r in rows[1:]])),
                   p95_ms=float(np.percentile([r['temporal_stages_ms'].get(k,0.) for r in rows[1:]],95))) for k in names}
    save(out,dict(frames=rows,summary=summary,all_outputs_exact=True,admission=admission,
        source_sha256={x.name:digest(x) for x in [Path(__file__),Path(__file__).with_name('deva_memory.py')]},
        reference_report_sha256=digest(reference/'report.json'),protocol_sha256=digest(PROTOCOL),
        scope='Single diagnostic replay with synchronization at disjoint stage boundaries; same32 outputs as original run. No GT. Not a new runtime or accuracy admission.'))
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['cache','source','pulp-source','weights','reference','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.cache,a.source,a.pulp_source,a.weights,a.reference,a.out)
