"""Fixed eight-image, multi-object SAM2 latency/causality feasibility probe."""
import argparse
from pathlib import Path
import subprocess,time
import numpy as np
from tools.instance_memory.data import read,save
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest


def run(source,weights,yolo,frames,images,out,image_size=1024):
    import cv2,torch
    from .fullrate import FullRateDetector
    from .sam2_memory import SAM2Pair
    from .kalman_cache import gpu_admission
    filename='sam2_memory_probe_20261029.json' if image_size==1024 else 'sam2_memory_probe_512_20261029.json'
    protocol_path=Path(__file__).resolve().parents[2]/'config'/filename
    protocol=read(protocol_path)
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=protocol['source_commit'] or digest(weights)!=protocol['checkpoint_sha256']:
        raise ValueError('Frozen official source/checkpoint mismatch')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise ValueError('Official implementation has local modifications')
    admission=gpu_admission();torch.set_num_threads(1);cv2.setNumThreads(1)
    if out.exists():raise ValueError('Refuse overwrite')
    sequence=read(frames)['scenes']['scene-0218'][:8]
    start=time.perf_counter();detector=FullRateDetector(yolo);model=SAM2Pair(source,weights,image_size=image_size);startup=time.perf_counter()-start
    rows=[];torch.cuda.reset_peak_memory_stats()
    for i,r in enumerate(sequence):
        path=DATA/r['path'] if r['keyframe'] is not None else images/Path(r['path']).name
        image=cv2.imread(str(path));frame,visual=detector.step(image,int(r['timestamp_us']*1000))
        masks,timing=model.step(image,frame)
        changed=int((masks['current']!=masks['temporal']).sum())
        if i==0 and changed:raise AssertionError('First-frame memory/current heads differ before any history')
        row=dict(frame=i,stamp_ns=frame.stamp_ns,image_sha256=digest(path),yolo_ms=visual['elapsed_ms'],**timing,changed_mask_pixels=changed)
        rows.append(row);print(row,flush=True)
    v=[r['encoder_ms']+r['temporal_ms'] for r in rows[1:]]
    report=dict(scene='scene-0218',frames=rows,metadata=model.meta,source_commit=commit,protocol=protocol,protocol_sha256=digest(protocol_path),frames_index_sha256=digest(frames),
        startup_seconds=startup,peak_allocated_bytes=torch.cuda.max_memory_allocated(),admission=admission,
        source_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('sam2_memory.py')]},
        sam_only_ms=dict(median=float(np.median(v)),p95=float(np.percentile(v,95)),max=max(v)),
        runtime_gate=dict(sam_incremental_p95_budget_ms=50,passed=float(np.percentile(v,95))<=50),
        scope='Eight predetermined development images, actual current detector prompts, no GT labels. Same encoded image and box prompts for zero-history/current and causal memory. No accuracy claim; timings exclude PTv3/ROS. Initial frame included separately.')
    save(out,report);print(report['sam_only_ms'],report['runtime_gate'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','weights','yolo','frames','images','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--image-size',type=int,choices=[512,1024],default=1024)
    a=p.parse_args();run(a.source,a.weights,a.yolo,a.frames,a.images,a.out,a.image_size)
