"""Same-forward adapter parity beside live PTv3, on the mismatching dev clip."""
import argparse
from pathlib import Path
import sys
import numpy as np
from tools.instance_memory.data import read,save
from tools.instance_memory.metrics import IdentityMetrics,identity_truth
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest


def run(index_path,frames_path,cache,images,weights,out):
    import cv2,torch
    from .cadence import PairedCadenceTrackers
    from .fullrate import FullRateDetector,native_visual_precision
    from .geometry import Camera,associate
    from .kalman_cache import gpu_admission
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from ptv3_worker import Segmenter
    gpu_admission();torch.set_num_threads(1);torch.manual_seed(20261029);cv2.setNumThreads(1)
    index=read(index_path);sequence=read(frames_path)['scenes']['scene-1074']
    targets=read(cache/'scene-1074.targets.json.gz')['frames'];rows=index['sequences']['scene-1074']
    segmenter=Segmenter(half=True,shuffle=False,intensity_scale=1.)
    with native_visual_precision():paired=PairedCadenceTrackers(weights)
    adapter=FullRateDetector(weights,visual_history=True)
    original=paired.model.predict;captured=[]
    def capture(*args,**kwargs):
        result=original(*args,**kwargs);captured[:]=result;return result
    paired.model.predict=capture
    # This diagnostic holds detector output fixed; it does not measure speed.
    adapter.model.predict=lambda *args,**kwargs:captured
    metrics={a:IdentityMetrics() for a in ['reference','adapter']}
    rows_out=[]
    for record in sequence:
        k=record['keyframe'];path=DATA/record['path'] if k is not None else images/Path(record['path']).name
        image=cv2.imread(str(path));stamp=int(record['timestamp_us']*1000)
        with native_visual_precision():ref,_=paired.step(image,stamp,k is not None)
        got,_=adapter.step(image,stamp)
        if k is None:continue
        row=rows[k];scan=np.fromfile(DATA/row['scan_path'],dtype='<f4').reshape(-1,5)[:,:4].copy();scan[:,3]/=255
        camera=Camera(image.shape[1],image.shape[0],row['camera_intrinsic'],np.zeros(4),np.linalg.inv(row['camera_transform'])@row['transform'])
        pred={a:associate(scan[:,:3],camera,f)['ids'] for a,f in [('reference',ref['fullrate']),('adapter',got)]}
        # Preserve the co-evaluation workload and its global environment.
        segmenter.segment(scan,return_features=True)
        target=targets[k];gt=identity_truth(target);inside=np.zeros(len(scan),bool);inside[target['camera_in_view_indices']]=True
        valid=inside&(np.asarray(target['gt'])>=0)&(np.asarray(target['panoptic'])>0)
        for a,p in pred.items():metrics[a].add(gt,p,k,valid)
        # Label-renaming-invariant correspondence: each nonzero reference ID has
        # one adapter ID and conversely; include zeros to expose coverage changes.
        pairs=np.unique(np.stack([pred['reference'],pred['adapter']],1),axis=0)
        equivalent=len(np.unique(pairs[:,0]))==len(pairs)==len(np.unique(pairs[:,1])) and np.array_equal(pred['reference']>0,pred['adapter']>0)
        rows_out.append(dict(frame=k,partition_equal=bool(equivalent),image_sha256=digest(path)))
    result=dict(scene='scene-1074',same_forward=True,ptv3_interleaved=True,identity={a:m.result() for a,m in metrics.items()},frames=rows_out,
        all_frame_partitions_equal=all(r['partition_equal'] for r in rows_out),source_sha256=digest(Path(__file__)),
        scope='Diagnostic of detector-output versus adapter differences, not a new accuracy evaluation or latency measurement.')
    save(out,result);print({k:v for k,v in result.items() if k!='frames'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['index','frames','cache','images','weights','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.index,a.frames,a.cache,a.images,a.weights,a.out)
