"""Diagnose the observed 24-point old-cache discrepancy, no parameter tuning."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tools.instance_memory.data import read,save
from tools.semantic_memory.nuscenes_pilot_data import DATA,digest


def audit(index_path,cache,weights,out):
    import cv2,torch,yaml
    from .cadence import PairedCadenceTrackers
    from .reid import NativeFeatureBOTSORT
    from .geometry import Camera,InstanceFrame,mask_frame,associate
    from .semantics import Config,component
    from .kalman_cache import gpu_admission
    gpu_admission();torch.manual_seed(20261029);torch.set_num_threads(1)
    index=read(index_path);name='scene-0218';cached=read(cache/(name+'.inputs.json.gz'))['frames']
    pair=PairedCadenceTrackers(weights)
    native=NativeFeatureBOTSORT(SimpleNamespace(**yaml.safe_load(Path(__file__).with_name('botsort.yaml').read_text())))
    results=[]
    for row,old in zip(index['sequences'][name],cached):
        image=cv2.imread(str(DATA/row['image_path']));stamp=int(row['camera_timestamp_us']*1000)
        frames,timing=pair.step(image,stamp,True);current=pair.model.predictor.results[0]
        tracks=native.update(current.boxes.cpu().numpy(),image)
        if len(tracks):
            s=current[tracks[:,-1].astype(int)];s.update(boxes=torch.as_tensor(tracks[:,:-1]));b=s.boxes
            f=mask_frame(s.masks.data.cpu().numpy()>.5,b.id.int().cpu().tolist(),b.cls.int().cpu().tolist(),
                b.conf.cpu().tolist(),stamp,'0'*32,image.shape[:2],class_scores=s.class_scores.float().cpu().numpy())
        else:f=InstanceFrame(stamp,'0'*32,np.zeros(image.shape[:2],np.uint16),np.empty(0,np.int64),np.empty(0,np.uint16),np.empty(0,np.float32))
        frames={'timestamp_capped':frames['keyframe'],'original_uncapped':f}
        scan=np.fromfile(DATA/row['scan_path'],dtype='<f4').reshape(-1,5)
        cam=Camera(image.shape[1],image.shape[0],row['camera_intrinsic'],np.zeros(4),np.linalg.inv(row['camera_transform'])@row['transform'])
        expected=np.zeros(len(scan),bool)
        for o in old['observations']:
            if o['trusted']:expected[o['indices']]=True
        masks={}
        for arm,frame in frames.items():
            ids=associate(scan[:,:3],cam,frame)['ids'];mask=np.zeros(len(scan),bool)
            for tid in np.unique(ids[ids>0]):
                at=np.flatnonzero(ids==tid);c=component(scan[at,:3],Config())
                if c is not None:mask[at[c]]=True
            masks[arm]=mask
        results.append(dict(frame=row['frame'],differences_from_old_cache={a:int((m!=expected).sum()) for a,m in masks.items()},
            common_vs_original=int((masks['timestamp_capped']!=masks['original_uncapped']).sum()),
            time_expired_tracks=timing['time_expired_tracks']['keyframe']))
    report=dict(scene=name,frames=results,source_sha256=digest(Path(__file__)),
        differences_from_old_cache={a:sum(r['differences_from_old_cache'][a] for r in results) for a in masks},
        common_vs_original=sum(r['common_vs_original'] for r in results),
        scope='Same YOLO forward, original tracker without added timestamp expiry vs timestamp-capped keyframe comparator. Identity numbering is ignored for coverage parity. No threshold change or GT input.')
    save(out,report);print({k:v for k,v in report.items() if k!='frames'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['index','cache','weights','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();audit(a.index,a.cache,a.weights,a.out)
