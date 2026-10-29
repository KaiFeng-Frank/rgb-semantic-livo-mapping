"""Reuse pinned BoT-SORT appearance memory without modifying installed packages.

Ultralytics 8.3.228 BOTrack calls update_features before initializing its deque
when a ReID feature is supplied. Initialize the unchanged parent with feat=None,
then supply a finite copy after its normal initialization. Association, Kalman,
GMC and appearance EMA remain the library implementation.
"""
import numpy as np
from ultralytics.trackers.bot_sort import BOTrack,BOTSORT


class InitializedBOTrack(BOTrack):
    def __init__(self,xywh,score,cls,feat=None,feat_history=50):
        super().__init__(xywh,score,cls,feat=None,feat_history=feat_history)
        if feat is not None:
            feat=np.asarray(feat,np.float32).copy()
            if not np.isfinite(feat).all() or np.linalg.norm(feat)<1e-12:raise ValueError('Invalid ReID feature')
            self.update_features(feat)


class NativeFeatureBOTSORT(BOTSORT):
    def init_track(self,results,img=None):
        if not self.args.with_reid:return super().init_track(results,img)
        if len(results)==0:return []
        if img is None or len(img)!=len(results):raise ValueError('Missing/misaligned native ReID features')
        bboxes=results.xywhr if hasattr(results,'xywhr') else results.xywh
        bboxes=np.concatenate([bboxes,np.arange(len(bboxes)).reshape(-1,1)],axis=-1)
        features=self.encoder(img,bboxes)
        return [InitializedBOTrack(box,score,cls,feat) for box,score,cls,feat in zip(bboxes,results.conf,results.cls,features)]


class FixedGMC:
    def __init__(self):self.warp=np.eye(2,3)
    def apply(self,img,detections):return self.warp.copy()
    def reset_params(self):pass


class PairedNativeTrackers:
    """One detector forward and one GMC estimate for both tracker hypotheses."""
    def __init__(self,weights):
        from pathlib import Path
        from types import SimpleNamespace
        import cv2,yaml,torch
        from ultralytics import YOLO
        from .evidence_predictor import EvidencePredictor
        from .geometry import COCO_TO_NATIVE
        self.torch=torch;cv2.setNumThreads(1);torch.set_num_threads(1)
        self.model=YOLO(str(Path(weights).resolve(strict=True)),task='segment')
        self.kwargs=dict(device='0',imgsz=640,rect=False,verbose=False,half=True,conf=.1,iou=.5,
            classes=list(COCO_TO_NATIVE),max_det=128,retina_masks=True,save=False,predictor=EvidencePredictor)
        for _ in range(3):self.model.predict(np.zeros((640,640,3),np.uint8),**self.kwargs)
        predictor=self.model.predictor
        def capture(module,inputs):predictor._feats=list(inputs[0])
        self.hook=self.model.model.model[-1].register_forward_pre_hook(capture)
        self.trackers={}
        for name,filename in [('baseline','botsort.yaml'),('native_reid','botsort_reid.yaml')]:
            cfg=yaml.safe_load(Path(__file__).with_name(filename).read_text())
            self.trackers[name]=NativeFeatureBOTSORT(SimpleNamespace(**cfg))
        self.shared_gmc=self.trackers['baseline'].gmc
        self.fixed_gmc=FixedGMC()
        for tracker in self.trackers.values():tracker.gmc=self.fixed_gmc
        self.last_stamp=None;self.epoch=0

    def step(self,image,stamp):
        import time,uuid
        from .geometry import InstanceFrame,mask_frame
        if self.last_stamp is not None and stamp<=self.last_stamp:raise ValueError('Nonincreasing camera time')
        if self.last_stamp is not None and stamp-self.last_stamp>750_000_000:
            for tr in self.trackers.values():tr.reset()
            self.shared_gmc.reset_params();self.epoch+=1
        self.last_stamp=stamp
        start=time.perf_counter();r=self.model.predict(image,**self.kwargs)[0]
        forward_ms=(time.perf_counter()-start)*1000
        if r.feats is None or len(r.feats)!=len(r.boxes):raise ValueError('Feature transport is missing or misaligned')
        det=r.boxes.cpu().numpy();start=time.perf_counter();gmc_error=None
        try:self.fixed_gmc.warp=self.shared_gmc.apply(image,det[det.conf>=self.trackers['baseline'].args.track_high_thresh].xyxy)
        except Exception as e:self.fixed_gmc.warp=np.eye(2,3);gmc_error=type(e).__name__+': '+str(e)
        gmc_ms=(time.perf_counter()-start)*1000;frames={};timing={}
        for name,tr in self.trackers.items():
            start=time.perf_counter();tracks=tr.update(det,image,r.feats)
            timing[name]=(time.perf_counter()-start)*1000
            # Distinct scene instances have their own tracker objects; epoch
            # namespaces prevent a reset from recycling an earlier identity.
            session=f'{self.epoch:032x}'
            if not len(tracks):
                frame=InstanceFrame(stamp,session,np.zeros(image.shape[:2],np.uint16),np.empty(0,np.int64),np.empty(0,np.uint16),np.empty(0,np.float32))
            else:
                selected=r[tracks[:,-1].astype(int)]
                selected.update(boxes=self.torch.as_tensor(tracks[:,:-1]))
                boxes=selected.boxes
                frame=mask_frame(selected.masks.data.cpu().numpy()>.5,boxes.id.int().cpu().tolist(),
                    boxes.cls.int().cpu().tolist(),boxes.conf.cpu().tolist(),stamp,session,image.shape[:2],
                    class_scores=selected.class_scores.float().cpu().numpy())
            frames[name]=frame
        return frames,dict(forward_ms=forward_ms,gmc_ms=gmc_ms,tracker_ms=timing,raw_detections=len(r.boxes),
            feature_dimension=int(r.feats.shape[1]),gmc_error=gmc_error)

    def close(self):self.hook.remove()
