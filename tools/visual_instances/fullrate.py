"""Single-stream deployment of the frozen full-rate tracking policy."""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import time
import uuid
import numpy as np
from .reid import PairedNativeTrackers,NativeFeatureBOTSORT
from .geometry import InstanceFrame,mask_frame,MAX_PIXELS
from .visual_memory import VisualMemory


class FullRateDetector(PairedNativeTrackers):
    def __init__(self,weights,imgsz=640,feature_embeddings=False,visual_history=False,keyframe_reference=False):
        import yaml,ultralytics
        if imgsz!=640:raise ValueError('Frozen full-rate policy uses imgsz=640')
        start=time.perf_counter();super().__init__(weights)
        self.hook.remove();self.model.predictor._feats=None
        cfg=yaml.safe_load(Path(__file__).with_name('botsort.yaml').read_text());cfg['track_buffer']=15 if keyframe_reference else 90
        self.tracker=NativeFeatureBOTSORT(SimpleNamespace(**cfg));self.trackers={}
        self.times={};self.session=uuid.uuid4().hex;self.imgsz=imgsz;self.last_stamp=None
        self.visual_features=None
        if feature_embeddings:
            from tools.instance_memory.data import VisualFeatures
            self.visual_features=VisualFeatures(self)
        self.visual_memory=VisualMemory() if visual_history else None
        self.meta=dict(weights=str(Path(weights).resolve()),sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
            ultralytics=ultralytics.__version__,tracking='keyframe_reference' if keyframe_reference else 'fullrate',tracker=cfg,real_time_expiry_s=7.5,gap_reset_s=.75,
            feature_embeddings=feature_embeddings,visual_memory=self.visual_memory.configuration() if self.visual_memory else None,
            cpu_affinity=sorted(os.sched_getaffinity(0)),training='none; frozen original YOLO11n-seg',
            startup_seconds=time.perf_counter()-start,paired_reference='40a38c4 fullrate arm')

    def step(self,image,stamp_ns):
        import cv2
        if (type(stamp_ns) is not int or stamp_ns<0 or self.last_stamp is not None and stamp_ns<=self.last_stamp
                or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or image.shape[0]*image.shape[1]>MAX_PIXELS):
            raise ValueError('Invalid full-rate image/timestamp')
        reset=self.last_stamp is not None and stamp_ns-self.last_stamp>750_000_000
        if reset:
            self.tracker.reset();self.times.clear();self.session=uuid.uuid4().hex
            if self.visual_memory:self.visual_memory.reset()
        self.last_stamp=stamp_ns;start=time.perf_counter()
        r=self.model.predict(image,**self.kwargs)[0];tr=self.tracker
        old=[t for t in tr.lost_stracks if stamp_ns-self.times[t.frame_id]>7_500_000_000]
        for t in old:t.mark_removed()
        retired={t.track_id for t in old};tr.lost_stracks=[t for t in tr.lost_stracks if t.track_id not in retired];tr.removed_stracks.extend(old)
        tracks=tr.update(r.boxes.cpu().numpy(),image);self.times[tr.frame_id]=stamp_ns
        retained={t.frame_id for t in tr.tracked_stracks+tr.lost_stracks}|{tr.frame_id}
        self.times={k:v for k,v in self.times.items() if k in retained}
        frame=InstanceFrame(stamp_ns,self.session,np.zeros(image.shape[:2],np.uint16),np.empty(0,np.int64),np.empty(0,np.uint16),np.empty(0,np.float32))
        if len(tracks):
            s=r[tracks[:,-1].astype(int)];s.update(boxes=self.torch.as_tensor(tracks[:,:-1]));b=s.boxes
            frame=mask_frame(s.masks.data.cpu().numpy()>.5,b.id.int().cpu().tolist(),b.cls.int().cpu().tolist(),b.conf.cpu().tolist(),stamp_ns,self.session,image.shape[:2],class_scores=s.class_scores.float().cpu().numpy())
        if len(frame.ids):
            hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV).reshape(-1,3);descriptors=[]
            for slot in range(len(frame.ids)):
                at=np.flatnonzero(frame.index.reshape(-1)==slot+1)
                if len(at)>1024:at=at[np.linspace(0,len(at)-1,1024,dtype=int)]
                h=np.concatenate([np.histogram(hsv[at,c],bins=8,range=(0,limit))[0] for c,limit in enumerate([180,256,256])]).astype(np.float32)+1e-3
                descriptors.append(h/h.sum())
            frame.appearance=np.stack(descriptors)
        if self.visual_features is not None:
            features=self.visual_features.pool(frame)
            frame.visual_features=np.asarray([features[int(t)][0] for t in frame.ids],np.float32).reshape(-1,64)
            frame.visual_feature_valid=np.asarray([features[int(t)][1] for t in frame.ids],bool)
        memory_info=None
        if self.visual_memory:
            frame.temporal_class_scores,frame.temporal_observations,memory_info=self.visual_memory.step(frame)
        return frame,dict(elapsed_ms=(time.perf_counter()-start)*1000,instances=len(frame.ids),reset=reset,
            expired_tracks=len(old),model_stages_ms=dict(r.speed),visual_memory=memory_info)
