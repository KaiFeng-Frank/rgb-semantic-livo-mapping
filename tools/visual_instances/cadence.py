"""Same current detections, different strictly causal tracking histories."""
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from .reid import PairedNativeTrackers,NativeFeatureBOTSORT
from .geometry import InstanceFrame,mask_frame


class PairedCadenceTrackers(PairedNativeTrackers):
    def __init__(self,weights):
        import yaml
        super().__init__(weights)
        # This experiment uses no appearance model or native ReID features.
        self.hook.remove();self.model.predictor._feats=None
        cfg=yaml.safe_load(Path(__file__).with_name('botsort.yaml').read_text())
        self.trackers={name:NativeFeatureBOTSORT(SimpleNamespace(**dict(cfg,track_buffer=buffer)))
            for name,buffer in [('keyframe',15),('fullrate',90)]}
        self.track_stamps={name:[None] for name in self.trackers}
        self.last_processed={name:None for name in self.trackers}
        self.last_stamp=None

    def step(self,image,stamp,is_keyframe):
        if self.last_stamp is not None and stamp<=self.last_stamp:raise ValueError('Noncausal image stream')
        self.last_stamp=stamp
        self.torch.cuda.synchronize();start=time.perf_counter();r=self.model.predict(image,**self.kwargs)[0]
        self.torch.cuda.synchronize();forward_ms=(time.perf_counter()-start)*1000
        det=r.boxes.cpu().numpy();frames={};timing={};expired={}
        for name,tr in self.trackers.items():
            if name=='keyframe' and not is_keyframe:continue
            last=self.last_processed[name]
            if last is not None and stamp-last>750_000_000:raise ValueError('Unexpected cadence gap; start a fresh scene')
            self.last_processed[name]=stamp;times=self.track_stamps[name]
            start=time.perf_counter();old=[t for t in tr.lost_stracks if stamp-times[t.frame_id]>7_500_000_000]
            expired[name]=len(old)
            for t in old:t.mark_removed()
            old_ids={t.track_id for t in old};tr.lost_stracks=[t for t in tr.lost_stracks if t.track_id not in old_ids]
            tr.removed_stracks.extend(old)
            tracks=tr.update(det,image);times.append(stamp)
            timing[name]=(time.perf_counter()-start)*1000
            if not is_keyframe:continue
            if not len(tracks):
                frame=InstanceFrame(stamp,'0'*32,np.zeros(image.shape[:2],np.uint16),np.empty(0,np.int64),np.empty(0,np.uint16),np.empty(0,np.float32))
            else:
                selected=r[tracks[:,-1].astype(int)];selected.update(boxes=self.torch.as_tensor(tracks[:,:-1]))
                boxes=selected.boxes
                frame=mask_frame(selected.masks.data.cpu().numpy()>.5,boxes.id.int().cpu().tolist(),
                    boxes.cls.int().cpu().tolist(),boxes.conf.cpu().tolist(),stamp,'0'*32,image.shape[:2],
                    class_scores=selected.class_scores.float().cpu().numpy())
            frames[name]=frame
        return frames,dict(forward_ms=forward_ms,tracking_including_gmc_ms=timing,raw_detections=len(r.boxes),
            time_expired_tracks=expired,is_keyframe=is_keyframe)

    def close(self):pass  # Owned feature hook was removed immediately after initialization.
