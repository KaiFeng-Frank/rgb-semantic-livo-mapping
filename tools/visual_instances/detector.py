"""Official YOLO11-seg + BoT-SORT; no extra training or semantic overwrites."""
import hashlib
import os
from pathlib import Path
import time
import uuid

import numpy as np

from .geometry import COCO_TO_NATIVE, InstanceFrame, MAX_INSTANCES, MAX_PIXELS, mask_frame


class Detector:
    def __init__(self, weights, device='0', imgsz=640, tracker=None, feature_embeddings=False):
        import cv2
        import torch
        from ultralytics import YOLO
        import ultralytics
        from .evidence_predictor import EvidencePredictor
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # The evaluator creates successive models in one interpreter.
        path = Path(weights).resolve(strict=True)  # No implicit model downloads.
        self.model = YOLO(str(path), task='segment')
        self.device, self.imgsz = device, imgsz
        self.tracker = str(tracker or Path(__file__).with_name('botsort.yaml'))
        self.session, self.last_stamp = uuid.uuid4().hex, None
        self.meta = dict(weights=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         ultralytics=ultralytics.__version__, imgsz=imgsz, tracker=Path(self.tracker).read_text(),
                         cpu_affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS,0),
                         classes=COCO_TO_NATIVE, training='none; official COCO pretrained weights')
        # Warm inference without creating track identities from synthetic inputs.
        for _ in range(3):
            self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), device=device,
                               imgsz=imgsz, rect=False, verbose=False, half=device != 'cpu',
                               predictor=EvidencePredictor)
        # Import/initialize tracking before declaring online readiness. Reset all
        # synthetic warmup state so it cannot contribute an observation or ID.
        self.model.track(np.zeros((imgsz, imgsz, 3), np.uint8), persist=True,
            tracker=self.tracker, device=device, imgsz=imgsz, verbose=False,
            classes=list(COCO_TO_NATIVE), retina_masks=True, rect=False, half=device != 'cpu')
        for tracker in self.model.predictor.trackers:
            tracker.reset()
        self.visual_features=None
        if feature_embeddings:
            from tools.instance_memory.data import VisualFeatures
            self.visual_features=VisualFeatures(self)
        self.meta['feature_embeddings']=bool(feature_embeddings)

    def step(self, image, stamp_ns):
        import cv2
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.shape[0]*image.shape[1] > MAX_PIXELS:
            raise ValueError('Expected bounded BGR image')
        if type(stamp_ns) is not int or stamp_ns < 0 or self.last_stamp is not None and stamp_ns <= self.last_stamp:
            raise ValueError('Visual timestamps must increase; use new Detector for new sequence')
        reset = self.last_stamp is not None and stamp_ns-self.last_stamp > 750_000_000
        if reset:
            # Tracker dynamics assume regular cadence; do not join across large gaps.
            for tracker in getattr(self.model.predictor, 'trackers', []):
                tracker.reset()
            self.session = uuid.uuid4().hex
        self.last_stamp = stamp_ns
        start = time.perf_counter()
        r = self.model.track(image, persist=True, tracker=self.tracker, device=self.device,
            imgsz=self.imgsz, conf=.1, iou=.5, classes=list(COCO_TO_NATIVE), max_det=MAX_INSTANCES,
            retina_masks=True, rect=False, verbose=False, half=self.device != 'cpu')[0]
        frame = InstanceFrame(stamp_ns, self.session, np.zeros(image.shape[:2], np.uint16),
                              np.empty(0, np.int64), np.empty(0, np.uint16), np.empty(0, np.float32))
        if r.masks is not None and r.boxes.id is not None:
            masks = r.masks.data.cpu().numpy() > .5
            boxes = r.boxes
            frame = mask_frame(masks, boxes.id.int().cpu().tolist(), boxes.cls.int().cpu().tolist(),
                               boxes.conf.cpu().tolist(), stamp_ns, self.session, image.shape[:2],
                               class_scores=r.class_scores.float().cpu().numpy())
            if len(frame.ids):
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape(-1, 3)
                descriptors = []
                for slot in range(len(frame.ids)):
                    at = np.flatnonzero(frame.index.reshape(-1) == slot+1)
                    if len(at) > 1024:
                        at = at[np.linspace(0, len(at)-1, 1024, dtype=int)]
                    hist = np.concatenate([np.histogram(hsv[at, c], bins=8, range=(0, limit))[0]
                                           for c,limit in enumerate([180, 256, 256])]).astype(np.float32)+1e-3
                    descriptors.append(hist/hist.sum())
                frame.appearance = np.stack(descriptors)
        if self.visual_features is not None:
            features=self.visual_features.pool(frame)
            frame.visual_features=np.asarray([features[int(t)][0] for t in frame.ids],np.float32).reshape(-1,64)
            frame.visual_feature_valid=np.asarray([features[int(t)][1] for t in frame.ids],bool)
        return frame, dict(elapsed_ms=(time.perf_counter()-start)*1000, instances=len(frame.ids), reset=reset,
                           model_stages_ms=dict(r.speed))
