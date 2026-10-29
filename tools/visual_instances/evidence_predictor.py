"""Keep YOLO class scores through NMS, empty-mask removal and tracker indexing.

Uses the public Ultralytics 8.3.228 predictor/NMS interfaces. Scores are independent
sigmoids, NOT a categorical posterior. No additional model forward is performed.
"""
import torch
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import nms, ops


class EvidenceResults(Results):
    def __init__(self, *args, class_scores=None, feats=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_scores = class_scores
        self.feats = feats
        self._keys = (*self._keys, 'class_scores', 'feats')

    def new(self):
        return EvidenceResults(orig_img=self.orig_img, path=self.path, names=self.names,
                               speed=self.speed)


class EvidencePredictor(SegmentationPredictor):
    def postprocess(self, preds, img, orig_imgs):
        prediction = preds[0]
        nc = len(self.model.names)
        if nc != 80 or prediction.ndim != 3 or getattr(self.model, 'end2end', False):
            raise ValueError('Class evidence adapter requires COCO YOLO11-seg dense output')
        # Clone before NMS: its box coordinate conversion may modify the tensor.
        scores = prediction[:, 4:4+nc].transpose(1, 2).clone()
        detections, indices = nms.non_max_suppression(prediction, self.args.conf,
            self.args.iou, self.args.classes, self.args.agnostic_nms,
            max_det=self.args.max_det, nc=nc, return_idxs=True)
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        normalized=[]
        for idx in indices:
            idx=idx.reshape(-1)
            if idx.numel()==0:idx=idx.to(torch.long)
            elif idx.dtype not in (torch.int32,torch.int64):raise ValueError('Non-integer NMS evidence indices')
            normalized.append(idx)
        # Follow the pinned official DetectionPredictor.get_obj_feats path.
        # The segmentation adapter must preserve the SAME NMS/mask row mapping.
        feature_maps=getattr(self,'_feats',None)
        object_features=self.get_obj_feats(feature_maps,normalized) if feature_maps is not None else [None]*len(normalized)
        results = []
        for det, idx, original, path, proto, score, features in zip(
                detections, normalized, orig_imgs, self.batch[0], protos, scores,object_features):
            result = super().construct_result(det, img, original, path, proto)
            # Ultralytics returns a default-float empty index tensor when NMS
            # finds no detections (including black-image warmup). Nonempty
            # indices must remain integers; never round malformed evidence.
            selected = score[idx]
            if features is not None and not isinstance(features,torch.Tensor):
                features=score.new_empty((0,min(f.shape[1] for f in feature_maps)))
            if len(result.boxes) != len(det):
                # construct_result scales det in-place, then may drop empty masks.
                match = (result.boxes.data[:, None, :6] == det[None, :, :6]).all(-1)
                if not torch.all(match.sum(1) == 1):
                    raise ValueError('Ambiguous mask-to-class-evidence correspondence')
                keep=match.to(torch.int64).argmax(1)
                selected = selected[keep]
                if features is not None:features=features[keep]
            results.append(EvidenceResults(orig_img=original, path=path, names=self.model.names,
                boxes=result.boxes.data, masks=result.masks.data if result.masks is not None else None,
                class_scores=selected,feats=features))
        return results
