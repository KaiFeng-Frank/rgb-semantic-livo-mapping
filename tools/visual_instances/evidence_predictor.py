"""Keep YOLO class scores through NMS, empty-mask removal and tracker indexing.

Uses the public Ultralytics 8.3.228 predictor/NMS interfaces. Scores are independent
sigmoids, NOT a categorical posterior. No additional model forward is performed.
"""
import torch
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import nms, ops


class EvidenceResults(Results):
    def __init__(self, *args, class_scores=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_scores = class_scores
        self._keys = (*self._keys, 'class_scores')

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
        results = []
        for det, idx, original, path, proto, score in zip(
                detections, indices, orig_imgs, self.batch[0], protos, scores):
            result = super().construct_result(det, img, original, path, proto)
            selected = score[idx.reshape(-1)]
            if len(result.boxes) != len(det):
                # construct_result scales det in-place, then may drop empty masks.
                match = (result.boxes.data[:, None, :6] == det[None, :, :6]).all(-1)
                if not torch.all(match.sum(1) == 1):
                    raise ValueError('Ambiguous mask-to-class-evidence correspondence')
                selected = selected[match.to(torch.int64).argmax(1)]
            results.append(EvidenceResults(orig_img=original, path=path, names=self.model.names,
                boxes=result.boxes.data, masks=result.masks.data if result.masks is not None else None,
                class_scores=selected))
        return results
