"""Use native cold-history inputs each frame, preserving temporal metric state.

This is a same-weight input intervention, not a trained single-frame model.
We keep the original memory attention, uncertainty head and output head. Only
inference history is reset to register_scene_buffer's initial values. Geometry
and GT/prediction bookkeeping for mSTCV continue normally across the scene.
"""
import torch


class ColdHistory:
    ZERO_FIELDS = ('global_feats', 'global_valid_masks', 'global_vis_cnt',
                   'global_pred_distribs')
    METRIC_FIELDS = ('global_gt', 'global_mask_gt', 'global_pred', 'global_mask_pred')

    def __init__(self, model):
        self.calls = 0
        self.attention_calls = 0
        self.last = {}
        native_update = model.update_global
        native_attention = model.temporal_encoder.forward

        def metric_versions():
            return [(name, id(tensor), tensor._version)
                    for name in self.METRIC_FIELDS
                    for tensor in (getattr(model, name, None) or []) if tensor is not None]

        def update(current, metadata, bda, previous_flow):
            before = metric_versions()
            with torch.no_grad():
                for name in self.ZERO_FIELDS:
                    for tensor in (getattr(model, name, None) or []):
                        if tensor is not None:
                            tensor.zero_()
                for tensor in (getattr(model, 'global_au', None) or []):
                    if tensor is not None:
                        tensor.fill_(1)
            model.global_frame_cnt = 0
            model.gt_flow = None  # upstream name; actually the preceding predicted flow
            assert before == metric_versions(), 'Intervention modified temporal metric state'
            self.calls += 1
            self.last = dict(reset_inference_history=True, metric_state_untouched=True,
                             predicted_history_flow_removed=True)
            return native_update(current, metadata, bda, None)

        def attention(*args, **kwargs):
            # The native empty-memory path uses current features as self-attention
            # keys/values. Verify the actual tensors, not just a configuration flag.
            assert torch.equal(args[0], kwargs['prev_bev'])
            assert kwargs.get('gt_flow') is None
            self.attention_calls += 1
            self.last['native_current_feature_attention_verified'] = True
            return native_attention(*args, **kwargs)

        model.update_global = update
        model.temporal_encoder.forward = attention
