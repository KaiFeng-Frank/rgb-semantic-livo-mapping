"""Experimental causal SAM 2 masks with matched current-only prompts.

Uses the official pretrained SAM2Base unchanged. A detection supplies each box
and identity; no annotation, future image or semantic posterior is an input.
Native SAM 2 recurrent mask features ARE retained (unlike raw-class memory).
This module has no deployment admission; latency/accuracy must both be checked.
"""
from collections import defaultdict
import hashlib
from pathlib import Path
import sys,time
import numpy as np
import torch


class SAM2Pair:
    def __init__(self,source,weights,image_size=1024,official_compile=False):
        if image_size not in (512,1024):raise ValueError('Only the two explicitly recorded inference sizes are supported')
        sys.path.insert(0,str(source))
        from sam2.build_sam import build_sam2
        from sam2.utils.transforms import SAM2Transforms
        overrides=[f'++model.image_size={image_size}']
        if official_compile:
            # Keep build_sam2's exact postprocessing settings. The video builder
            # would additionally binarize prompted memories and fill holes.
            # The official VOS subclass only adds compilation/output clones to
            # the low-level methods used here; no predictor session is involved.
            overrides += ['++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS',
                          '++model.compile_image_encoder=True']
        self.model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(weights),apply_postprocessing=True,
            hydra_overrides_extra=overrides)
        if official_compile:
            # RoPE initializes for the official 1024 grid. At the frozen 512
            # size, first use would otherwise persist a tensor CREATED inside
            # the compiled attention graph, whose storage can later be reused.
            # Precompute the identical native frequencies in owned eager memory.
            from sam2.modeling.sam.transformer import RoPEAttention
            for module in self.model.modules():
                if isinstance(module,RoPEAttention):
                    side=image_size//self.model.backbone_stride
                    module.freqs_cis=module.compute_cis(end_x=float(side),end_y=float(side)).to(self.model.device).clone()
            # On the installed Torch build, a later compiled prompt encoder can
            # recycle the memory-attention CUDAGraph output before the mask head
            # reads it. Own this intermediate just as official VOS owns its
            # image/head/memory outputs. Clone is OUTSIDE the compiled graph.
            compiled_attention=self.model.memory_attention.forward
            def owned_attention(*args,**kwargs):
                return compiled_attention(*args,**kwargs).clone()
            self.model.memory_attention.forward=owned_attention
        self.transforms=SAM2Transforms(self.model.image_size,mask_threshold=0.)
        self.histories={};self.last_seen={};self.stamp=None;self.session=None;self.frame=0
        self.meta=dict(weights_sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),num_maskmem=self.model.num_maskmem,image_size=image_size,
            max_tracks=128,absent_ttl_s=2.,max_batch=4,object_pointer_history=self.model.max_obj_ptrs_in_encoder,
            prompts='same current detector-mask boxes for both arms',history='official recurrent mask features; no semantic-logit feedback',
            model='official SAM2.1 Hiera Tiny, pretrained, no fine tuning',
            precision='bfloat16 autocast, official VOS compilation' if official_compile else 'bfloat16 autocast, uncompiled')

    def _batched_history(self,keys):
        result={}
        first=self.histories[keys[0]]
        for kind in ['cond_frame_outputs','non_cond_frame_outputs']:
            result[kind]={}
            for f in first[kind]:
                rows=[self.histories[k][kind][f] for k in keys]
                result[kind][f]=dict(obj_ptr=torch.cat([r['obj_ptr'] for r in rows]),
                    maskmem_features=(torch.cat([r['maskmem_features'] for r in rows]) if rows[0]['maskmem_features'] is not None else None),
                    maskmem_pos_enc=([torch.cat([r['maskmem_pos_enc'][j] for r in rows]) for j in range(len(rows[0]['maskmem_pos_enc']))] if rows[0]['maskmem_pos_enc'] is not None else None))
        return result

    def _retain(self,key,result,slot,is_new):
        # Clone per object so an expired object's shared batch storage is freed.
        row=dict(obj_ptr=result['obj_ptr'][slot:slot+1].detach().clone(),
            maskmem_features=result['maskmem_features'][slot:slot+1].detach().clone(),
            maskmem_pos_enc=[x[slot:slot+1].detach().clone() for x in result['maskmem_pos_enc']])
        state=self.histories.setdefault(key,dict(cond_frame_outputs={},non_cond_frame_outputs={}))
        kind='cond_frame_outputs' if is_new else 'non_cond_frame_outputs';state[kind][self.frame]=row
        for f in list(state['non_cond_frame_outputs']):
            if f<=self.frame-self.model.max_obj_ptrs_in_encoder:del state['non_cond_frame_outputs'][f]
            elif f<=self.frame-(self.model.num_maskmem-1):
                state['non_cond_frame_outputs'][f]['maskmem_features']=None
                state['non_cond_frame_outputs'][f]['maskmem_pos_enc']=None

    @torch.inference_mode()
    def step(self,image,instances):
        if self.stamp is not None and instances.stamp_ns<=self.stamp:raise ValueError('Noncausal camera input')
        if image.shape[:2]!=instances.index.shape or image.dtype!=np.uint8:raise ValueError('Invalid image')
        if len(instances.ids)>128:raise ValueError('Object capacity exceeded')
        if self.session!=instances.session or self.stamp is not None and instances.stamp_ns-self.stamp>750000000:
            self.histories.clear();self.last_seen.clear();self.frame=0
        self.session=instances.session;self.stamp=instances.stamp_ns
        for key in list(self.histories):
            if self.stamp-self.last_seen[key]>2000000000:del self.histories[key];del self.last_seen[key]
        supported=[s for s in range(len(instances.ids)) if np.any(instances.index==s+1)]
        keys=[int(instances.ids[s]) for s in supported]
        for key in keys:self.last_seen[key]=self.stamp
        # Current detections have priority; evict least recently observed others.
        for key in sorted(self.histories,key=lambda k:self.last_seen[k]):
            if len(set(self.histories)|set(keys))<=128:break
            if key not in keys:del self.histories[key];del self.last_seen[key]
        empty=np.zeros((len(instances.ids),*image.shape[:2]),bool)
        if not keys:self.frame+=1;return dict(current=empty,temporal=empty),dict(objects=0,encoder_ms=0.,current_ms=0.,temporal_ms=0.)
        boxes=[]
        for slot in supported:
            y,x=np.nonzero(instances.index==slot+1)
            if not len(x):raise ValueError('Empty detector instance')
            boxes.append([x.min(),y.min(),x.max()+1,y.max()+1])
        box=torch.as_tensor(np.asarray(boxes),dtype=torch.float32,device='cuda')
        prompts=self.transforms.transform_boxes(box,normalize=True,orig_hw=image.shape[:2])
        torch.cuda.synchronize();start=time.perf_counter()
        tensor=self.transforms(image[:,:,::-1].copy()).unsqueeze(0).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            backbone=self.model.forward_image(tensor)
            _,features,positions,sizes=self.model._prepare_backbone_features(backbone)
        torch.cuda.synchronize();encoder_ms=(time.perf_counter()-start)*1000
        masks={a:[None]*len(keys) for a in ['current','temporal']};cost={};read_objects=0
        for arm in masks:
            torch.cuda.synchronize();start=time.perf_counter();groups=defaultdict(list)
            for slot,key in enumerate(keys):
                state=self.histories.get(key) if arm=='temporal' else None
                signature=(tuple(state['cond_frame_outputs']),tuple(state['non_cond_frame_outputs'])) if state else None
                groups[signature].append(slot)
            for signature,slots in groups.items():
                for begin in range(0,len(slots),4):
                    chosen=slots[begin:begin+4];batch_keys=[keys[s] for s in chosen];n=len(chosen)
                    state=self._batched_history(batch_keys) if signature is not None else dict(cond_frame_outputs={},non_cond_frame_outputs={})
                    assert all(f<self.frame for d in state.values() for f in d),'Future/self memory'
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        result=self.model.track_step(frame_idx=self.frame,is_init_cond_frame=signature is None,
                            current_vision_feats=[x.expand(-1,n,-1) for x in features],current_vision_pos_embeds=[x.expand(-1,n,-1) for x in positions],
                            feat_sizes=sizes,point_inputs=dict(point_coords=prompts[chosen],point_labels=torch.tensor([[2,3]],device='cuda').expand(n,-1)),
                            mask_inputs=None,output_dict=state,num_frames=self.frame+1,run_mem_encoder=arm=='temporal')
                    full=torch.nn.functional.interpolate(result['pred_masks_high_res'].float(),image.shape[:2],mode='bilinear',align_corners=False)[:,0]>0
                    for b,slot in enumerate(chosen):masks[arm][slot]=full[b].cpu().numpy()
                    if arm=='temporal':
                        read_objects+=n if signature is not None else 0
                        for b,key in enumerate(batch_keys):self._retain(key,result,b,signature is None)
            torch.cuda.synchronize();cost[arm]=(time.perf_counter()-start)*1000
        self.frame+=1
        outputs={a:empty.copy() for a in masks}
        for a,v in masks.items():outputs[a][supported]=np.stack(v)
        return outputs,dict(objects=len(keys),input_objects=len(instances.ids),encoder_ms=encoder_ms,current_ms=cost['current'],temporal_ms=cost['temporal'],
            objects_reading_history=read_objects,tracks=len(self.histories),memory_entries=sum(sum(len(d) for d in s.values()) for s in self.histories.values()))
