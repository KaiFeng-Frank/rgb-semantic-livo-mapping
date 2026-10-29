"""Causal DEVA adapter for the frozen current YOLO instance-mask stream.

Experimental only: native detection merging stores its own recurrent predictions.
This is not the previous raw-probability, feedback-free memory. No GT is an input.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from .geometry import InstanceFrame, COCO_TO_NATIVE


@contextmanager
def checkpoint_only_initialization(resnet):
    """Skip redundant ImageNet downloads; strict whole-model load follows."""
    old = {name:getattr(resnet,name) for name in ('resnet18','resnet50')}
    try:
        for name, fn in old.items():
            def build(*args,_fn=fn,**kwargs):
                kwargs['pretrained']=False
                return _fn(*args,**kwargs)
            setattr(resnet,name,build)
        yield
    finally:
        for name,fn in old.items():setattr(resnet,name,fn)


def input_tensor(image,device,short_side=480):
    """RGB/ImageNet normalization and bilinear min-side resize, on GPU."""
    x=torch.from_numpy(image[:,:,::-1].copy()).to(device=device).permute(2,0,1).float()/255.
    mean=x.new_tensor([.485,.456,.406])[:,None,None]
    std=x.new_tensor([.229,.224,.225])[:,None,None]
    x=(x-mean)/std
    scale=short_side/min(image.shape[:2])
    shape=tuple(int(n*scale) for n in image.shape[:2])
    return F.interpolate(x[None],shape,mode='bilinear',align_corners=False)[0]


class DEVAPair:
    def __init__(self,source,weights,config):
        sys.path.insert(0,str(source))
        from deva.model.network import DEVA
        from deva.model import resnet
        from deva.inference.inference_core import DEVAInferenceCore
        from deva.inference.image_feature_store import ImageFeatureStore
        from deva.inference.object_info import ObjectInfo
        from deva.utils.tensor_utils import pad_divide_by
        self.config=dict(config)
        if self.config['temporal_setting']!='online' or self.config['detection_every']!=1:
            raise ValueError('This adapter uses every current detection without future voting')
        with checkpoint_only_initialization(resnet):self.model=DEVA(self.config)
        state=torch.load(weights,map_location='cpu',weights_only=True)
        self.model.load_state_dict(state,strict=True)
        self.model=self.model.cuda().eval()
        self.ObjectInfo=ObjectInfo;self.pad=pad_divide_by

        class SharedFrameStore(ImageFeatureStore):
            def delete(self,index):
                # Both arms read the same current encoding. Caller releases it
                # after the paired step; no future image can enter this store.
                pass
            def release(self,index):super().delete(index)

        class OnlineCore(DEVAInferenceCore):
            reads=0
            def add_to_temporary_buffer(self,*args,**kwargs):raise RuntimeError('Future-frame buffering forbidden')
            def vote_in_temporary_buffer(self,*args,**kwargs):raise RuntimeError('Semi-online voting forbidden')
            def _segment(self,*args,**kwargs):
                if self.frame_buffer or self.last_mem_ti>=self.curr_ti:raise RuntimeError('Noncausal memory read')
                self.reads+=1
                return super()._segment(*args,**kwargs)

        self.Core=OnlineCore
        self.store=SharedFrameStore(self.model)
        self.core=None;self.stamp=None;self.input_session=None;self.frame=0
        self.output_session=uuid.uuid4().hex
        self.meta=dict(source='official DEVA propagation, strict checkpoint load',
            checkpoint_sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
            precision='float16 autocast, no compilation',config=self.config,
            memory='native recurrent mask/sensory and compressed long-term memory; prediction feedback',
            controls='same image encoding and detections, fresh current-only core versus persistent strictly causal core',
            attribution='pretrained propagation+native matching/voting, no new training',
            limitation='Native category/score lists and historical ID set are not yet bounded for an indefinitely running node')

    def reset(self):
        self.core=self.Core(self.model,self.config,image_feature_store=self.store)
        self.core.enabled_long_id()
        self.stamp=None;self.input_session=None;self.frame=0
        self.output_session=uuid.uuid4().hex

    def _output(self,value,manager,frame,shape):
        # Follow the official result writer: resize returned scores before argmax.
        idx=F.interpolate(value[:,None].float(),shape,mode='bilinear',align_corners=False)[:,0].argmax(0)
        idx=idx.to(torch.int32).cpu().numpy().astype(np.uint16)
        objects=[manager.tmp_id_to_obj[i] for i in range(1,manager.num_obj+1)]
        ids=np.asarray([o.id for o in objects],np.int64)
        labels=np.asarray([o.vote_category_id() for o in objects],np.uint16)
        scores=np.asarray([o.vote_score() for o in objects],np.float32)
        classes=np.zeros((len(ids),80),np.float32)
        back={v:k for k,v in COCO_TO_NATIVE.items()}
        for i,c in enumerate(labels):classes[i,back[int(c)]]=1.
        return InstanceFrame(frame.stamp_ns,self.output_session,idx,ids,labels,scores,class_scores=classes)

    @torch.inference_mode()
    def step(self,image,frame):
        if self.stamp is not None and frame.stamp_ns<=self.stamp:raise ValueError('Non-increasing image time')
        if image.dtype!=np.uint8 or image.shape[:2]!=frame.index.shape:raise ValueError('Image/mask mismatch')
        if self.core is None or frame.session!=self.input_session or self.stamp is not None and frame.stamp_ns-self.stamp>750000000:
            self.reset()
        self.stamp=frame.stamp_ns;self.input_session=frame.session
        torch.cuda.synchronize();start=time.perf_counter()
        x=input_tensor(image,'cuda',self.config['size'])
        index=F.interpolate(torch.from_numpy(frame.index.astype(np.int32)).to('cuda')[None,None].float(),
            x.shape[-2:],mode='nearest')[0,0].long()
        supported=torch.unique(index).cpu().tolist();supported=[s for s in supported if s]
        padded,_=self.pad(x,16)
        with torch.autocast('cuda',dtype=torch.float16):
            self.store.get_ms_features(self.frame,padded[None])
        torch.cuda.synchronize();encode_ms=(time.perf_counter()-start)*1000
        output={};cost={};before_reads=self.core.reads
        for arm in ('current','temporal'):
            core=self.Core(self.model,self.config,image_feature_store=self.store) if arm=='current' else self.core
            if arm=='current':core.enabled_long_id()
            segments=[self.ObjectInfo(id=int(s),category_id=int(frame.labels[s-1]),isthing=True,
                                     score=float(frame.scores[s-1])) for s in supported]
            torch.cuda.synchronize();t=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16):
                scores=core.incorporate_detection(x,index,segments,image_ti_override=self.frame,incremental=False)
            output[arm]=self._output(scores,core.object_manager,frame,image.shape[:2])
            torch.cuda.synchronize();cost[arm]=(time.perf_counter()-t)*1000
            if core.frame_buffer:raise RuntimeError('Unexpected temporal buffer')
            del scores
            if arm=='current':del core
        self.store.release(self.frame)
        assert len(self.store)==0
        self.frame+=1
        torch.cuda.synchronize();total=(time.perf_counter()-start)*1000
        def tokens(bank):return sum(int(v.shape[-1]) for v in bank.key.values())
        timing=dict(wall_both_arms_ms=total,prep_encode_ms=encode_ms,current_ms=cost['current'],temporal_ms=cost['temporal'],
            deploy_component_ms=total-cost['current'],input_objects=len(supported),objects=self.core.object_manager.num_obj,
            memory_reads=self.core.reads-before_reads,buffered_future_frames=len(self.core.frame_buffer),
            work_tokens=tokens(self.core.memory.work_mem),long_tokens=tokens(self.core.memory.long_mem),
            metadata_observations=sum(len(o.category_ids) for o in self.core.object_manager.obj_to_tmp_id))
        return output,timing
