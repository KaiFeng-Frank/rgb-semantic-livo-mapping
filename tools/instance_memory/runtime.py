"""One causal step shared by training, cached evaluation and online integration."""
import time
import numpy as np
import torch

from .memory import ObjectMemory
from .model import batch_inputs,point_logits


class Runtime:
    def __init__(self,model,identity,mode='feature_memory'):
        self.model,self.identity,self.mode=model,identity,mode
        self.memory=ObjectMemory()

    def reset(self):
        self.memory.reset()

    def step(self,frame):
        """Only frame['observations'] and timestamp are accepted, no targets."""
        start=time.perf_counter()
        obs=frame['observations'];stamp=frame['stamp']
        ids=self.memory.candidates(stamp)
        device=self.identity.mean.device
        with torch.no_grad():
            x=torch.as_tensor(np.stack([o['x'] for o in obs]),dtype=torch.float32,device=device) if obs else torch.empty((0,320),device=device)
            # Re-encode raw history with the frozen identity head. Caches cannot
            # silently mix embeddings from different optimizer iterations.
            old=self.memory.representative_inputs(ids)
            old_x=torch.as_tensor(np.stack(old),dtype=torch.float32,device=device) if old else torch.empty((0,320),device=device)
            embeddings=self.identity.embeddings(x).cpu().numpy()
            old_emb=self.identity.embeddings(old_x).cpu().numpy()
        track_ids,histories,stats=self.memory.match(obs,embeddings,ids,old_emb,stamp)
        if obs:
            temporal=self.mode not in ('matched_current','same_memory_head_without_history','uniform_history')
            x,memory,meta,mask=batch_inputs(self.model,obs,histories,stamp,temporal=temporal)
            if self.model.training and temporal:
                # Missing observations happen online; train with causal history
                # dropout, keeping the same inference operation and no hard gate.
                mask=mask & (torch.rand(mask.shape,device=mask.device)>.2)
            delta=self.model(x,memory,meta,mask)
            outputs=[]
            for i,o in enumerate(obs):
                logits=point_logits(o,delta[i])
                if self.mode=='uniform_history' and histories[i]:
                    # Same identity and same single-frame learned head; simple
                    # averaging of RAW past posteriors is the temporal control.
                    old=np.mean([r['raw_mean'] for r in histories[i]],axis=0)
                    old=torch.as_tensor(old,dtype=logits.dtype,device=logits.device)
                    logits=(.5*logits.softmax(-1)+.5*old[None]).clamp_min(1e-8).log()
                outputs.append(logits)
        else:outputs=[]
        self.memory.write(obs,track_ids,stamp)
        stats.update(self.memory.stats(),history_observations=sum(bool(h) for h in histories),
                     elapsed_ms=(time.perf_counter()-start)*1000)
        return outputs,track_ids,stats
