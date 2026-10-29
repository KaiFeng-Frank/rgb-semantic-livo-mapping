"""One bounded instance-query reader, shared by matched temporal/current arms.

Inputs are raw frozen encoder observations. Neither GT identity nor a recursively
enhanced feature enters memory. This implements a small adaptation of query-memory
ideas, not an AutoSeg3D reproduction or a claim of architectural novelty.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

INPUT_DIM = 320


class MemoryHead(nn.Module):
    def __init__(self, mean, scale, width=64):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        self.encoder=nn.Sequential(nn.Linear(INPUT_DIM,width),nn.LayerNorm(width),nn.GELU())
        self.identity=nn.Linear(width,32)
        self.age=nn.Sequential(nn.Linear(5,width),nn.Tanh())
        self.attention=nn.MultiheadAttention(width,4,batch_first=True,dropout=0.)
        self.norm=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,width))
        self.delta=nn.Linear(width,16)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)

    def encode(self,x):
        return self.encoder(((x-self.mean)/self.scale).clamp(-8,8))

    def embeddings(self,x):
        return F.normalize(self.identity(self.encode(x)),dim=-1,eps=1e-6)

    def forward(self,x,memory,meta,mask):
        """x[B,320], memory[B,K,320], meta[B,K,5], mask[B,K].

        A current token is always present, including in the matched control.
        All modules and parameters are retained in both arms; only past content
        is unavailable in current-only. Training and inference have one path.
        """
        q=self.encode(x)
        hist=self.encode(memory)+self.age(meta)
        values=torch.cat([q[:,None],hist],dim=1)
        padding=torch.cat([torch.zeros((len(x),1),dtype=torch.bool,device=x.device),~mask],dim=1)
        read,_=self.attention(q[:,None],values,values,key_padding_mask=padding,need_weights=False)
        h=self.norm(q+read[:,0]);h=h+self.ff(h)
        return self.delta(h)


def batch_inputs(model,observations,histories,now,temporal=True):
    n=len(observations)
    device=model.mean.device
    x=torch.as_tensor(np.stack([o['x'] for o in observations]),dtype=torch.float32,device=device)
    memory=np.zeros((n,6,INPUT_DIM),np.float32)
    meta=np.zeros((n,6,5),np.float32)
    mask=np.zeros((n,6),bool)
    for i,(o,records) in enumerate(zip(observations,histories)):
        for j,r in enumerate(records[:6] if temporal else []):
            age=(now-r['stamp'])*1e-9
            if not 0 < age <= 8.:raise ValueError('Noncausal or expired memory')
            memory[i,j]=r['x']
            # Relative positions are in the object's predicted current frame.
            # A single motion estimate is only a coarse center alignment; the
            # network also sees relative age/extent to handle shape uncertainty.
            predicted=np.asarray(r['center'])+np.asarray(r.get('velocity',[0,0,0]))*age
            delta=(predicted-np.asarray(o['center']))/10.
            meta[i,j]=[age/8.,*delta,float(r.get('quality',0.))]
            mask[i,j]=True
    return (x,torch.as_tensor(memory,device=device),torch.as_tensor(meta,device=device),
            torch.as_tensor(mask,device=device))


def point_logits(observation,delta):
    raw=torch.as_tensor(observation['raw_probabilities'],dtype=delta.dtype,device=delta.device)
    return raw.clamp_min(1e-5).log()+delta
