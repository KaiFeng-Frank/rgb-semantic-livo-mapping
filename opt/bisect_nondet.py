import os,sys,glob,numpy as np
os.environ["PTV3_SHUFFLE"]="0"; os.environ["PTV3_HALF"]="0"
sys.path.insert(0,"/data/livo_sem/src")
import torch, ptv3_worker as W
D="/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/"
p=np.fromfile(sorted(glob.glob(D+"*.bin"))[0],dtype=np.float32).reshape(-1,4)
seg=W.Segmenter(); bb=seg.model.backbone
order=[]; rec={}
def grab(o):
    t=o
    if hasattr(t,'feat'): t=t.feat
    if hasattr(t,'features'): t=t.features
    if isinstance(t,dict):
        for v in t.values():
            if torch.is_tensor(v): return v
        return None
    return t if torch.is_tensor(t) else None
def mk(name):
    def h(mod,i,o):
        t=grab(o)
        if t is None: return
        rec.setdefault(name,[]).append(t.detach().float().cpu().numpy().copy())
        if name not in order: order.append(name)
    return h
hs=[]
for name,mod in bb.named_modules():
    if name=="" : continue
    if len(list(mod.children()))==0 or name.count(".")<=2:
        hs.append(mod.register_forward_hook(mk(name)))
for _ in range(2): seg.segment(p)
first=None
for name in order:
    v=rec.get(name,[])
    if len(v)<2: continue
    half=len(v)//2
    a,b=v[0],v[half]
    if a.shape!=b.shape: continue
    d=np.abs(a-b).max()
    if d>0 and first is None:
        first=name
        print("FIRST DIVERGENCE at %-45s maxdiff=%.3e"%(name,d))
    if name.count(".")<=1 or first==name:
        print("  %-48s shape=%-16s maxdiff=%.3e"%(name,str(a.shape),d))
print("total hooked:",len(order))
