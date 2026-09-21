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
    return t if torch.is_tensor(t) else None
def mk(name):
    def h(mod,i,o):
        t=grab(o)
        if t is None: return
        rec.setdefault(name,[]).append(t.detach().float().cpu().numpy().copy())
        order.append(name)
    return h
for name,mod in bb.named_modules():
    if name: mod.register_forward_hook(mk(name))
# also capture the serialized order/inverse of the pooled points
snap={}
import pointcept.models.point_transformer_v3.point_transformer_v3m1_base as M
orig=M.SerializedPooling.forward
cnt=[0]
def patched(self, point):
    out=orig(self,point)
    k="pool%d"%(cnt[0]%4); cnt[0]+=1
    snap.setdefault(k+"_order",[]).append(out.serialized_order.cpu().numpy().copy())
    snap.setdefault(k+"_code",[]).append(out.serialized_code.cpu().numpy().copy())
    snap.setdefault(k+"_gc",[]).append(out.grid_coord.cpu().numpy().copy())
    snap.setdefault(k+"_feat",[]).append(out.feat.detach().float().cpu().numpy().copy())
    return out
M.SerializedPooling.forward=patched
for _ in range(2): seg.segment(p)
seen=[]
for name in order[:len(order)//2]:
    v=rec[name]; half=len(v)//2
    a,b=v[0],v[half]
    if a.shape!=b.shape: continue
    d=float(np.abs(a-b).max())
    seen.append((name,a.shape,d))
    if d>0: break
for name,sh,d in seen[-8:]:
    print("  %-46s %-16s maxdiff=%.3e"%(name,str(sh),d))
print()
for k in sorted(snap):
    v=snap[k]
    if len(v)<2: continue
    a,b=v[0],v[len(v)//2]
    print("  %-16s equal=%s  shape=%s"%(k,np.array_equal(a,b),a.shape))
