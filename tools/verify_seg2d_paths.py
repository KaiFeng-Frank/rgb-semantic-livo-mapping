import os, sys, time
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path.insert(0,"/data/livo_sem/src")
import numpy as np, torch
torch.set_num_threads(4)
from seg2d_infer import (Seg2DSegmenter, SCALE_PRESETS, CITYSCAPES19,
                         CITYSCAPES19_TO_COARSE, COARSE, COARSE_VOID)
MK = sys.argv[1] if len(sys.argv) > 1 else "mask2former"
from kitti_calib import KittiCalib

PNG="/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/image_02/data/0000000000.png"
BIN="/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/0000000000.bin"

print("=== SCALE_PRESETS -> padded input size ===")
for k,v in SCALE_PRESETS.items():
    w,h=int(round(1226*v)),int(round(370*v))
    print("  %-11s s=%.4f  %4dx%-4d -> padded %4dx%-4d  (%.2f Mpix)"
          %(k,v,w,h,w+(-w)%32,h+(-h)%32,(w+(-w)%32)*(h+(-h)%32)/1e6))

print("\n=== Cityscapes-19 -> coarse-13 LUT ===")
for i,n in enumerate(CITYSCAPES19):
    c=CITYSCAPES19_TO_COARSE[i]
    print("  %2d %-14s -> %s"%(i,n,"VOID (unscorable)" if c==COARSE_VOID else COARSE[c]))
cov=set(COARSE[c] for c in CITYSCAPES19_TO_COARSE if c!=COARSE_VOID)
print("  coarse classes REACHABLE by the 2D arm: %d/%d -> %s"%(len(cov),len(COARSE),sorted(cov)))
print("  UNREACHABLE: %s"%sorted(set(COARSE)-cov))

# --- geometry path: non-integer scale, pad, crop, native resample -------------
print("\n=== path: scale=d2_maxsize (1.670), tta=none ===")
s=Seg2DSegmenter(MK, device="cpu",scale="d2_maxsize",tta="none")
t0=time.time(); lab,conf=s.segment_image(PNG); print("  %.1fs shapes"%(time.time()-t0),lab.shape,conf.shape,lab.dtype,conf.dtype)
assert lab.shape==(370,1226) and conf.shape==(370,1226)
assert lab.max()<19 and 0.0<=conf.min() and conf.max()<=1.0
bot=lab[3*370//4:].ravel(); print("  bottom quarter top class:",CITYSCAPES19[np.bincount(bot,minlength=19).argmax()])

# --- sample_points must agree with segment_image at pixel centres ------------
uv=np.stack(np.meshgrid(np.arange(1226)+0.0,np.arange(370)+0.0),-1).reshape(-1,2)[::977]
lp,cp=s.sample_points(PNG,uv)
ref=lab[uv[:,1].astype(int),uv[:,0].astype(int)]
print("  sample_points vs segment_image at integer centres: %d/%d identical"%( (lp==ref).sum(),len(ref)))
assert (lp==ref).all()

# --- real projected LiDAR points ---------------------------------------------
calib=KittiCalib("/data/livo_sem/data/raw/2011_09_30")
pts=np.fromfile(BIN,np.float32).reshape(-1,4)
uv,depth,mask=calib.project_velo_to_cam2(pts[:,:3],img_shape=(370,1226),return_mask=True)
lp,cp=s.sample_points(PNG,uv)
co=s.to_coarse(lp)
print("  projected %d/%d pts (%.1f%%); labelled %d; sky-VOID %d (%.2f%%); mean conf %.3f"
      %(mask.sum(),len(pts),100*mask.sum()/len(pts),len(lp),(co==COARSE_VOID).sum(),
        100.0*(co==COARSE_VOID).mean(),cp.mean()))
h=np.bincount(co,minlength=COARSE_VOID+1)
for i in np.argsort(-h):
    if h[i]: print("     %-14s %6d (%.1f%%)"%(("VOID/sky" if i==COARSE_VOID else COARSE[i]),h[i],100*h[i]/h.sum()))

# --- TTA paths ---------------------------------------------------------------
print("\n=== path: tta=flip, scale=native ===")
s2=Seg2DSegmenter(MK, device="cpu",scale="native",tta="flip",verbose=False)
t0=time.time(); l2,c2=s2.segment_image(PNG); print("  %.1fs ok"%(time.time()-t0),l2.shape,"mean conf %.3f"%c2.mean())
print("=== path: tta=ms_flip, scale=native, ms=(0.75,1.0,1.25) ===")
s3=Seg2DSegmenter(MK, device="cpu",scale="native",tta="ms_flip",verbose=False)
t0=time.time(); l3,c3=s3.segment_image(PNG); print("  %.1fs ok"%(time.time()-t0),l3.shape,"mean conf %.3f"%c3.mean())
ln,_=Seg2DSegmenter(MK, device="cpu",scale="native",tta="none",verbose=False).segment_image(PNG)
print("  label agreement  none/flip %.2f%%   none/ms_flip %.2f%%   flip/ms_flip %.2f%%"
      %(100*np.mean(ln==l2),100*np.mean(ln==l3),100*np.mean(l2==l3)))
print("\nALL PATHS OK")
