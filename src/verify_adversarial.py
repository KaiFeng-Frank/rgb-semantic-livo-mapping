#!/usr/bin/env python3
"""Independent adversarial check: does the 'nothing helps, it is pure domain gap'
claim survive a MULTI-FRAME test on both a held-out sequence and the main one?"""
import sys, glob, numpy as np
sys.path.insert(0, "/data/livo_sem/src")
from ptv3_infer import (PTv3Segmenter, read_bin, sk_labels_to_coarse,
                        NUSC16_TO_COARSE, COARSE, IGNORE, NUSCENES16)
D = "/data/livo_sem/data"
SEQ = {"04": (f"{D}/raw/2011_09_30/2011_09_30_drive_0016_sync/velodyne_points/data",
              f"{D}/odometry/dataset/sequences/04/labels"),
       "07": (f"{D}/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data",
              f"{D}/odometry/dataset/sequences/07/labels")}
def frames(seq, n, stride, off=0):
    sd, ld = SEQ[seq]
    s = sorted(glob.glob(sd+"/*.bin")); l = sorted(glob.glob(ld+"/*.label"))
    idx = list(range(off, min(len(s), len(l)), stride))[:n]
    return [(s[i], l[i]) for i in idx]
seg = PTv3Segmenter()
seg.warmup()
K = len(COARSE)
def run(fl, tag, iscale=1.0, zsh=0.0, yaw=0.0):
    seg.intensity_scale = iscale; seg.z_shift = zsh
    inter=np.zeros(K,np.int64); pc=np.zeros(K,np.int64); gc=np.zeros(K,np.int64)
    ok=tot=0; hist=np.zeros(16,np.int64)
    for sp, gp in fl:
        p = read_bin(sp).copy(); g = sk_labels_to_coarse(gp)
        if yaw:
            th=np.deg2rad(yaw); R=np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]],np.float32)
            p[:,:3] = p[:,:3] @ R.T
        lb,_ = seg.segment(p); hist += np.bincount(lb.astype(np.int64), minlength=16)
        pr = NUSC16_TO_COARSE[lb.astype(np.int64)]; m = g != IGNORE
        ok += int((pr[m]==g[m]).sum()); tot += int(m.sum())
        for k in range(K):
            pk=(pr==k)&m; gk=(g==k)&m
            inter[k]+=int((pk&gk).sum()); pc[k]+=int(pk.sum()); gc[k]+=int(gk.sum())
    ious=[inter[k]/max(pc[k]+gc[k]-inter[k],1) for k in range(K) if gc[k]>0]
    road=C_road=COARSE.index("road")
    rr = inter[road]/max(gc[road],1)
    riou = inter[road]/max(pc[road]+gc[road]-inter[road],1)
    print("%-34s acc=%6.2f%% mIoU=%6.2f%% | road recall=%6.2f%% road IoU=%6.2f%% | top: %s"
          % (tag, 100*ok/max(tot,1), 100*np.mean(ious), 100*rr, 100*riou,
             " ".join("%s=%.0f%%"%(NUSCENES16[i],100*hist[i]/hist.sum()) for i in np.argsort(-hist)[:3])))
    return 100*ok/max(tot,1), 100*np.mean(ious)
CFG = [("base                    ",1.0,0.0,0.0),
       ("intens0.5               ",0.5,0.0,0.0),
       ("z-0.5                   ",1.0,-0.5,0.0),
       ("z-1.0                   ",1.0,-1.0,0.0),
       ("z-1.5                   ",1.0,-1.5,0.0),
       ("z-1.0+intens0.5         ",0.5,-1.0,0.0),
       ("z-1.0+yaw90             ",1.0,-1.0,90.0),
       ("z-1.0+intens0.5+yaw90   ",0.5,-1.0,90.0)]
for seq, n, st in (("04",10,25),("07",12,90)):
    print("\n==== seq %s : %d frames ====" % (seq,n))
    fl = frames(seq,n,st)
    for tag,i,z,y in CFG: run(fl, tag, i, z, y)
