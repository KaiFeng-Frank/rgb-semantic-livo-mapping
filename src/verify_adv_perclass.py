#!/usr/bin/env python3
import sys, glob, numpy as np
sys.path.insert(0, "/data/livo_sem/src")
from ptv3_infer import (PTv3Segmenter, read_bin, sk_labels_to_coarse,
                        NUSC16_TO_COARSE, COARSE, IGNORE)
D="/data/livo_sem/data"
sd=f"{D}/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data"
ld=f"{D}/odometry/dataset/sequences/07/labels"
s=sorted(glob.glob(sd+"/*.bin")); l=sorted(glob.glob(ld+"/*.label"))
fl=[(s[i],l[i]) for i in range(0,1101,90)][:12]
seg=PTv3Segmenter(); seg.warmup(); K=len(COARSE)
def run(tag,iscale,zsh):
    seg.intensity_scale=iscale; seg.z_shift=zsh
    inter=np.zeros(K,np.int64); pc=np.zeros(K,np.int64); gc=np.zeros(K,np.int64); ok=tot=0
    for sp,gp in fl:
        p=read_bin(sp); g=sk_labels_to_coarse(gp); lb,_=seg.segment(p)
        pr=NUSC16_TO_COARSE[lb.astype(np.int64)]; m=g!=IGNORE
        ok+=int((pr[m]==g[m]).sum()); tot+=int(m.sum())
        for k in range(K):
            pk=(pr==k)&m; gk=(g==k)&m
            inter[k]+=int((pk&gk).sum()); pc[k]+=int(pk.sum()); gc[k]+=int(gk.sum())
    print("\n### %s   acc=%.2f%%"%(tag,100*ok/tot))
    print("  class             GTpts    predpts   recall     IoU")
    ious=[]
    for k in range(K):
        if gc[k]==0 and pc[k]==0: continue
        u=pc[k]+gc[k]-inter[k]; iou=inter[k]/u if u else 0
        if gc[k]>0: ious.append(iou)
        print("  %-14s%9d%10d%9.2f%%%8.2f%%"%(COARSE[k],gc[k],pc[k],100*inter[k]/max(gc[k],1),100*iou))
    print("  mIoU over %d GT classes = %.2f%%"%(len(ious),100*np.mean(ious)))
run("DEFAULT (shipped: intens=1.0 z=0.0)",1.0,0.0)
run("intens=0.5 z=-1.0",0.5,-1.0)
