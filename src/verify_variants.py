import sys, glob, numpy as np, torch
sys.path.insert(0, "/data/livo_sem/src")
from ptv3_infer import (PTv3Segmenter, read_bin, sk_labels_to_coarse,
                        NUSC16_TO_COARSE, COARSE, IGNORE)
S = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/0000000000.bin"
pts = read_bin(S)

seg = PTv3Segmenter(half=False, voxel_backend="pointcept")
seg.seed = 1; a1, _ = seg.segment(pts)
seg.seed = 2; a2, _ = seg.segment(pts)
seg.seed = 3; a3, _ = seg.segment(pts)
print("Pointcept voxelizer, two different random representatives: agreement %.2f%% / %.2f%%"
      % (100*(a1 == a2).mean(), 100*(a1 == a3).mean()))
seg.voxel_backend = "gpu"
g1, _ = seg.segment(pts); g2, _ = seg.segment(pts)
print("GPU voxelizer, repeated (deterministic representative): agreement %.2f%%" % (100*(g1 == g2).mean()))
print("GPU vs Pointcept: %.2f%%" % (100*(g1 == a1).mean()))

D = "/data/livo_sem/data"
sc = sorted(glob.glob(D + "/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/*.bin"))
lb = sorted(glob.glob(D + "/odometry/dataset/sequences/07/labels/*.label"))
fl = [(sc[i], lb[i]) for i in range(0, 300, 30)]

def ev(seg, tag):
    K = len(COARSE); inter = np.zeros(K, np.int64); pc = np.zeros(K, np.int64); gc = np.zeros(K, np.int64)
    ok = tot = 0
    for sp, gp in fl:
        p = read_bin(sp); g = sk_labels_to_coarse(gp)
        l, _ = seg.segment(p); pr = NUSC16_TO_COARSE[l.astype(np.int64)]; m = g != IGNORE
        ok += int((pr[m] == g[m]).sum()); tot += int(m.sum())
        for k in range(K):
            pk = (pr == k) & m; gk = (g == k) & m
            inter[k] += int((pk & gk).sum()); pc[k] += int(pk.sum()); gc[k] += int(gk.sum())
    ious = [inter[k]/max(pc[k]+gc[k]-inter[k],1) for k in range(K) if gc[k] > 0]
    print("%-46s acc=%5.2f%%  mIoU=%5.2f%%" % (tag, 100*ok/tot, 100*np.mean(ious)))

ev(PTv3Segmenter(half=False, voxel_backend="pointcept"), "fp32 weights + Pointcept GridSample (reference)")
ev(PTv3Segmenter(half=False, voxel_backend="gpu"),       "fp32 weights + GPU voxelizer")
ev(PTv3Segmenter(half=True,  voxel_backend="gpu"),       "fp16 weights + GPU voxelizer")
