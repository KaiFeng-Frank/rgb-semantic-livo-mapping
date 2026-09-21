import sys, numpy as np, torch
sys.path.insert(0, "/data/livo_sem/src")
from ptv3_infer import PTv3Segmenter, read_bin
S = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/0000000000.bin"
pts = read_bin(S)
seg = PTv3Segmenter(half=False)

a = seg.voxelize_cpu(pts)                       # Pointcept GridSample
b = seg.voxelize_gpu(pts)                       # ours
bi = b["inverse"].cpu().numpy(); bg = b["grid_coord"].cpu().numpy()
print("voxels: pointcept=%d  gpu=%d  equal=%s" % (len(a["coord"]), len(bg), len(a["coord"]) == len(bg)))
print("same cell set:", np.array_equal(np.unique(a["grid_coord"], axis=0), np.unique(bg, axis=0)))
gc_all = np.floor(pts[:, :3] / seg.grid_size).astype(np.int64)
gc_all -= gc_all.min(0)
print("gpu inverse valid (grid_coord[inverse] == own cell):", np.array_equal(bg[bi], gc_all.astype(np.int32)))
print("pointcept inverse valid:", np.array_equal(a["grid_coord"][a["inverse"]], gc_all.astype(a["grid_coord"].dtype)))

for bk in ("pointcept", "gpu"):
    seg.voxel_backend = bk
    seg.half = False; seg.feat_dtype = torch.float32
    l32, _ = seg.segment(pts)
    if bk == "gpu": g32 = l32
    else: p32 = l32
print("label agreement pointcept-vox vs gpu-vox (fp32): %.2f%%" % (100*(p32 == g32).mean()))

seg.voxel_backend = "gpu"; seg.model.half(); seg.half = True; seg.feat_dtype = torch.float16
g16, c16 = seg.segment(pts)
print("label agreement fp32 vs fp16 weights (gpu-vox): %.2f%%" % (100*(g32 == g16).mean()))
