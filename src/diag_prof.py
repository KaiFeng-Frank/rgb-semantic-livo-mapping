import sys, time, numpy as np, torch
sys.path.insert(0, "/data/livo_sem/src")
from ptv3_infer import PTv3Segmenter, read_bin
S = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/0000000000.bin"
pts = read_bin(S)
seg = PTv3Segmenter()
for _ in range(5): seg.segment(pts)
torch.cuda.synchronize()

t = []
for _ in range(20):
    t0 = time.perf_counter(); d = seg.voxelize(pts); t.append(time.perf_counter()-t0)
print("CPU voxelize (Pointcept GridSample): mean %.1f ms" % (1000*np.mean(t)))

dev = seg.device
coord = torch.from_numpy(d["coord"]).to(dev).float()
grid = torch.from_numpy(d["grid_coord"]).to(dev).int()
st = torch.from_numpy(d["strength"]).to(dev).float()
feat = torch.cat([coord, st], 1)
inp = dict(coord=coord, grid_coord=grid, feat=feat, offset=torch.tensor([coord.shape[0]], device=dev))
with torch.inference_mode():
    torch.cuda.synchronize(); t = []
    for _ in range(20):
        t0 = time.perf_counter(); seg.model(inp); torch.cuda.synchronize(); t.append(time.perf_counter()-t0)
print("GPU forward: mean %.1f ms" % (1000*np.mean(t)))

# half precision model
seg.model.half()
inp2 = dict(coord=coord.half(), grid_coord=grid, feat=feat.half(),
            offset=torch.tensor([coord.shape[0]], device=dev))
with torch.inference_mode():
    try:
        out = seg.model(inp2)["seg_logits"]
        torch.cuda.synchronize(); t = []
        for _ in range(20):
            t0 = time.perf_counter(); seg.model(inp2); torch.cuda.synchronize(); t.append(time.perf_counter()-t0)
        print("GPU forward fp16 weights: mean %.1f ms  logits dtype %s" % (1000*np.mean(t), out.dtype))
    except Exception as e:
        print("fp16 weights FAILED:", type(e).__name__, str(e)[:200])
