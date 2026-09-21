import sys, glob, time
sys.path.insert(0,"/data/livo_sem/src")
import numpy as np, os
os.environ.setdefault("PTV3_GRID_SIZE","0.05")
import ptv3_worker as W
D="/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/"
files=sorted(glob.glob(D+"*.bin"))
idx=[0,250,500,750,1000,1100]
for gs in (0.05,0.075,0.10,0.15,0.20):
  allok=True
  for i in idx:
    p=np.fromfile(files[i],dtype=np.float32).reshape(-1,4)
    c=np.ascontiguousarray(p[:,:3],dtype=np.float32); s=np.ascontiguousarray(p[:,3:4]*0.2,dtype=np.float32)
    a=W.voxelize(c,s,gs,None,fast=False)
    b=W.voxelize(c,s,gs,None,fast=True)
    ok=all(np.array_equal(x,y) for x,y in zip(a,b))
    allok&=ok
  print("grid %.3f  bit-identical=%s" % (gs,allok))
# timing, one core
p=np.fromfile(files[0],dtype=np.float32).reshape(-1,4)
c=np.ascontiguousarray(p[:,:3],dtype=np.float32); s=np.ascontiguousarray(p[:,3:4]*0.2,dtype=np.float32)
for fast in (False,True):
    ts=[]
    for _ in range(30):
        t0=time.perf_counter(); W.voxelize(c,s,0.05,None,fast=fast); ts.append((time.perf_counter()-t0)*1e3)
    print("fast=%s  numpy voxelize total %.2f ms (incl np.unique)" % (fast, np.median(ts)))
