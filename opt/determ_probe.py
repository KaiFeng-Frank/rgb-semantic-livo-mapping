import os,sys,glob,numpy as np
os.environ["PTV3_SHUFFLE"]="0"; os.environ["PTV3_HALF"]=sys.argv[1] if len(sys.argv)>1 else "0"
sys.path.insert(0,"/data/livo_sem/src")
import ptv3_worker as W
D="/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/"
f=sorted(glob.glob(D+"*.bin"))[0]
p=np.fromfile(f,dtype=np.float32).reshape(-1,4)
seg=W.Segmenter()
outs=[]
for i in range(4):
    l,c=seg.segment(p); outs.append((l.copy(),c.copy()))
for i in range(1,4):
    same=(outs[0][0]==outs[i][0]).mean()*100
    cmax=np.abs(outs[0][1]-outs[i][1]).max()
    print("within-process rep %d: label agreement %.4f%%  conf maxdiff %.3e"%(i,same,cmax))
np.save("/tmp/determ_%s.npy"%os.environ["PTV3_HALF"], outs[0][0])
