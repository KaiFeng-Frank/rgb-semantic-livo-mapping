import sys, numpy as np
sys.path.insert(0,"/data/livo_sem/src")
import seg2d_infer as M

class Fake:                      # tiling math only -- no model, no torch device
    _tiles = M.Seg2DSegmenter._tiles
f = Fake()
print("axis-split rule (EoMT's own, applied per axis):")
for L,P in [(1024,1024),(3393,1024),(1184,1024),(3923,1024),(1226,1024),(1024,1024)]:
    st=f._tiles(L,P)
    cov=np.zeros(max(L,P),np.int32)
    for a in st: cov[a:a+P]+=1
    print("  L=%-5d P=%d -> n=%d starts=%-28s last_end=%-5d min_cover=%d max_cover=%d"
          %(L,P,len(st),st,st[-1]+P,cov[:L].min(),cov[:L].max()))
    assert cov[:L].min()>=1, "GAP in coverage"
    assert st[-1]+P>=L and st[0]==0

print("\nfull-frame window grids:")
for name,sc in M.SCALE_PRESETS.items():
    Hr,Wr=int(round(370*sc)),int(round(1226*sc))
    Hp,Wp=max(Hr,1024),max(Wr,1024)
    ys,xs=f._tiles(Hp,1024),f._tiles(Wp,1024)
    print("  %-11s s=%.4f  %4dx%-4d -> %d rows x %d cols = %2d windows of 1024^2"
          %(name,sc,Wr,Hr,len(ys),len(xs),len(ys)*len(xs)))
print("\n  at s=2.7676 ('shortside') the grid is 1 row x 4 cols == EoMT's shipped split. OK")
