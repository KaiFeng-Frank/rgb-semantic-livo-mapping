import os,sys,glob,numpy as np
os.environ["PTV3_SHUFFLE"]="0"; os.environ["PTV3_HALF"]=os.environ.get("H","0")
sys.path.insert(0,"/data/livo_sem/src")
import torch, ptv3_worker as W
D="/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/"
p=np.fromfile(sorted(glob.glob(D+"*.bin"))[0],dtype=np.float32).reshape(-1,4)
seg=W.Segmenter()
import pointcept.models.point_transformer_v3.point_transformer_v3m1_base as M
if os.environ.get("STABLE","0")=="1":
    import math, torch_scatter, spconv.pytorch as spconv
    from pointcept.models.utils import Point
    from addict import Dict
    def fwd(self, point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth: pooling_depth = 0
        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True, return_counts=True)
        _, indices = torch.sort(cluster, stable=True)          # <-- the only change
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        code = code[:, head_indices]
        order = torch.argsort(code, stable=True)               # <-- and here
        inverse = torch.zeros_like(order).scatter_(dim=1, index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1))
        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0]); code=code[perm]; order=order[perm]; inverse=inverse[perm]
        point_dict = Dict(
            feat=torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce),
            coord=torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code, serialized_order=order, serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth, batch=point.batch[head_indices])
        if "condition" in point.keys(): point_dict["condition"] = point.condition
        if "context" in point.keys(): point_dict["context"] = point.context
        if self.traceable:
            point_dict["pooling_inverse"] = cluster; point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None: point = self.norm(point)
        if self.act is not None: point = self.act(point)
        point.sparsify()
        return point
    M.SerializedPooling.forward = fwd
o=[seg.segment(p) for _ in range(5)]
for i in range(1,5):
    print("rep %d  label agreement %.4f%%  conf maxdiff %.3e"%(i,(o[0][0]==o[i][0]).mean()*100,np.abs(o[0][1]-o[i][1]).max()))
