"""Apply one exact integer-shape workaround for torch 1.12/cu113 on Ada.

Upstream issue: https://github.com/pytorch/pytorch/issues/87595
The source constructs spatial_shapes from (height,width) integer pairs. Explicit
H*W equals prod(dim=1) exactly, avoiding the unsupported NVRTC reduction JIT.
"""
import difflib
import hashlib
import json
from pathlib import Path

import torch

ROOT=Path('/home/wuyou/stocc_reference')
REL='mmdet3d/models/stocc/view_transformation/backward_projection/bevformer_utils/bevformer.py'
OLD='spatial_shapes.prod(1)'
NEW='(spatial_shapes[:, 0] * spatial_shapes[:, 1])'


def main():
    path=ROOT/'upstream'/REL
    before=path.read_text()
    if NEW in before:
        assert (ROOT/'compatibility.json').exists()
        print('compatibility patch already recorded')
        return
    assert before.count(OLD)==1
    # Actual CUDA execution versus CPU integer reduction, including real shapes.
    generator=torch.Generator().manual_seed(0)
    shapes=torch.cat([torch.tensor([[16,44],[100,100],[1,1]],dtype=torch.int64),
                      torch.randint(1,4097,(10000,2),generator=generator,dtype=torch.int64)])
    gpu=shapes.cuda()
    actual=(gpu[:,0]*gpu[:,1]).cpu()
    assert torch.equal(actual,shapes.prod(1))
    after=before.replace(OLD,NEW)
    patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                                     fromfile='a/'+REL,tofile='b/'+REL))
    (ROOT/'compatibility.patch').write_text(patch)
    path.write_text(after)
    record=dict(file=REL,before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                after_sha256=hashlib.sha256(after.encode()).hexdigest(),
                change='exact two-dimensional integer H*W instead of NVRTC-JIT prod',
                integer_cuda_parity_rows=len(shapes),parity_passed=True,
                upstream_issue='https://github.com/pytorch/pytorch/issues/87595')
    (ROOT/'compatibility.json').write_text(json.dumps(record,indent=2));print(record)


if __name__=='__main__': main()
