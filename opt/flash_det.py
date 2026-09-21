import torch, flash_attn, numpy as np
torch.manual_seed(0)
H, D = 4, 16; K = 1024
N = 50362
npad = ((N + K - 1)//K)*K
qkv = torch.randn(npad, 3, H, D, device="cuda", dtype=torch.float16)
cu = torch.arange(0, npad+1, K, dtype=torch.int32, device="cuda")
outs=[]
for _ in range(6):
    o = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu, max_seqlen=K, dropout_p=0, softmax_scale=D**-0.5)
    outs.append(o.float().cpu().numpy().copy())
d=[np.abs(outs[0]-o).max() for o in outs[1:]]
print("flash varlen forward, identical qkv, 6 reps: maxdiff", d)
# and a Linear
lin = torch.nn.Linear(H*D, H*D).cuda().half()
o2=[lin(qkv[:,0].reshape(npad,-1)).float().cpu().numpy().copy() for _ in range(4)]
print("Linear maxdiff", [np.abs(o2[0]-x).max() for x in o2[1:]])
