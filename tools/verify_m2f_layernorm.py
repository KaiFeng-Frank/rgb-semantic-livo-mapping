import os
os.environ.setdefault("HF_HOME","/data/hf_cache"); os.environ["HF_HUB_OFFLINE"]="1"
import torch, numpy as np
from transformers import Mask2FormerForUniversalSegmentation
torch.set_num_threads(4)
MID="facebook/mask2former-swin-large-cityscapes-semantic"
m=Mask2FormerForUniversalSegmentation.from_pretrained(MID, dtype=torch.float32).eval()

swin = m.model.pixel_level_module.encoder.swin
print("has .layernorm:", hasattr(swin,"layernorm"))
fired={"n":0}
if hasattr(swin,"layernorm") and swin.layernorm is not None:
    swin.layernorm.register_forward_hook(lambda *a: fired.__setitem__("n",fired["n"]+1))
# also hook the per-stage hidden_states_norms that Mask2Former actually uses
names=[n for n,_ in swin.named_modules() if "norm" in n and n.count(".")<=1]
print("top-level norm modules on the backbone:", names)

x=torch.randn(1,3,256,512)
with torch.no_grad():
    o1=m(pixel_values=x)
print("swin.layernorm forward-hook fired:", fired["n"], "times  <-- 0 means the MISSING keys are dead weight")

# decisive: perturb the missing layernorm massively; if output is identical it is unused
if hasattr(swin,"layernorm") and swin.layernorm is not None:
    with torch.no_grad():
        swin.layernorm.weight.fill_(137.0); swin.layernorm.bias.fill_(-42.0)
    with torch.no_grad():
        o2=m(pixel_values=x)
    dm=(o1.masks_queries_logits-o2.masks_queries_logits).abs().max().item()
    dc=(o1.class_queries_logits-o2.class_queries_logits).abs().max().item()
    print("after setting swin.layernorm to garbage: max|d mask logits| = %.3e, max|d class logits| = %.3e"%(dm,dc))
    print("VERDICT:", "UNUSED - warning is benign" if max(dm,dc)==0.0 else "USED - PROBLEM")
