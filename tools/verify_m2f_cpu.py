import os, json, time
os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, numpy as np
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

MID = "facebook/mask2former-swin-large-cityscapes-semantic"
torch.set_num_threads(12)

t0 = time.time()
m = Mask2FormerForUniversalSegmentation.from_pretrained(MID, dtype=torch.float32)
m.eval()
print("[load] %.1fs  params=%.1fM" % (time.time()-t0, sum(p.numel() for p in m.parameters())/1e6))

cfg = m.config
print("[cfg] num_labels(len id2label) =", len(cfg.id2label))
print("[cfg] num_queries =", cfg.num_queries)
print("[cfg] class_pred head out_features =", m.class_predictor.out_features)
print("[cfg] backbone =", cfg.backbone_config.model_type, "embed_dim", cfg.backbone_config.embed_dim)
print("[id2label shipped by the model]")
for i in range(len(cfg.id2label)):
    print("   %2d  %s" % (i, cfg.id2label[i]))

# what does the SHIPPED processor do to a KITTI-shaped image?
ip = AutoImageProcessor.from_pretrained(MID)
print("[proc] size =", ip.size, " size_divisor =", getattr(ip,"size_divisor",None))
dummy = np.zeros((370, 1226, 3), dtype=np.uint8)
out = ip(images=dummy, return_tensors="pt")
print("[proc] DEFAULT processor turns 1226x370 into", tuple(out["pixel_values"].shape),
      "  <-- aspect ratio destroyed" )

# CPU forward on a small dummy, exercising the real path
H, W = 256, 640
x = torch.randn(1, 3, H, W)
t0 = time.time()
with torch.no_grad():
    o = m(pixel_values=x)
print("[fwd] cpu %dx%d ok in %.1fs" % (H, W, time.time()-t0))
print("[fwd] class_queries_logits", tuple(o.class_queries_logits.shape),
      "(Q=%d, C+1=%d)" % (o.class_queries_logits.shape[1], o.class_queries_logits.shape[2]))
print("[fwd] masks_queries_logits", tuple(o.masks_queries_logits.shape))
assert o.class_queries_logits.shape[-1] == len(cfg.id2label) + 1, "head size mismatch"
print("[fwd] HEAD SIZE OK: %d = 19 Cityscapes classes + 1 null" % o.class_queries_logits.shape[-1])
json.dump({str(i): cfg.id2label[i] for i in range(len(cfg.id2label))},
          open("/data/livo_sem/tools/m2f_id2label.json","w"), indent=1)
