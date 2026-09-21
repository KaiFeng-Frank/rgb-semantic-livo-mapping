from huggingface_hub import snapshot_download
p = snapshot_download("facebook/mask2former-swin-large-cityscapes-semantic",
                      allow_patterns=["*.json","*.txt","model.safetensors","README.md"],
                      max_workers=8)
print("OK", p)
