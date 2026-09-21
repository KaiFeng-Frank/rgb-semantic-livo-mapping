from huggingface_hub import snapshot_download
p = snapshot_download("tue-mps/cityscapes_semantic_eomt_large_1024",
                      allow_patterns=["*.json","*.txt","*.safetensors","README.md"],
                      max_workers=8)
print("OK", p)
