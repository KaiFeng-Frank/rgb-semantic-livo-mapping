#!/bin/bash
set -x
export TMPDIR=/data/tmp
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /opt/miniconda3/envs/ptv3
pip install spconv-cu124
echo "--- torch_scatter ---"
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html --no-index || \
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html --no-index
echo "--- misc ---"
pip install timm addict einops yacs termcolor numpy==1.26.4 scipy
echo "--- flash-attn wheel via gh-proxy ---"
cd /data/tmp
FA=flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
curl -L -f -o $FA "https://gh-proxy.com/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$FA" && pip install ./$FA || echo "FLASH_FAIL"
echo "--- verify ---"
python - <<'PY'
import torch, spconv.pytorch as spconv, torch_scatter, timm, numpy
print("torch", torch.__version__, torch.cuda.is_available())
print("dev", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("spconv OK", spconv.__name__)
print("torch_scatter OK", torch_scatter.__version__)
print("numpy", numpy.__version__)
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn MISSING:", e)
PY
echo "MKENV2_DONE"
