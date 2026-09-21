#!/bin/bash
set -x
export CONDA_PKGS_DIRS=/data/conda_pkgs
export TMPDIR=/data/tmp
mkdir -p $TMPDIR $CONDA_PKGS_DIRS
source /opt/miniconda3/etc/profile.d/conda.sh
conda create -y -p /opt/miniconda3/envs/ptv3 python=3.10 || exit 1
conda activate /opt/miniconda3/envs/ptv3
python -V
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U pip setuptools wheel
# torch cu124 for sm_89
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 || exit 2
python -c "import torch;print('torch',torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_arch_list())"
echo "MKENV_DONE"
