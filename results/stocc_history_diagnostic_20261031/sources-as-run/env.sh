#!/usr/bin/env bash
# Dedicated ST-Occ runtime. Does not alter the TASeg environment or local VPN.
export STOCC_ROOT=/home/wuyou/stocc_reference
export CUDA_HOME="$STOCC_ROOT/cuda113"
export PATH="$STOCC_ROOT/env/bin:$CUDA_HOME/bin:$PATH"
# The restarted host now has matching kernel/system driver 595.99.02.
# Override only if a later host audit identifies a different matching library.
export STOCC_DRIVER_DIR="${STOCC_DRIVER_DIR:-/usr/lib/x86_64-linux-gnu}"
export LD_LIBRARY_PATH="$STOCC_DRIVER_DIR:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export PYTHONPATH="$STOCC_ROOT/upstream"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MAX_JOBS=4
# CUDA 11.3 emits Ampere code; the 4090 executes compatible sm_86 binaries.
export TORCH_CUDA_ARCH_LIST=8.6
export FORCE_CUDA=1
