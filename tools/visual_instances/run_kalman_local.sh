#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PTV3_PYTHON=/home/frank/semantic_livo_runtime/venv/bin/python
export PTV3_POINTCEPT=/home/frank/semantic_livo_runtime/src/Pointcept_v151
export PTV3_CONFIG=/home/frank/semantic_livo_runtime/weights/nuscenes-semseg-pt-v3m1-0-base/config.py
export PTV3_CHECKPOINT=/home/frank/semantic_livo_runtime/v06-armB0_s1-student.pth
export YOLO_CONFIG_DIR=/home/frank/semantic_livo_runtime/visual-config
export PYTHONPATH=/home/frank/semantic_livo_runtime/visual-deps:${PYTHONPATH:-}
exec "$PTV3_PYTHON" "$@"
