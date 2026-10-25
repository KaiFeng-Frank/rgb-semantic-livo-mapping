#!/usr/bin/env bash
set -eo pipefail
cd /home/frank/semantic_livo_realtime_main
PYTHONPATH=/home/frank/semantic_livo_runtime/visual-deps \
YOLO_CONFIG_DIR=/home/frank/semantic_livo_runtime/visual-config \
YOLO_AUTOINSTALL=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/frank/semantic_livo_runtime/venv/bin/python -m tools.visual_instances.evaluate \
  --index /home/frank/semantic_livo_runtime/consistency-20261026-index/index.json \
  --weights /home/frank/semantic_livo_runtime/visual-weights/yolo11n-seg.pt \
  --out /home/frank/semantic_livo_runtime/visual-instance-20261026-final-eval \
  > /home/frank/semantic_livo_runtime/visual-instance-20261026-final-eval.log 2>&1
for arm in control-final active-final; do
  VISUAL_CPU_AFFINITY=14,15,30,31 taskset -c 0-13,16-29 \
    bash /home/frank/semantic_backend_runtime/run-visual-instance-20261026.sh "$arm" \
    > "/home/frank/semantic_backend_runtime/visual-instance-20261026-$arm-full.log" 2>&1
done
