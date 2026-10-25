#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /home/frank/semantic_livo_realtime_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=94
export CYCLONEDDS_URI=file:///home/frank/semantic_livo_realtime_main/tools/loop_closure/local_dds.xml
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SEMANTIC_ROS_PYTHONPATH=/home/frank/semantic_livo_runtime/ros-deps
export VISUAL_PYTHONPATH=/home/frank/semantic_livo_runtime/visual-deps
export YOLO_CONFIG_DIR=/home/frank/semantic_livo_runtime/visual-config
cd /home/frank/semantic_livo_realtime_main
extra=()
if [[ "$1" == active* ]]; then
  extra=(--visual-instances --visual-weights /home/frank/semantic_livo_runtime/visual-weights/yolo11n-seg.pt)
fi
/usr/bin/python3 -m tools.loop_closure.realtime_replay \
  --workspace /home/frank/semantic_livo_realtime_ws \
  --worker /home/frank/semantic_backend_runtime/venv/bin/python \
  --bag '/media/frank/Extreme SSD/data/fast_livo_mcap/FAST-LIVO/hku1' \
  --out "/home/frank/semantic_backend_runtime/visual-instance-20261026-$1-full" \
  --backend --semantics --semantic-runtime /home/frank/semantic_livo_runtime \
  --seconds 130 "${extra[@]}"
