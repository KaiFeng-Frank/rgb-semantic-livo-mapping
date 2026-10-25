#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=95
export CYCLONEDDS_URI=file:///home/frank/semantic_livo_realtime_main/tools/loop_closure/local_dds.xml
export SEMANTIC_ROS_PYTHONPATH=/home/frank/semantic_livo_runtime/ros-deps
export VISUAL_PYTHONPATH=/home/frank/semantic_livo_runtime/visual-deps
export YOLO_CONFIG_DIR=/home/frank/semantic_livo_runtime/visual-config
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="/home/frank/semantic_livo_runtime/ros-deps:${PYTHONPATH:-}"
cd /home/frank/semantic_livo_realtime_main
/usr/bin/python3 -m tools.visual_instances.real_ros \
  --index /home/frank/semantic_livo_runtime/consistency-20261026-index/index.json \
  --runtime /home/frank/semantic_livo_runtime \
  --weights /home/frank/semantic_livo_runtime/visual-weights/yolo11n-seg.pt \
  --out "/home/frank/semantic_backend_runtime/visual-instance-20261026-real-ros${1:-}"
