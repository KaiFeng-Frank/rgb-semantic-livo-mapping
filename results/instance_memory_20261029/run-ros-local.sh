#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=96
export CYCLONEDDS_URI=file:///home/frank/semantic_livo_instance_memory/tools/loop_closure/local_dds.xml
export SEMANTIC_ROS_PYTHONPATH=/home/frank/semantic_livo_runtime/ros-deps VISUAL_PYTHONPATH=/home/frank/semantic_livo_runtime/visual-deps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export YOLO_CONFIG_DIR=/home/frank/semantic_livo_runtime/visual-config
export PYTHONPATH=/home/frank/semantic_livo_runtime/ros-deps:$PYTHONPATH
cd /home/frank/semantic_livo_instance_memory
runtime=/home/frank/semantic_livo_runtime/instance-memory-20261029
case "$1" in
  current) mode=matched_current; checkpoint=current;;
  memory) mode=feature_memory; checkpoint=temporal;;
  *) exit 2;;
esac
/usr/bin/python3 -u -m tools.visual_instances.real_ros \
 --index /home/frank/semantic_livo_runtime/consistency-20261026-index/index.json \
 --runtime /home/frank/semantic_livo_runtime \
 --weights /home/frank/semantic_livo_runtime/visual-weights/yolo11n-seg.pt \
 --out "$runtime/ros-$1-$2" --scene scene-0519 --instance-semantics active \
 --instance-memory-checkpoint "$runtime/round0/seed-0/$checkpoint.pt" --memory-mode "$mode"
