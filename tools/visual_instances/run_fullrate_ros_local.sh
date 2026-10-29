#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=96
export CYCLONEDDS_URI=file:///home/frank/semantic_livo_instance_memory/tools/loop_closure/local_dds.xml
export SEMANTIC_ROS_PYTHONPATH=/home/frank/semantic_livo_runtime/ros-deps
export VISUAL_PYTHONPATH=/home/frank/semantic_livo_runtime/visual-deps
export VISUAL_CPU_AFFINITY=14
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export YOLO_CONFIG_DIR=/home/frank/semantic_livo_runtime/visual-config
export PYTHONPATH=/home/frank/semantic_livo_runtime/ros-deps:$PYTHONPATH
cd /home/frank/semantic_livo_instance_memory
case "$1" in
  keyframe) cadence_args=(--keyframes-only);;
  fullrate) cadence_args=();;
  *) exit 2;;
esac
exec /usr/bin/python3 -u -m tools.visual_instances.real_ros \
 --index /home/frank/semantic_livo_runtime/instance-independent-dev-20261029/index/index.json \
 --runtime /home/frank/semantic_livo_runtime \
 --weights /home/frank/semantic_livo_runtime/visual-weights/yolo11n-seg.pt \
 --out "$2" --scene scene-0218 --instance-semantics off \
 --fullrate-frames /home/frank/semantic_livo_runtime/instance-fullrate-20261029/index/frames.json \
 --fullrate-images /home/frank/semantic_livo_runtime/instance-fullrate-20261029/data/images \
 --identity-cache /home/frank/semantic_livo_runtime/instance-independent-dev-20261029/cache \
 --visual-history "${cadence_args[@]}"
