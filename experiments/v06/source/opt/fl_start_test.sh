#!/usr/bin/env bash
# fl_start_test.sh -- how often does fastlivo_mapping survive startup (camera params via blackboard)?
cd /data/livo_sem
source /opt/ros/jazzy/setup.bash; source ros2_ws/install/setup.bash
pkill -9 -f "[p]arameter_blackboard"; pkill -9 -f "[f]astlivo_mapping"; sleep 1
for T in LARGE_DATA DEFAULT; do
  if [ $T = LARGE_DATA ]; then export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA; else unset FASTDDS_BUILTIN_TRANSPORTS; fi
  ros2 run demo_nodes_cpp parameter_blackboard --ros-args -r __node:=parameter_blackboard --params-file config/camera_kitti_cam2.yaml > /dev/null 2>&1 &
  BB=$!; sleep 4
  ok=0
  for k in 1 2 3 4; do
    ros2 run fast_livo fastlivo_mapping --ros-args -r __node:=laserMapping --params-file config/kitti_velodyne64.yaml > out/v06/tmp/fltest_${T}_$k.log 2>&1 &
    FL=$!; sleep 6
    if kill -0 $FL 2>/dev/null; then ok=$((ok+1)); fi
    kill -9 $FL 2>/dev/null; sleep 1
  done
  echo "transport=$T survived $ok/4"
  kill -9 $BB 2>/dev/null; sleep 1
done
pkill -9 -f "[p]arameter_blackboard"; pkill -9 -f "[f]astlivo_mapping"; echo done
