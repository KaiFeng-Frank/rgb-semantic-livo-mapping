#!/usr/bin/env bash
# Run FAST-LIVO2 (ROS2 Jazzy port) on a KITTI bag and collect the TUM trajectory.
#   usage: run_fastlivo2_kitti.sh <seq_name> <bag_dir> [rate] [extra yaml overrides...]
# Writes:  <ROOT>/Log/result/<seq_name>.txt  -> copied to out/<seq_name>_fastlivo2_tum.txt
set +u
SEQ=${1:-kitti_seq04}
BAG=${2:-/data/livo_sem/bags/kitti_seq04_us}
RATE=${3:-0.5}
BASE=/data/livo_sem
WS=$BASE/ros2_ws
PKG=$WS/src/FAST-LIVO2
RES=$PKG/Log/result/$SEQ.txt
LOG=$BASE/logs/fastlivo2_$SEQ.log

source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash
export RCUTILS_LOGGING_BUFFERED_STREAM=0

mkdir -p $PKG/Log/result $BASE/out $BASE/logs
rm -f "$RES"

# tell the node which seq_name to write
CFG=$BASE/config/kitti_velodyne64.yaml
TMP=$(mktemp /tmp/kitti_cfg_XXXX.yaml)
sed "s|seq_name: \".*\"|seq_name: \"$SEQ\"|" "$CFG" > "$TMP"

pkill -9 -f parameter_blackboard 2>/dev/null
pkill -9 -f fastlivo_mapping 2>/dev/null
sleep 1

ros2 run demo_nodes_cpp parameter_blackboard --ros-args -r __node:=parameter_blackboard \
     --params-file $BASE/config/camera_kitti_cam2.yaml > $BASE/logs/blackboard_$SEQ.log 2>&1 &
BB=$!
sleep 3

ros2 run fast_livo fastlivo_mapping --ros-args -r __node:=laserMapping \
     --params-file "$TMP" > "$LOG" 2>&1 &
FL=$!
sleep 5
if ! kill -0 $FL 2>/dev/null; then
  echo "[run] fastlivo_mapping died during startup:"; tail -40 "$LOG"; kill -9 $BB; exit 1
fi
echo "[run] node up (pid $FL). playing $BAG at rate $RATE ..."

ros2 bag play "$BAG" --rate "$RATE" > $BASE/logs/bagplay_$SEQ.log 2>&1
echo "[run] bag finished; draining 20 s ..."
sleep 20

kill -INT $FL 2>/dev/null; sleep 3
kill -9 $FL $BB 2>/dev/null
pkill -9 -f parameter_blackboard 2>/dev/null
pkill -9 -f fastlivo_mapping   2>/dev/null

if [ -s "$RES" ]; then
  cp "$RES" $BASE/out/${SEQ}_fastlivo2_tum.txt
  echo "[run] TUM poses: $(wc -l < $RES)  -> $BASE/out/${SEQ}_fastlivo2_tum.txt"
else
  echo "[run] NO TRAJECTORY WRITTEN. tail of node log:"; tail -40 "$LOG"; exit 1
fi
rm -f "$TMP"
