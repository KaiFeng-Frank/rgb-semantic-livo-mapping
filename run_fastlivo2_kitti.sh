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
rm -f $BASE/out/${SEQ}_fastlivo2_tum.txt   # a failed run must not leave the previous result
sleep 1

CAM=$BASE/config/camera_kitti_cam2.yaml
ros2 run demo_nodes_cpp parameter_blackboard --ros-args -r __node:=parameter_blackboard \
     --params-file $CAM > $BASE/logs/blackboard_$SEQ.log 2>&1 &
BB=$!
sleep 3

# vikit's getRemoteParam() gives each camera parameter one 100 ms wait_for_service and
# otherwise returns the default: a missed cam_model aborts the node, a missed cam_fx etc.
# would silently become 0. Require the printed intrinsics to match the yaml, else restart.
EXP=$(awk -F': *' '/cam_fx/{fx=$2} /cam_fy/{fy=$2} /cam_cx/{cx=$2} /cam_cy/{cy=$2}
      END{printf "intrinsic: %.6f, %.6f, %.6f, %.6f", fx, fy, cx, cy}' $CAM)
for TRY in 1 2 3 4 5; do
  ros2 run fast_livo fastlivo_mapping --ros-args -r __node:=laserMapping \
       --params-file "$TMP" > "$LOG" 2>&1 &
  FL=$!
  for _ in $(seq 30); do
    grep -q "^intrinsic:" "$LOG" 2>/dev/null && break
    kill -0 $FL 2>/dev/null || break
    sleep 0.5
  done
  sleep 2
  if kill -0 $FL 2>/dev/null && grep -qxF "$EXP" "$LOG"; then break; fi
  echo "[run] startup attempt $TRY failed: $(grep -m1 -E '^intrinsic:|what\(\)' "$LOG")"
  kill -9 $FL 2>/dev/null; pkill -9 -f fastlivo_mapping 2>/dev/null; sleep 1
  if [ $TRY = 5 ]; then tail -40 "$LOG"; kill -9 $BB; exit 1; fi
done
echo "[run] node up (pid $FL, $EXP). playing $BAG at rate $RATE ..."

# ros2 bag play publishes as soon as its publishers exist; without a delay, messages sent
# before DDS discovery matches the subscriber are lost (measured on one host: the first
# 0.52 s of the bag, which moves IMU initialisation). PLAY_DELAY=0 restores the old behaviour.
ros2 bag play "$BAG" --rate "$RATE" --delay "${PLAY_DELAY:-3}" > $BASE/logs/bagplay_$SEQ.log 2>&1
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
