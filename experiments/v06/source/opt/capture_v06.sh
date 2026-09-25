#!/usr/bin/env bash
# opt/capture_v06.sh TAG NPZ FX FY FZ DIST PITCH YAW MODE(rgb|class) [PTSIZE] [SETTLE_S]
# opt/capture_v05.sh with the v0.6 output directory: the saved map .npz is re-published on
# /semantic_map (RELIABLE / TRANSIENT_LOCAL, latched) by opt/publish_npz.py, a generated RViz2
# config is opened on DISPLAY=:0 and the root window is captured.  Only after every timed run.
set -e
cd /data/livo_sem
source /opt/ros/jazzy/setup.bash
export DISPLAY=:0
export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA
TAG=$1; NPZ=$2; FX=$3; FY=$4; FZ=$5; DIST=$6; PITCH=$7; YAW=$8; MODE=${9:-class}; PS=${10:-3}; SETTLE=${11:-18}
if [ "$MODE" = "rgb" ]; then RGB=true; CLS=false; else RGB=false; CLS=true; fi
CFG=/data/livo_sem/out/v06/tmp/v06_$TAG.rviz
bash rviz/make_rviz.sh $CFG $RGB $CLS false $FX $FY $FZ $DIST $PITCH $YAW $PS >/dev/null
pkill -9 -f '[r]viz2' 2>/dev/null || true
pkill -9 -f '[p]ublish_npz.py' 2>/dev/null || true
sleep 2
python3 opt/publish_npz.py --npz "$NPZ" --seconds $((SETTLE + 30)) > out/v06/tmp/pub_$TAG.log 2>&1 &
PUB=$!
sleep 4
ros2 run rviz2 rviz2 -d $CFG > out/v06/tmp/rviz_$TAG.log 2>&1 &
RV=$!
sleep $SETTLE
import -window root out/v06/rviz_$TAG.png
kill $RV $PUB 2>/dev/null || true
sleep 2
pkill -9 -f '[r]viz2' 2>/dev/null || true
pkill -9 -f '[p]ublish_npz.py' 2>/dev/null || true
echo "captured out/v06/rviz_$TAG.png  ($(head -1 out/v06/tmp/pub_$TAG.log))"
