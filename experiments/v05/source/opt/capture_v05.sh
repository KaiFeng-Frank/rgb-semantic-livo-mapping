#!/usr/bin/env bash
# opt/capture_v05.sh TAG NPZ FX FY FZ DIST PITCH YAW MODE(rgb|class) [PTSIZE] [SETTLE_S]
# Same mechanism as opt/capture_v03.sh: the saved map .npz is re-published on /semantic_map
# (RELIABLE / TRANSIENT_LOCAL, i.e. latched) by opt/publish_npz.py, a generated RViz2 config
# (rviz/make_rviz.sh, the same two displays as rviz/semantic_map.rviz) is opened on
# DISPLAY=:0, and the root window is captured with ImageMagick.  Nothing here runs while the
# pipeline is timed; the visualiser never touches the timing numbers.
set -e
cd /data/livo_sem
source /opt/ros/jazzy/setup.bash
export DISPLAY=:0
export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA
TAG=$1; NPZ=$2; FX=$3; FY=$4; FZ=$5; DIST=$6; PITCH=$7; YAW=$8; MODE=${9:-class}; PS=${10:-3}; SETTLE=${11:-25}
if [ "$MODE" = "rgb" ]; then RGB=true; CLS=false; else RGB=false; CLS=true; fi
CFG=/tmp/v05_$TAG.rviz
bash rviz/make_rviz.sh $CFG $RGB $CLS false $FX $FY $FZ $DIST $PITCH $YAW $PS >/dev/null
pkill -9 -f '[r]viz2' 2>/dev/null || true
pkill -9 -f '[p]ublish_npz.py' 2>/dev/null || true
sleep 2
python3 opt/publish_npz.py --npz "$NPZ" --seconds $((SETTLE + 30)) > /tmp/pub_v05_$TAG.log 2>&1 &
PUB=$!
sleep 4
ros2 run rviz2 rviz2 -d $CFG > /tmp/rviz_v05_$TAG.log 2>&1 &
RV=$!
sleep $SETTLE
import -window root out/v05/rviz_$TAG.png
kill $RV $PUB 2>/dev/null || true
sleep 2
pkill -9 -f '[r]viz2' 2>/dev/null || true
pkill -9 -f '[p]ublish_npz.py' 2>/dev/null || true
echo "captured out/v05/rviz_$TAG.png  ($(head -1 /tmp/pub_v05_$TAG.log))"
