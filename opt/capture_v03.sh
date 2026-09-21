#!/usr/bin/env bash
# opt/capture_v03.sh TAG NPZ FX FY FZ DIST PITCH YAW MODE(rgb|class)
set -e
cd /data/livo_sem
source /opt/ros/jazzy/setup.bash
export DISPLAY=:0
export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA
TAG=$1; NPZ=$2; FX=$3; FY=$4; FZ=$5; DIST=$6; PITCH=$7; YAW=$8; MODE=${9:-class}
if [ "$MODE" = "rgb" ]; then RGB=true; CLS=false; else RGB=false; CLS=true; fi
CFG=/tmp/v03_$TAG.rviz
bash rviz/make_rviz.sh $CFG $RGB $CLS false $FX $FY $FZ $DIST $PITCH $YAW 3 >/dev/null
pkill -9 -f '[r]viz2' 2>/dev/null || true
pkill -9 -f '[p]ublish_npz.py' 2>/dev/null || true
sleep 2
python3 opt/publish_npz.py --npz "$NPZ" --seconds 45 > /tmp/pub_$TAG.log 2>&1 &
PUB=$!
sleep 4
ros2 run rviz2 rviz2 -d $CFG > /tmp/rviz_$TAG.log 2>&1 &
RV=$!
sleep 22
import -window root /data/livo_sem/out/rviz_v03_$TAG.png
kill $RV $PUB 2>/dev/null || true
sleep 2
pkill -9 -f '[r]viz2' 2>/dev/null || true
pkill -9 -f '[p]ublish_npz.py' 2>/dev/null || true
echo "captured out/rviz_v03_$TAG.png"
