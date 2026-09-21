#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
S=/data/livo_sem/src
R=/data/livo_sem/data/raw/2011_09_30
X=/data/livo_sem/data/extract/2011_09_30
B=/data/livo_sem/bags
run(){ d=$1; n=$2; o=$3; shift 3
  echo "########## $o"; rm -rf $B/$o
  /usr/bin/python3 $S/kitti_to_ros2bag.py \
    --drive $R/2011_09_30_drive_${d}_sync \
    --oxts100 $X/2011_09_30_drive_${d}_extract/oxts \
    --out $B/$o --end $n "$@"
}
run 0016  271 kitti_seq04
run 0016  271 kitti_seq04_us --time-unit us
run 0027 1101 kitti_seq07
run 0027 1101 kitti_seq07_us --time-unit us
echo "########## ALL BAGS DONE"; du -sh $B/*
