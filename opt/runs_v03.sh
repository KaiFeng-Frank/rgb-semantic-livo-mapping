#!/usr/bin/env bash
# opt/runs_v03.sh <what>   -- ONE AT A TIME.  The GPU is exclusive (M1).
cd /data/livo_sem
B7=bags/kitti_seq07_us
T7=out/kitti_seq07_fastlivo2_tum.txt
BASE="--conf-gate 0.5 --expect-voxels 4000000"
# FROZEN v0.3 operating point -- see opt/out/sweep/ for the curve each value sits on
DYN="--dyn --dyn-n-az 150 --dyn-n-el 64 --dyn-margin 0.6 --dyn-r-min 3.0 \
     --dyn-r-max 25.0 --dyn-dil-el 1 --dyn-dil-az 1 --dyn-reset-on-seen 0 --dyn-k-free 24"
case "$1" in
  # ---- S4 real time: saturated (bag rate 2.0), node-intrinsic, GPU exclusive
  cap_off) ./opt/run.sh v03_cap_off src $B7 $T7 2.0 --reliable $BASE --idle-finish 15 ;;
  cap_on)  ./opt/run.sh v03_cap_on  src $B7 $T7 2.0 --reliable $BASE $DYN --idle-finish 15 ;;
  cap_off2) ./opt/run.sh v03_cap_off2 src $B7 $T7 2.0 --reliable $BASE --idle-finish 15 ;;
  cap_on2) ./opt/run.sh v03_cap_on2 src $B7 $T7 2.0 --reliable $BASE $DYN --idle-finish 15 ;;
  # ---- 10 Hz reference runs
  rt_off)  ./opt/run.sh v03_rt_off src $B7 $T7 1.0 --reliable $BASE \
               --npz-out opt/out/map_v03_rt_off.npz ;;
  rt_on)   ./opt/run.sh v03_rt_on  src $B7 $T7 1.0 --reliable $BASE $DYN \
               --npz-out opt/out/map_v03_rt_on.npz ;;
  # ---- S1/S2/S3 scoring runs: mask dump ON (I/O -- never a timing run)
  score)   ./opt/run.sh v03_score_$2 src $B7 $T7 1.0 --reliable $BASE $DYN \
               --mask-dir out/dump_$2 --npz-out opt/out/map_v03_score_$2.npz --npz-out-raw opt/out/map_v03_raw_$2.npz ;;
  *) echo "usage: $0 {cap_off|cap_on|cap_off2|cap_on2|rt_off|rt_on|score <tag>}" ;;
esac
