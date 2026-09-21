#!/usr/bin/env bash
# end-to-end node runs.  ONE AT A TIME -- the GPU is exclusive.
cd /data/livo_sem
B7=bags/kitti_seq07_us
T7=out/kitti_seq07_fastlivo2_tum.txt
COMMON="--conf-gate 0.5 --expect-voxels 4000000"
case "$1" in
 before)  ./opt/run.sh before src_before $B7 $T7 1.0 --reliable --conf-gate 0.5 \
              --npz-out opt/out/map_before.npz ;;
 after)   ./opt/run.sh after  src       $B7 $T7 1.0 --reliable $COMMON \
              --npz-out opt/out/map_after.npz ;;
 after_be) ./opt/run.sh after_be src    $B7 $T7 1.0 $COMMON ;;
 cap)     ./opt/run.sh cap    src       $B7 $T7 2.0 --reliable $COMMON --idle-finish 15 ;;
 cap_before) ./opt/run.sh cap_before src_before $B7 $T7 2.0 --reliable --conf-gate 0.5 --idle-finish 15 ;;
 after_novis) ./opt/run.sh after_novis src $B7 $T7 1.0 --reliable $COMMON --map-rate 0.001 ;;
esac
