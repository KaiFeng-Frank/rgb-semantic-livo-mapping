#!/usr/bin/env bash
# opt/runs_v05.sh -- v0.5 re-qualification runs.  ONE AT A TIME; the GPU is exclusive and
# nothing else may load the CPU while a run is timed (stage B is CPU-bound).
#
#   runs_v05.sh sat <ZS|B0|RP> <rep>   saturated (bag rate 2.0), node-intrinsic frame period
#   runs_v05.sh rt  <ZS|B0|RP>         full bag at rate 1.0 -> processed/dropped, map .npz
#   runs_v05.sh all                    a,b saturated reps interleaved over the 3 models, then rt x3
#
# Node arguments are the v0.2 `cap` / `after3` configuration verbatim (opt/runs_node.sh):
#   --reliable --conf-gate 0.5 --expect-voxels 4000000 (+ --idle-finish 15 for the saturated runs)
# The ONLY difference between the three arms is --ptv3-ckpt (absent = released weights).
set -u
cd /data/livo_sem
B7=bags/kitti_seq07_us
T7=out/kitti_seq07_fastlivo2_tum.txt
BASE="--reliable --conf-gate 0.5 --expect-voxels 4000000"
W=/data/livo_sem/weights/v05
ck() { case "$1" in
  ZS) echo "" ;;
  B0) echo "--ptv3-ckpt $W/B0_student.pth" ;;
  RP) echo "--ptv3-ckpt $W/Rprime_noKL_student.pth" ;;
  *) echo "bad model $1" >&2; exit 2 ;;
esac; }
gpu_free_or_die() {
  local n; n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  if [ "$n" -ne 0 ]; then echo "REFUSING: $n compute process(es) on the GPU" >&2; nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2; exit 1; fi
  local l; l=$(uptime | sed 's/.*load average: //')
  echo "[v05] $(date '+%F %T') GPU idle; load average $l"
}
clk() { nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate --format=csv,noheader >> logs/gpu_v05_$1.txt; }
case "${1:-}" in
  sat) M=$2; I=$3; TAG=v05_sat_${M}_${I}; gpu_free_or_die; clk $TAG
       ./opt/run.sh $TAG src $B7 $T7 2.0 $BASE --idle-finish 15 $(ck $M); clk $TAG ;;
  rt)  M=$2; TAG=v05_rt_${M}; gpu_free_or_die; clk $TAG
       ./opt/run.sh $TAG src $B7 $T7 1.0 $BASE --npz-out out/v05/map_${M}.npz $(ck $M); clk $TAG ;;
  all) for i in a b; do for m in ZS B0 RP; do bash $0 sat $m $i; sleep 8; done; done
       for m in ZS B0 RP; do bash $0 rt $m; sleep 8; done; echo "[v05] ALL DONE" ;;
  *) echo "usage: $0 {sat M rep | rt M | all}"; exit 2 ;;
esac
