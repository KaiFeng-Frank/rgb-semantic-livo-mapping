#!/usr/bin/env bash
# Staged driver for the "why not just project a 2D segmentation?" study.
# Every GPU stage is guarded: it refuses to start while someone else holds the GPU.
set -euo pipefail
R=/data/livo_sem
PY=/opt/miniconda3/envs/ptv3/bin/python
OUT=$R/out/vs2d
LOG=$R/logs2d
mkdir -p "$OUT" "$LOG"

gpu_guard () {
  while true; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v "^$" | wc -l)
    [ "$n" -eq 0 ] && break
    echo "GPU busy ($n compute apps); waiting..." ; sleep 30
  done
}

case "${1:-}" in
  sweep)      # STEP 1, seq04, 2D only
    gpu_guard
    $PY $R/tools/sweep9_seq04.py --nframes 55 --stride 5 \
        --models eomt mask2former \
        --scales native d2_maxsize half_focal shortside focal focal_125 \
        --tta none flip --json-out $OUT/sweep9_seq04.json 2>&1 | tee $LOG/sweep9_seq04.log ;;

  cache2d)    # $2 model  $3 scale  $4 tta  $5 tag
    gpu_guard
    $PY $R/tools/cache_seg2d.py --model "$2" --scale "$3" --tta "$4" --seq 07 \
        --out $OUT/seg2d_$5 2>&1 | tee $LOG/cache2d_$5.log ;;

  cache3d)    # $2 tag
    gpu_guard
    $PY $R/tools/cache_ptv3.py --seq 07 --out $OUT/ptv3_$2 2>&1 | tee $LOG/cache3d_$2.log ;;

  prep)       # CPU: boundary-set cache
    $PY $R/src/score_2d_vs_3d.py prep --seq 07 --frames all \
        --bcache $OUT/bcache --workers 10 2>&1 | tee $LOG/prep.log ;;

  score)      # CPU
    shift; $PY $R/src/score_2d_vs_3d.py score "$@" ;;

  bench)      # STEP 7, exclusive GPU
    gpu_guard
    $PY $R/tools/bench_latency.py "${@:2}" ;;

  mapcov)
    $PY $R/src/score_2d_vs_3d.py mapcov --seq 07 --frames all \
        --out $OUT/mapcov.json 2>&1 | tee $LOG/mapcov.log ;;

  *) echo "usage: $0 {sweep|cache2d|cache3d|prep|score|bench|mapcov}" ; exit 2 ;;
esac
