#!/usr/bin/env bash
# ============================================================================
#  run_main.sh -- stages 2..8 of the 2D-vs-3D study, after STEP 1 froze the
#  2D arm's configuration on seq 04.
#
#  Usage: run_main.sh <stage>
#     cache2d | cache3d | score | bench | fig | report | all
#  The frozen configuration is read from FROZEN.env, written by STEP 1.
# ============================================================================
set -euo pipefail
R=/data/livo_sem
PY=/opt/miniconda3/envs/ptv3/bin/python
OUT=$R/out/vs2d
LOG=$R/logs2d
mkdir -p "$OUT" "$LOG"
source $OUT/FROZEN.env          # EOMT_SCALE EOMT_TTA M2F_SCALE M2F_TTA

gpu_free_or_die () {
  local n
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  if [ "$n" -ne 0 ]; then
    echo "REFUSING: $n compute process(es) on the GPU." >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
    exit 1
  fi
}

cache2d () {
  gpu_free_or_die
  $PY $R/tools/cache_seg2d.py --model eomt --scale "$EOMT_SCALE" --tta "$EOMT_TTA" \
      --seq 07 --frames all --out $OUT/seg2d_eomt        2>&1 | tee $LOG/cache2d_eomt.log
  $PY $R/tools/cache_seg2d.py --model mask2former --scale "$M2F_SCALE" --tta "$M2F_TTA" \
      --seq 07 --frames all --out $OUT/seg2d_m2f         2>&1 | tee $LOG/cache2d_m2f.log
  # second independent pass of the primary 2D arm, 40 frames, for the determinism check
  $PY $R/tools/cache_seg2d.py --model eomt --scale "$EOMT_SCALE" --tta "$EOMT_TTA" \
      --seq 07 --frames 0:440:11 --out $OUT/seg2d_eomt_rerun 2>&1 | tee $LOG/cache2d_eomt_rerun.log
}

cache3d () {
  gpu_free_or_die
  for r in r1 r2 r3; do
    $PY $R/tools/cache_ptv3.py --seq 07 --frames all --out $OUT/ptv3_$r \
        2>&1 | tee $LOG/cache3d_$r.log
  done
}

score () {
  # determinism of the 2D arm, on the frames both passes cover
  $PY - <<'P'
import numpy as np, os
A="/data/livo_sem/out/vs2d/seg2d_eomt"; B=A+"_rerun"
fs=[int(f[1:7]) for f in sorted(os.listdir(B)) if f.endswith(".npz")]
bad=[f for f in fs if not np.array_equal(np.load("%s/f%06d.npz"%(A,f))["seg"],
                                         np.load("%s/f%06d.npz"%(B,f))["seg"])]
print("2D DETERMINISM over %d frames: %s" % (len(fs), "FAIL %s"%bad[:5] if bad else "PASS (bitwise identical)"))
P
  # PRIMARY: EoMT 2D arm vs 3D arm, 3 independent 3D draws, all 1101 frames
  $PY $R/src/score_2d_vs_3d.py score --seq 07 --frames all \
      --arm2d $OUT/seg2d_eomt --arm3d $OUT/ptv3_r1,$OUT/ptv3_r2,$OUT/ptv3_r3 \
      --bcache $OUT/bcache_07 --occ-win-sweep 0,1,2,3 \
      --out $OUT/results_eomt.json 2>&1 | tee $LOG/score_eomt.log
  # SECONDARY 2D checkpoint, identical everything else
  $PY $R/src/score_2d_vs_3d.py score --seq 07 --frames all \
      --arm2d $OUT/seg2d_m2f --arm3d $OUT/ptv3_r1,$OUT/ptv3_r2,$OUT/ptv3_r3 \
      --bcache $OUT/bcache_07 --skip-naive-2d \
      --out $OUT/results_m2f.json 2>&1 | tee $LOG/score_m2f.log
  # SENSITIVITY: sky counted as an abstention instead of wrong (pro-2D variant)
  $PY $R/src/score_2d_vs_3d.py score --seq 07 --frames all \
      --arm2d $OUT/seg2d_eomt --arm3d $OUT/ptv3_r1 \
      --bcache $OUT/bcache_07 --sky abstain --skip-naive-2d \
      --out $OUT/results_eomt_skyabstain.json 2>&1 | tee $LOG/score_sky.log
}

bench () {
  gpu_free_or_die
  $PY $R/tools/bench_latency.py --arm 3d --seq 07 --nframes 100 --stride 11 \
      --out $OUT/timing_3d.json            > $LOG/bench_3d.log 2>&1
  $PY $R/tools/bench_latency.py --arm 2d --model eomt --scale "$EOMT_SCALE" \
      --tta "$EOMT_TTA" --seq 07 --nframes 100 --stride 11 \
      --out $OUT/timing_2d_tta.json        > $LOG/bench_2d_tta.log 2>&1
  $PY $R/tools/bench_latency.py --arm 2d --model eomt --scale "$EOMT_SCALE" \
      --tta none --seq 07 --nframes 100 --stride 11 \
      --out $OUT/timing_2d_nott.json       > $LOG/bench_2d_nott.log 2>&1
  $PY $R/tools/bench_latency.py --arm 2d --model mask2former --scale "$M2F_SCALE" \
      --tta "$M2F_TTA" --seq 07 --nframes 100 --stride 11 \
      --out $OUT/timing_2d_m2f.json        > $LOG/bench_2d_m2f.log 2>&1
  echo "--- bench done"
}

fig () {
  for f in 380 700; do
    $PY $R/tools/fig_2d_vs_3d.py --frame $f --seq 07 \
        --arm3d $OUT/ptv3_r1 --arm2d $OUT/seg2d_eomt --out $R/out
  done
}

report () {
  $PY $R/tools/report9.py $OUT/results_eomt.json > $OUT/TABLE_eomt.md
  $PY $R/tools/report9.py $OUT/results_m2f.json  > $OUT/TABLE_m2f.md
  echo "wrote $OUT/TABLE_eomt.md $OUT/TABLE_m2f.md"
}

case "${1:-all}" in
  cache2d) cache2d ;; cache3d) cache3d ;; score) score ;;
  bench) bench ;; fig) fig ;; report) report ;;
  all) cache2d; cache3d; score; fig; report ;;
  *) echo "usage: $0 {cache2d|cache3d|score|bench|fig|report|all}"; exit 2 ;;
esac
