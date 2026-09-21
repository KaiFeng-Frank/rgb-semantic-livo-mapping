#!/usr/bin/env bash
# =============================================================================
#  run_2d_vs_3d.sh -- the whole 2D-vs-3D experiment, in order.
#
#  Answers: "you have RGB, why not segment the image and project the labels onto
#  the points?"  With measurements.  The 2D arm is given every advantage; see the
#  PROTOCOL docstring at the top of src/score_2d_vs_3d.py for the full list.
#
#  Run stages individually:   ./run_2d_vs_3d.sh 0     (CPU)
#                             ./run_2d_vs_3d.sh 1     (GPU) ... etc
#  Or everything:             ./run_2d_vs_3d.sh all
#
#  STAGE 0  CPU   boundary-set cache (both sequences)       ~12 min, 1 core
#  STAGE 1  GPU   tune the 2D input scale ON SEQ 04         4 scales x 271 frames
#  STAGE 2  GPU   2D arm on seq 07 at the FROZEN scale      1101 frames
#  STAGE 3  GPU   3D arm on seq 07, 3 independent draws     3 x 1101 frames
#  STAGE 4  CPU   score all arms, 3 repeats, occ-win sweep
#  STAGE 5  CPU   accumulated map coverage
#  STAGE 6  CPU   markdown report
# =============================================================================
set -euo pipefail

ROOT=/data/livo_sem
OUT=$ROOT/opt/out2d
PY3=/opt/miniconda3/envs/ptv3/bin/python      # PTv3 env (3D arm + scorer)
PY2=/opt/miniconda3/envs/ags/bin/python       # transformers env (2D arm)
export HF_HOME=/data/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$OUT"

# The frozen 2D input scale.  STAGE 1 decides it; write the winner here and never
# touch it again.  3.1998 = Cityscapes fx / KITTI fx, i.e. the scale at which a
# Cityscapes-trained network sees KITTI objects at the pixel size it was trained on.
SCALE=${SCALE:-3.1998}
FAMILY=${FAMILY:-mask2former}

gpu_free_or_die () {
  local n
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$n" -gt 0 ]; then
    echo "REFUSING: $n compute process(es) already on the GPU." >&2
    echo "Timed benchmarks and this run would corrupt each other.  Wait." >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
    exit 1
  fi
}

stage0 () {   # ---------------------------------------------------- CPU ----
  echo "== STAGE 0: boundary-set cache =="
  # The GT-kNN semantic-boundary set (P7/B1) is the only expensive CPU step and it
  # is prediction-independent, so it is computed once and reused by every arm and
  # every repeat.  Single-threaded ON PURPOSE: do not contend with a timed GPU run.
  $PY3 $ROOT/src/score_2d_vs_3d.py prep --seq 07 --frames all \
       --bcache "$OUT/bcache_07" --workers 1
  $PY3 $ROOT/src/score_2d_vs_3d.py prep --seq 04 --frames all \
       --bcache "$OUT/bcache_04" --workers 1
}

stage1 () {   # ---------------------------------------------------- GPU ----
  echo "== STAGE 1: tune the 2D input scale ON SEQ 04 (NOT the eval sequence) =="
  # Choosing the scale on seq 07 and then reporting seq 07 would be tuning on the
  # test set.  Not tuning it at all would cripple the 2D arm, since a Cityscapes
  # model at KITTI's native resolution sees every object 3.2x too small.  seq 04 is
  # the same rig, the same calibration day, a disjoint drive.  Four principled
  # landmarks, not a grid sweep:
  #   1.00   native KITTI resolution (the naive choice)
  #   2.00   round factor
  #   2.7676 height -> 1024, the Cityscapes training crop height
  #   3.1998 angular match: Cityscapes fx / KITTI fx
  gpu_free_or_die
  for s in 1.0 2.0 2.7676 3.1998; do
    tag=$(echo "$s" | tr -d .)
    $PY2 $ROOT/src/seg2d_cityscapes.py run --family "$FAMILY" --seq 04 \
         --scale "$s" --tta hflip --out "$OUT/tune04_s$tag" \
         --timing-out "$OUT/tune04_s$tag.timing.json"
    $PY3 $ROOT/src/score_2d_vs_3d.py score --seq 04 --frames all \
         --arm2d "$OUT/tune04_s$tag" --bcache "$OUT/bcache_04" \
         --out "$OUT/tune04_s$tag.results.json"
  done
  echo "--- seq 04 tuning result (pick the best, then set SCALE and never revisit) ---"
  for f in "$OUT"/tune04_s*.results.json; do
    $PY3 - "$f" "${f%.results.json}.timing.json" <<'P'
import json, os, sys
r = json.load(open(sys.argv[1]))
a = r["agg"]["2d"]
scale = "?"
if os.path.exists(sys.argv[2]):
    scale = json.load(open(sys.argv[2])).get("input_scale", "?")
print("scale %-8s  in-frustum acc %6.2f %%   mIoU expr %6.2f %%   "
      "in-frustum coverage %5.2f %%"
      % (scale, a["frustum"]["acc_abstain_excluded"]["mean"],
         a["frustum"]["miou_expr_abstain_excluded"]["mean"],
         100 * a["frustum"]["coverage"]["mean"]))
P
  done
}

stage2 () {   # ---------------------------------------------------- GPU ----
  echo "== STAGE 2: 2D arm on seq 07 at the frozen scale $SCALE =="
  gpu_free_or_die
  $PY2 $ROOT/src/seg2d_cityscapes.py run --family "$FAMILY" --seq 07 \
       --scale "$SCALE" --tta hflip --out "$OUT/seg2d_a" \
       --timing-out "$OUT/timing_2d.json"
  # determinism: a fixed-weight eval-mode network must be bitwise reproducible.
  # If it is not, every 2D number needs the same repeat treatment as the 3D arm.
  $PY2 $ROOT/src/seg2d_cityscapes.py run --family "$FAMILY" --seq 07 \
       --scale "$SCALE" --tta hflip --f1 30 --out "$OUT/seg2d_b"
  $PY3 - "$OUT/seg2d_a" "$OUT/seg2d_b" <<'P'
import sys, numpy as np
a, b = sys.argv[1], sys.argv[2]
bad = [f for f in range(30)
       if not np.array_equal(np.load("%s/f%06d.npz" % (a, f))["seg"],
                             np.load("%s/f%06d.npz" % (b, f))["seg"])]
if bad:
    print("2D DETERMINISM: FAIL on frames %s -- the 2D arm needs the same repeat "
          "treatment as the 3D arm; run it 3x and pass 3 dirs to --arm2d" % bad[:5])
    sys.exit(1)
print("2D DETERMINISM: PASS (30 frames bitwise identical)")
P
  # latency without TTA, for the honest deployment number
  $PY2 $ROOT/src/seg2d_cityscapes.py run --family "$FAMILY" --seq 07 \
       --scale "$SCALE" --tta none --f1 120 --out "$OUT/seg2d_nott" \
       --timing-out "$OUT/timing_2d_nott.json"
}

stage3 () {   # ---------------------------------------------------- GPU ----
  echo "== STAGE 3: 3D arm on seq 07, 3 independent draws =="
  # The PTv3 forward is non-deterministic (~4-5 % of points change label between
  # two forwards).  Three draws let the scorer report a measured half-range instead
  # of asserting that the noise is small.
  gpu_free_or_die
  for r in a b c; do
    $PY3 $ROOT/opt/cache_pred.py --out "$OUT/ptv3_$r" --f0 0 --f1 1101
  done
  echo "NOTE: fill $OUT/timing_3d.json from a timed PTv3 pass (see TIMING_SCHEMA"
  echo "      in src/score_2d_vs_3d.py; stages pre/forward/post/total)."
}

stage4 () {   # ---------------------------------------------------- CPU ----
  echo "== STAGE 4: score =="
  T=""
  [ -f "$OUT/timing_2d.json" ] && T="$OUT/timing_2d.json"
  [ -f "$OUT/timing_3d.json" ] && T="$T,$OUT/timing_3d.json"
  $PY3 $ROOT/src/score_2d_vs_3d.py score --seq 07 --frames all \
       --arm2d "$OUT/seg2d_a" \
       --arm3d "$OUT/ptv3_a,$OUT/ptv3_b,$OUT/ptv3_c" \
       --bcache "$OUT/bcache_07" \
       --occ-win-sweep 0,1,2,3 \
       ${T:+--timing "$T"} \
       --out "$OUT/results_seq07.json"
  # the 100-frame noise set, for the spread quoted next to every headline number
  $PY3 $ROOT/src/score_2d_vs_3d.py score --seq 07 --frames stride:11 \
       --arm2d "$OUT/seg2d_a" \
       --arm3d "$OUT/ptv3_a,$OUT/ptv3_b,$OUT/ptv3_c" \
       --bcache "$OUT/bcache_07" --out "$OUT/results_noise100.json"
}

stage5 () {   # ---------------------------------------------------- CPU ----
  echo "== STAGE 5: accumulated map coverage =="
  # Pre-empts the obvious rebuttal: "per-scan coverage is 16 %, but the camera
  # sweeps as you drive, so the MAP is covered."  Answered with a number.
  $PY3 $ROOT/src/score_2d_vs_3d.py mapcov --seq 07 --frames all --voxel 0.2 \
       --out "$OUT/mapcov.json"
}

stage6 () {   # ---------------------------------------------------- CPU ----
  echo "== STAGE 6: report =="
  $PY3 $ROOT/opt/report_2d_vs_3d.py "$OUT/results_seq07.json" \
       --mapcov "$OUT/mapcov.json" > "$OUT/TABLE_2d_vs_3d.md"
  echo "wrote $OUT/TABLE_2d_vs_3d.md"
}

case "${1:-all}" in
  0) stage0 ;; 1) stage1 ;; 2) stage2 ;; 3) stage3 ;;
  4) stage4 ;; 5) stage5 ;; 6) stage6 ;;
  all)
     stage0
     stage1
     echo
     echo "STOP.  Stage 1 printed four scales.  Put the winner in SCALE= at the top"
     echo "of this script (or export SCALE=...), then run stages 2 3 4 5 6.  The"
     echo "scale is frozen from that moment: it was chosen on seq 04 and must not be"
     echo "revisited against seq 07 results." ;;
  *) echo "usage: $0 {0|1|2|3|4|5|6|all}"; exit 2 ;;
esac
