#!/usr/bin/env bash
# opt/score_arms.sh  -- S1 / S2 / S3 from the SAME dumps, then the K4 spread.
set -e
cd /data/livo_sem
mkdir -p opt/out/score
for i in r1 r2 r3; do
  for arm in main noop naive; do
    python3 opt/finalize_masks.py --dump out/dump_$i --out out/masks_${arm}_$i --arm $arm
  done
done
echo "=== S3  NO-OP ARM (v0.2 as shipped) ==="
python3 opt/score_dynamic.py --mask-dir out/masks_noop_r1 \
    --json-out opt/out/score/s3_noop.json
echo
echo "=== S2  NAIVE CLASS-DELETE ARM (real PTv3 argmax, same frames) ==="
python3 opt/score_dynamic.py --mask-dir out/masks_naive_r1 \
    --json-out opt/out/score/s2_naive.json
echo
echo "=== S2b NAIVE ARM, ANALYTIC UPPER BOUND (perfect classifier) ==="
python3 opt/score_dynamic.py --policy oracle-naive --json-out opt/out/score/s2_oracle.json
echo
echo "=== S1  v0.3 MAIN ARM, 3 repeats (K4) ==="
python3 opt/score_dynamic.py --mask-dir out/masks_main_r1 \
    --repeat-dirs out/masks_main_r1 out/masks_main_r2 out/masks_main_r3 \
    --compare-dir out/masks_noop_r1 --json-out opt/out/score/s1_main.json
echo SCORING_DONE
