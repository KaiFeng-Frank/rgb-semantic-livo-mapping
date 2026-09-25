#!/usr/bin/env bash
# three independent offline draws of arm Rprime_noKL over seq 07 through the ORIGINAL
# DistilSegmentorMiB path (tools/cache_trained.py), exactly as every v0.4 arm was scored.
cd /data/livo_sem
P=/data/miniconda3/envs/ptv3/bin/python
until grep -q EXTRACT_CHAIN_DONE logs/extract_v05_RP.log 2>/dev/null; do sleep 15; done
for r in 1 2 3; do
  until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" = "0" ]; do sleep 10; done
  $P tools/cache_trained.py --ckpt exp/sk/armRprime_noKL/model/model_best.pth --seq 07 --frames all --out out/v05/RP_r$r > logs/rp_cache_v05_r$r.log 2>&1
  tail -1 logs/rp_cache_v05_r$r.log
done
echo RP_CACHE_DONE
