#!/usr/bin/env bash
cd /data/livo_sem
for a in cap_off cap_on cap_off2 cap_on2; do
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> logs/gpu_pre_$a.txt
  ./opt/runs_v03.sh $a
  sleep 10
done
echo TIMING_BATCH_DONE
