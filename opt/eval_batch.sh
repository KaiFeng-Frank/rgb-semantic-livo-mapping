#!/usr/bin/env bash
# S5: the per-point accuracy harness, DEPLOYED settings, all 1101 frames, 3 repeats.
cd /data/livo_sem
P=/opt/miniconda3/envs/ptv3/bin/python
for i in 1 2 3; do
  $P src/eval_report.py --nframes 1101 --stride 1 \
     --half 1 --shuffle 0 --fast-voxel 1 --gpu-voxel 1 --fast-hilbert 1 \
     --intensity-scale 0.2 --grid-size 0.05 --quiet \
     --tag v03_r$i --json-out opt/logs/v03_full_r$i.json
done
echo EVAL_DONE
