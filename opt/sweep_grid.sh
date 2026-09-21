#!/usr/bin/env bash
# grid_size sweep AT INTENSITY 0.2 on the fixed harness (the old sweep used the
# broken intensity 1.0 AND the broken HEAD loader, and is void twice over).
cd /data/livo_sem
P=/opt/miniconda3/envs/ptv3/bin/python
for g in 0.05 0.06 0.08 0.10 0.15; do
  $P src/eval_report.py --nframes 200 --stride 5 --grid-size $g --intensity-scale 0.2 \
     --shuffle 0 --half 1 --fast-voxel 1 --gpu-voxel 1 --fast-hilbert 1 \
     --tag grid_$g --json-out opt/logs/grid_$g.json --quiet 2>opt/logs/grid_$g.err | tail -1
done
