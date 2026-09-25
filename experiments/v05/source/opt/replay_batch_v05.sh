#!/usr/bin/env bash
# opt/replay_batch_v05.sh -- CPU replays, 4 in parallel.  NEVER while a timing run is on.
cd /data/livo_sem
P=/data/miniconda3/envs/ags/bin/python
run() { tag=$1; pred=$2; $P opt/replay_v05.py --pred $pred --pred-order raw --tag $tag --json-out out/v05/map_$tag.json > logs/replay_v05_$tag.log 2>&1; echo "done $tag $(tail -1 logs/replay_v05_$tag.log | cut -c1-60)"; }
export -f run; export P
printf "%s\n" "ZS_r1 out/vs2d/ptv3_r1" "ZS_r2 out/vs2d/ptv3_r2" "ZS_r3 out/vs2d/ptv3_r3" "ZS_r4 out/v04/ptv3_r4" "ZS_r5 out/v04/ptv3_r5" "ZS_r6 out/v04/ptv3_r6" "B0_r1 out/v04/arms/B0_r1" "B0_r2 out/v04/arms/B0_r2" "B0_r3 out/v04/arms/B0_r3" | xargs -P 4 -L 1 bash -c 'run $0 $1'
echo REPLAY_BATCH1_DONE
