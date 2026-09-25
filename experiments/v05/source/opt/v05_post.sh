#!/usr/bin/env bash
# opt/v05_post.sh -- CPU work that must NOT overlap the timing runs: live-map lookups, zoom
# region search, frozen-harness scoring of the Rprime_noKL caches.
cd /data/livo_sem
P2=/data/miniconda3/envs/ags/bin/python
until grep -q ORCH_TIMING2_DONE logs/v05_orchestrate2.log 2>/dev/null; do sleep 20; done
echo "[post $(date "+%T")] timing done; live lookups"
for spec in "ZS out/vs2d/ptv3_r1" "B0 out/v04/arms/B0_r1" "RP out/v05/RP_r1"; do set -- $spec
  $P2 opt/replay_v05.py --pred $2 --pred-order raw --live-npz out/v05/map_$1.npz --tag live_$1 \
      --json-out out/v05/live_$1.json --save-map out/v05/replaymap_$1.npz > logs/replay_v05_live_$1.log 2>&1 &
done; wait
for m in ZS B0 RP; do tail -2 logs/replay_v05_live_$m.log; done
echo "[post $(date "+%T")] zoom regions"
$P2 tools/v05_zoom_region.py --a out/v05/map_ZS.npz --b out/v05/map_B0.npz --gt out/v05/replaymap_ZS.npz --json-out out/v05/zoom_ZS_B0.json
$P2 tools/v05_zoom_region.py --a out/v05/map_B0.npz --b out/v05/map_RP.npz --gt out/v05/replaymap_ZS.npz --json-out out/v05/zoom_B0_RP.json
echo "[post $(date "+%T")] frozen harness on the local Rprime_noKL caches"
$P2 src/score_2d_vs_3d.py score --arm3d out/v05/RP_r1,out/v05/RP_r2,out/v05/RP_r3 --no-boundary --seq 07 --frames all \
    --out out/v05/score_RP.json > logs/score_RP_v05.log 2>&1; tail -3 logs/score_RP_v05.log
echo POST_DONE
