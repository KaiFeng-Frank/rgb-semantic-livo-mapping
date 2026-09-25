#!/usr/bin/env bash
# opt/v05_orchestrate.sh -- sequencing for the v0.5 re-qualification.
# GPU jobs never overlap each other; the timing runs start only when the box is quiet.
cd /data/livo_sem
P3=/data/miniconda3/envs/ptv3/bin/python
P2=/data/miniconda3/envs/ags/bin/python
log() { echo "[orch $(date "+%F %T")] $*"; }
until grep -q RP_CACHE_DONE logs/rp_cache_v05.log 2>/dev/null; do sleep 15; done
log "RP caches done"
# ---- 2a: S4 re-run, 101 frames, relative verdict rule (GPU, serial)
( for spec in "ZS weights/nuscenes-semseg-pt-v3m1-0-base/model/model_best.pth -" \
              "B0 exp/sk/armB0/model/model_best.pth weights/v05/ZS_student.pth" \
              "Rprime_noKL exp/sk/armRprime_noKL/model/model_best.pth weights/v05/B0_student.pth"; do
    set -- $spec; NEG=""; [ "$3" != "-" ] && NEG="--neg $3"
    $P3 tools/extract_student.py --ckpt $2 --tag $1 --out weights/v05/$1_student.pth \
        --report out/v05/extract_$1.json $NEG > logs/extract_v05b_$1.log 2>&1
    echo "$1: $(tail -1 logs/extract_v05b_$1.log)"
  done; echo EXTRACT2_DONE ) > logs/extract_v05b.log 2>&1 &
# ---- 2b: Rprime_noKL replays (CPU), in parallel with 2a
( for r in 1 2 3; do
    $P2 opt/replay_v05.py --pred out/v05/RP_r$r --pred-order raw --tag RP_r$r \
        --json-out out/v05/map_RP_r$r.json > logs/replay_v05_RP_r$r.log 2>&1 &
  done; wait; echo REPLAY_RP_DONE ) > logs/replay_batch2_v05.log 2>&1 &
wait
log "stage 2 done: $(tail -1 logs/extract_v05b.log) / $(tail -1 logs/replay_batch2_v05.log)"
until grep -q REPLAY_BATCH1_DONE logs/replay_batch1_v05.log 2>/dev/null; do sleep 15; done
sleep 20
log "box quiet (load $(cut -d" " -f1 /proc/loadavg)); starting the timing runs"
bash opt/runs_v05.sh all > logs/runs_v05_all.log 2>&1
log "timing runs done"
echo ORCH_TIMING_DONE
