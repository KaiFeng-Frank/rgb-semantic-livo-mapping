#!/usr/bin/env bash
# opt/v06_all.sh -- the v0.6 run matrix, ONE run at a time, nothing else on the box.
# Per repetition i = 1..3 (interleaved so machine-state drift is spread over the arms):
#   fl_i          FAST-LIVO2 alone                                   (contention baseline, ATE)
#   off_i         semantic node alone, offline TUM (= v0.5 rt B0)    (contention baseline, OFF map)
#   onopt_i       FAST-LIVO2 + node on /aft_mapped_to_init, cv       (ON-opt map, stream_i)
#   iso_i         node alone on stream_onopt_i.tum                   (CAUSAL-ISO map)
#   onimu_i       FAST-LIVO2 (imu_rate_odom) + node on /LIVO2/imu_propagate   (ON-imu map)
#   onopthold_i   as onopt but --pose-extrap hold                    (extrapolation-policy ablation)
#   onopttx_i     as onopt with the tuned LARGE_DATA transport          (the /semantic_scan delivery fix under full load)
# then the per-run analysis (CPU only, after every timed run is over).
cd /data/livo_sem
source /opt/ros/jazzy/setup.bash
LOG=logs/v06_all.log
log() { echo "[all $(date '+%F %T')] $*" | tee -a $LOG; }
run() { # mode tag [env...]
  local mode=$1 tag=$2; shift 2
  log "run $tag ($mode) $*"
  env "$@" bash opt/run_v06.sh $mode $tag >> logs/v06_run_$tag.log 2>&1
  tail -3 logs/v06_run_$tag.log | tee -a $LOG
  sleep 8
}
for i in 1 2 3; do
  run fl        fl_$i
  run off       off_$i
  run onopt     onopt_$i
  if [ -s out/v06/runs/stream_onopt_$i.tum ]; then
    run iso     iso_$i     ISO_TRAJ=/data/livo_sem/out/v06/runs/stream_onopt_$i.tum
  else
    log "no stream for onopt_$i -- iso_$i skipped"
  fi
  run onimu     onimu_$i
  run onopt     onopthold_$i  EXTRAP=hold
  run onopt     onopttx_$i    'FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA?max_msg_size=5MB&sockets_size=20MB&non_blocking=false'
done
log "timed runs done; analysis"
for i in 1 2 3; do for m in fl off onopt iso onimu onopthold onopttx; do t=${m}_$i
  python3 tools/v06_analyze.py --tag $t --ate 2>&1 | tail -2 | tee -a $LOG
done; done
log "V06_ALL_DONE"
