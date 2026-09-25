#!/usr/bin/env bash
# third interleaved saturated rep (c) for all three models: after the first chain, before any CPU post-work
cd /data/livo_sem
until grep -q ORCH_TIMING_DONE logs/v05_orchestrate.log 2>/dev/null; do sleep 15; done
sleep 15
echo "[orch2 $(date "+%F %T")] load $(cut -d" " -f1 /proc/loadavg); rep c"
for m in ZS B0 RP; do bash opt/runs_v05.sh sat $m c >> logs/runs_v05_all.log 2>&1; sleep 8; done
echo ORCH_TIMING2_DONE
