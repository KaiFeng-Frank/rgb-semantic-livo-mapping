#!/usr/bin/env bash
cd /data/livo_sem
for i in r1 r2 r3; do
  rm -rf out/dump_$i
  ./opt/runs_v03.sh score $i
  sleep 5
done
echo SCORE_RUNS_DONE
