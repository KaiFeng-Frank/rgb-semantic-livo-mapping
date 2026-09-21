#!/usr/bin/env bash
# opt/sweep_v03.sh <parallel> <name=args>...
cd /data/livo_sem
P=$1; shift
mkdir -p opt/out/sweep
i=0
for spec in "$@"; do
  name="${spec%%=*}"; args="${spec#*=}"
  OMP_NUM_THREADS=3 nice -n 5 python3 opt/replay_v03.py $args --tag "$name" \
      --json-out opt/out/sweep/$name.json > opt/out/sweep/$name.log 2>&1 &
  i=$((i+1))
  if [ $((i % P)) -eq 0 ]; then wait; fi
done
wait
echo "SWEEP DONE"
