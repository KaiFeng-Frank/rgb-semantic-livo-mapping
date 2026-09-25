#!/usr/bin/env bash
set -u
cd /data/livo_sem
P=/data/miniconda3/envs/ags/bin/python
O=out/v06_sparsediag
say() { echo "[$(date +'%F %T')] $*" | tee -a $O/run.log; }
[ -f $O/PREREG.md ] || { say "no PREREG.md"; exit 1; }
say "PREREG sha256 $(sha256sum $O/PREREG.md | cut -c1-16)  ($(tail -1 $O/PREREG.md))"
for tp in "B0_r1 out/v04/arms/B0_r1" "B0_r2 out/v04/arms/B0_r2" "ZS_r1 out/vs2d/ptv3_r1"; do
    set -- $tp
    say "run $1"
    $P tools/sparse_diag.py --seq 07 --pred "$2" --pred-order raw --tag "$1" \
        --json-out "$O/diag_$1.json" > "logs/sdiag_$1.log" 2>&1 \
        || { say "FAILED $1"; tail -8 "logs/sdiag_$1.log" | tee -a $O/run.log; exit 1; }
    say "  done $1"
done
$P tools/sparse_diag_report.py > $O/REPORT.txt 2>&1
say "report rc=$?"
say "SDIAG_DONE"
