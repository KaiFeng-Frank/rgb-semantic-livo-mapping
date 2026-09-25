#!/usr/bin/env bash
set -u
cd /data/livo_sem
P=/data/miniconda3/envs/ags/bin/python
O=out/v06_sparsediag
say() { echo "[$(date +'%F %T')] $*" | tee -a $O/run_confirm.log; }
say "PREREG sha256 $(sha256sum $O/PREREG.md | cut -c1-16)"
for tp in "B1_r1 out/v04/arms/B1_r1" "D_r1 out/v04/arms/D_r1" "C_r1 out/v04/arms/C_r1" "R_r1 out/v04/arms/R_r1" "RP_r1 out/v05/RP_r1"; do
    set -- $tp
    say "run $1"
    $P tools/sparse_diag.py --seq 07 --pred "$2" --pred-order raw --tag "$1" \
        --json-out "$O/diag_$1.json" > "logs/sdiag_$1.log" 2>&1 \
        || { say "FAILED $1"; tail -8 "logs/sdiag_$1.log" | tee -a $O/run_confirm.log; exit 1; }
done
$P tools/sparse_diag_confirm.py > $O/REPORT_confirm.txt 2>&1
say "report rc=$?"
say "CONFIRM_DONE"
