#!/usr/bin/env bash
# opt/run_anatomy.sh -- residual anatomy for the completion-head question.  CPU only,
# one replay at a time (~3 GB each).  Default OOM priority on purpose: if the machine is
# exhausted again, this analysis should die before the training queue (oom_score_adj -900).
set -u
cd /data/livo_sem
P=/data/miniconda3/envs/ags/bin/python
O=out/v06_anatomy
mkdir -p $O logs
say() { echo "[$(date +'%F %T')] $*" | tee -a $O/run.log; }
[ -f $O/PREREG.md ] || { say "no PREREG.md -- refusing to run"; exit 1; }
say "PREREG sha256 $(sha256sum $O/PREREG.md | cut -c1-16)  ($(tail -1 $O/PREREG.md))"
for tp in "ZS_r1 out/vs2d/ptv3_r1" "B0_r1 out/v04/arms/B0_r1" "B0_r2 out/v04/arms/B0_r2" "RP_r1 out/v05/RP_r1"; do
    set -- $tp
    say "replay $1  <- $2"
    $P tools/residual_anatomy.py --pred "$2" --pred-order raw --tag "$1" \
        --json-out "$O/anat_$1.json" > "logs/anat_$1.log" 2>&1 \
        || { say "FAILED $1"; tail -8 "logs/anat_$1.log" | tee -a $O/run.log; exit 1; }
    say "  done $1  $(grep -E "frames [0-9]+" logs/anat_$1.log | tail -1 | cut -c1-90)"
done
$P tools/anatomy_report.py > $O/REPORT.txt 2>&1
rc=$?
cat $O/REPORT.txt >> $O/run.log
say "report rc=$rc"
say "ANATOMY_DONE"
