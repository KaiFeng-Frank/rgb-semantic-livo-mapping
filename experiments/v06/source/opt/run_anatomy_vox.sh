#!/usr/bin/env bash
# opt/run_anatomy_vox.sh -- POST-HOC voxel-weighted anatomy.  CPU only, one replay at a time.
set -u
cd /data/livo_sem
P=/data/miniconda3/envs/ags/bin/python
O=out/v06_anatomy
say() { echo "[$(date +'%F %T')] $*" | tee -a $O/run_vox.log; }
[ -f $O/PREREG_voxel.md ] || { say "no PREREG_voxel.md -- refusing to run"; exit 1; }
say "PREREG_voxel sha256 $(sha256sum $O/PREREG_voxel.md | cut -c1-16)  ($(tail -1 $O/PREREG_voxel.md))"
for tp in "ZS_r1 out/vs2d/ptv3_r1" "B0_r1 out/v04/arms/B0_r1" "B0_r2 out/v04/arms/B0_r2" "RP_r1 out/v05/RP_r1"; do
    set -- $tp
    say "replay $1"
    $P tools/residual_anatomy_vox.py --pred "$2" --pred-order raw --tag "$1" \
        --json-out "$O/anatv_$1.json" > "logs/anatv_$1.log" 2>&1 \
        || { say "FAILED $1"; tail -8 "logs/anatv_$1.log" | tee -a $O/run_vox.log; exit 1; }
done
$P tools/anatomy_voxel_report.py > $O/REPORT_voxel.txt 2>&1
say "report rc=$?"
say "VOX_DONE"
