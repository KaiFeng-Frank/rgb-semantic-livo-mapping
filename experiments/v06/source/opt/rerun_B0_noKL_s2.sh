#!/usr/bin/env bash
# Clean from-scratch rerun of armB0_noKL_s2.  The 2026-09-25 resume of this seed loaded the
# frozen KL-anchor weights into the student (Pointcept v1.5.1 CheckpointLoader strips 7 key
# characters on a single GPU: frozen_backbone.* -> backbone.*), so its epoch 10 trained the
# released model, not the seed.  No resume is used here.
set -u
cd /data/livo_sem
echo -900 | sudo -n tee /proc/$$/oom_score_adj >/dev/null 2>&1
PY=/data/miniconda3/envs/ptv3/bin/python
L=out/v06_train
echo "[$(date +"%F %T")] rerun armB0_noKL_s2 (clean, no resume) -> exp/sk2/armB0_noKL_s2_clean" | tee -a $L/queue.log
t0=$(date +%s)
$PY tools/train_distil_v2.py --config-file src/Pointcept_v151/configs/semantic_kitti/arm_B0_noKL_v2_s2.py \
    --options save_path=/data/livo_sem/exp/sk2/armB0_noKL_s2_clean > $L/armB0_noKL_s2_clean.log 2>&1
rc=$?; dt=$(( ($(date +%s) - t0) / 60 ))
best=$(grep -o "Best mIoU: [0-9.]*" $L/armB0_noKL_s2_clean.log | tail -1)
if [ $rc -eq 0 ] && [ -f exp/sk2/armB0_noKL_s2_clean/model/model_best.pth ]; then
    echo "[$(date +"%F %T")] rerun armB0_noKL_s2_clean  OK   ${dt} min  $best" | tee -a $L/queue.log
else
    echo "[$(date +"%F %T")] rerun armB0_noKL_s2_clean  FAILED rc=$rc ${dt} min" | tee -a $L/queue.log
fi
echo "RERUN_DONE" >> $L/queue.log
