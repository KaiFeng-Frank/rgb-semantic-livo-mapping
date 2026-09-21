#!/usr/bin/env bash
cd /data/livo_sem
# GPU clocks locked (see logs/gpu_clock_lock.txt); arms strictly interleaved so any
# residual drift is charged equally to both.
./opt/run.sh v03_warm src bags/kitti_seq07_us out/kitti_seq07_fastlivo2_tum.txt 2.0 \
     --reliable --conf-gate 0.5 --expect-voxels 4000000 --idle-finish 12 >/dev/null 2>&1
for i in a b c; do
  for arm in cap_off cap_on; do
    nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader >> logs/gpu_clk_${arm}_$i.txt
    sed "s/v03_${arm}/v03_${arm}_$i/" /dev/null >/dev/null
    ARM=$arm IDX=$i bash -c '
      cd /data/livo_sem
      B7=bags/kitti_seq07_us; T7=out/kitti_seq07_fastlivo2_tum.txt
      BASE="--conf-gate 0.5 --expect-voxels 4000000"
      DYN="--dyn --dyn-n-az 150 --dyn-n-el 64 --dyn-margin 0.6 --dyn-r-min 3.0 --dyn-r-max 25.0 --dyn-dil-el 1 --dyn-dil-az 1 --dyn-reset-on-seen 0 --dyn-k-free 24"
      if [ "$ARM" = "cap_on" ]; then EX="$DYN"; else EX=""; fi
      ./opt/run.sh v03_T_${ARM}_${IDX} src $B7 $T7 2.0 --reliable $BASE $EX --idle-finish 15'
    sleep 8
  done
done
echo TIMING2_DONE
