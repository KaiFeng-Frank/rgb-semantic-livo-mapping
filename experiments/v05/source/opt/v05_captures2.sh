#!/usr/bin/env bash
# opt/v05_captures2.sh -- M3 screenshots (all timing runs are finished; POST_DONE is in logs/v05_post.log).
# Viewpoints: v0.2 top-down overview (20,93,-3 / 360 / pitch 1.50), v0.2 oblique street view
# (4.6,21.7,-0.5 / 55 / 0.42), and two zoom tiles chosen by tools/v05_zoom_region.py on the LIVE maps:
#   zA (-115.1,143.8,-1.1): the 10 m tile where ZS and B0 disagree most (48.3 % of voxels); B0 relabels
#      manmade->vegetation and is WRONG there (ZS 87.8 %, B0 43.5 %, Rprime 91.0 % of GT-labelled voxels correct)
#   zB (-95.9,4.4,4.0): B0 relabels the zero-shot road->sidewalk and is RIGHT (ZS 42.6 %, B0 84.9 %)
cd /data/livo_sem
grep -q POST_DONE logs/v05_post.log || { echo "post not done"; exit 1; }
rm -f out/v05/rviz_ZS_zoom_class.png out/v05/rviz_ZS_zoom_rgb.png
for M in ZS B0 RP; do
  N=out/v05/map_$M.npz
  bash opt/capture_v05.sh ${M}_top_class  $N 20 93 -3 360 1.50 1.5708 class 3 18
  bash opt/capture_v05.sh ${M}_top_rgb    $N 20 93 -3 360 1.50 1.5708 rgb 3 18
  bash opt/capture_v05.sh ${M}_obl_class  $N 4.6 21.7 -0.5 55 0.42 1.5708 class 4 18
  bash opt/capture_v05.sh ${M}_obl_rgb    $N 4.6 21.7 -0.5 55 0.42 1.5708 rgb 4 18
  bash opt/capture_v05.sh ${M}_zA_class   $N -115.1 143.8 -1.1 30 0.85 1.5708 class 6 18
  bash opt/capture_v05.sh ${M}_zA_rgb     $N -115.1 143.8 -1.1 30 0.85 1.5708 rgb 6 18
  bash opt/capture_v05.sh ${M}_zB_class   $N -95.9 4.4 4.0 30 0.85 1.5708 class 6 18
  bash opt/capture_v05.sh ${M}_zB_rgb     $N -95.9 4.4 4.0 30 0.85 1.5708 rgb 6 18
done
echo CAPTURES2_DONE
