#!/usr/bin/env bash
# opt/v05_captures.sh -- M3 screenshots, AFTER every timing run has finished (POST_DONE).
# Same viewpoints as the v0.2 deliverables: top-down overview (focal 20,93,-3 / dist 360 /
# pitch 1.50) and the oblique street view (4.6,21.7,-0.5 / dist 55 / pitch 0.42); plus the
# zoom tile where the ZS and B0 live maps disagree most (tools/v05_zoom_region.py, top tile).
cd /data/livo_sem
until grep -q POST_DONE logs/v05_post.log 2>/dev/null; do sleep 20; done
read ZX ZY ZZ < <(python3 -c "import json; t=json.load(open(out/v05/zoom_ZS_B0.json))[tiles][0]; print(%.1f %.1f %.1f % tuple(t[center]))")
echo "[cap $(date "+%T")] zoom centre $ZX $ZY $ZZ"
cp -n out/rviz_class_legend.png out/v05/rviz_class_legend_rainbow0-15.png 2>/dev/null
for M in ZS B0 RP; do
  N=out/v05/map_$M.npz
  bash opt/capture_v05.sh ${M}_top_class  $N 20 93 -3 360 1.50 1.5708 class 3
  bash opt/capture_v05.sh ${M}_top_rgb    $N 20 93 -3 360 1.50 1.5708 rgb 3
  bash opt/capture_v05.sh ${M}_obl_class  $N 4.6 21.7 -0.5 55 0.42 1.5708 class 4
  bash opt/capture_v05.sh ${M}_obl_rgb    $N 4.6 21.7 -0.5 55 0.42 1.5708 rgb 4
  bash opt/capture_v05.sh ${M}_zoom_class $N $ZX $ZY $ZZ 30 0.85 1.5708 class 6
  bash opt/capture_v05.sh ${M}_zoom_rgb   $N $ZX $ZY $ZZ 30 0.85 1.5708 rgb 6
done
echo CAPTURES_DONE
