#!/usr/bin/env bash
# opt/v06_post.sh -- CPU post-processing AFTER every timed run (waits for V06_ALL_DONE).
#   1. map-level scoring: opt/replay_v06.py, 4 in parallel, run i paired with cached draw B0_r{i}
#      off_i      --traj offline TUM,   --live-npz map_off_i        (OFF replay + OFF live)
#      onopt_i    --traj stream_onopt_i --live-npz map_onopt_i --pose-used pused_onopt_i
#                                                                    (CAUSAL-ISO replay + ON live causal + geometry)
#      iso_i      --traj stream_onopt_i --live-npz map_iso_i        (CAUSAL-ISO live)
#      onimu_i / onopthold_i / onopttx_i   as onopt_i with their own streams
#   2. trajectory divergence in the shared W frame (tools/v06_traj_diff.py)
#   3. zoom tiles (tools/v05_zoom_region.py): ON-opt vs CAUSAL-ISO (same frame), ON-opt vs OFF
#   4. RViz2 captures (opt/capture_v06.sh) -- latched map, after everything else
cd /data/livo_sem
P2=/data/miniconda3/envs/ags/bin/python
R=out/v06/runs
log() { echo "[post $(date '+%F %T')] $*"; }
until grep -q V06_ALL_DONE logs/v06_all.log 2>/dev/null; do sleep 30; done
sleep 10
log "timed runs done (load $(cut -d' ' -f1 /proc/loadavg)); replays"
rep() { # tag traj live pose_used(or -) draw
  local tag=$1 traj=$2 live=$3 pu=$4 draw=$5 extra=""
  [ "$pu" != "-" ] && extra="--pose-used $pu --diag-out $R/diag_$tag.npz"
  [ "$tag" = "off_1" ] && extra="$extra --save-map $R/replaymap_off_1.npz"
  [ "$tag" = "onopt_1" ] && extra="$extra --save-map $R/replaymap_onopt_1.npz"
  $P2 opt/replay_v06.py --pred out/v04/arms/B0_r$draw --pred-order raw --traj $traj --live-npz $live \
      --tag $tag --json-out $R/replay_$tag.json $extra > logs/replay_v06_$tag.log 2>&1
  echo "done $tag: $(grep -E 'live_lookup_causal|live_lookup_interp' logs/replay_v06_$tag.log | tail -1 | cut -c1-110)"
}
export -f rep; export P2 R
OFF=/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt
{
for i in 1 2 3; do
  [ -f $R/map_off_$i.npz ] && echo "off_$i $OFF $R/map_off_$i.npz - $i"
  for m in onopt onimu onopthold onopttx; do
    [ -f $R/map_${m}_$i.npz ] && echo "${m}_$i $R/stream_${m}_$i.tum $R/map_${m}_$i.npz $R/pused_${m}_$i.npz $i"
  done
  [ -f $R/map_iso_$i.npz ] && echo "iso_$i $R/stream_onopt_$i.tum $R/map_iso_$i.npz - $i"
done
} | xargs -P 4 -L 1 bash -c 'rep $0 $1 $2 $3 $4'
log "replays done; trajectory divergence"
for i in 1 2 3; do
  for m in onopt onimu onopthold onopttx; do
    [ -f $R/stream_${m}_$i.tum ] && python3 tools/v06_traj_diff.py --a $R/stream_${m}_$i.tum --b $OFF --json-out $R/trajdiff_${m}_${i}_vs_offline.json
  done
  [ -f $R/fl_evo_fl_$i.tum ] && python3 tools/v06_traj_diff.py --a $R/fl_evo_fl_$i.tum --b $OFF --json-out $R/trajdiff_fl_${i}_vs_offline.json
  [ -f $R/fl_evo_fl_$i.tum ] && [ -f $R/stream_onopt_$i.tum ] && python3 tools/v06_traj_diff.py --a $R/stream_onopt_$i.tum --b $R/fl_evo_fl_$i.tum --json-out $R/trajdiff_onopt_${i}_vs_fl_$i.json
done
log "zoom tiles"
$P2 tools/v05_zoom_region.py --a $R/map_iso_1.npz --b $R/map_onopt_1.npz --gt $R/replaymap_onopt_1.npz --json-out out/v06/zoom_iso_onopt.json | tail -6
$P2 tools/v05_zoom_region.py --a $R/map_off_1.npz --b $R/map_onopt_1.npz --gt $R/replaymap_off_1.npz --json-out out/v06/zoom_off_onopt.json | tail -6
$P2 tools/v05_zoom_region.py --a $R/map_onopt_1.npz --b $R/map_onimu_1.npz --gt $R/replaymap_onopt_1.npz --json-out out/v06/zoom_onopt_onimu.json | tail -6
log "captures"
read ZX ZY ZZ < <(python3 -c "import json; t=json.load(open('out/v06/zoom_iso_onopt.json'))['tiles'][0]; print('%.1f %.1f %.1f' % tuple(t['center']))")
log "zoom centre (ISO vs ON-opt top tile) $ZX $ZY $ZZ"
cp -n out/rviz_class_legend.png out/v06/rviz_class_legend_rainbow0-15.png 2>/dev/null
for M in off_1 iso_1 onopt_1 onimu_1; do
  N=$R/map_$M.npz
  bash opt/capture_v06.sh ${M}_top_class $N 20 93 -3 360 1.50 1.5708 class 3 18
  bash opt/capture_v06.sh ${M}_top_rgb   $N 20 93 -3 360 1.50 1.5708 rgb 3 18
  bash opt/capture_v06.sh ${M}_obl_class $N 4.6 21.7 -0.5 55 0.42 1.5708 class 4 18
  bash opt/capture_v06.sh ${M}_obl_rgb   $N 4.6 21.7 -0.5 55 0.42 1.5708 rgb 4 18
done
for M in off_1 iso_1 onopt_1; do
  bash opt/capture_v06.sh ${M}_zoom_class $R/map_$M.npz $ZX $ZY $ZZ 30 0.85 1.5708 class 6 18
  bash opt/capture_v06.sh ${M}_zoom_rgb   $R/map_$M.npz $ZX $ZY $ZZ 30 0.85 1.5708 rgb 6 18
done
log "POST_DONE"
