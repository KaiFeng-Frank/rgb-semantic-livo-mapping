#!/usr/bin/env bash
# opt/run_v06.sh MODE TAG -- one v0.6 online-integration run (B0 checkpoint, bag rate 1.0).
#   MODE  fl      FAST-LIVO2 alone (parameter_blackboard + fastlivo_mapping), no semantic node
#         off     semantic node alone on the offline TUM  (== v0.5 rt B0 arguments) + probe
#         onopt   FAST-LIVO2 + node --pose-topic /aft_mapped_to_init (causal) + probe
#         onimu   FAST-LIVO2 (uav.imu_rate_odom:=true) + node --pose-topic /LIVO2/imu_propagate + probe
#         iso     semantic node alone on $ISO_TRAJ (a recorded ON stream, non-causal) + probe
# env   DUR       playback duration in bag seconds (empty = whole bag)
#       WAIT_MS   --pose-wait-ms for the on* modes (default 0)
#       EXTRAP    --pose-extrap  (default cv)
#       NOPROBE=1 skip the subscriber-side probe
#       NODE_EXTRA  extra node arguments (e.g. --scan-reliable 1)
# Node arguments are opt/runs_v05.sh's `rt` line verbatim; only the v0.6 measurement
# outputs and (for on*/iso) the pose source differ.  Everything is written under out/v06.
set +u
MODE=$1; TAG=$2
B=/data/livo_sem; O=$B/out/v06; L=$B/logs; R=$O/runs
source /opt/ros/jazzy/setup.bash
source $B/ros2_ws/install/setup.bash
export FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-LARGE_DATA}   # v0.5 setting unless overridden (transport experiments)
export RCUTILS_LOGGING_BUFFERED_STREAM=0
B7=$B/bags/kitti_seq07_us
T7=$B/out/kitti_seq07_fastlivo2_tum.txt
W=$B/weights/v05
BASE="--reliable --conf-gate 0.5 --expect-voxels 4000000 --ptv3-ckpt $W/B0_student.pth"
FLPKG=$B/ros2_ws/src/FAST-LIVO2
mkdir -p $R $O/tmp $L
log() { echo "[v06 $(date '+%F %T') $TAG] $*"; }
kill_all() {
  pkill -9 -f '[s]emantic_map_node.py' 2>/dev/null; pkill -9 -f '[p]tv3_worker.py' 2>/dev/null
  pkill -9 -f '[b]ag play' 2>/dev/null; pkill -9 -f '[f]astlivo_mapping' 2>/dev/null
  pkill -9 -f '[p]arameter_blackboard' 2>/dev/null; pkill -9 -f '[t]opic_probe_v06.py' 2>/dev/null
  pkill -9 -f '[r]es_sampler_v06.py' 2>/dev/null; pkill -9 -f '[s]tatic_transform_publisher' 2>/dev/null
}
kill_all; sleep 2
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
if [ "$n" -ne 0 ]; then log "REFUSING: $n compute process(es) on the GPU"; exit 1; fi
log "start mode=$MODE load=$(cut -d' ' -f1-3 /proc/loadavg) DUR=${DUR:-all} WAIT_MS=${WAIT_MS:-0} EXTRAP=${EXTRAP:-cv}"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate --format=csv,noheader >> $L/gpu_v06_$TAG.txt

python3 $B/opt/res_sampler_v06.py --out $R/res_$TAG.csv > /dev/null 2>&1 &
SAMP=$!

NEED_FL=0; NEED_NODE=0; NEED_PROBE=1
case $MODE in
  fl) NEED_FL=1; NEED_PROBE=0 ;;
  off|iso) NEED_NODE=1 ;;
  onopt|onimu) NEED_FL=1; NEED_NODE=1 ;;
  *) log "bad mode $MODE"; kill $SAMP; exit 2 ;;
esac
[ "${NOPROBE:-0}" = "1" ] && NEED_PROBE=0

FL=""; BBP=""
if [ $NEED_FL = 1 ]; then
  CFG=$O/tmp/kitti_cfg_$TAG.yaml
  sed "s|seq_name: \".*\"|seq_name: \"v06_$TAG\"|" $B/config/kitti_velodyne64.yaml > $CFG
  rm -f $FLPKG/Log/result/v06_$TAG.txt
  ros2 run demo_nodes_cpp parameter_blackboard --ros-args -r __node:=parameter_blackboard \
       --params-file $B/config/camera_kitti_cam2.yaml > $L/blackboard_$TAG.log 2>&1 &
  BBP=$!
  # the camera loader asks the blackboard for its parameters through a parameter client;
  # under FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA discovery takes longer than the 3 s the
  # offline runner slept (measured: FL aborted with "Camera model not correctly
  # specified") -- wait until the parameter is actually reachable.
  for i in $(seq 1 30); do
    timeout 10 ros2 param get /parameter_blackboard cam_model 2>/dev/null | grep -q Pinhole && break
    sleep 1
  done
  log "blackboard reachable after ${i} probe(s)"
  sleep 2
  EXTRA=""
  [ "$MODE" = "onimu" ] && EXTRA="-p uav.imu_rate_odom:=true"
  ros2 run fast_livo fastlivo_mapping --ros-args -r __node:=laserMapping --params-file $CFG $EXTRA \
       > $L/fl_$TAG.log 2>&1 &
  FL=$!
  sleep 5
  if ! kill -0 $FL 2>/dev/null; then log "fastlivo_mapping DIED at startup"; tail -20 $L/fl_$TAG.log; kill $SAMP $BBP; exit 1; fi
  log "FAST-LIVO2 up (pid $FL) $EXTRA"
fi

NODE=""
if [ $NEED_NODE = 1 ]; then
  NODELOG=$L/node_$TAG.log
  case $MODE in
    off)   POSE="--traj $T7" ;;
    iso)   POSE="--traj $ISO_TRAJ" ;;
    onopt) POSE="--pose-topic /aft_mapped_to_init --pose-extrap ${EXTRAP:-cv} --pose-wait-ms ${WAIT_MS:-0} --pose-record-tum $R/stream_$TAG.tum --pose-record-npz $R/stream_$TAG.npz --pose-used-out $R/pused_$TAG.npz" ;;
    onimu) POSE="--pose-topic /LIVO2/imu_propagate --pose-depth 5000 --pose-extrap ${EXTRAP:-cv} --pose-wait-ms ${WAIT_MS:-0} --pose-record-tum $R/stream_$TAG.tum --pose-record-npz $R/stream_$TAG.npz --pose-used-out $R/pused_$TAG.npz" ;;
  esac
  python3 $B/src/semantic_map_node.py $POSE --bag-rate 1.0 $BASE ${NODE_EXTRA:-} \
      --npz-out $R/map_$TAG.npz --frame-log $R/flog_$TAG.npz \
      --stats-out $R/stats_$TAG.json --ptv3-log $L/ptv3_$TAG.log \
      --ros-args -p use_sim_time:=true > $NODELOG 2>&1 &
  NODE=$!
  for i in $(seq 1 300); do
    grep -q "PTv3 worker ready" $NODELOG && break
    kill -0 $NODE 2>/dev/null || { log "NODE DIED"; tail -30 $NODELOG; kill_all; exit 1; }
    sleep 1
  done
  grep -q "PTv3 worker ready" $NODELOG || { log "worker never ready"; tail -30 $NODELOG; kill_all; exit 1; }
  log "node up after ${i}s"
fi

PROBE=""
if [ $NEED_PROBE = 1 ]; then
  python3 $B/opt/topic_probe_v06.py --out $R/probe_$TAG.npz --idle 25 > $L/probe_$TAG.log 2>&1 &
  PROBE=$!
  sleep 3
fi

DURARG=""
[ -n "$DUR" ] && DURARG="--playback-duration $DUR"
T0=$(date +%s.%N)
ros2 bag play --clock --rate 1.0 -d 3 $DURARG $B7 > $L/bagplay_$TAG.log 2>&1   # -d 3: let discovery finish before the first sample
T1=$(date +%s.%N)
log "bag wall $(echo "$T1 - $T0" | bc) s"

if [ -n "$NODE" ]; then
  for i in $(seq 1 120); do grep -q "bag finished" $NODELOG && break; sleep 1; done
  grep -q "bag finished" $NODELOG && log "node finished (+${i}s)" || log "WARNING: node did not report finish"
fi
if [ -n "$FL" ]; then
  sleep 8            # drain (the evo file is appended per frame; nothing is buffered)
  kill -INT $FL 2>/dev/null; sleep 2
  kill -9 $FL $BBP 2>/dev/null
  if [ -s $FLPKG/Log/result/v06_$TAG.txt ]; then
    cp $FLPKG/Log/result/v06_$TAG.txt $R/fl_evo_$TAG.tum
    log "FL evo poses: $(wc -l < $R/fl_evo_$TAG.tum)"
  else
    log "WARNING: no FL evo trajectory written"
  fi
fi
if [ -n "$PROBE" ]; then
  for i in $(seq 1 40); do grep -q "probe:" $L/probe_$TAG.log && break; sleep 1; done
  grep -q "probe:" $L/probe_$TAG.log || { kill -INT $PROBE 2>/dev/null; sleep 3; }
  tail -1 $L/probe_$TAG.log
fi
kill -TERM $SAMP 2>/dev/null; sleep 2
kill_all
[ -n "$NODE" ] && [ -f $R/stats_$TAG.json ] && python3 - "$R/stats_$TAG.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
pm = d.get("pose_mode") or {}
print("[v06] recv %d proc %d bp-drop %d no-pose %d stale %s | voxels %d | period %.1f ms | latency p50 %.1f p95 %.1f | pose msgs %s extrap p50 %s p95 %s"
      % (d["scans_received"], d["scans_processed"], d["scans_dropped_backpressure"], d["scans_no_pose"],
         pm.get("scans_no_pose_stale", "-"), d["map_voxels"], d["frame_period_ms"]["mean"],
         d["latency_ms"]["p50"], d["latency_ms"]["p95"], pm.get("msgs", "-"),
         ("%.0f" % pm["extrap_span_ms"]["p50"]) if pm else "-", ("%.0f" % pm["extrap_span_ms"]["p95"]) if pm else "-"))
EOF
log "DONE"
