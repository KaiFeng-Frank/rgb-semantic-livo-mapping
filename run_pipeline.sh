#!/usr/bin/env bash
# Self-contained end-to-end run.  Launch detached:
#   setsid nohup /data/livo_sem/run_pipeline.sh TAG BAG TRAJ RATE [node args] \
#       > /data/livo_sem/logs/pipe_TAG.log 2>&1 &
#
# NOTE on pkill: every pattern below is bracketed ([s]emantic...) so it cannot
# match this script's own command line.  An unbracketed `pkill -f "ros2 bag play"`
# once killed the wrapper shell that launched it.
set +u
TAG=$1; BAG=$2; TRAJ=$3; RATE=$4; shift 4
B=/data/livo_sem
source /opt/ros/jazzy/setup.bash
# MEASURED: the stock UDP builtin transports lose ~63% of 2.7 MB best-effort
# PointCloud2 samples on this box; LARGE_DATA (TCP for big samples) cuts it to ~22%.
export FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-LARGE_DATA}
mkdir -p $B/logs $B/out
pkill -9 -f '[s]emantic_map_node.py' 2>/dev/null
pkill -9 -f '[p]tv3_worker.py' 2>/dev/null
pkill -9 -f '[b]ag play' 2>/dev/null
sleep 2

NODELOG=$B/logs/node_$TAG.log
python3 $B/src/semantic_map_node.py \
    --traj "$TRAJ" --bag-rate "$RATE" \
    --stats-out $B/out/stats_$TAG.json \
    --npz-out  $B/out/map_$TAG.npz \
    --ptv3-log $B/logs/ptv3_$TAG.log \
    --hold "$@" --ros-args -p use_sim_time:=true > $NODELOG 2>&1 &
NODE=$!
echo "[pipe] node pid $NODE -> $NODELOG"
for i in $(seq 1 300); do
  grep -q "PTv3 worker ready" $NODELOG && break
  kill -0 $NODE 2>/dev/null || { echo "[pipe] NODE DIED"; tail -40 $NODELOG; exit 1; }
  sleep 1
done
grep -q "PTv3 worker ready" $NODELOG || { echo "[pipe] worker never ready"; tail -40 $NODELOG; exit 1; }
echo "[pipe] worker ready after ${i}s"

ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 \
     --qx 0 --qy 0 --qz 0 --qw 1 --frame-id map --child-frame-id velodyne \
     > /dev/null 2>&1 &
TFPID=$!
echo $NODE  > /tmp/sem_$TAG.node.pid
echo $TFPID > /tmp/sem_$TAG.tf.pid

if [ "${SEM_RVIZ:-0}" = "1" ]; then
  DISPLAY=:0 rviz2 -d ${SEM_RVIZ_CFG:-$B/rviz/semantic_map_rgb.rviz} \
      > $B/logs/rviz_$TAG.log 2>&1 &
  echo $! > /tmp/sem_$TAG.rviz.pid
  echo "[pipe] rviz2 started on :0"
  sleep 8
fi

# resource sampler: VRAM + RSS of node and worker, every 2 s
RES=$B/logs/res_$TAG.csv
( echo "t,vram_mib,node_rss_kb,worker_rss_kb,gpu_util";
  while true; do
    V=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    NR=$(ps -o rss= -p $NODE 2>/dev/null | tr -d ' '); NR=${NR:-0}
    WP=$(pgrep -f '[p]tv3_worker.py' | head -1)
    WR=$(ps -o rss= -p ${WP:-0} 2>/dev/null | tr -d ' '); WR=${WR:-0}
    echo "$(date +%s),${V%,*},$NR,$WR,${V#*,}"
    sleep 2
  done ) > $RES 2>/dev/null &
SAMP=$!
echo $SAMP > /tmp/sem_$TAG.samp.pid

echo "[pipe] BAGSTART $(date +%s.%N)"
ros2 bag play --clock --rate "$RATE" "$BAG" > $B/logs/bagplay_$TAG.log 2>&1
echo "[pipe] BAGEND $(date +%s.%N)"

for i in $(seq 1 120); do
  grep -q "bag finished" $NODELOG && break
  sleep 2
done
kill $SAMP 2>/dev/null
echo "[pipe] DONE"
