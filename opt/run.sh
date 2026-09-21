#!/usr/bin/env bash
# opt/run.sh TAG SRC BAG TRAJ RATE [node args...]
set +u
TAG=$1; SRCDIR=$2; BAG=$3; TRAJ=$4; RATE=$5; shift 5
B=/data/livo_sem
source /opt/ros/jazzy/setup.bash
export FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-LARGE_DATA}
mkdir -p $B/logs $B/opt/out
pkill -9 -f '[s]emantic_map_node.py' 2>/dev/null
pkill -9 -f '[p]tv3_worker.py' 2>/dev/null
pkill -9 -f '[b]ag play' 2>/dev/null
sleep 2
NODELOG=$B/logs/node_$TAG.log
python3 $B/$SRCDIR/semantic_map_node.py \
    --traj "$TRAJ" --bag-rate "$RATE" \
    --stats-out $B/opt/out/stats_$TAG.json \
    --ptv3-log  $B/logs/ptv3_$TAG.log \
    "$@" --ros-args -p use_sim_time:=true > $NODELOG 2>&1 &
NODE=$!
for i in $(seq 1 300); do
  grep -q "PTv3 worker ready" $NODELOG && break
  kill -0 $NODE 2>/dev/null || { echo "[run] NODE DIED"; tail -30 $NODELOG; exit 1; }
  sleep 1
done
grep -q "PTv3 worker ready" $NODELOG || { echo "[run] worker never ready"; tail -30 $NODELOG; exit 1; }
echo "[run] $TAG worker ready after ${i}s"
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 \
     --qx 0 --qy 0 --qz 0 --qw 1 --frame-id map --child-frame-id velodyne >/dev/null 2>&1 &
TFPID=$!
RES=$B/logs/res_$TAG.csv
( echo "t,vram_mib,node_rss_kb,worker_rss_kb,gpu_util";
  while true; do
    V=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    NR=$(ps -o rss= -p $NODE 2>/dev/null | tr -d ' '); NR=${NR:-0}
    WP=$(pgrep -f '[p]tv3_worker.py' | head -1)
    WR=$(ps -o rss= -p ${WP:-0} 2>/dev/null | tr -d ' '); WR=${WR:-0}
    echo "$(date +%s),${V%,*},$NR,$WR,${V#*,}"
    sleep 1
  done ) > $RES 2>/dev/null &
SAMP=$!
T0=$(date +%s.%N)
ros2 bag play --clock --rate "$RATE" "$BAG" > $B/logs/bagplay_$TAG.log 2>&1
T1=$(date +%s.%N)
echo "[run] bag wall $(echo "$T1 - $T0" | bc)"
for i in $(seq 1 90); do grep -q "bag finished" $NODELOG && break; sleep 1; done
kill $SAMP $TFPID 2>/dev/null
pkill -9 -f '[s]emantic_map_node.py' 2>/dev/null
pkill -9 -f '[p]tv3_worker.py' 2>/dev/null
echo "[run] $TAG DONE"
