#!/usr/bin/env bash
# Launch the existing LIVO frontend and its independent online geometric backend.
set -euo pipefail
if (( $# < 3 || $# > 5 )); then
  echo 'usage: run_online.sh WORKSPACE WORKER_PYTHON OUTPUT_DIR [FRONTEND_YAML] [CAMERA_YAML]' >&2
  exit 2
fi
WORKSPACE=$(realpath "$1")
# Do not realpath a venv executable: that silently bypasses the venv.
WORKER_PYTHON=$2
[[ "$WORKER_PYTHON" = /* ]] || WORKER_PYTHON="$PWD/$WORKER_PYTHON"
OUTPUT_DIR=$(realpath -m "$3")
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
FRONTEND_CONFIG=${4:-"$REPO_DIR/config/kitti_velodyne64.yaml"}
CAMERA_CONFIG=${5:-"$REPO_DIR/config/camera_kitti_cam2.yaml"}
[[ -x "$WORKER_PYTHON" && -f "$WORKSPACE/install/setup.bash" ]]
mkdir "$OUTPUT_DIR"  # Each run has separate evidence; never overwrite old logs.
set +u
source /opt/ros/jazzy/setup.bash
source "$WORKSPACE/install/setup.bash"
set -u
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=1
cd "$REPO_DIR"
pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
/usr/bin/python3 tools/loop_closure/frontend/parameter_blackboard.py --ros-args --params-file "$CAMERA_CONFIG" > "$OUTPUT_DIR/camera.log" 2>&1 &
pids+=("$!")
/usr/bin/python3 -m tools.loop_closure.online_node --worker-python "$WORKER_PYTHON" \
  --log "$OUTPUT_DIR/backend.jsonl" > "$OUTPUT_DIR/backend.log" 2>&1 &
pids+=("$!")
"$WORKSPACE/install/fast_livo/lib/fast_livo/fastlivo_mapping" --ros-args \
  --params-file "$FRONTEND_CONFIG" --params-file "$REPO_DIR/tools/loop_closure/frontend/backend.yaml" \
  > "$OUTPUT_DIR/frontend.log" 2>&1 &
pids+=("$!")
echo "Online backend launched; publish LiDAR/IMU/images. Logs: $OUTPUT_DIR"
echo 'Outputs: /backend/pose /backend/path /backend/map /backend/status and map -> camera_init -> backend_imu TF'
wait -n "${pids[@]}"
