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
if [[ "${INSTANCE_SEMANTICS:-off}" != off && -z "${VISUAL_INSTANCE_WEIGHTS:-}" ]]; then
  echo 'INSTANCE_SEMANTICS requires VISUAL_INSTANCE_WEIGHTS.' >&2; exit 2
fi
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
backend_args=()
visual_args=()
if [[ -n "${VISUAL_INSTANCE_WEIGHTS:-}" ]]; then
  [[ -n "${SEMANTIC_RUNTIME:-}" && -f "$VISUAL_INSTANCE_WEIGHTS" && "${SEMANTIC_CONSISTENCY:-off}" == off ]] || {
    echo 'Visual instances require SEMANTIC_RUNTIME, an existing weights file, and SEMANTIC_CONSISTENCY=off.' >&2; exit 2;
  }
  visual_args=(--visual-instances --camera "$CAMERA_CONFIG" --instance-semantics "${INSTANCE_SEMANTICS:-off}")
  /usr/bin/python3 -m tools.visual_instances.node \
    --worker-python "$SEMANTIC_RUNTIME/venv/bin/python" --weights "$VISUAL_INSTANCE_WEIGHTS" \
    --out "$OUTPUT_DIR/vision" --frontend "$FRONTEND_CONFIG" --camera "$CAMERA_CONFIG" \
    > "$OUTPUT_DIR/vision.log" 2>&1 &
  pids+=("$!")
fi
if [[ -n "${SEMANTIC_RUNTIME:-}" ]]; then
  for f in "$SEMANTIC_RUNTIME/venv/bin/python" "$SEMANTIC_RUNTIME/v06-armB0_s1-student.pth" \
           "$SEMANTIC_RUNTIME/weights/nuscenes-semseg-pt-v3m1-0-base/config.py"; do
    [[ -f "$f" ]] || { echo "Missing semantic runtime file: $f" >&2; exit 2; }
  done
  backend_args+=(--semantic-trajectory)
  PTV3_PYTHON="$SEMANTIC_RUNTIME/venv/bin/python" \
  PTV3_POINTCEPT="$SEMANTIC_RUNTIME/src/Pointcept_v151" \
  PTV3_CONFIG="$SEMANTIC_RUNTIME/weights/nuscenes-semseg-pt-v3m1-0-base/config.py" \
  PTV3_CHECKPOINT="$SEMANTIC_RUNTIME/v06-armB0_s1-student.pth" \
  PYTHONPATH="${SEMANTIC_ROS_PYTHONPATH:+${SEMANTIC_ROS_PYTHONPATH}:}${PYTHONPATH:-}" \
  OMP_NUM_THREADS=1 /usr/bin/python3 -m tools.loop_closure.semantic_node \
    --frontend "$FRONTEND_CONFIG" --out "$OUTPUT_DIR/semantics" \
    --reflectance-divisor "${SEMANTIC_REFLECTANCE_DIVISOR:-1}" \
    --consistency "${SEMANTIC_CONSISTENCY:-off}" \
    "${visual_args[@]}" \
    > "$OUTPUT_DIR/semantics.log" 2>&1 &
  pids+=("$!")
fi
/usr/bin/python3 -m tools.loop_closure.online_node --worker-python "$WORKER_PYTHON" "${backend_args[@]}" \
  --log "$OUTPUT_DIR/backend.jsonl" > "$OUTPUT_DIR/backend.log" 2>&1 &
pids+=("$!")
"$WORKSPACE/install/fast_livo/lib/fast_livo/fastlivo_mapping" --ros-args \
  --params-file "$FRONTEND_CONFIG" --params-file "$REPO_DIR/tools/loop_closure/frontend/backend.yaml" \
  > "$OUTPUT_DIR/frontend.log" 2>&1 &
pids+=("$!")
echo "Online backend launched; publish LiDAR/IMU/images. Logs: $OUTPUT_DIR"
echo 'Outputs: /backend/pose /backend/path /backend/map /backend/status and map -> camera_init -> backend_imu TF'
if [[ -n "${SEMANTIC_RUNTIME:-}" ]]; then
  echo 'Semantic outputs: /backend/semantic_scan /backend/semantic_map /backend/semantic_map_info /backend/semantic_status'
  echo 'Wait for /backend/semantic_status ready=true before publishing sensors.'
fi
if [[ -n "${VISUAL_INSTANCE_WEIGHTS:-}" ]]; then
  echo 'Instance outputs: /backend/semantic_instances + /backend/semantic_instance_info (pair by stamp; semantic identity includes its epoch).'
  echo "Also wait for /backend/visual_status ready=true. Instance semantics mode: ${INSTANCE_SEMANTICS:-off}."
fi
wait -n "${pids[@]}"
