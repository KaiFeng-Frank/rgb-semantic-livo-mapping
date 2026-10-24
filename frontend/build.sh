#!/usr/bin/env bash
set -euo pipefail
WS=${1:?usage: frontend/build.sh WORKSPACE}
FRONTEND=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Keep a system ROS build independent of a shell's conda activation.
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin"
unset CONDA_PREFIX
set +u
source /opt/ros/jazzy/setup.bash
set -u
cd "$WS"
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-2}
colcon build --packages-select sophus livox_ros_driver2 vikit_common vikit_ros fast_livo \
  --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DBUILD_SOPHUS_TESTS=OFF -DBUILD_SOPHUS_EXAMPLES=OFF -DBUILD_TESTING=OFF \
  -DSEMANTIC_FRONTEND_SOURCE_DIR="$FRONTEND" \
  '-DCMAKE_CXX_FLAGS=-DEIGEN_MAX_ALIGN_BYTES=16 -DEIGEN_MAX_STATIC_ALIGN_BYTES=16' \
  -Dfmt_DIR=/usr/lib/x86_64-linux-gnu/cmake/fmt
