#!/usr/bin/env bash
# Build the FAST-LIVO2 (ROS 2 Jazzy) workspace without apt access to packages.ros.org,
# using RoboStack/conda-forge. Recreates HANDOFF_fastlivo2_trajectory.md S6 on a
# fresh Ubuntu 24.04 host; used for docs/fastlivo2_reproduction.md.
#
#   usage: tools/setup_fastlivo2_conda.sh [BASE=/data/livo_sem]
#
# Result: $BASE/ros2_ws/install with sophus, livox_ros_driver2 (msg-only),
# vikit_common, vikit_ros, fast_livo; /opt/mm/ros_env.sh activates the env.
# Pins follow Ubuntu 24.04 where it matters numerically: Eigen 3.4.0, gcc 13,
# CMake 3.28, Sophus = the ros-jazzy-sophus 1.22.9102 release source.
set -euo pipefail
BASE=${1:-/data/livo_sem}
MM=${MM:-/opt/mm}
WS=$BASE/ros2_ws
export MAMBA_ROOT_PREFIX=$MM/root

# ---- micromamba (micro.mamba.pm may be unreachable; conda-forge serves the same binary)
if [ ! -x $MM/bin/micromamba ]; then
  mkdir -p $MM
  curl -sSL -o $MM/mm.tar.bz2 https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.9.0-0.tar.bz2
  tar -xjf $MM/mm.tar.bz2 -C $MM bin/micromamba
fi

# ---- ROS 2 Jazzy + build deps
if [ ! -d $MAMBA_ROOT_PREFIX/envs/ros ]; then
  $MM/bin/micromamba create -y -q -n ros -c robostack-jazzy -c conda-forge \
    ros-jazzy-ros-base ros-jazzy-rosbag2 ros-jazzy-rosbag2-storage-mcap ros-jazzy-pcl-ros \
    ros-jazzy-pcl-conversions ros-jazzy-cv-bridge ros-jazzy-image-transport ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs ros-jazzy-visualization-msgs ros-jazzy-nav-msgs \
    ros-jazzy-demo-nodes-cpp ros-jazzy-rosidl-default-generators \
    "eigen=3.4.0" "ceres-solver=2.2" "cmake>=3.28,<3.29" "gxx_linux-64=13" "gcc_linux-64=13" \
    glog fmt compilers make ninja pkg-config colcon-common-extensions \
    numpy scipy matplotlib pillow
fi
cat > $MM/ros_env.sh <<EOF
export MAMBA_ROOT_PREFIX=$MAMBA_ROOT_PREFIX
eval "\$($MM/bin/micromamba shell hook -s bash)"
micromamba activate ros
EOF
# run_fastlivo2_kitti.sh / build_bags.sh source /opt/ros/jazzy/setup.bash
if [ ! -e /opt/ros/jazzy/setup.bash ]; then
  mkdir -p /opt/ros/jazzy
  echo "source $MM/ros_env.sh  # shim: ROS 2 Jazzy from the RoboStack env" > /opt/ros/jazzy/setup.bash
fi

# ---- sources, at the commits recorded in the handoff
mkdir -p $WS/src && cd $WS/src
[ -d FAST-LIVO2 ] || { git clone -q -b humble https://github.com/Robotic-Developer-Road/FAST-LIVO2;
                       git -C FAST-LIVO2 checkout -q 837b7bbc1431cb04cf936528e52c83c835efba8e; }
[ -d rpg_vikit ]  || { git clone -q https://github.com/Robotic-Developer-Road/rpg_vikit;
                       git -C rpg_vikit checkout -q 4b7abc838f5d2ca9137f70f122eaaeff9eaf0f50;
                       rm -rf rpg_vikit/vikit_py; }
[ -d sophus ]     || git clone -q --depth 1 -b release/jazzy/sophus/1.22.9102-2 \
                       https://github.com/ros2-gbp/sophus-release sophus

# ---- livox_ros_driver2: message-only stand-in (the AVIA path is never entered on KITTI)
if [ ! -d livox_ros_driver2 ]; then
  mkdir -p livox_ros_driver2/msg
  cat > livox_ros_driver2/msg/CustomPoint.msg <<'EOF'
uint32 offset_time      # offset time relative to the base time
float32 x               # X axis, unit:m
float32 y               # Y axis, unit:m
float32 z               # Z axis, unit:m
uint8 reflectivity      # reflectivity, 0~255
uint8 tag               # livox tag
uint8 line              # laser number in lidar
EOF
  cat > livox_ros_driver2/msg/CustomMsg.msg <<'EOF'
std_msgs/Header header    # ROS standard message header
uint64 timebase           # The time of first point
uint32 point_num          # Total number of pointclouds
uint8  lidar_id           # Lidar device id number
uint8[3]  rsvd            # Reserved use
CustomPoint[] points      # Pointcloud data
EOF
  cat > livox_ros_driver2/package.xml <<'EOF'
<?xml version="1.0"?>
<package format="3">
  <name>livox_ros_driver2</name>
  <version>1.0.0</version>
  <description>Message-only stand-in for livox_ros_driver2; wire-compatible CustomMsg/CustomPoint.</description>
  <maintainer email="none@example.com">livo_sem</maintainer>
  <license>MIT</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>rosidl_default_generators</buildtool_depend>
  <depend>std_msgs</depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>
  <export><build_type>ament_cmake</build_type></export>
</package>
EOF
  cat > livox_ros_driver2/CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.8)
project(livox_ros_driver2)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME} "msg/CustomPoint.msg" "msg/CustomMsg.msg" DEPENDENCIES std_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
EOF
fi

# ---- the build-system-only patches of HANDOFF S6 (no estimator source is touched)
python3 - <<'EOF'
import re
FALLBACK = """
# [livo_sem build patch] ros-jazzy-sophus exports only the Sophus::Sophus target
if(NOT Sophus_INCLUDE_DIRS AND TARGET Sophus::Sophus)
  get_target_property(Sophus_INCLUDE_DIRS Sophus::Sophus INTERFACE_INCLUDE_DIRECTORIES)
endif()
"""
def patch(path, find_re, extra=()):
    s = open(path).read()
    if "[livo_sem build patch]" in s:
        return
    s, n = re.subn(find_re, lambda m: m.group(0) + FALLBACK, s, count=1)
    assert n == 1, path
    for a, b in extra:
        assert a in s, (path, a)
        s = s.replace(a, b, 1)
    open(path, "w").write(s)
    print("patched", path)

patch("rpg_vikit/vikit_common/CMakeLists.txt", r"FIND_PACKAGE\(Sophus REQUIRED\)", [
    ('-std=c++0x")', '-std=c++17")  # [livo_sem build patch] was c++0x'),
    ('-std=c++11")', '-std=c++17")  # [livo_sem build patch] was c++11'),
    ("  ${Sophus_LIBRARIES}\n  fmt::fmt)", "  ${Sophus_LIBRARIES}\n  Sophus::Sophus\n  fmt::fmt)"),
])
patch("rpg_vikit/vikit_ros/CMakeLists.txt", r"find_package\(Sophus REQUIRED\)", [
    ("target_link_libraries(${PROJECT_NAME}\n  ${cpp_typesupport_target}\n)",
     "target_link_libraries(${PROJECT_NAME}\n  ${cpp_typesupport_target}\n  Sophus::Sophus\n)"),
])
patch("FAST-LIVO2/CMakeLists.txt", r"find_package\(Sophus REQUIRED\)", [
    ("set(COMMON_DEPENDENCIES OpenMP::OpenMP_CXX fmt::fmt)",
     "set(COMMON_DEPENDENCIES OpenMP::OpenMP_CXX fmt::fmt Sophus::Sophus)"),
])
EOF

# ---- build
set +u; source $MM/ros_env.sh; set -u
cd $WS
colcon build --packages-select sophus livox_ros_driver2 vikit_common vikit_ros fast_livo \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_SOPHUS_TESTS=OFF \
               -DBUILD_SOPHUS_EXAMPLES=OFF -DBUILD_TESTING=OFF
echo "[setup] done. source $MM/ros_env.sh && source $WS/install/setup.bash"
