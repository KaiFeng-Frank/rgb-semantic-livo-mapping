#!/bin/bash
set -x
export DEBIAN_FRONTEND=noninteractive
echo "=== [1] 基础工具 ==="
sudo -n apt-get install -y curl gnupg lsb-release software-properties-common
echo "=== [2] 根分区空间检查 (ROS2 装 /opt/ros, 属根分区) ==="
df -h /
echo "=== [3] 加 ROS2 源 (用清华镜像加速) ==="
sudo -n curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg 2>/dev/null \
  || sudo -n curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
  || sudo -n apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu noble main" | sudo -n tee /etc/apt/sources.list.d/ros2.list
echo "=== [4] apt update ==="
sudo -n apt-get update
echo "=== [5] 装 ros-jazzy-desktop (含 RViz2) ==="
sudo -n apt-get install -y ros-jazzy-desktop ros-jazzy-pcl-ros ros-jazzy-pcl-conversions ros-jazzy-cv-bridge ros-jazzy-tf2-ros ros-jazzy-rosbag2-storage-mcap python3-colcon-common-extensions
echo "=== [6] 结果 ==="
ls /opt/ros/ ; df -h /
echo "=== DONE rc=$? ==="
