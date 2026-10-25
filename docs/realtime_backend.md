# 在线几何后端与实时性验证

此入口以实时 LIVO 位姿和去畸变点云为输入，边接收传感器数据边检索回环、
执行增量位姿图优化并更新地图；不读取预计算轨迹。源码固定在
FAST-LIVO2 ROS 2 提交 `837b7bbc1431cb04cf936528e52c83c835efba8e`。
独立补丁包含构建兼容、图像所有权、整数采集时间戳、有界后端导出、有限深度
传感器订阅、原生压缩图像入口和直接 LIVO 的调度修复，
不应用研究分支的语义特征、边界、关联或视觉语义补丁。

## 运行

需要 ROS 2 Jazzy、Cyclone DDS、PCL/OpenCV/Eigen/fmt 和 colcon。几何优化使用
独立 Python 环境；依赖固定在 `tools/loop_closure/requirements.txt`。

```bash
python3 tools/loop_closure/frontend/bootstrap.py --workspace /path/to/backend_ws
bash tools/loop_closure/frontend/build.sh /path/to/backend_ws
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$PWD/tools/loop_closure/local_dds.xml"
bash tools/loop_closure/run_online.sh /path/to/backend_ws \
  /path/to/geometry-venv/bin/python /path/to/new-run
```

默认使用 KITTI 的 LiDAR/IMU/相机配置。更换传感器时，第四、第五个参数必须
传入对应前端与相机配置。每次运行创建新的日志目录。bootstrap 校验固定上游
文件的 Git blob 哈希，拒绝覆盖不同版本的工作空间；不会下载数据或模型。

上述实测启动配置使用 `tools/loop_closure/local_dds.xml`；同机传感器驱动与
处理进程须使用一致的设置。它固定 loopback、单播发现、适合本机 MTU 的
分片和 4 MiB socket 缓冲。
它仅用于同一主机；跨机器 ROS 驱动应使用对应网络配置，不能套用 loopback
的大报文设置。无需更改全局 sysctl，操作系统达不到缓冲要求时启动失败。
HKU 配置开启 `common.img_compressed`，直接解码原始相机 JPEG 消息，不重新编码。

## 执行路径

- 前端仍执行原始直接 LIVO。后端只复用去畸变、下采样的几何，每秒导出
  至多 12,000 点；每次视觉更新后发布同一采集时刻的位姿。
- 前端复用一个 ROS executor。当前图像对应的 LIO 已完成后，立即运行
  VIO，不再因原始点云/IMU输入队列暂时为空而等待下一扫描。视觉残差和
  状态更新公式不变；回归检查要求最后一帧不依赖未来点云才能输出。
- 位姿 ROS 进程立即应用最近已完成、非未来的 `map → camera_init` 校正。
  因子图和 ICP 在独立、单线程、低优先级进程中运行。
- 地图和路径在第三个进程构造及发布。即使显示进程暂停，也不会占住位姿
  executor 或图优化进程。显示只使用最新快照，输出 QoS 为 best effort、深度 1。
- 待处理关键帧最多两个，超载时替换最旧待处理帧；按跨过的时间间隔扩大
  里程计边的不确定度。过期队列丢弃，迟到结果计数并显示状态。独立 watchdog
  为完整 IPC 交换计时，包括半包导致的阻塞读取。
- 图/地图进程健康状态分别报告；地图发布有确认消息，卡住超过两秒后
  显式告警，恢复后清除。期限取消、超时认领和恢复使用各自状态锁，避免
  根据旧期限误杀已完成的优化进程或错误报告显示卡死。

输出为 `/backend/pose`、`/backend/path`、`/backend/map`、`/backend/status`，
以及 `map → camera_init → backend_imu` TF。原始里程计继续定义图的边，
不会把修正位姿回灌为独立 IEKF 测量。无回环时保持严格单位校正。

`/backend/map` 是 XYZ 几何图。它保留关键帧局部点云，并按优化位姿重新摆放
历史几何；它还不是回环后可撤回、重新融合的语义地图。

## 可复核测量

先 source ROS 和工作空间，再用系统 Python 运行：

```bash
python3 -m tools.loop_closure.realtime_replay \
  --workspace /path/to/backend_ws --worker /path/to/geometry-venv/bin/python \
  --bag /path/to/original/FAST-LIVO/hku1 --out /path/to/new-evidence --backend
```

不加 `--backend` 是相同导出配置的前端对照。回放使用原始 MCAP，原速发布
LiDAR/IMU/图像，不复制数据。图像解码及播放器调度迟到单独记录；每个输出
必须精确匹配整数图像时间戳。输入发布至最终位姿接收的延迟包含 ROS 排队，
不能用优化器耗时或回调内部计时替代。

`validate_online_ros` 另行暂停本次启动的优化进程和地图进程，检查位姿仍连续
输出。测试是故障隔离验证，不替代真实序列测量。

## 适用边界

当前是有界会话：默认最多 1,000 个关键帧，达到后明确报告
`capacity_reached`，继续输出位姿但停止图增长。约 1 Hz 关键帧时对应约
17 分钟；这不是支持无限时长的子图后端。不要仅增大容量来掩盖这一限制。
实时性结果必须连同序列、时长、关键帧数和图规模一起报告。

这是几何 Pose3 图，不是含视觉重投影、IMU 预积分、速度和偏置状态的联合
全局 LIVO 优化。这里的时间测量也不包含 PTv3 语义推理，不能代表完整语义
建图链路。回环被接纳不等于定位精度得到提高，准确率需要另外的配对评估。
