# 在线回环语义重融合

入口为 `tools.loop_closure.semantic_node`。实时传感器驱动 LIVO，PTv3 对到达的
去畸变关键帧在线推理，语义地图接收已经完成的图优化版本。整个过程不读取最终
离线轨迹或预测缓存。默认关键帧约 **1 Hz**；位姿仍逐帧输出。此入口不等同于
每个 10 Hz 扫描都运行语义分割，也没有接入相机 RGB 着色。

## 数据与更新

1. `/backend/input/semantic_cloud` 输出完整去畸变 XYZI，坐标系为
   `backend_lidar`，时间戳与几何后端关键帧严格一致。它与最多 12,000 点的
   回环几何输入分开；超过语义 200,000 点上限时整帧拒绝，不静默抽样。
   前端在 LIO 阶段缓存待导出的关键帧，在同一时刻 VIO 完成后发布，避免
   `Process2` 在视觉阶段清空原始去畸变缓存。
2. PTv3 使用现有 B0、Pointcept v1.5.1 和严格权重加载。KITTI 反射强度归一化
   除数为 1；本次 Livox 的 uint8 反射强度显式除以 255，再应用模型的 0.2
   强度缩放。不把该传感器迁移解释为已经证明的语义准确率。
3. 图优化器独立发布 `/backend/trajectory`：整数时间戳、完整关键帧位姿和
   图版本。它的进程独立于 XYZ 地图显示。语义点只有找到完全相同时间戳的
   已接纳关键帧才能入图，不使用最近邻时间戳、未来插值或尚未完成的优化。
4. 外参取同一前端 YAML，且只应用一次：
   `p_map = T_map_IMU × T_IMU_LiDAR × p_LiDAR`。
   原始局部语义证据保留为只读数组。每个关键帧拥有自己的分块贡献；位姿变化时
   替换旧贡献并重新规约受影响的旧、新空间块。重复更新不会增加投票次数。
5. 每个发布快照对应一个完整应用的图版本。若处理期间又有新回环到达，先发布
   当前完整版本，再处理最新版本；状态同时给出最新接收版本与已完成版本。
   不把混合新旧位姿的中间地图标记成已经完成。

采用简单的置信度加权投票，不额外加入尚未证实更优的重访均衡机制。
地图 `confidence` 是胜出类别的票重占比，不是校准后的概率。显示点位于体素中心。
默认体素 0.2 m，空间块每边 20 个体素。局部更新只重算受影响块；如果回环改变
全部关键帧，必须更新全部对应贡献，不能声称成本恒定。显示快照的全局排序同样
计入计时，显示最多 200,000 点，内部融合与一致性验证使用完整地图。

## 动态类别与容量

当前扫描保留全部 16 类。长期静态地图排除 bicycle、bus、car、construction_vehicle、
motorcycle、pedestrian、trailer、truck，并过滤低于 0.35 的置信度。
这是保守的类别策略，不是逐点运动估计：静止车辆也不会长期写入静态地图。
`barrier`、`traffic_cone` 等仍可入图；没有承诺识别所有实际移动物体。

默认最多 1,000 个关键帧、4,000,000 个过滤后保留的局部点。达到容量明确报告
`capacity_reached`，停止接纳新地图证据；已有贡献仍能跟随位姿修正。
这是一段有界会话，没有完成无限时长的子图淘汰与图压缩。

## 运行

先按[几何后端说明](realtime_backend.md)建立本版本前端工作空间；旧补丁工作空间
不能直接当成本版本二进制使用。语义环境目录结构与恢复出的本机环境相同：
`venv/bin/python`、`src/Pointcept_v151`、`v06-armB0_s1-student.pth` 和
`weights/nuscenes-semseg-pt-v3m1-0-base/config.py`。不自动下载数据或权重。

```bash
export SEMANTIC_RUNTIME=/path/to/semantic_runtime
export SEMANTIC_REFLECTANCE_DIVISOR=255  # 本例 Livox；KITTI 为 1
export CYCLONEDDS_URI="file://$PWD/tools/loop_closure/local_dds.xml"
bash tools/loop_closure/run_online.sh /path/to/backend_ws \
  /path/to/geometry-venv/bin/python /path/to/new-run \
  tools/loop_closure/frontend/hku.yaml tools/loop_closure/frontend/camera_hku.yaml
```

等 `/backend/semantic_status` 的 `ready=true` 后发布传感器数据。同机 DDS 配置
不能直接用于跨主机网络。关闭进程时仅回收本次启动的推理和显示子进程。

| 输出 | 内容 |
|---|---|
| `/backend/semantic_scan` | 当前关键帧全部类别，`backend_imu` 坐标，原采集时间戳 |
| `/backend/semantic_map` | 已重融的静态地图，`map` 坐标 |
| `/backend/semantic_map_info` | 本次发布 ID、图时间戳/版本、处理耗时与地图规模 |
| `/backend/semantic_status` | 就绪、丢弃、容量、队列、版本落后和显示进程健康 |

点云字段为 `x,y,z`、`label:uint16`、`confidence:float32`；发布端明确声明字段
偏移和点步长。当前扫描在自己的采集时刻坐标系中表达，回环一致性以静态地图
及其版本元数据为准。

当前扫描与静态地图分别在独立进程发布。两者各保留一个在途输出和最新快照；
静态地图显示暂停不会阻塞当前扫描。当前扫描自身的显示若持续阻塞，中间显示
快照被覆盖的数量会由 `scan_superseded` 明确报告；已融合的原始证据不因此删除。
地图元数据包含点云内容 SHA-256，验证器实际订阅地图并匹配最终发布版本与哈希，
不以“已生成”或“已经 publish”替代订阅端收到。

## 验证与边界

冻结预算在 `tools/loop_closure/semantic_realtime_protocol.json`。实采验证原速读取
HKU1 MCAP，实时运行 LIVO、PTv3、在线回环和语义重融。对照也订阅同样的密集
导出，以计入相同的前端导出负担。ROS 接收端测量位姿、语义扫描与地图延迟，
队列必须排空，不能靠丢帧通过。独立故障测试暂停语义进程/显示，检查位姿继续。

`--verify-final` 在输入结束后，以保留的原始局部预测和最终已到达位姿重新构建
参考地图，比较全部体素键、类别、票重与证据计数。此检查验证撤回/重融正确性，
它是结束后的诊断，不算作在线运行或语义准确率评价。核心测试另用独立逐点
规约器交叉检查，防止两个相同实现互相证明。

`--predictions-input` 仅供明确标注的合成 ROS 合约测试，生产入口默认实时推理。
HKU1 没有本地语义真值；已有 Seq07 的离线 mIoU 增益不能移作本在线入口的成绩。
实测结果见[在线重融合记录](../results/online_semantic_refusion_20261026/README.md)。

可选的 `--consistency shadow|active` 为当前扫描增加短时跨帧一致性；默认关闭，
长期地图仍保存原始分类证据。完整概率传输、几何/置信度门控、回环重对齐、
依赖隔离及实测限制见[跨帧一致性说明](semantic_consistency.md)。

实采复核使用系统 ROS Python，先 source ROS 与前端工作空间：

```bash
python3 -m tools.loop_closure.realtime_replay \
  --workspace /path/to/backend_ws --worker /path/to/geometry-venv/bin/python \
  --bag /path/to/FAST-LIVO/hku1 --out /path/to/new-control --backend --semantic-export
python3 -m tools.loop_closure.realtime_replay \
  --workspace /path/to/backend_ws --worker /path/to/geometry-venv/bin/python \
  --bag /path/to/FAST-LIVO/hku1 --out /path/to/new-online --backend \
  --semantics --semantic-runtime /path/to/semantic_runtime
python3 -m tools.loop_closure.assess_semantic_realtime \
  --control /path/to/new-control --online /path/to/new-online --out /path/to/new-assessment.json
```
