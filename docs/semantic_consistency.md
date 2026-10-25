# 在线短时跨帧语义一致性

实现为 `src/semantic_consistency.py`，在线入口为
`tools.loop_closure.semantic_node --consistency active`。它使用同一次 PTv3
前向的完整 16 类概率，对当前扫描施加有限的历史支持；不重训练、不增加定位前端
残差，也不把逐类概率从 top-1 标签和置信度反推出来。

这是**推理时的短时一致性基线**，不是训练一致性损失、动态目标跟踪或完整地图上下文
网络。回环只负责把保留的上一帧原始证据重新对齐到已经到达的图版本。

## 因果性与更新

- 每次只保留上一帧的原始传感器坐标、完整原始概率、传感器时间和位姿。历史最大
  1.5 秒、60,000 点；超过点数上限时当前输出完全退回单帧，并清空旧历史。
- 在当前传感器坐标系建立双向最近邻，距离不超过 0.25 m，要求互为最近邻；
  两个传感器视线方向上的距离误差分别不超过 0.15 m。先让全部点参与几何匹配，
  再过滤类别，避免删除前景后误接后面的静态表面。
- 当前和历史的预测类别都须属于静态候选；历史 top-1 概率至少 0.8，并比当前
  高至少 0.1；当前概率达到 0.8 时否决历史。这是置信度启发式，没有声称概率已校准。
  当前帧即可确定的类别/置信度否决在搜索之前执行；两棵搜索树仍包含全部点，
  不删除潜在遮挡物。重复查询坐标和重复历史候选分别只执行一次查询。真实评估额外执行全点
  查询参考，逐帧核对完整输出概率相等；参考搜索不计入在线模块耗时。
- 通过门控时 `p_out = (1-w)*p_current + w*p_previous_raw`，
  `w = 0.35*exp(-age_seconds/2)`。未通过门控的点逐值保留原始概率。
- 写入历史的始终是 `p_current`，不把 `p_out` 当成新的独立证据。
  每次图更新先修正历史传感器位姿；锚点缺失清空历史，序列重置必须显式 reset。

几何接近和高置信不等于可靠真值。此版本没有完整遮挡推理、表面法向验证或物体
运动估计；静态类别也可能实际移动。错误高置信历史仍能影响弱但正确的当前预测，
因此实测同时保留纠错、致错、跳变和连续错误计数。

## 在线接口

`--consistency off|shadow|active` 默认 `off`。`shadow` 执行相同计算并记录统计，
当前扫描仍发布原始结果；`active` 发布增强后的当前扫描。两者的长期地图都保存
**原始分类证据**，避免在已经跨帧投票的地图里再次重复计算历史支持。
后端回环重融照常撤回、重投原始证据，不重跑分割。

`PTv3Client(probability_classes=16).segment_probabilities(points)` 经共享内存返回
同一次前向的标签、置信度和完整概率。旧 `segment()` 保持二元组契约；新开关不会
从外部环境意外继承，禁止与实验 learned-memory 路径叠加。默认四槽、200,000 点
配置额外分配 51.2 MB 共享内存；短时历史最多约 5.28 MB，不随序列增长。

在线 ROS Python 需要 NumPy 和 SciPy（本机验证 SciPy 1.13.1）。若几何 worker
和 ROS 使用不同 Python 版本，不能把 ROS 的二进制依赖全局放进 `PYTHONPATH`。
可以为语义进程独立指定可选依赖目录：

```bash
export SEMANTIC_CONSISTENCY=active
export SEMANTIC_ROS_PYTHONPATH=/path/to/ros-python-compatible-dependencies
# 其余 SEMANTIC_RUNTIME、反射率与工作空间设置见 online_semantic_refusion.md
bash tools/loop_closure/run_online.sh WORKSPACE GEOMETRY_PYTHON NEW_OUTPUT_DIR
```

直接使用 `semantic_node` 时，将相同依赖路径只提供给该进程。记录中的
`consistency.json` 固定算法参数；每个 keyframe 事件包含配对、使用、改类、退回原因、
保留字节数和模块耗时。语义仍是约 1 Hz 关键帧，不声称逐扫描 10 Hz。

## 效果与复核

真实评估入口 `tools.semantic_memory.consistency_eval`；从 nuScenes 官方验证集
选择六个未参加此前 20 场景实验的采集日志，每个日志一个场景。选集和参数先冻结，
每帧只执行一次前向，四组共享其输出。数据原位读取，不保存大型特征缓存。
类别真值和物体框只在预测完成后用于评估。该数据是原始 keyframe + 数据集提供的
ego pose，不是已去畸变的在线定位输入，也不是 KITTI19 榜单成绩。

```bash
python -m tools.semantic_memory.consistency_eval prepare \
  --old-index PRIOR_20_SCENE_INDEX --out NEW_INDEX_DIRECTORY
python -m tools.semantic_memory.consistency_eval evaluate \
  --index NEW_INDEX_DIRECTORY/index.json --out NEW_RESULT_DIRECTORY
```

冻结的 B0 checkpoint、模型运行配置、代码和输入文件哈希均记录在结果中。
同一几何匹配上的普通平均融合是必要对照；不能只报告条件版本相对单帧的小幅提升。
稳定性在**相同的几何匹配且 GT 类别不变的点对**上统计，不使用门控各自筛选的子集。
置信区间按采集日志成组抽样，不能把数百万相关点当成独立样本。

另有真实 ROS 合约 `tools.semantic_memory.validate_consistency_ros`，输入是明确标注
的人工概率，用实际订阅结果验证增强扫描、当前观测推翻历史、原始地图证据与图修正。
其数据不用于准确率或延迟结论。完整 HKU1 原速回放使用真实 PTv3、LIVO 和在线
回环，专门验证运行与交付；HKU1 没有本地语义 GT，不能据此声称分割精度提高。

结果和结论见[实测记录](../results/semantic_consistency_20261026/README.md)。
