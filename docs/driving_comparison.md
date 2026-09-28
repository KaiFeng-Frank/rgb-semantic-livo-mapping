# 原始 ZS 与训练后 B0：同帧驾驶语义对比

[高清 MP4](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/releases/download/driving-comparison-20260928/driving_zs_vs_b0_seq07.mp4)
· [循环 GIF](img/driving_zs_vs_b0.gif)
· [完整附件](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/releases/tag/driving-comparison-20260928)

MP4 约 27 MB；README 的 GIF 约 8.5 MB，1008 × 235、106 帧、7.5 fps、14.03 秒，
保留原速并循环播放。GIF 是降低帧率的预览，全部 131 个扫描帧请看 MP4。

左边是**原始预训练 PTv3（ZS）**，右边是**研究方法训练后的 v0.6 B0，seed 1**。
B0 使用经过 filter E 筛选的相机伪标签蒸馏、全网络微调和相机视野外的 KL 防遗忘约束。
原始 ZS 从 nuScenes 发布权重提取，没有在 KITTI 上微调。

两边都使用单帧 LiDAR 输入。RGB 只用于展示点云投影；蓝色为汽车，橙色为行人。
画面中是模型预测；点更多、颜色更连贯本身不等于准确率更高。

## 同一时间段如何对齐

- 固定为[之前驾驶演示](driving_demo.md)的 KITTI 07 连续帧 **970–1100**，共 131 帧。
- 每个画面左右使用**同一帧点云、同一张相机图像**，131 个原始点云的 SHA-256
  逐一匹配之前演示的输入记录。没有按任一模型的效果删帧或换片段。
- 两组均使用原实验保存的**第一轮评分预测 r1**，统一原始点序、相机标定、投影视角、
  类别配色、绘点大小和 **0.5 置信度阈值**。
- 预测的公共设置为 Pointcept v1.5.1、488 个模型张量、强度乘 0.2、0.05 m 体素、
  FP16、关闭 shuffle 和 TTA。原始评分脚本及其校验值随附件保存。
- 按原始相机时间戳以 1 倍速更新，MP4 为 2520 × 588、30 fps、421 帧、14.03 秒，
  保持与之前演示相同的视频时长，结尾短暂保持最后一帧。

本次是**逐帧对齐的离线效果对比**。它不模拟两个模型各自算完的时刻，不能从左右
画面判断哪个推理更快。之前的[延迟演示](driving_demo.md)仍单独保留。
这里没有接入多帧地图融合、动态过滤或 FAST-LIVO2 在线联动。

B0 权重与之前的延迟演示相同，但这里复用评分缓存，之前演示使用另一次实时测量产生的预测。
PTv3 本身存在运行间差异，因此这两次 B0 输出不保证逐点相同。

## person 的量化参照

已有整序列评测支持 B0 的行人分割能力提升。下面是 **common-9、相机视野外、单帧预测的
person IoU（%）**，不是这 14 秒前视视频的评分，也不是目标检测召回率。

| 序列 | 原始 ZS | B0 | 提升（百分点） |
|---|---:|---:|---:|
| 07 | 38.61 ± 0.29 | 52.73 ± 1.44 | +14.11 |
| 09（独立测试） | 59.15 ± 0.05 | 69.54 ± 0.79 | +10.39 |

数据来自[逐次运行汇总](../results/v06/scoring/runs_summary.json)中
`outside.offline_all.per_class_iou.person`。ZS 为 2 次推理，B0 为 3 个训练种子 × 2 次推理，
表中为平均值 ± 总体标准差。视频展示的是其中的 B0 seed 1；完整评测口径见
[v0.6 结果](v06_results.md)。

## 模型和恢复证据

| 左右位置 | 模型 | 已备份的权重 |
|---|---|---|
| 左 | ZS，原始预训练 PTv3 | `v05-ZS-student.pth`，SHA-256 `4c1d540a2b0dac5f3dff41c24e76d7f531a9dd3a3c48aac55ee243be7d7df287` |
| 右 | v0.6 B0 seed 1，蒸馏 + KL | `v06-armB0_s1-student.pth`，SHA-256 `35f5d3aec63ae1c383f4a4146e157aa054860cb12b178d899571287e6f97e5f1` |

权重、完整评分缓存和原始运行时位于[实验归档 Release](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/releases/tag/v0.6-artifacts-20260928)。
原服务器关闭后，本次在 GitHub Actions 的 CPU 环境恢复缓存，并从 KITTI 官方数据源
按范围读取这 131 帧的图像和点云；没有重新训练或重新推理，原始数据未上传到仓库或 Release。
本次范围读取约 361 MB，无须下载完整 4.4 GB ZIP；所有原始输入只在云端临时使用。

[逐帧来源与校验记录](../results/v06/driving_comparison_20260928/comparison.json)
· [恢复记录](../results/v06/driving_comparison_20260928/input-recovery.json)
· [视频校验](../results/v06/driving_comparison_20260928/video-check.json)
· [GIF 校验](../results/v06/driving_comparison_20260928/gif-check.json)
· [发布回执](../results/v06/driving_comparison_20260928/publication.json)

[本次视频附件](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/releases/tag/driving-comparison-20260928)
保存视频、GIF、预览图、262 个确切的逐帧预测及以下证据：

- `comparison.json`：模型、输入、预测、标定的校验值和逐帧来源。
- `input-recovery.json`：官方数据来源与原演示输入一致性检查。
- `video-check.json`：完整解码检查以及每个视频画面对应的扫描帧号。
- `gif-check.json`：GIF 时长、帧率、循环和保留类别配色的导出记录。
- `archive-check.json`：预测子集归档内每个成员的校验。
- `scoring-sources.json`：从归档恢复并校验的两组评分脚本。
- `comparison-manifest.json` 和 `SHA256SUMS`：发布文件清单和校验值。

## 复现

已有原始数据、预测缓存和归档索引时，直接运行
[渲染脚本](../tools/compare_driving_semantics.py)。服务器已释放时，可运行
[云端恢复工作流](../.github/workflows/driving-comparison.yml)，其中
[输入恢复脚本](../tools/restore_driving_comparison.py)会校验 GitHub 归档和原始点云。
需要 NumPy、OpenCV、Pillow、Requests 以及 H.264 编码器；不需要 GPU 或模型权重下载。

```bash
python tools/restore_driving_comparison.py --root /tmp/comparison-inputs --out out/comparison
python tools/compare_driving_semantics.py --root /tmp/comparison-inputs --out out/comparison
```

云端工作流创建草稿 Release 供视觉检查；正式发布后，重复运行应使用新的 Release 标签，
避免覆盖已保存的历史产物。
