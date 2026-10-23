# SemanticKITTI 19 类验证准备

## 路线与比较口径

先建立使用官方训练集三维标注的 **GT19 参考模型**，再决定是否开展同监督、同训练预算下的方法比较。它与原来的相机伪标签蒸馏 B0 是两种监督条件，不能将新增三维标注带来的提升写成 B0 的方法收益。原 v04/v05/v06 代码、权重和无目标域三维训练标注的结果保留。

当前工具支持训练、恢复、全点单帧推理、官方 19 类评估和提交 ZIP 检查。**完整 08 验证及隐藏测试提交尚未完成。** 小样本可行性训练使用 04 的八帧标注；它不能证明泛化能力或上榜水平。初版参考训练只使用交叉熵，没有进行超参数搜索，也不是已经收敛的强基线。

训练默认禁止 08 及 11–21。`--mode full` 要求完整的官方训练序列 00–10（排除 08）。评估默认要求 08 全部 4,071 帧；提交要求 11–21 共 20,351 帧的逐点标签。评估不会筛掉低置信度、视野外或远处点，预测为空的有效 GT 点仍计为错误。

## 已完成的旧模型诊断

将归档的 nuScenes 16 类预测按预先冻结的映射转换后，在全部点、全部官方 19 类上计算：

| 公共序列 | 原始 ZS mIoU19 | B0 seed 1 mIoU19 | 差值 |
|---|---:|---:|---:|
| 07 | 29.88% | 33.34% | +3.46 个百分点 |
| 09 | 28.56% | 32.28% | +3.72 个百分点 |

这不是新训练的 19 类模型，也不是官方测试成绩。固定映射无法输出 bicyclist、motorcyclist、parking、trunk、pole、traffic-sign 六类；这些类别仍参与平均。原 common-9 地图分数与本表不直接比较。

可追溯证据：

- [完整 JSON、逐类 IoU、输入哈希](../results/semantickitti19_readiness_20261024/)
- [成功的 CPU 重评分运行](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/actions/runs/40031940747)
- [归档发布](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/releases/tag/sk19-readiness-40031940747-1)

## 本机 GPU 兼容性

实测设备为 RTX 5070 Ti Laptop，12,227 MiB 显存。PyTorch `2.11.0.dev20260201+cu128`、spconv-cu126 `2.3.8`、cumm-cu126 `0.7.11`。Pointcept 固定在 v1.5.1 的 `72a799353c568c29a3365c02a15129ebe7fe637a`，486 个主干张量严格载入，16 类输出头明确更换为随机初始化的 19 类头。

适配使用 PyTorch SDPA 替代原 FlashAttention，使用 `torch.segment_reduce` 替代 CSR pooling 扩展。独立稠密公式验证了注意力/池化的数值和梯度；这是新硬件后端，不声称与旧 4090 环境逐位相同。没有修改 Pointcept 的跟踪源码。

真实 08/000000 点云有 123,389 点、94,444 体素。完整模型三次前后向及优化器更新成功：热身后 0.405–0.517 秒/步，峰值 allocated 6,284 MiB（约 6.14 GiB），reserved 6,766 MiB。这次使用随机目标，**仅是显存/运算测试**。随后使用真实 04 训练标注的 pilot 单独保存；它的训练损失不是验证指标。

真实标注 pilot 已完成 100 次更新，再从保存的优化器、GradScaler 和随机状态恢复并完成第 101 次更新。每次累积四帧，热身后更新耗时中位数约 1.30 秒，训练峰值 allocated 5,338 MiB，无跳过更新。另对未训练的 08 序列三帧生成了完整逐点预测，全部通过点数、标签范围和哈希检查。三帧的诊断度量只保存在证据 JSON 内，不作为正式验证成绩。

数据、环境和检查点放在 `/home/frank/.local/share/semantic-kitti/`，Git 只保留源码及小型证据。HTTP range 下载仅提取明确指定的 ZIP 成员，校验 CRC/SHA256，并检查下载预算和磁盘保留空间；不会整包下载 80GB 原始压缩包。

## 运行命令

以下使用本机已经验证的独立环境（继承已有的 CUDA 12.8 PyTorch，不改共享环境）。关键包版本另见运行证据中的 `local-environment.json`。

新建相同环境时，用已有 CUDA PyTorch 环境的 Python 执行 `-m venv --system-site-packages`，在新 venv 中安装 `tools/semantic_kitti/requirements-gpu.txt`；Pointcept 从官方仓库检出上面的精确提交。源权重从 [v0.6 recovery release](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/releases/download/v0.6-artifacts-20260928/v05-ZS-student.pth) 恢复，SHA256 必须为 `4c1d540a2b0dac5f3dff41c24e76d7f531a9dd3a3c48aac55ee243be7d7df287`，可用 `data.download_verified()` 断点下载并校验。GPU 型号或 PyTorch 版本变化后，应重新执行 `gpu_smoke`。

```bash
SK_ROOT=/home/frank/.local/share/semantic-kitti
SK_PY="$SK_ROOT/venv/bin/python"

# CPU 官方度量、提交约束，以及硬件移植的独立公式测试
"$SK_PY" -m unittest tools.semantic_kitti.test_contract tools.semantic_kitti.test_runtime -v

# 少量真实训练帧，和官方划分中的验证帧；总预算分别不超过 0.1 GiB
"$SK_PY" -m tools.semantic_kitti.data --root "$SK_ROOT/data" \
  --sequence 04 --frames 0,40,80,120,160,200,240,270 --labels --max-gib 0.1
"$SK_PY" -m tools.semantic_kitti.data --root "$SK_ROOT/data" \
  --sequence 08 --frames 0,1000,2000 --labels --max-gib 0.1

# 输出目录需为空；恢复时加 --resume。固定学习率允许显式增加 --steps。
"$SK_PY" -m tools.semantic_kitti.train --dataset "$SK_ROOT/data" \
  --pointcept "$SK_ROOT/Pointcept-v1.5.1" \
  --source "$SK_ROOT/weights/v05-ZS-student.pth" \
  --out "$SK_ROOT/runs/gt19-pilot-new" --steps 100 --accumulation 4

# pilot 只能生成显式诊断输出，不能打包为正式提交。
"$SK_PY" -m tools.semantic_kitti.infer --dataset "$SK_ROOT/data" \
  --pointcept "$SK_ROOT/Pointcept-v1.5.1" \
  --checkpoint "$SK_ROOT/runs/gt19-pilot-new/last.pth" \
  --out "$SK_ROOT/predictions/gt19-pilot-new" --partial-diagnostic
```

标准流程在完整数据和正式训练检查点就绪后运行 `infer`（不带 `--partial-diagnostic`），然后：

```bash
"$SK_PY" -m tools.semantic_kitti.evaluate --dataset "$SK_ROOT/data" \
  --predictions "$SK_ROOT/predictions/gt19-validation" --sequence 08 \
  --out "$SK_ROOT/runs/gt19-validation.json"
"$SK_PY" -m tools.semantic_kitti.submission --dataset "$SK_ROOT/data" \
  --predictions "$SK_ROOT/predictions/gt19-test" \
  --manifest "$SK_ROOT/predictions/gt19-test/inference.json" \
  --out "$SK_ROOT/submissions/gt19.zip"
```

打包器仅生成本地 ZIP，不上传。它要求原生 19 类、正式训练、完整预测、无未来帧/地图融合，并逐文件检查点数、哈希和原始标签 ID。原始来源是 nuScenes 有监督预训练，须在参赛信息中如实披露外部训练数据。

## 官方依据与剩余工作

- [SemanticKITTI tasks](https://www.semantic-kitti.org/tasks.html)：single-scan 19 类；官方主站展示还涉及论文/预印本链接。
- [官方 API](https://github.com/PRBonn/semantic-kitti-api)：固定到 `a9c749e8124b2243b6eef1b8bcf971a9f1173a2d`，原文保留在 `tools/semantic_kitti/vendor/`，附 MIT 许可和哈希。
- [单帧 CodaBench 入口](https://www.codabench.org/competitions/12448/)：当前账户/外部训练数据的具体条款尚未核实，不把本地检查通过等同于主办方准入。

尚需完成：具有代表性的训练和完整 08 评估；在相同监督条件下核实方法收益；根据验证结果决定是否运行完整测试并提交。训练前先根据小规模实测确定时间、数据及显存预算，避免无依据扩展为多组大实验。
