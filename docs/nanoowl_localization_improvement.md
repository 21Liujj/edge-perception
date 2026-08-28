# NanoOWL BBox Localization Refinement 实验报告

## 背景

Stage F 确认 EdgePerception 当前第一算法瓶颈是 **NanoOWL bbox 定位精度**，而非 prompt engineering 或 NanoSAM 分割质量：
- 50 例联合 pipeline 评测中，17 例为 B_localization_propagation（检测成功但框不准，导致 mask 质量下降）
- Box IoU 与 Mask IoU 的 Pearson 相关 r=0.931 (p<0.001)，定位误差强烈传递到下游分割

本阶段（Stage G）针对这一瓶颈设计并验证 **training-free** 的轻量 bbox refinement 方法。

## 当前系统的 Box Selection 机制审计

### OWL-ViT / NanoOWL 内部流程
1. Vision encoder 输出 576 个 patch embeddings（24×24 grid，每个 patch 为原图 32×32 像素区域）
2. 每个 patch 产生：
   - `image_class_embeds` [576, 512]：与 text embedding 对齐的表示
   - `pred_boxes` [576, 4]：该 patch 的 bbox regression 输出（xyxy，原图像素坐标）
3. `decode` 阶段：
   - 计算每个 patch 对 text query 的 classification score（L2-normalize + cosine + logit_shift/scale + sigmoid）
   - 应用 `score > threshold` 过滤（默认 0.1）
   - **返回所有通过 threshold 的 patches，没有 NMS，没有 top-k 排序**

### Project-side 选择规则
当前 `scripts/eval_joint_pipeline_coco.py` 和 `scripts/run_edge_perception.py` 均采用：
- 对单个 text query（如 COCO category name），从 `decode` 返回的所有候选中取 **最高分的一个 patch (top-1)** 作为最终检测框
- 完全不利用其他高分 patch 的空间信息

**假设（待验证）**：如果其他高分 patches 包含更准确的空间信息，则融合或共识机制可能改善定位精度。

## 候选框空间分析（Top-K Candidates）

对 E4 固定 50 例，每例提取 top-5 候选（按 language score 降序），分析发现：

### 关键观察 1：更优候选确实存在
17 例 B_localization_propagation 中，**8 例的非 top-1 候选具有更高的 GT-IoU**（比 top-1 高出 >0.01）。

示例（ann=2221879, pizza）：
- Top-1（rank=0）：score=0.410, GT-IoU=0.017
- Rank=2：score=0.249, GT-IoU=**0.908**

这证明语言匹配最高分的 patch 不一定空间定位最准，存在算法改进空间。

### 关键观察 2：优质候选空间孤立
但深入分析这 8 例发现，**更优候选与 top-1 的 pairwise IoU 极低**：
- ann=1556646 (orange): top-1 与 best-alt (rank=3, GT-IoU 0.638) 的 pairwise IoU = 0.000
- ann=2221879 (pizza): top-1 与 best-alt (rank=2, GT-IoU 0.908) 的 pairwise IoU = 0.017
- ann=1742434 (person): top-1 与 best-alt (rank=3, GT-IoU 0.842) 的 pairwise IoU = 0.171

**这意味着"更优候选"不是 top-1 的空间邻居（轻微偏移），而是完全不同的图像区域被 language matching 打了高分。**

这一发现直接否定了两个 training-free refinement 的前提假设：
1. Score-Weighted Fusion 假设优质候选与 top-1 相邻，可以加权平均得到更准的框 ❌
2. Spatial Consensus 假设优质候选有邻近其他候选的支持（高 pairwise IoU），可通过 consensus score 选中 ❌

## Training-Free Refinement 方法定义

为确保实验的科学性，两个方法的公式和超参在查看任何评估数据前冻结（记录于 `frozen_methods.json`）：

### Method A: Score-Weighted Box Fusion
对 top-K=5 候选进行 softmax 加权融合：
```
weight_i = exp(score_i / temperature) / Σ exp(score_j / temperature)
fused_box = Σ weight_i * box_i
```
- Temperature = 0.1（检测领域标准 softmax temperature，低温使高分权重更集中）

### Method B: Best Spatial Consensus Candidate
从 top-K=5 候选中选择 consensus score 最高的：
```
consensus_i = score_i * (mean_pairwise_IoU_i) ^ gamma
best_box = argmax_i(consensus_i)
```
- Gamma = 2.0（平方惩罚，孤立框的 mean_pairwise_IoU ≈ 0 时 consensus ≈ 0）

### 数据集划分
- E4 固定 50 例按 (size_bucket, scene_type) 分层随机划分（seed=20260828）
- Development subset: 17 例（仅用于观察候选分布，**未用于任何参数调优**）
- Evaluation subset: **33 例**（冻结，唯一允许报告最终结果的集合）

## 实验结果（Frozen Evaluation Split, n=33）

### 整体指标对比

| Metric | Baseline (E4 top-1) | Method A (Fusion) | Method B (Consensus) |
|---|---|---|---|
| Detection success rate | 84.8% | 84.8% | 84.8% |
| Mean Box IoU | 0.532 | **0.479 (↓ 0.054)** | 0.535 (↑ 0.003) |
| Median Box IoU | 0.779 | **0.567 (↓ 0.212)** | 0.779 (≈) |
| Box IoU ≥ 0.5 | 51.5% | **45.5% (↓)** | 51.5% (≈) |
| Box IoU ≥ 0.75 | 42.4% | **36.4% (↓)** | 42.4% (≈) |
| Joint Mask IoU (mean) | 0.407 | **0.378 (↓ 0.029)** | 0.407 (≈) |
| Joint Mask IoU (median) | 0.398 | **0.367 (↓)** | 0.398 (≈) |
| Refinement latency (mean) | — | +0.124 ms | +0.329 ms |

**参考上界（GT-box → NanoSAM）**：mask IoU = 0.728（同一 eval split）

### Paired Case-Level 分析

**Method A** (33 cases):
- Box IoU: 1 improved, **6 degraded**, 26 unchanged
- Mask IoU: 2 improved, **3 degraded**, 28 unchanged
- **净效应：负面**（fusion 把优质 top-1 与低质量候选混合，拉低精度）

**Method B** (33 cases):
- Box IoU: 1 improved, 1 degraded, **31 unchanged**
- Mask IoU: **0 improved**, 0 degraded, 33 unchanged
- **净效应：基本无作用**（多数情况仍选中 top-1 本身）

### B_localization_propagation 恢复情况

Eval split 中 9 例 B_localization_propagation cases：
- **Method A**：0 例恢复（box_iou 从 <0.5 提升到 ≥0.5），2 例明显变差
- **Method B**：0 例恢复，1 例轻微变差

两个方法均未能修复任何一例定位传播失败。

### Small / Medium / Large Object 分解

| Size | n | Baseline Box IoU | Method A | Method B | Baseline Mask IoU | Method A | Method B |
|---|---|---|---|---|---|---|---|
| Small | 12 | 0.292 | **0.252 (↓)** | 0.302 (≈) | 0.234 | **0.225 (↓)** | 0.235 (≈) |
| Medium | 12 | 0.582 | **0.470 (↓)** | 0.580 (≈) | 0.453 | **0.381 (↓)** | 0.453 (≈) |
| Large | 9 | 0.770 | 0.772 (≈) | 0.770 (≈) | 0.575 | 0.577 (≈) | 0.575 (≈) |

**Refinement 并未特别帮助 small object（NanoOWL 历史弱点），Method A 反而在 medium object 上退化最明显。**

## 失败原因分析

### Method A (Fusion) 为何退化？
Fusion 假设"多个高分候选应该聚集在正确位置附近"，加权平均能去噪。但实际情况：
- 候选数量不足（9/33 例仅 1 候选，无融合空间）
- 次高分候选往往是**错误区域**而非 top-1 的微调版本
- Fusion 把正确的 top-1 与错误候选混合 → 框位置被拉偏 → box IoU 下降

### Method B (Consensus) 为何无效？
Consensus 假设"优质候选有邻近其他候选的空间支持"。但实际：
- 优质非 top-1 候选**空间孤立**（pairwise IoU 与其他候选都很低）
- Consensus mechanism 反而惩罚这些孤立的优质候选（mean_pairwise_IoU^2 → 0）
- 最终仍选中 top-1 本身，或选中一个比 top-1 还差的"有邻居"候选

### 根本问题：候选冗余度不足
NanoOWL 的 576 个 patch embeddings 本质是 24×24 的空间网格，粒度较粗（每个 patch 覆盖 32×32 像素）。当目标本身小或边界不对齐 patch grid 时：
- 正确区域可能只有 1 个 patch 高分
- 其他高分 patch 往往是误报（背景区域或错误目标）
- 没有"同一目标的不同 patch"形成 spatial cluster

这与 anchor-based / NMS-heavy 检测器（如 YOLO/Faster R-CNN）的冗余检测框（同一目标产生多个重叠框需 NMS 去重）完全不同。

## 决策与建议

### 本阶段结论
**TRAINING_FREE_LOCALIZATION_REFINEMENT_INSUFFICIENT**

两个 training-free 方法均未展现稳定收益，反而 Method A 造成明显退化。根本原因是 NanoOWL 的候选框冗余度不足以支撑 spatial voting / fusion 类后处理。

### 下一步方向（超出本阶段 scope，留作未来工作）

**A. OWL-ViT detection/box head fine-tuning（推荐）**
- 在本项目实际错误分布（small objects、crowded scenes、COCO category distribution）上微调 box regression head
- 让模型直接学习产生更准的框，而非依赖后处理补救
- 需要 5090 训练资源 + 标注数据（COCO bbox 已有）

**B. 更强的 VLM grounding 模型（外部知识参考，本项目未部署）**
- 如 Grounding DINO（Transformer-based grounding，box regression head 更强）
- 但模型更大、推理更慢，需权衡 Jetson 部署成本

**不建议**：继续堆叠第 3、4、5 个 training-free heuristic——本次实验已证明问题不是"后处理不够复杂"，而是"候选质量/冗余度不支持后处理"。

## 附录：Evidence 文件

- `baseline_selection.json`: NanoOWL 当前 box selection 机制源码审计
- `candidate_analysis.csv`: 50 例 top-5 候选框提取 + 空间分析
- `development_split.json` / `evaluation_split.json`: 数据集划分（seed=20260828）
- `frozen_methods.json`: 方法定义与超参冻结文档
- `baseline_results.csv` / `method_a_results.csv` / `method_b_results.csv`: 33 例评估完整结果
- `paired_analysis.csv`: Case-level Δbox_iou, Δmask_iou
- `size_breakdown.csv`: Small/Medium/Large 分解
- `latency.json`: 延迟开销测量
- `stage_summary.md`: 本阶段技术总结（中文）
