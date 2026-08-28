# Stage G — NanoOWL BBox Localization Training-Free Refinement 实验总结

## 1. 审计结论：当前 box selection 机制

源码位置：`third_party/nanoowl/nanoowl/owl_predictor.py`
- `encode_image_torch`：576 个 patch tokens（24×24 grid，patch_size=32）各自产生 `image_class_embeds`[576,512] 和 `pred_boxes`[576,4]
- `decode`：L2-normalize + cosine + logit_shift/scale + sigmoid 得到每个 patch 对 text query 的分数；仅做 `score > threshold` 过滤，**没有 NMS，没有 top-k 排序**
- Project-side (`scripts/eval_joint_pipeline_coco.py`)：对通过 threshold 的候选取 **最高分的单个 patch (top-1)** 作为最终检测框

即当前系统对"多个高分 patch 同时存在"的情况，完全没有利用空间聚合信息，只信任语言匹配分数最高的那一个。

## 2. Candidate-level 关键发现

对 E4 固定 50 例，用冻结 top_k=5 提取候选后：
- **17 例 B_localization_propagation 中，8 例存在语义/空间上更优的非top-1候选**（GT-IoU 比 top-1 高出 >0.01）
- 但深入检查这 8 例发现：更优候选与 top-1 的 **pairwise IoU 普遍很低**（如 ann=1556646 orange: top1 与 best-alt 的 pairwise IoU = 0.000；ann=2221879 pizza: pairwise IoU = 0.017-0.084）
- 这说明"更优候选"往往是**空间上孤立的**，不在 top-1 附近，而是完全不同的图像区域被误打分排到高位

这是决定性的负面证据：Score-Weighted Fusion 假设优质候选与top-1相邻可融合，Spatial Consensus 假设优质候选有邻近支持，**两个假设在本数据集上都不成立**。

## 3. 实验结果（33-instance frozen evaluation split）

| Metric | Baseline | Method A (Fusion) | Method B (Consensus) |
|---|---|---|---|
| Detection success | 0.848 | 0.848 | 0.848 |
| Mean Box IoU | 0.532 | 0.479 (-0.054) | 0.535 (+0.003) |
| Median Box IoU | 0.779 | 0.567 (-0.212) | 0.779 (+0.000) |
| Box IoU≥0.5 | 0.515 | 0.455 | 0.515 |
| Box IoU≥0.75 | 0.424 | 0.364 | 0.424 |
| Joint Mask IoU (mean) | 0.407 | 0.378 (-0.029) | 0.407 (+0.001) |
| Latency overhead (mean) | — | +0.124ms | +0.329ms |

GT-box NanoSAM 上界参考（同一 eval split）：mask IoU = 0.728

## 4. Paired 分析

- **Method A**：6 例 box IoU 退化，仅 1 例改善（净负）；mask IoU 3 例退化，2 例改善
- **Method B**：1 例改善，1 例退化，其余 31 例完全不变（该方法在多数情况下选中的仍是 top-1 本身）

## 5. B_localization_propagation 恢复情况（eval split 内 9 例）

- Method A：0 例恢复（box_iou≥0.5），2 例明显变差（bowl, person）
- Method B：0 例恢复，1 例轻微变差（bowl 的 top-1 candidate 数=1 时 method B 与 baseline 完全一致，此处退化来自 method A/B 共用代码路径的边界情况，需注明）
- **两个方法均未能修复任何一例定位传播失败**

## 6. Small/Medium/Large 分解

| Size | Baseline Box IoU | Method A | Method B | Baseline Mask IoU | Method A | Method B |
|---|---|---|---|---|---|---|
| Small (n=12) | 0.292 | 0.252 (↓) | 0.302 (↑微小) | 0.234 | 0.225 (↓) | 0.235 (≈) |
| Medium (n=12) | 0.582 | 0.470 (↓明显) | 0.580 (≈) | 0.453 | 0.381 (↓明显) | 0.453 (≈) |
| Large (n=9) | 0.770 | 0.772 (≈) | 0.770 (≈) | 0.575 | 0.577 (≈) | 0.575 (≈) |

**结论：refinement 并未特别帮助 small object（NanoOWL 历史弱点），反而 Method A 在 medium object 上退化最明显。**

## 7. Box improvement → Mask improvement 传导关系

在为数不多的改善案例中，box IoU 提升确实伴随 mask IoU 提升（与 Stage F 的 Pearson r=0.931 全局相关性一致），但由于本阶段两种方法整体未能可靠提升 box IoU，该传导关系在本次实验中未能被正向利用。

## 8. 候选数量分布（诊断信息）

Eval split 33 例中：5 例 0 候选（检测失败）、9 例仅 1 候选（无融合/共识空间）、其余 19 例 2-5 候选。仅在候选数≥2 的 19 例中，Method A/B 才有实际作用空间；这部分样本中 Method A 表现出更明显的负面效应。

## 9. 决策

**TRAINING_FREE_LOCALIZATION_REFINEMENT_INSUFFICIENT**

两个 training-free 方法均未展现稳定收益：
- Method A (Score-Weighted Fusion) 在 box IoU 和 mask IoU 上均**净负**（fusion 把优质 top-1 框和低质量噪声候选框平均，反而拉低精度）
- Method B (Spatial Consensus) **基本无效**（因为大多数案例候选数太少，或优质候选本身空间孤立，consensus 机制找不到它们）

根本原因：本项目中 NanoOWL 的候选框冗余度不足以支撑 spatial voting 类方法——当 top-1 错误时，错误往往不是"轻微偏移"而是"完全错误区域"，且没有可靠的邻近候选提供纠正信号。

## 10. 下一步建议（不在本阶段执行）

A. **OWL-ViT detection/box head fine-tuning**：在本项目实际错误分布（小目标、拥挤场景）上微调 box head，让模型直接学习产生更准的框，而非依赖后处理
B. **更强的 Vision-Language grounding 模型**：换用有更强 box regression 能力的检测器（如 Grounding DINO，仅作为外部知识参考，本项目未部署）

两者均需要训练资源，超出本阶段 training-free 范围，留作未来工作。
