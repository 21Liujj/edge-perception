# NanoOWL 开放词汇目标检测算法

## 闭集检测与开放词汇检测

闭集检测器只能预测训练时预先定义的类别。典型的 YOLO 类 COCO 检测器在推理时只接收图像，因为固定类别已经编码在训练得到的分类头中。要加入真正的新类别，通常需要新的标注数据，并重新训练或微调模型。

NanoOWL 是基于 OWL-ViT 的开放词汇检测器。它在推理时同时接收图像和文本提示词，由提示词定义当前有效的类别集合。因此，用户可以查询未固化在项目专用 COCO 分类头中的概念。该能力并不保证所有提示词、同义词、目标尺寸或数据集命名规则都具有相同质量。

| 属性 | NanoOWL / OWL-ViT | YOLO 类闭集检测器 |
|---|---|---|
| 类别定义 | 推理时传入文本提示词 | 训练时学习并固定输出类别 |
| 训练依赖 | 预训练视觉语言对齐可支持零样本类别 | 目标类别通常需要带标注的检测训练 |
| 推理输入 | 图像和文本提示词集合 | 图像 |
| 新增词汇 | 修改提示词，无需重建分类头 | 通常需要重新训练或微调 |
| 分类机制 | 图像区域嵌入与文本嵌入相似度 | 固定维度的已学习类别 logits |
| 部署复杂度 | 包含文本分词、文本编码、图像编码和提示词相关解码 | 通常可部署为单个固定检测图 |

二者代表不同的工作方式。NanoOWL 的零样本 COCO 结果必须如实报告，但不能直接作为与 COCO 监督训练检测器的同口径排名。

## OWL-ViT / NanoOWL Pipeline

当前部署使用 `google/owlvit-base-patch32` 和已验收的 NanoOWL TensorRT FP16 图像编码器：

```text
文本提示词 -> tokenizer -> OWL-ViT 文本编码器 -> 文本嵌入
图像 -> 预处理 -> TensorRT 图像编码器 -> patch 嵌入与候选框
patch/文本相似度 -> shift 和 scale -> sigmoid -> 提示词 argmax -> threshold -> 检测结果
```

提示词集合固定时，文本嵌入只需计算一次，并可在所有图像间复用。NanoOWL 对 patch 嵌入和提示词嵌入做归一化，计算相似度 logits，应用模型学习得到的 shift 和 scale，再转换为 sigmoid 分数。每个 patch 只保留得分最高的提示词。当前解码器随后按阈值过滤，不执行 NMS，也不实现 top-k。

## 检测输出

每个有效候选包含：

- `label`：提示词列表中的零基索引
- `score`：该 patch 获胜提示词的 sigmoid 分数
- `box`：关闭方形 padding 时，原图像素坐标中的 `[x1, y1, x2, y2]`

COCO 评测会将提示词索引映射为 category ID，把框裁剪到图像边界，并转换为 `[x, y, width, height]`。`score` 是排序和置信信号，不应解释为目标必然正确的已校准概率。

## IoU、Precision 与 Recall

Intersection over Union（IoU）衡量预测框与 GT 框的重叠程度：

```text
IoU = 交集面积 / 并集面积
```

在指定 IoU 阈值和类别下，匹配成功的预测为 true positive；未匹配预测为 false positive；未被匹配的 GT 目标为 false negative。

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
```

提高分数阈值通常会移除低分检测，可能减少 false positive，也可能降低 recall。限制 top-k 会使每张图只保留得分最高的若干检测，对拥挤场景影响尤其明显。

## AP、AP50、AP75 与 AR

Average Precision（AP）在预测按分数排序后，对 precision-recall 曲线进行汇总。COCO 主指标 AP 对 IoU 0.50 到 0.95、步长 0.05 的多个阈值取平均，因此同时要求目标被发现且定位足够准确。

- `AP50`：IoU 0.50 下的 AP，对定位误差更宽容
- `AP75`：IoU 0.75 下的 AP，对框定位要求更严格
- `APsmall`、`APmedium`、`APlarge`：按 COCO 目标面积范围分组的 AP
- `AR1`、`AR10`、`AR100`：每张图最多保留 1、10、100 个检测时的平均召回率

小目标 AP 往往会暴露总 AP 隐藏的分辨率与定位限制。分析面积分组结果时，还应同时查看类别频率和预测数量。

## Threshold 与 top-k 策略

B2 主策略使用 threshold `0.10`、top-k `100`。阈值消融在 top-k 固定为 100 时评测 `0.05`、`0.10`、`0.20`、`0.30`；top-k 消融在 threshold 固定为 0.10 时评测 `10`、`50`、`100`。

图像只在 threshold 0.05 下推理一次。更高阈值与 top-k 限制离线应用于同一候选缓存，避免重复推理干扰比较。由于 COCO AP 本身已经按分数排序，不能预设提高前置过滤阈值一定会改善 AP。

## Failure Analysis

正式质量指标只来自 `pycocotools.COCOeval`。辅助性的类别感知贪心匹配器使用 IoU 0.50，仅支持诊断。它将代表性图像分为成功检测、false-positive-heavy、false-negative-heavy、小目标漏检和语义混淆等组。

类别分析保留 GT 数量与预测数量。重点模式包括高频低 AP 类别、大目标 AP 较好但小目标 AP 较差的类别，以及 car/truck、chair/couch 等语义相近类别的混淆。统一模板会保留 `a scissors` 等语法不自然的提示词，以维持协议公平；提示词敏感性属于开放词汇部署契约的一部分。

不能用“零样本开放词汇”掩盖实际较低的 COCO AP：该结果是当前提示词协议下真实的部署限制。反过来，它也不能证明开放词汇检测整体不如任务专用检测器，因为两者的类别定义、监督方式和推理输入不同。

## 评测边界

NanoOWL B2 衡量图像加文本到检测框的质量。NanoSAM B1 衡量给定 GT 框提示后的 mask 质量。NanoOWL 到 NanoSAM 的联合 Pipeline 同时包含检测框误差和分割误差。三者的 AP 与 mIoU 描述不同契约，不能合并解释。

B2 evaluation throughput 包含图像加载、预处理、TensorRT 图像编码、NanoOWL 解码、CPU 传输和验证。它不是 Day 2 稳定态核心推理 FPS，也不是联合 Pipeline 的端到端吞吐率。
