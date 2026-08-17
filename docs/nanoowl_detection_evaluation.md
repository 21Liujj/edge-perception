# NanoOWL 开放词汇检测评测协议

## 评测边界

Stage B2 评测已验收 NanoOWL 部署的开放词汇检测质量。它不是经过项目专用监督训练后的闭集 COCO 检测器评测。模型为 `google/owlvit-base-patch32`，图像编码器为已验收的 NanoOWL FP16 TensorRT Engine；项目未针对当前 COCO 80 类进行微调。

评测路径为：

```text
COCO 图像 + 80 个文本提示词 -> NanoOWL -> 带类别和分数的检测框 -> COCO bbox 评测
```

NanoSAM 与联合 Pipeline 的指标边界不同：

- NanoSAM B1 以 COCO GT 框为提示，衡量条件分割 mask 质量。
- NanoOWL B2 以图像和文本为输入，衡量开放词汇检测框质量。
- 联合 Pipeline 使用 NanoOWL 框提示 NanoSAM，因此最终输出同时包含检测误差和分割误差。

三类指标不能当作同一任务直接比较。

## 数据集与 subset 协议

数据源为现有 COCO 2017 validation：

- 图像：`data/coco/val2017`，5,000 张 JPEG
- 标注：`data/coco/annotations/instances_val2017.json`
- 类别：80
- GT annotations：36,781
- 评测器：`pycocotools 2.0.11`、`COCOeval(iouType="bbox")`

不下载或修改图像与标注。QUICK 和 STANDARD 从排好序的 COCO image ID 中使用 Python `random.Random(2027).sample` 做确定性 image-level 采样。选中图像的全部 annotation 仍保留在原始 COCO 数据库中；subset 通过 `coco_eval.params.imgIds` 限定，不创建局部 annotation 数据库。

| Scope | 图像数 |
|---|---:|
| SMOKE | 20 |
| QUICK | 100 |
| STANDARD | 500 |
| FULL | 5,000 |

SMOKE 使用 QUICK manifest 的前 20 个 ID。仅在 STANDARD 通过并确认 wall time、磁盘和运行行为合理后执行 FULL。

## 提示词协议

主提示词模板统一为：

```text
a {category_name}
```

例如 `a person`、`a bicycle`、`a dog`。一致性优先于对个别类别做语法或同义词优化。由此可能产生 `a scissors`、`a skis` 等不自然表达；该限制保留并如实报告，不为少数类别定向调优。

提示词索引与 `instances_val2017.json` 中按 category ID 排序后的 COCO 类别一一映射。评测器将 80 个完整映射写入 `category_prompt_mapping.json`，并校验 prompt index 和 category ID 的唯一性。

80 个提示词只执行一次 tokenizer 与 OWL-ViT text encoder，得到的文本嵌入供所有图像复用，不逐图重复编码。

## NanoOWL 输出与过滤语义

已验收的 NanoOWL `OwlPredictor` 为每个 visual patch 生成候选框。解码阶段计算归一化图像与文本嵌入的相似度 logits，应用学习得到的 shift/scale 与 sigmoid，再为每个 patch 选择得分最高的提示词。返回的 `labels` 是零基 prompt index，`scores` 是获胜提示词的 sigmoid 分数；关闭方形 padding 时，`boxes` 是原图像素坐标 `[x1, y1, x2, y2]`。

解码在 per-patch prompt argmax 后应用 `score > threshold`。当前实现没有 NMS，也没有 top-k。B2 因此只在 threshold `0.05` 下执行一次推理并缓存合法候选，然后离线应用 threshold 和每图 top-k，避免为消融重复推理。

输出 COCO 结果前，将每个框独立裁剪到 `[0,width] x [0,height]`，并转换为 `[x, y, width, height]`：

```text
width = x2 - x1
height = y2 - y1
```

非有限或退化框会被拒绝并计数；坐标轴不交换。在调用 `COCO.loadRes` 前，评测器校验有限分数与坐标、合法类别映射以及严格为正的 width/height。

## 主策略与消融策略

主策略：

- score threshold：`0.10`
- 每图 top-k：`100`
- prompt template：`a {category_name}`
- square padding：关闭，与已验收回归路径一致

threshold 消融固定 top-k 100，评测 `0.05`、`0.10`、`0.20`、`0.30`。COCO AP 已经对置信分数排序，前置过滤可能移除候选并降低 recall，因此不预设高阈值会提升 AP。

top-k 消融固定 threshold 0.10，评测每图 `10`、`50`、`100` 个检测，按图内 score 降序截断。

## 正式指标与辅助分析

正式指标只来自 `pycocotools.cocoeval.COCOeval`：

- AP@[0.50:0.95]、AP50、AP75
- APsmall、APmedium、APlarge
- AR1、AR10、AR100
- ARsmall、ARmedium、ARlarge

类别级 AP 和 AP50 从 COCOeval 累积后的 precision tensor 读取。每条类别记录包含 GT 数与预测数；排名排除低于指定最小 GT 数的类别，所有结论保留样本量。

辅助诊断包括每图预测数、零预测图像数、每图 GT 数、分数分布、预测框面积分布，以及 IoU >= 0.5 的 analysis-only greedy matcher。该 matcher 仅用于可视化和 failure grouping，不替代 COCOeval，也不定义正式 AP。

## Timing 边界

B2 记录模型加载、一次性文本编码、图像循环、images/s 与评测器总 wall time。图像循环吞吐包含图像加载、预处理、TensorRT 图像编码、NanoOWL 解码、GPU 到 CPU 传输与输出验证。它是 evaluation throughput，不是 Day 2 core-inference FPS。

## 解释约束

实测较低 COCO AP 必须直接报告，并作为当前提示词协议下的实际部署限制。该结果本身不能证明开放词汇检测作为问题设定不如闭集检测，也不能直接与针对 COCO 标签监督训练的现代 YOLO 类检测器横向排名。解释必须同时保留两个事实：NanoOWL 使用文本定义类别进行零样本评测；当前部署的实测质量就是当前部署质量。

Stage B2 不训练、不微调、不修改模型权重、不重建 TensorRT Engine、不做精度转换、不修改 CUDA/C++，也不修改 `third_party/nanoowl` 或 `third_party/nanosam`。

## 实际执行结果

所有 scope 按顺序完成。SMOKE 20 和 QUICK 100 通过后，STANDARD 500 在总计 44.52 秒内完成，图像循环 16.92 秒，生成 evidence 约 4.1 MB，因此 FULL 5,000 在资源上合理。FULL 随后完成全部 5,000 张图像和 36,781 个 GT 实例评测。

FULL 主策略（`threshold=0.10`、`top-k=100`）输出 44,962 个检测，82 张图像零预测。COCOeval 结果：

| 指标 | 数值 |
|---|---:|
| AP | 0.258616 |
| AP50 | 0.407885 |
| AP75 | 0.272948 |
| APsmall | 0.095263 |
| APmedium | 0.264089 |
| APlarge | 0.442186 |
| AR1 | 0.255193 |
| AR10 | 0.365058 |
| AR100 | 0.372312 |

图像循环 165.26 秒，吞吐 `30.26 images/s`；评测器总 wall time 349.17 秒。后者包含多组离线 COCOeval、类别分析、失败挖掘、序列化和 15 张可视化，不是纯检测 FPS。

threshold `0.05` 获得本次测试最高 AP `0.270547` 和 AR100 `0.427018`，同时产生 111,588 个预测。threshold 提高到 0.10、0.20、0.30 时 AP 单调下降。在 threshold 0.10 下，top-k 50 与 100 基本相同（AP `0.258508` 对 `0.258616`）；top-k 10 降至 AP `0.249041`、AR100 `0.344323`。

APlarge `0.442186` 明显高于 APsmall `0.095263`。较强类别包括 cat `0.615114`、train `0.592233`、airplane `0.575426`；弱势高频类别包括 book `0.041912`、handbag `0.052681`、backpack `0.059292` 和 chair `0.107673`。辅助 matcher 发现 car/truck、chair/couch、knife/fork、snowboard/skis、microwave/oven 等重复语义混淆；这些诊断不替代 COCOeval。

Day 2 NanoOWL regression 5/5 通过，并复现 owl anchor score `0.3419235944747925`。Day 4 joint regression 返回 `status=OK`，复现相同 owl 框和分数，继续使用 NanoSAM fixed slot 0，得到有限非空 mask，selected predicted IoU 为 `0.720552`。

精简 evidence 保存于 `results/algorithm_b2_nanoowl/`；大型预测 JSON 和重复中间 scope 输出仅作为本地生成物保留，不纳入源码证据集。
