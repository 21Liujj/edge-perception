# Day 9 — Perception Algorithm Deep Dive

## Stage B1 — NanoSAM Mask Selection Ablation

### WHAT

Evaluated fixed slot 0, predicted-IoU argmax, and GT oracle best-of-4 over the unchanged Day 5 STANDARD 500-instance manifest. Each instance ran one decoder call; all four masks and scores were retained for offline policy, calibration, size, category, failure-case, and runtime analysis.

### WHY

Determine whether the historical slot-0 baseline limits segmentation quality and whether the decoder's predicted IoU scores provide a safe deployable selector.

### INPUT

- Manifest: `data/eval_manifests/standard_manifest.json`
- SHA256: `19a00a836c2f1d1f236ea34175d12c2afc670b9cb5136b30e6b39bf7ca78f015`
- 500 instances, 455 unique images
- COCO GT box prompts
- Accepted NanoSAM FP16 image encoder and FP32 mask decoder

### OUTPUT

Evidence is under `results/algorithm_b1/`, including 500-row CSV, policy and calibration summaries, category/size analyses, failure mining, runtime measurements, and 20 representative visualizations.

### RESULT

Fixed slot 0 reproduced Day 5 exactly at mIoU `0.718128`. Predicted argmax fell to `0.661484` (`-0.056644`) and small-object mIoU fell from `0.605320` to `0.492933`. Oracle reached `0.748094`, giving `+0.029966` overall headroom and `+0.034884` for small objects.

Predicted argmax selected slot 2 for all 500 instances and matched the oracle only 91 times (18.2%). Across 2,000 candidates, predicted/actual IoU Pearson correlation was `-0.023302` and MAE was `0.201619`.

### DEPLOYMENT DECISION

`KEEP_FIXED_SLOT0_BASELINE`

Predicted argmax overhead is small relative to the full pipeline, but quality is unstable and materially worse: 126 instances improve and 374 degrade. Historical Day 5 evidence is not overwritten.

### NEXT STEP

Stop after B1. A separately authorized stage may investigate why the deployed IoU head is slot-biased or evaluate a validated selector. No training, TensorRT rebuild, INT8, CUDA optimization, or NanoOWL experiment is started.

## Stage B2 — NanoOWL 开放词汇检测质量评测

### 评测内容

在 COCO 2017 val 上评测未修改的 `google/owlvit-base-patch32` NanoOWL 部署，80 个类别统一生成提示词 `a {category_name}`。TensorRT FP16 图像编码器和 COCO 标注保持不变；文本嵌入只编码一次，并在全部图像间复用。

### 评测协议

按 SMOKE 20、QUICK 100、STANDARD 500、FULL 5,000 张图像的顺序执行。QUICK 和 STANDARD 使用 seed `2027` 做确定性 image-level 采样；subset COCOeval 通过 `params.imgIds` 限定图像。主策略为 threshold `0.10`、每图 top-k `100`、关闭方形 padding、无 NMS。threshold 与 top-k 对比复用 threshold `0.05` 单次推理产生的候选。

正式指标使用 `pycocotools 2.0.11` 的 `COCOeval(iouType="bbox")`。IoU 0.50 贪心 matcher 仅用于失败分析和可视化。

### FULL 结果

- 图像数 / GT 实例数：`5,000 / 36,781`
- 预测数 / 零预测图像数：`44,962 / 82`
- AP / AP50 / AP75：`0.258616 / 0.407885 / 0.272948`
- APsmall / APmedium / APlarge：`0.095263 / 0.264089 / 0.442186`
- AR1 / AR10 / AR100：`0.255193 / 0.365058 / 0.372312`
- 图像循环：`165.26 s`，`30.26 images/s`
- 评测器总 wall time：`349.17 s`

Evaluation throughput 包含图像加载、预处理、TensorRT 图像编码、解码、传输和输出验证，不等同于 Day 2 稳定态 inference-only FPS。

### 消融与失败分析

本次测试策略中，threshold `0.05` 最佳，AP 为 `0.270547`、AR100 为 `0.427018`；提高 threshold 会持续降低 recall 和 AP。在 threshold 0.10 下，top-k 50 与 100 几乎相同，而 top-k 10 将 AP 降至 `0.249041`。

大目标质量明显高于小目标。较强类别包括 cat、train 和 airplane；弱势高频类别包括 book、handbag、backpack 和 chair。辅助语义混淆分析发现 car/truck、chair/couch、knife/fork、snowboard/skis 和 microwave/oven 等模式。15 张代表性可视化覆盖成功检测、FP-heavy、FN-heavy、小目标失败和语义混淆。

当前 AP 是该零样本提示词协议下真实的部署限制。由于监督方式、类别定义和推理输入不同，它不能直接与针对 COCO 专项训练的 YOLO 类检测器横向比较。

### 回归与证据

Day 2 NanoOWL regression 的 `5/5` cases 全部通过，并复现 owl anchor score `0.3419235944747925`。Day 4 joint regression 返回 `OK`，复现 owl 检测结果，NanoSAM 保持 fixed slot 0，并得到有限非空 mask，selected IoU 为 `0.720552`。

Evidence 位于 `results/algorithm_b2_nanoowl/`。协议与解释见 `docs/nanoowl_detection_evaluation.md`；算法概念见 `docs/nanoowl_algorithm.md`。

### 决策

`ESTABLISH_NANOOWL_COCO_BASELINE`

B2 已完成。未进行训练、微调、Engine 重建、精度实验、CUDA/C++ 优化或第三方模型修改。Stage B2 后停止。
