# Day 4

## 今日目标

完成 NanoOWL → NanoSAM 首次自动联合 Pipeline：Image + Text Prompt → NanoOWL → Bounding Box → NanoSAM Box Prompt → Segmentation Mask。本阶段只验证联合正确性，不做正式 Benchmark / COCO / INT8 / MP4 / ROS2 / stability。

## Stage 4A 架构

- 单进程内联合（不通过 subprocess 拼两个 CLI），复用 `scripts/run_nanoowl.py` 已验证的 OwlPredictor 逻辑与官方 `nanosam.utils.predictor.Predictor`，不复制 third_party 模型核心实现，不修改 upstream。
- CLI（`scripts/run_edge_perception.py`）：`--image --prompts --threshold --target-label --nanoowl-engine --nanosam-image-encoder-engine --nanosam-mask-decoder-engine --output-dir`。
- 流程：NanoOWL 开放词汇检测 → selection → bridge（box 转换）→ NanoSAM Box Prompt → mask。

## NanoOWL → NanoSAM bridge 规则

1. NanoOWL 输出候选 detection：label / score / box [x1,y1,x2,y2]。
2. Coordinate convention（读两端源码确认，非记忆）：
   - NanoOWL：OwlPredictor.predict 返回 output.boxes 为 **xyxy，原图像素空间**（pad_square=False）。见 owl_predictor `_owl_box_roi_to_box_global` 将归一化 roi box 映射到原图 (x0,y0,wh) 像素。
   - NanoSAM：Predictor.predict(points, point_labels) 的 points 在 **原图 XY 像素空间**；preprocess_points 按 orig_image_encoder_size/max(W,H) 缩放。Box prompt 编码为 points=[[x1,y1],[x2,y2]]，labels=[2,3]（2=box 左上，3=box 右下）。
   - 两端均为原图 [x,y]/xyxy，**约定一致，无需交换 xy 顺序**：NanoOWL box [x1,y1,x2,y2] 直接映射 NanoSAM point_coords [[x1,y1],[x2,y2]]。
3. Bridge 必查并记录：原图 NanoOWL box、image W/H、box 是否越界、clipping 后 box、NanoSAM point_coords、labels [2,3]。owl case 中 box 未越界（clipped_box == original_box）。

## detection selection rule

- 若指定 `--target-label`：从 label 匹配该 target 的 detection 中选 score 最高（detections 已按 score 降序，取首个匹配）。
- 若未指定：取所有 detection 中 score 最高的一个（dets[0]）。
- 该 rule 写入输出 JSON（`nanoowl.selection_rule`）。owl case rule = "highest-score detection matching target-label an owl"。

## mask selection rule

- MobileSAM decoder 输出 4 候选 mask（low_res_masks[1,4,256,256]、iou_predictions[1,4]）。
- 官方 examples/basic_usage.py 选 **固定 slot 0**（`mask[0,0]>0`，非 IoU argmax）。本阶段沿用，selected_mask_index=0；selected_predicted_iou=iou_predictions[0]。

## owl case 结果（PASS）

- 输入：third_party/nanoowl/assets/owl_glove_small.jpg（W=756 H=499）；prompts `an owl,a glove,a finger`；target-label `an owl`；threshold 0.10。
- NanoOWL：1 detection；selected label **an owl**，score **0.3419**（与 Day1/2 baseline 完全一致），box [275.1943, 82.3545, 463.6406, 399.3462]（与 baseline 一致）。
- Bridge：box 未越界；clipped_box == original_box；point_coords [[275.1943,82.3545],[463.6406,399.3462]]；labels [2,3]。
- NanoSAM：selected_mask_index **0**；predicted_ious [0.720552, 0.733915, 0.847281, 0.790455]；selected_iou **0.720552**；mask_shape [499,756]；nonzero_pixels 34065；mask_area_ratio 0.0903；无 NaN/Inf。
- status: OK。联合可视化包含 NanoOWL box + label + score + NanoSAM mask overlay。

## NO_DETECTION case 结果（execution PASS）

- 输入：third_party/nanoowl/assets/camera.jpg（W=640 H=480）；prompts `a camera,a hand,a person`；threshold 0.1（复用 Day2 regression camera_scene case）。
- NanoOWL：0 detections → selection_rule="no detections"。
- Pipeline：返回结构化 `status=NO_DETECTION`，**不调用 NanoSAM**（bridge=null, nanosam=null），程序正常退出。
- NO_DETECTION 属于 execution PASS，不是模型/程序失败。

## Timing（correctness timing，非正式 Benchmark）

owl case（timing.json，GPU async 操作后 cuda.synchronize）：
- nanoowl_ms 462.748（含 text encode + detect）
- nanosam_set_image_ms 700.932
- nanosam_decode_ms 28.003
- pipeline_total_ms 1191.683（=nanoowl + set_image + decode；不含 model_load）

NO_DETECTION case：pipeline_total_ms 446.234（仅 NanoOWL，无 NanoSAM）。

注：以上为 Stage 4A correctness timing，**不作为正式 NanoSAM E2E baseline**；不沿用 Day 3 的 52.186 ms 作为正式 baseline。

## 约束遵守

- 未修改 third_party/nanoowl、third_party/nanosam；未重建任何 Engine；未安装新依赖。
- 未做正式 Benchmark / COCO / INT8 / MP4 / ROS2 / stability。
- 核心环境版本保持不变：torch 2.5.0a0+872d972e41.nv24.08 / torchvision 0.20.0a0+afc54f7 / numpy 1.23.5 / tensorrt 10.3.0 / transformers 4.46.3 / Pillow 12.3.0 / OpenCV 4.10.0 / torch2trt importable / timm MISSING。
- 未 remote / push / GitHub 上传；未继续 Stage 4B。

## evidence 路径

- results/joint_pipeline/owl/：pipeline_result.json、input.jpg、nanoowl_visualization.jpg、nanosam_mask.npy、joint_visualization.jpg、timing.json、inference.log
- results/joint_pipeline/no_detection/：pipeline_result.json、input.jpg、nanoowl_visualization.jpg、timing.json、inference.log
- 脚本：scripts/run_edge_perception.py

## 结论

Stage 4A：PASS — NanoOWL → NanoSAM 首次自动联合 Pipeline 正确性验证通过（owl 检测→分割正确、NO_DETECTION 优雅降级）。

---

## Stage 4B — Joint Pipeline 正式稳定态 E2E Benchmark（PASS）

## Benchmark methodology

- 单进程：`scripts/benchmark_edge_perception.py`，复用 `run_edge_perception.py` 的 detection / bridge / selection 逻辑（不 subprocess、不复制 third_party 核心）。
- 模型只加载一次、Engines 只加载一次（NanoOWL OwlPredictor + NanoSAM Predictor 各一次）。
- 固定输入：owl_glove_small.jpg（756×499）；prompts `an owl,a glove,a finger`；target-label `an owl`；threshold 0.10。
- text encodings 在 warmup 前编码一次并复用（prompts 固定，符合稳定态推理）。
- mask selection 保持官方 baseline：slot 0（非 IoU argmax）。未做 INT8 / 优化 / COCO / MP4 / ROS2 / stability。
- 所有 GPU timing 在 `torch.cuda.synchronize()` 后记录；未用未同步 wall-clock 作 latency 结论。
- timing 边界（每轮）：nanoowl（ready input→selected box）/ bridge（box→clip→box prompt）/ nanosam_set_image（preprocess+encoder→embedding）/ nanosam_decode（embedding+prompt→slot0 mask）/ pipeline_e2e（ready input→final mask）。不含 model load / 磁盘 I/O / JSON / 可视化。

## warmup / iterations

- warmup 30（不计入统计）；正式 test 300 轮。
- 每轮强制正确性检查：selected label=an owl、box 有效、mask 非空、无 NaN/Inf。任一轮异常则记录并 Stage FAIL（不静默跳过）。本次 300/300 全通过，errors=0。
- 额外 cached-embedding 实验 100 轮（set_image 仅一次，复用 embedding，每轮 = NanoOWL box→bridge→decoder）。

## Module breakdown（mean ms，n=300）

- nanoowl：mean **38.6133** / P50 37.7168 / P95 44.9096 / P99 45.1861
- bridge：mean **0.1046** / P50 0.1035 / P95 0.1148 / P99 0.1457
- nanosam_set_image：mean **60.7769** / P50 60.2287 / P95 70.437 / P99 72.9323
- nanosam_decode：mean **17.2227** / P50 16.6307 / P95 20.0158 / P99 20.6568
- module sum (mean)=116.7175；pipeline_e2e mean=117.0351；**unaccounted_overhead_ms=0.3176**（基本闭合，仅记录真实结果，未强行解释）。

## E2E P50/P95/P99 与 FPS

- pipeline_e2e：mean **117.0351** / min 87.8657 / max 139.8605 / P50 115.8755 / P95 132.4133 / P99 136.7383 ms。
- steady-state E2E FPS = 1000/mean = **8.54**（inference pipeline FPS，不含磁盘 I/O / 可视化）。
- model_load_ms（单独测量，不计入 E2E）= 2220.798（nanoowl 1594.971 + nanosam 625.831）。

## Cached-embedding 实验

- cached-embedding pipeline（set_image 一次，复用 embedding，每轮 NanoOWL+bridge+decoder）：mean **34.0178** / P50 33.5994 / P95 36.3087 / P99 40.0005 ms（n=100）。
- 量化“image embedding 可复用”情况下的 prompt-update latency。**标记为 cached-image scenario，非完整 Pipeline FPS**。
- 含义：当图像 embedding 缓存复用时，每 prompt-update 耗时约 34ms（≈29 FPS 量级），主要成分为 NanoOWL 检测 + NanoSAM decoder（已省去 ~61ms 的每轮 NanoSAM image encoder）。

## System resource（tegrastats 300 轮 E2E 负载期间，n=190）

- RAM used：mean 4139.795 / peak 4154 MB
- GR3D FREQ：mean 51.542% / max 99%（min 0% 为采样间隙，按观测记录）
- GPU temp：mean 56.204 / max 57.031 °C
- CPU temp：mean 54.41 / max 55.375 °C
- VDD_IN (avg)：mean 9749.558 / max 13067 mW
- EMC FREQ：N/A（当前 L4T 未上报该字段）
- benchmark 后 `tegrastats --stop`，确认无残留 daemon 进程。

## Correctness regression

- owl joint case：PASS — an owl score 0.3419（与 Day1/2/4A baseline 一致），box [275.19,82.35,463.64,399.35]（一致），mask slot 0、selected IoU 0.7206、nonzero 34065、无 NaN/Inf。
- NO_DETECTION case：PASS — camera.jpg 0 detections → 结构化 NO_DETECTION（不调用 NanoSAM），正常退出。

## evidence 路径

- results/joint_benchmark/：benchmark.json、e2e_latencies.csv、module_latencies.csv、cached_embedding_latencies.csv、tegrastats.log、tegrastats_parsed.json、summary.md
- 脚本：scripts/benchmark_edge_perception.py
- regression：results/joint_pipeline/owl_regression/、results/joint_pipeline/no_detection_regression/

## 约束遵守

- 未修改 third_party/nanoowl、third_party/nanosam；未重建 Engine；未安装依赖。
- 未做 INT8 / COCO / MP4 / ROS2 / stability / 性能优化 / IoU argmax；mask 仍 slot 0。
- core env 保持不变：torch 2.5.0a0+872d972e41.nv24.08 / torchvision 0.20.0a0+afc54f7 / numpy 1.23.5 / tensorrt 10.3.0 / transformers 4.46.3 / Pillow 12.3.0 / OpenCV 4.10.0 / torch2trt importable / timm MISSING。
- 未 remote / push / GitHub 上传；未继续 Stage 4C。

## 结论

Stage 4B：PASS — NanoOWL → NanoSAM joint pipeline 正式稳定态 E2E Benchmark 完成，module-level latency 分解与 system resource 采集齐全，300/300 正确性通过，owl/NO_DETECTION 回归 PASS。

---

## Stage 4C — Day 4 工程收口（仅整理已有结果，不新增功能 / 不重新 Benchmark）

本阶段不产生新的测量、不新增功能、不重建 Engine、不运行新 benchmark。仅对已有 Stage 4A/4B 结果做收口与口径固化。

### Latency Budget 小结

基于 n=300 稳定态 E2E Benchmark（mean ms）：

| component | mean ms | 占 E2E mean 比例 |
|---|---|---|
| NanoSAM Image Encoder (set_image) | 60.7769 | ~51.9% |
| NanoOWL (detect + postprocess + selected box) | 38.6133 | ~33.0% |
| NanoSAM Decoder | 17.2227 | ~14.7% |
| coordinate bridge | 0.1046 | ~0.09% |
| unaccounted overhead | 0.3176 | ~0.27% |
| **pipeline_e2e mean** | **117.0351** | 100% |

明确结论（基于真实 Benchmark，非推测）：

- NanoSAM Image Encoder 是当前最大 latency component，约占整线一半（~51.9%）。
- NanoOWL 是第二大 component（~33.0%）。
- NanoSAM Decoder 次之（~14.7%）。
- coordinate bridge 成本可忽略（~0.09%，mean 0.1046 ms）。

边界声明：当前仅证明了 latency component distribution（各模块在 E2E 中的耗时占比）。
尚未通过 Nsight / kernel-level profiling 证明 compute / memory / kernel 层根因，
因此不在此写为“硬件底层瓶颈”结论。Latency distribution 是事实陈述，
root cause（如是否 EMC 带宽 / SM 占用 / 显存拷贝主导）留待 Nsight 阶段验证。

### 性能口径区别（Day 2 vs Day 4）

- Day 2 NanoOWL ~22.49 ms：属于较窄的 core inference benchmark（owl 单模型推理层面，
  timing boundary = 纯 OwlPredictor core inference）。
- Day 4 NanoOWL ~38.61 ms：属于 Joint Pipeline 中包含 detection / postprocess / selected-box
  路径的 stage latency（timing boundary = ready input → selected box，含 selection / 后处理）。

两者 timing boundary 不同，不直接互相替代，不能解释为“Day4 比 Day2 慢”或“回归劣化”。
Day4 的 38.61 ms 是更宽边界的 pipeline-stage 耗时，Day2 的 22.49 ms 是更窄边界的 core inference 耗时。

### Cached Embedding 实验结论

- cached-image / reused-embedding scenario：mean 34.0178 / P50 33.5994 / P95 36.3087 / P99 40.0005 ms（n=100）。
- 明确标记：这是 cached-image / reused-embedding scenario，不是完整 E2E Pipeline FPS（已省去每轮 ~61 ms 的 NanoSAM Image Encoder）。
- 它证明：同一图像多 Prompt 场景可以复用 NanoSAM image embedding，
  避免对同一图像重复执行 NanoSAM Image Encoder；
  每 prompt-update 约 34 ms（≈29 FPS 量级），主要成分为 NanoOWL 检测 + NanoSAM decoder。
- 不作为正式 E2E baseline，仅作为“embedding 可复用性”的量化记录。

### Day 4 Evidence 路径

- results/joint_pipeline/ — owl / owl_regression / no_detection / no_detection_regression joint pipeline 产物
- results/joint_benchmark/ — benchmark.json / e2e_latencies.csv / module_latencies.csv / cached_embedding_latencies.csv / tegrastats.log / tegrastats_parsed.json / summary.md

### Day 4 Final Result

PASS — NanoOWL and NanoSAM have been integrated into a reproducible
open-vocabulary detection-to-segmentation TensorRT pipeline on Jetson,
with stable E2E latency and module-level performance characterized.

### 当前仍未做

- precision / FP32-FP16 quality-performance ablation
- COCO accuracy / mIoU
- INT8
- MP4 realtime pipeline
- ROS2
- long-run stability
- Nsight profiling（compute/memory/kernel 层根因尚未验证）

### Day 5 起点（确定）

基于当前 Benchmark，Day 5 起点确定为：Precision / Quality / Performance Ablation。

重点：

- 保持当前 FP16 / mixed-precision baseline 不变（作为对照基准）。
- 建立可比较的精度 / 性能实验。
- 优先关注 NanoSAM Image Encoder 和 NanoOWL（当前第一 / 第二 latency component）。
- 开始引入正式数据集 / COCO 评测口径。
- 不盲目优化 coordinate bridge 或 Mask Decoder（成本可忽略 / 次之）。

明确：Day 4 不提前执行 Day 5 内容。

### Stage 4C 约束遵守

- 仅整理已有结果，未新增功能、未重新 Benchmark、未重建 Engine、未下载数据集。
- 未 remote / push / GitHub 上传。
- 核心环境版本未变（见 Stage 4A/4B 记录）。


---

## Stage 4C — Day 4 工程收口（仅整理已有结果，不新增功能 / 不重新 Benchmark）

本阶段不产生新的测量、不新增功能、不重建 Engine、不运行新 benchmark。仅对已有 Stage 4A/4B 结果做收口与口径固化。

### Latency Budget 小结

基于 n=300 稳定态 E2E Benchmark（mean ms）：

| component | mean ms | 占 E2E mean 比例 |
|---|---|---|
| NanoSAM Image Encoder (set_image) | 60.7769 | ~51.9% |
| NanoOWL (detect + postprocess + selected box) | 38.6133 | ~33.0% |
| NanoSAM Decoder | 17.2227 | ~14.7% |
| coordinate bridge | 0.1046 | ~0.09% |
| unaccounted overhead | 0.3176 | ~0.27% |
| **pipeline_e2e mean** | **117.0351** | 100% |

明确结论（基于真实 Benchmark，非推测）：

- NanoSAM Image Encoder 是当前最大 latency component，约占整线一半（~51.9%）。
- NanoOWL 是第二大 component（~33.0%）。
- NanoSAM Decoder 次之（~14.7%）。
- coordinate bridge 成本可忽略（~0.09%，mean 0.1046 ms）。

边界声明：当前仅证明了 latency component distribution（各模块在 E2E 中的耗时占比）。
尚未通过 Nsight / kernel-level profiling 证明 compute / memory / kernel 层根因，
因此不在此写为“硬件底层瓶颈”结论。Latency distribution 是事实陈述，
root cause（如是否 EMC 带宽 / SM 占用 / 显存拷贝主导）留待 Nsight 阶段验证。

### 性能口径区别（Day 2 vs Day 4）

- Day 2 NanoOWL ~22.49 ms：属于较窄的 core inference benchmark（owl 单模型推理层面，
  timing boundary = 纯 OwlPredictor core inference）。
- Day 4 NanoOWL ~38.61 ms：属于 Joint Pipeline 中包含 detection / postprocess / selected-box
  路径的 stage latency（timing boundary = ready input → selected box，含 selection / 后处理）。

两者 timing boundary 不同，不直接互相替代，不能解释为“Day4 比 Day2 慢”或“回归劣化”。
Day4 的 38.61 ms 是更宽边界的 pipeline-stage 耗时，Day2 的 22.49 ms 是更窄边界的 core inference 耗时。

### Cached Embedding 实验结论

- cached-image / reused-embedding scenario：mean 34.0178 / P50 33.5994 / P95 36.3087 / P99 40.0005 ms（n=100）。
- 明确标记：这是 cached-image / reused-embedding scenario，不是完整 E2E Pipeline FPS（已省去每轮 ~61 ms 的 NanoSAM Image Encoder）。
- 它证明：同一图像多 Prompt 场景可以复用 NanoSAM image embedding，
  避免对同一图像重复执行 NanoSAM Image Encoder；
  每 prompt-update 约 34 ms（≈29 FPS 量级），主要成分为 NanoOWL 检测 + NanoSAM decoder。
- 不作为正式 E2E baseline，仅作为“embedding 可复用性”的量化记录。

### Day 4 Evidence 路径

- results/joint_pipeline/ — owl / owl_regression / no_detection / no_detection_regression joint pipeline 产物
- results/joint_benchmark/ — benchmark.json / e2e_latencies.csv / module_latencies.csv / cached_embedding_latencies.csv / tegrastats.log / tegrastats_parsed.json / summary.md

### Day 4 Final Result

PASS — NanoOWL and NanoSAM have been integrated into a reproducible
open-vocabulary detection-to-segmentation TensorRT pipeline on Jetson,
with stable E2E latency and module-level performance characterized.

### 当前仍未做

- precision / FP32-FP16 quality-performance ablation
- COCO accuracy / mIoU
- INT8
- MP4 realtime pipeline
- ROS2
- long-run stability
- Nsight profiling（compute/memory/kernel 层根因尚未验证）

### Day 5 起点（确定）

基于当前 Benchmark，Day 5 起点确定为：Precision / Quality / Performance Ablation。

重点：

- 保持当前 FP16 / mixed-precision baseline 不变（作为对照基准）。
- 建立可比较的精度 / 性能实验。
- 优先关注 NanoSAM Image Encoder 和 NanoOWL（当前第一 / 第二 latency component）。
- 开始引入正式数据集 / COCO 评测口径。
- 不盲目优化 coordinate bridge 或 Mask Decoder（成本可忽略 / 次之）。

明确：Day 4 不提前执行 Day 5 内容。

### Stage 4C 约束遵守

- 仅整理已有结果，未新增功能、未重新 Benchmark、未重建 Engine、未下载数据集。
- 未 remote / push / GitHub 上传。
- 核心环境版本未变（见 Stage 4A/4B 记录）。
