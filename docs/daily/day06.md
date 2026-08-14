# Day 6

## 今日目标

建立 MP4 → NanoOWL → NanoSAM → Overlay → MP4 的基础视频推理闭环（Stage 6A）。只建立正确、可复用的视频 Pipeline，不做正式实时优化。不修改 NanoOWL/NanoSAM upstream，不安装新依赖，不继续 Stage 6B。

---

## Stage 6A — MP4 Joint Pipeline

### 视频来源

- 本地搜索范围 `third_party/nanoowl/`、`third_party/nanosam/`、`data/`、`assets/`、`results/` 均无可用 MP4/AVI/MOV。
- **未联网下载**随机视频。从本地图像 `third_party/nanoowl/assets/owl_glove_small.jpg` 生成最小 synthetic MP4（明确为工程 Pipeline 验证用合成视频，非真实视频数据集）。
- 合成方法：owl 图按 per-frame 水平正弦平移（max 30px），恒定 scale，OpenCV `mp4v` 写入；fps=10，60 帧，6 秒，分辨率 756×499（源图分辨率）。
- manifest：`results/video_pipeline/baseline/synthetic_video_manifest.json`（source image / fps / resolution / frame count / generation method / 标注 engineering-only）。

### 输入/解码路径

- OpenCV `cv2.VideoCapture` 解码 MP4；frame 为 BGR。
- BGR → RGB（`cv2.cvtColor`）→ `PIL.Image` RGB，与 Day 4 单图 Pipeline（PIL RGB 打开）一致；**不因视频路径偷改预处理**。
- 输入 metadata：60 帧，10 fps，756×499（CAP_PROP 偶报 498 高度，frame shape 实测 498×756×3，一致无 mismatch）。

### Per-frame joint pipeline

- 模型**只加载一次**（NanoOWL 一次 + NanoSAM 一次），全视频复用；text_encodings 一次编码复用。禁止逐帧 reload Engine。
- 每帧：decode → BGR→RGB PIL → NanoOWL(检测) → selection(target-label “an owl” 最高分) → box clip → NanoSAM box prompt(points=[[x1,y1],[x2,y2]] labels=[2,3]，原图 px) → mask slot 0 → overlay(box+mask+status 文字) → encode/write。
- 坐标约定与 Day 4 一致：NanoOWL box xyxy 原图像素（`pad_square=False`）；NanoSAM point 原图像素，无轴交换；mask selection = slot 0（fixed，非 IoU argmax）。
- NanoSAM `set_image` 每处理帧调用一次（合成视频有平移，embedding 每帧变化；不跨不同帧复用 embedding —— 这是正确路径，非缓存策略）。

### NO_DETECTION 处理

- 无匹配 / 无检测 → **不调用 NanoSAM**，写原帧 + “status=NO_DETECTION” 标记，pipeline 不 crash。
- 验证：用 target-label “a zebra”（5 帧）触发 NO_DETECTION 分支：5/5 no_detection，帧正常写出，输出可重读，无 crash。

### Timing boundary

- 每帧记录：decode_ms / nanoowl_ms / bridge_ms / nanosam_set_image_ms / nanosam_decode_ms / overlay_ms / encode_write_ms / frame_total_ms；GPU timing 均 `torch.cuda.synchronize()`。
- model_load_ms 单独记录（nanoowl 2307.6ms + nanosam 706.4ms = 3014.0ms）。
- Timing 仅作 correctness profiling；**不把 `1000/frame_total_ms` 称为 realtime FPS**（Stage 6B 再正式定义 throughput，明确写入 timing_summary 注释）。

### Output artifacts

`results/video_pipeline/baseline/`：

- `input.mp4` —— 输入合成视频
- `output_overlay.mp4` —— overlay 输出视频
- `frames.jsonl` —— 每帧记录（frame_index / input_timestamp_ms / status / selected_label / score / box / mask_area_ratio / timing）
- `timing_summary.json` —— 汇总
- `run.log`
- `synthetic_video_manifest.json`

### 输出视频正确性

- 可被 OpenCV 重新 `VideoCapture` 打开：`output_reread_ok=true`。
- frame count > 0：写出 60 帧，重读 60 帧。
- resolution 正确：756×498，与输入一致。
- FPS metadata 合理：10.0 fps。
- 每个写出 frame shape 一致：均为 498×756×3（input_frame_shapes_seen 仅 1 种 shape；output_frame_shape_first 与之一致）。

### Smoke 结果（max_frames=60）

| 项 | 值 |
|---|---|
| input FPS | 10.0 |
| input frame count | 60 |
| processed frames | 60 |
| detected frames | 60 |
| no_detection frames | 0（NO_DETECTION 分支单独用 “a zebra” 5 帧验证 PASS） |
| output frame count | 60（重读 60） |
| output_reread_ok | true |
| 无 NaN/Inf | 是（60/60） |
| 无 frame shape mismatch | 是 |
| 无 per-frame engine reload | 是（模型各加载一次） |
| 无内存持续增长 | 是（alloc 643.6→643.8 MB，平稳） |

平均 correctness timing（processed frames，ms）：

| decode | nanoowl | bridge | nanosam_set_image | nanosam_decode | overlay | encode_write | frame_total |
|---|---|---|---|---|---|---|---|
| 0.1 | 44.2 | 1.8 | 59.8 | 18.5 | 10.2 | 14.9 | 149.6 |

### 单图回归

- 重跑 Day 4 owl 单图 joint regression（`scripts/run_edge_perception.py`，owl_glove_small.jpg，prompts “an owl,a glove,a finger”，target-label “an owl”，threshold 0.10）。
- **PASS**：label “an owl”，score 0.3419，box [275.19, 82.35, 463.64, 399.35]（与 Day1/2/4 baseline 一致），mask slot 0、selected IoU 0.7206、nonzero 34065、无 NaN/Inf。
- 结论：视频代码未破坏单图 Pipeline（视频脚本独立，未改动 `run_edge_perception.py`）。

### Evidence 路径

- `scripts/run_edge_perception_video.py`
- `results/video_pipeline/baseline/{input.mp4,output_overlay.mp4,frames.jsonl,timing_summary.json,run.log,synthetic_video_manifest.json}`
- `results/joint_pipeline/owl_regression/pipeline_result.json`（单图回归）

---

## 约束遵守 / Day 6 收口

- 未做 backpressure / latest-frame 策略 / producer-consumer 线程 / queue / realtime FPS 正式 benchmark / long-run stability / ROS2 / camera / INT8 / 新精度实验 / COCO / Engine 重建。
- 未修改 `third_party/nanoowl`、`third_party/nanosam`；未安装新依赖。
- 未 remote / push / GitHub 上传（无 remote）。
- 未提前开始视频实时优化；**未继续 Stage 6B**。
- 核心环境保持：torch 2.5.0a0+872d972e41.nv24.08 / tensorrt 10.3.0 / numpy 1.23.5 / torchvision 0.20.0a0+afc54f7 / opencv 4.10.0。无残留视频推理进程。

### 默认配置（未变）

- NanoOWL：`owlvit_image_encoder_patch32.engine`（FP16）
- NanoSAM Image Encoder：`resnet18_image_encoder.engine`（FP16）
- NanoSAM Mask Decoder：`mobile_sam_mask_decoder.engine`（FP32）

## Day 6 结论

Stage 6A：MP4 → NanoOWL → NanoSAM → Overlay → MP4 基础视频推理闭环已建立。60 帧 smoke PASS（60 detection，NO_DETECTION 分支 PASS，输出可重读，无 NaN/Inf/shape mismatch/per-frame reload/内存增长），单图回归 PASS。Stage 6B 将在此基础上正式定义 throughput / backpressure / latest-frame。

---

## Stage 6B — Realtime Video Runtime / Backpressure / Latest-Frame 策略对比

### 目标

在**真实帧到达节奏**（producer 按视频原始 FPS=10 释放帧，即每 100ms 一帧；**非磁盘解码速度**）下，对比三种 runtime 队列/背压策略的吞吐、延迟、丢帧、队列行为与资源占用。此为 runtime 策略 benchmark，**非算法质量 benchmark**（输入为 Stage 6A 合成视频，仅作 runtime-controlled 输入）。

### 三种策略定义

| 策略 | 机制 | 丢帧 | 背压 |
|---|---|---|---|
| sequential | read→inference→next，无 queue | 否 | N/A（无 queue） |
| fifo | producer thread → 有界 FIFO queue（maxsize=8，`--queue-size`）→ consumer thread | 否（满则 BLOCK） | 是：queue 满时 producer 阻塞 |
| latest | 有效容量 1；新帧到达且 consumer 忙时丢弃被取代的旧 pending 帧 | 是（reason=`superseded_by_newer_frame`） | 否（drop-old 替代 block） |

### 实时回放实现

- producer 按 source FPS 节奏释放帧：`target_wall = replay_start + src_idx * (1/FPS)`，`time.sleep` 到点；**不是磁盘解码速度**。
- `source_timestamp_ms = src_idx * (1000/FPS)`，跨 `--loops 5` 单调递增（loop-2 frame-0 不重置为 0）。
- 源帧一次性读入内存（60 帧很小），保证回放节奏只受 FPS 节拍控制，不受 IO/decode 抖动污染。
- 每源帧记录：`source_timestamp_ms` / `scheduled_arrival_ms` / `arrival_wall_time` / `enqueue_time` / `dequeue_time` / `inference_start` / `inference_end` / `completion_time`。
- 关键实时指标 `live_e2e_latency_ms = completion_time - scheduled_arrival_time`（completion 相对 replay 开始，减去该帧的 scheduled 到达时间，即"这一帧从该到达时刻到处理完成的真实端到端延迟"）。

### 复用与一致性

- 复用 Stage 6A / Day 4 已验证推理逻辑：`detect_on_image`（NanoOWL `pad_square=False`）/ `select_detection` / `clip_box` / NanoSAM `set_image`+box prompt(slot 0) `/ NaN-Inf guard。BGR→RGB PIL 与 Day 4 一致。
- 模型**只加载一次**（NanoOWL 1459.5ms + NanoSAM 627.4ms = 2086.9ms），三模式共用同一 instance，无 per-frame reload。
- 输出视频写入**已禁用**（VideoWriter IO 会污染 runtime timing）；在 summary 中显式标注。
- processing throughput 为 realtime-paced 输入下的离线 pipeline throughput，**不声称**为 realtime 摄像头 FPS。

### 输出 artifacts

`results/video_runtime_benchmark/`：
- `sequential|fifo|latest/{summary.json, frames.jsonl, tegrastats.log, tegrastats_parsed.json, run.log}`
- `comparison.json` / `comparison.md`

### 结果汇总表（--loops 5，300 源帧，10 fps，756×498）

| Mode | Proc FPS | Processed | Dropped | Drop Rate | Live Lat P50 (ms) | P95 (ms) | P99 (ms) | Queue Wait P95 (ms) | Frame Age P95 (ms) | Max Q Depth | Producer Blocked (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sequential | 7.87 | 300 | 0 | 0.000 | 4130.47 | 7868.44 | 8172.94 | N/A | 0.013 | N/A | N/A |
| fifo | 8.51 | 300 | 0 | 0.000 | 2763.56 | 5027.58 | 5316.07 | 1100.08 | 1100.10 | 1 | 28.30 |
| latest | 8.59 | 260 | 40 | 0.133 | 289.01 | 334.73 | 343.98 | 95.74 | 95.80 | N/A | N/A |

资源占用（tegrastats，每模式独立采样）：

| Mode | RAM Peak (MB) | GR3D Mean (%) | GPU Temp Peak (C) | CPU Temp Peak (C) | Power Mean (mW) |
|---|---|---|---|---|---|
| sequential | 3502 | 54.54 | 55.75 | 54.06 | 8140.86 |
| fifo | 3629 | 51.53 | 57.06 | 55.34 | 8340.84 |
| latest | 3683 | 55.17 | 58.19 | 56.25 | 8417.07 |

单帧处理时延（ms；三模式几乎一致，差异在测量噪声范围内，**不据此称某模式更快**）：

| Mode | mean | P50 | P95 | P99 | max |
|---|---|---|---|---|---|
| sequential | 126.53 | 125.70 | 142.76 | 149.68 | 267.95 |
| fifo | 117.06 | 119.19 | 124.22 | 135.57 | 151.76 |
| latest | 115.86 | 117.64 | 123.21 | 133.66 | 139.02 |

### 数据解读（data-driven，不强行结论）

- **源供给侧**：10 fps；**处理侧**：~126ms/帧（sequential，含 TRT default-stream 同步开销）→ ~7.9 fps。处理 < 源供给 → 持续积压，这是三模式分化的根因。
- **sequential**：无 escape valve，积压线性累积。`live_e2e_latency_ms` 随帧单调增长至 **P99=8173ms / max=8236ms**（300 帧末端帧的实际端到端延迟已超 8 秒）。frame_age 始终 ~0（每帧到达即立即处理，但处理的是"很早之前该到达的帧"）。零丢帧，但延迟不可接受。
- **fifo**：有界 queue(maxsize=8) 吸收突发、queue 满时 producer block（实测 `producer_blocked=28.3ms`，`queue_depth_max=8` 触发背压）。零丢帧，但 queue_wait / frame_age P95≈1100ms —— 延迟被缓冲而非消除，仍持续上升至 P99=5316ms。适合"不能丢帧、可接受可控延迟"的场景。
- **latest**：drop-old 换低延迟。latency 有界：P50=289ms / P99=344ms，frame_age P95=96ms —— 始终处理近实时帧。代价是丢 40/300=13.3% 帧（`reason=superseded_by_newer_frame`，不丢 in-flight 帧仅丢 pending 旧帧）。丢帧率与吞吐缺口一致：理论 max drop ≈ 1 - 8.0/10 = 20%，实测 13.3%（稳态尾部追回部分帧）。
- **资源**：三模式 RAM 3.5–3.7 GB、GR3D ~52–55%、GPU temp peak 56–58°C、power ~8.1–8.4 W，差异在噪声范围。latest 因更密集调度（CPU 线程唤醒更频繁）峰值 temp 略高（58.2°C），无热异常。
- **correctness**：三模式 0 NaN/Inf；sequential/fifo 300 detection；latest 260 detection（其余 40 帧为 superseded drop，未被推理）。latest 不丢 in-flight 帧（被推理中的帧不被取代），仅丢 consumer 忙时堆积的旧 pending 帧。

### 结论（基于上述数据）

- 若需**零丢帧**：fifo（可控延迟）优于 sequential（延迟爆炸）。sequential 在处理 < 供给时不可用。
- 若需**低实时延迟**：latest 用 13.3% 丢帧换 P99=344ms 有界延迟，适合作业容忍丢帧的实时场景。本合成视频 owl 始终居中、帧间位移小（max 30px 正弦），丢帧对"目标持续在场"语义影响小；**真实场景是否可丢帧需结合作业判定，本 benchmark 不替场景下该结论**。
- 三模式单帧处理时延差异在测量噪声内，**不据此排序处理速度**。
- 不声称任一策略"最优"；选择取决于 业务对丢帧 vs 延迟 的权衡。

### 单图回归

- 重跑 Day 4 owl 单图 joint regression（`scripts/run_edge_perception.py`，owl_glove_small.jpg，prompts “an owl,a glove,a finger”，target-label “an owl”，threshold 0.10）。
- **PASS**：label “an owl”，score 0.3419，box [275.19, 82.35, 463.64, 399.35]（与 Day1/2/4/6A baseline 一致），mask slot 0、selected IoU 0.7206、nonzero 34065、无 NaN/Inf。
- 结论：新增 runtime benchmark 脚本未改动 `run_edge_perception.py`，单图 Pipeline 未受影响。

### Evidence 路径

- `scripts/benchmark_video_runtime.py`
- `results/video_runtime_benchmark/{sequential,fifo,latest}/{summary.json,frames.jsonl,tegrastats.log,tegrastats_parsed.json,run.log}`
- `results/video_runtime_benchmark/comparison.json` / `comparison.md`

### 约束遵守（Stage 6B）

- 未 remote / push / GitHub 上传（无 remote，仅 local commit）。
- 未修改 `third_party/nanoowl`、`third_party/nanosam`；未安装新依赖。
- 未 rebuild engine / 未 INT8 / 未新精度 / 未 COCO / 未 ROS2 / 未 camera / 未 long-run stability。
- **未继续 Stage 6C**。

### 默认配置（未变）

- NanoOWL：`owlvit_image_encoder_patch32.engine`（FP16）
- NanoSAM Image Encoder：`resnet18_image_encoder.engine`（FP16）
- NanoSAM Mask Decoder：`mobile_sam_mask_decoder.engine`（FP32）

## Day 6 结论（更新）

Stage 6A：MP4 基础视频推理闭环已建立，60 帧 smoke PASS，单图回归 PASS。

Stage 6B：在真实帧到达节奏（source FPS pacing）下对比 sequential / fifo / latest 三种 runtime 策略（300 源帧）。结果：sequential 无 escape valve 致 live latency 爆炸（P99=8173ms）；fifo 零丢帧但延迟被缓冲持续上升（P99=5316ms，背压 28.3ms，queue depth max=8）；latest 用 13.3% superseded drop 换有界延迟（P99=344ms，frame age P95=96ms）。三模式单帧处理 ~116–127ms、资源占用相近、0 NaN/Inf。选择取决于 丢帧 vs 延迟 权衡，不强行称某策略最优。单图回归 PASS。环境保持，无残留进程，无 remote/git push。
