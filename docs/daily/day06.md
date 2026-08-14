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
