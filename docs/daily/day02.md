# Day 2

## 今日目标

将 Day 1 NanoOWL baseline 工程化，并建立 correctness regression 与 performance benchmark。

## Stage 2A — 可复用 NanoOWL 推理 CLI

- 目标：在 Day 1 `scripts/nanoowl_infer.py` 与 baseline 基础上整理出可复用 CLI，不重新实现模型逻辑；复用固定 NanoOWL `OwlPredictor` API 与现有 FP16 engine。
- 输入：
  - Engine: `third_party/nanoowl/data/owlvit_image_encoder_patch32.engine`（Day 1 构建，FP16，SHA256 `d3f723f97d5e466805105f8b25da83d898e215c09598fb2b41c06d5254d09037`）
  - Model: `google/owlvit-base-patch32`（离线 HF cache，rev `cbc355fb...`）
  - 回归图: `third_party/nanoowl/assets/owl_glove_small.jpg`
  - Prompts: `an owl,a glove,a finger`，threshold `0.10`
- 产出：`scripts/run_nanoowl.py`
  - CLI 参数：`--image`、`--prompts`、`--threshold`、`--engine`、`--output-dir`（另含 `--model`、`--pad-square`）
  - 输出目录至少生成：`detections.json`、`result.jpg`、`timing.json`
  - `detections.json` 包含：prompts、threshold、label、score、box（xyxy）
  - `timing.json` 记录：`model_load_ms`、`inference_ms`、`total_ms`（非正式 Benchmark，无统计）
- 兼容性处理：保留 Day 1 验证的 OpenCV readonly-array 兼容——在自有脚本内对图像做可写 `.copy()` 后再 `draw_owl_output`，不修改 `third_party/nanoowl`。
- 回归结果（与 Day 1 baseline 一致）：
  - num_detections: 1
  - label: `an owl`，score: `0.3419235944747925`，box: `[275.19, 82.35, 463.64, 399.35]`
  - 仍检测到 `an owl` ✓
- timing（本次回归，单次冷启动，非正式 Benchmark）：
  - model_load_ms: ~1878.9
  - inference_ms: ~441.7
  - total_ms: ~2327.3
- 证据目录：`results/nanoowl_cli/`（`detections.json`、`result.jpg`、`timing.json`）
- 完整性：不修改模型/Engine/核心依赖；不做 NanoSAM；不做正式 Benchmark；不下载新模型；无 remote/push/GitHub 上传；`third_party/nanoowl` 未改。
- Stage 2A status: PASS。

## Stage 2B — 固定回归测试集与批量正确性验证

- 目标：建立 NanoOWL 固定 regression manifest + 批量正确性 runner；复用 `scripts/run_nanoowl.py` 推理逻辑，不复制/重写模型核心。
- 图片来源（仅本地，未联网/下载）：`third_party/nanoowl/assets/` 原始照片（排除 `*_out.jpg` 可视化输出）。本地不同有效图片 >=3，使用 5 张建立 5 个 case。样本覆盖有限（仅仓库自带 asset），已在 manifest 旁注明。
- Manifest：`tests/nanoowl_regression_manifest.json`，每 case 记录 `case_id`、`image`、`prompts`、`threshold`（owl 基准 case 含 `expect_label: "an owl"`）。
- Runner：`scripts/run_nanoowl_regression.py`（import 复用 `run_nanoowl.load_predictor` / `run_once`，单次模型加载后批量推理）。每 case 单独保存 `detections.json`、`result.jpg`、`timing.json`、`status.json`。
- 判定规则：零检测 ≠ FAIL；仅模型加载/推理异常/输出非法记 execution FAIL。Day 1 owl case 必须仍检测到 "an owl"，否则整 Stage FAIL。
- 结果（summary.json）：status=PASS，owl_baseline_pass=true，total_cases=5，pass_cases=5，failed_cases=0，distinct_images=5。

  | case_id | image | prompts | detection_count | labels (score) |
  |---|---|---|---|---|
  | owl_glove | owl_glove_small.jpg | an owl,a glove,a finger | 1 | an owl (0.3419) |
  | owl_glove_full | owl_glove.jpg | an owl,a glove | 1 | an owl (0.3649) |
  | frog | frog.jpg | a frog,a leaf,a hand | 1 | a frog (0.4309) |
  | classroom | class.jpg | a person,a backpack,a chair | 2 | a chair (0.1201), a person (0.1096) |
  | camera_scene | camera.jpg | a camera,a hand,a person | 0 | — (零检测，execution PASS) |

- 证据目录：`results/nanoowl_regression/`（`summary.json` + 每个子目录 `detections.json`/`result.jpg`/`timing.json`/`status.json`）。
- 完整性：不做正式 latency benchmark / mAP/mIoU / NanoSAM / INT8 / MP4；不装新依赖；不改模型/Engine；`third_party/nanoowl` 未改；无 remote/push/GitHub 上传。`scripts/run_nanoowl.py` 仅重构出 `load_predictor`/`run_once` 供复用，推理逻辑与 Day 1 一致（owl 基准 score 0.3419 复现）。
- Stage 2B status: PASS。

## Stage 2C — NanoOWL 稳定态性能 Benchmark

- 目标：测量 NanoOWL TensorRT baseline 稳定态推理性能 + 同步 tegrastats 系统资源；不做优化、不解释未验证瓶颈。
- 工具：`scripts/benchmark_nanoowl.py`（复用 `scripts/run_nanoowl.py` 的 `load_predictor`，单次模型加载后 warmup + 定时推理，无 per-iter 重载）。
- 配置：owl_glove_small.jpg，prompts `an owl,a glove,a finger`，threshold 0.10；FP16 engine 固定。
- 流程：warmup 40 次 -> 正式 300 次定时推理 -> tegrastats 在定时窗口内 500ms 采样。每轮校验：无异常 / 输出无 NaN / owl 检测仍存在。
- 结果（真实测量）：
  - model_load_ms: 1860.80
  - latency(ms): mean 22.49 / min 20.58 / max 26.07 / p50 22.47 / p95 22.85 / p99 24.38
  - estimated FPS: 44.46
  - validity: inference_errors 0, NaN false, owl 每轮存在 true
- 系统资源（tegrastats，14 样本）：RAM mean 3121MB(peak 3124/7620)；GR3D freq mean 79.57%(max 94%)；GPU temp mean 56.26C(max 57.19)；CPU temp mean 53.24C；VDD_IN mean 16148.71mW(max 16830)；EMC tegrastats 未上报（记为不可用）。
- tegrastats 进程在 benchmark 结束后正常停止（仅本阶段进程，未动系统其他进程）。
- 后置回归：owl baseline 重跑 PASS，检测 "an owl" score 0.3419（与 Day 1 一致）。
- 证据目录：`results/nanoowl_benchmark/`（benchmark.json、latencies.csv、tegrastats.log、tegrastats_stats.json、summary.md、post_bench_regression/）。
- 完整性：不改模型/Engine/源码/核心依赖；不做 NanoSAM/INT8/MP4/ROS2；不改 power mode、不跑 jetson_clocks；不重建 Engine；不装新依赖；无 remote/push。
- Stage 2C status: PASS。

---

## Day 2 Evidence

- `results/nanoowl_cli/` — Stage 2A reusable CLI run output（detections.json / result.jpg / timing.json）
- `results/nanoowl_regression/` — Stage 2B regression suite（summary.json + 每 case 子目录 detections.json / result.jpg / timing.json / status.json）
- `results/nanoowl_benchmark/` — Stage 2C benchmark（benchmark.json / latencies.csv / tegrastats.log / tegrastats_stats.json / summary.md / post_bench_regression/）

### Stage 2C 关键数据（真实测量）

- model_load_ms = 1860.80
- warmup = 40，iterations = 300
- mean = 22.49 ms / P50 = 22.47 ms / P95 = 22.85 ms / P99 = 24.38 ms / max = 26.07 ms
- estimated inference-only FPS = 44.46
- Peak RAM = 3124 MB
- GR3D mean = 79.57% / GR3D max = 94%
- GPU temp mean = 56.26 C / GPU temp max = 57.19 C
- CPU temp mean = 53.24 C
- VDD_IN mean = 16148.71 mW / VDD_IN max = 16830 mW
- EMC = N/A（当前 L4T tegrastats 未上报）
- post benchmark regression：an owl，score 0.3419，PASS

### 性能口径说明

44.46 FPS 当前属于 **steady-state inference-only estimate**，不是完整 image I/O / preprocessing / visualization / NanoSAM pipeline 的 end-to-end FPS。

---

## Day 2 Final Result

**PASS — NanoOWL has been converted from an upstream baseline demo into a reusable, regression-tested and benchmarked Jetson inference module.**

- CLI（Stage 2A）：PASS
- Regression suite（Stage 2B）：PASS（5 cases / 5 images，owl 基准稳定复现）
- Performance benchmark（Stage 2C）：PASS（mean 22.49ms，~44.46 inference-only FPS，0 异常 / 无 NaN / owl 每轮存在）

### Day 2 仍未做（明确列出）

- 正式 COCO accuracy benchmark
- NanoSAM
- NanoOWL + NanoSAM
- INT8
- MP4 realtime pipeline
- ROS2
- stability test

---

## Day 3 起点

NanoSAM TensorRT baseline：Image + Box/Point Prompt -> Segmentation Mask。
