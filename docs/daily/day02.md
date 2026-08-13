# Day 2

## 今日目标

将 Day 1 NanoOWL baseline 整理为可复用的工程化推理 CLI，固定 API 与现有 FP16 TensorRT Engine，保证回归与 Day 1 一致，为固定测试集 + 初步 Benchmark（Stage 2B）做铺垫。

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

## Day 2 后续（未做，见 Stage 2B 起）

- 固定测试集
- 初步性能 Benchmark（多轮统计）

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
