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
