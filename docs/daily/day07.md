# Day 07 — Stage 7A: CUDA-aware Runtime Optimization of NanoSAM set_image / Host→GPU 数据路径

基于 Day 6 / Stage 6C Nsight Systems Profiling 结果（整体 GPU 空闲 46.1%，
set_image GPU 空闲 ~36.7ms，H2D ~44.7 copies/iter），对 EdgePerception Joint
Pipeline 的 NanoSAM `set_image` / Host→GPU 数据路径实施 CUDA-aware Runtime
优化，并完成严格的 baseline vs optimized 对照验证。

## 约束（遵守）

- 禁止修改 `third_party/nanoowl` / `third_party/nanosam` upstream。
- 不重建 Engine，不做 INT8。
- 不实现 fused `__global__` preprocess kernel，不做 TensorRT Plugin，不写 C++ Runtime。
- 不为性能改变 preprocessing 数值定义（PIL.resize BILINEAR + mean/std + 零填充 完全保留）。
- 显式 CUDA Stream 使用真实依赖（event.synchronize / s.synchronize），无伪异步。
- 不为了让项目"看起来优化成功"保留负收益代码；Negative result 也是有效工程结果。
- 仅本地 commit，禁止 remote / push / GitHub 上传。

## 实现的优化

复用 upstream engine loader（`load_image_encoder_engine` / `load_mask_decoder_engine`），
仅重构 `set_image` / `predict` 数据路径：

- **OPT1** Pinned host staging buffer（`pin_memory=True`，`is_pinned()` 已校验），复用。
- **OPT2** 预分配 GPU 输入 tensor `(1,3,1024,1024)` 复用，每帧不再 `torch.zeros`。
- **OPT3** GPU-resident mean/std 常量缓存一次，不再每帧重建。
- **OPT4** 静态 decoder 输入在 device 上缓存（`mask_input_zero`、`has_mask_input_zero/one`、`labels_gpu`）。
- **OPT6** Point/label 模板复用，每帧仅更新 4 个 point 坐标（pinned 拷贝）。
- **OPT5** 显式 `torch.cuda.Stream` + Event（`use_stream=True`）；`--no-stream` 变体保留 OPT1–4,6 在默认流。

文件：`edge_perception/runtime/nanosam_optimized.py`（OptimizedPredictor）。

## set_image 数据路径审计结论

逐段计时（owl 图 756×499，100 iter 均值）：PIL.resize → 1024 占 **15.5ms**，
normalize 占 **4.9ms**，pad 1.6ms，H2D 2.1ms，encoder GPU 仅 **8.95ms**。
**根因：GPU encoder 只 ~9ms，但 PIL.resize(CPU 15.5ms)+normalize(CPU 4.9ms)
在主机串行执行时 GPU 空闲。** 缓冲复用无法移除 PIL.resize —— 这是 set_image
GPU 空闲持续存在的真正瓶颈，需 fused CUDA 前处理 kernel 才能解决（本 Stage 不做）。

## 验证结果

### Correctness gates（owl / dogs，baseline & optimized）

- owl: 检出 owl，`mask_nonzero = 34065`（与 baseline 完全一致），无 NaN/Inf。PASS
- dogs: 检出，`mask_nonzero = 375624`（与 baseline 完全一致），无 NaN/Inf。PASS

### COCO QUICK 100-instance 回归（vs Day5 baseline）

| 指标 | baseline | optimized |
|---|---|---|
| mIoU | 0.721328258289041 | 0.721328258289041 |
| mIoU small / medium / large | 0.6598 / 0.7279 / 0.7925 | 0.6598 / 0.7279 / 0.7925 |
| IoU≥0.50 / ≥0.75 | 0.88 / 0.53 | 0.88 / 0.53 |
| valid / failed | 100 / 0 | 100 / 0 |

- **mean abs IoU delta = 0.0，max abs IoU delta = 0.0**（100 全一致）。
- 数值语义零偏移。

### Benchmark（warmup≥30 / iter≥200，P50/P95/P99/max）

**set_image（micro）**

| ms | baseline | opt stream | opt no-stream |
|---|---|---|---|
| P50 | 66.14 | **61.53** | 67.04 |
| P95 | 69.13 | **64.67** | 72.30 |
| P99 | 70.55 | **66.21** | 73.32 |
| mean | 63.95 | **59.51** | 65.43 |

- stream speedup **1.075x**（P95 −4.46ms，超出噪声）。**ACCEPTED**。
- no-stream 0.977x（P95 更差）。REJECTED。

**joint（E2E：NanoOWL + NanoSAM）**

| ms | baseline | opt stream | opt no-stream |
|---|---|---|---|
| P50 | 105.12 | 113.87 | 110.07 |
| P95 | 114.39 | 115.19 | 116.08 |
| P99 | 115.62 | 116.23 | 116.40 |
| mean | 105.16 | 110.72 | 108.79 |

- stream speedup **0.950x**（慢，P95 +0.8ms 更差）。
- no-stream 0.967x（慢，P95 +1.7ms 更差）。
- **REJECTED（两个变体）** —— set_image 的收益未撑过 joint 流水：显式 stream 的
  event/sync 开销 + staging 跨默认流边界，超过了减小 H2D 拷贝带来的收益，且
  NanoOWL↔NanoSAM 串行无 overlap 窗口。

### Nsight re-profile（warmup20/iters15）

| 指标 | Stage6C baseline | Stage7A optimized |
|---|---|---|
| 整体 GPU busy | 53.9% | 50.5% |
| 整体 GPU idle | 46.1% | 49.5% |
| set_image wall ms | 55.7 | 57.92 |
| set_image GPU busy % | 34.1% | 29.5% |
| set_image GPU idle ms | ~36.7 | ~40.8 |
| **H2D copies/iter** | **44.7** | **4.0** |

- **H2D 拷贝数 44.7 → 4.0/iter（OPT4 静态缓存的客观收益）**，但 set_image wall
  与整体 GPU busy 未改善（PIL.resize 主机瓶颈未动，且 staging 开销落在 GPU 空闲窗）。
- Stage 6C evidence 保留未覆盖。

### tegrastats 资源权衡

| 资源 | baseline | optimized | Δ |
|---|---|---|---|
| RAM mean / max (MB) | 3116 / 3665 | 3122 / 3692 | +6 / +27 |
| GR3D mean / max (%) | 20.7 / 99 | 21.3 / 99 | ≈ |
| GPU temp mean (°C) | 53.15 | 53.7 | +0.5 (噪声) |
| VDD_IN mean / max (mW) | 6135 / 8438 | 6126 / 8582 | ≈ / +144 |

- RAM peak +27MB（pinned/preallocated buffer 常驻代价），可接受；功耗基本持平。

## 采纳决策

| 路径 | 结论 | 原因 |
|---|---|---|
| set_image (stream) | **ACCEPTED** | 1.075x，P95/P99 更好，数值一致 |
| joint (stream) | REJECTED | 0.950x 慢，P95/P99 更差 |
| joint (no-stream) | REJECTED | 0.967x 慢，P95 更差 |

不将优化 runtime 提升为默认；保留为 `--runtime optimized` 可选项。
**联合（生产）路径的负收益是有效工程结果**，诚实记录。

## 结论与下一步建议

- OPT1–4,6 在数据路径效率上确有客观收益（H2D 44.7→4.0/iter），set_image 在隔离
  benchmark 上 1.075x 提速；但 **joint 生产路径负收益**，原因为 PIL.resize(CPU 15.5ms)
  主机瓶颈未动，且显式 stream 在串行流水无 overlap 窗口时开销大于收益。
- **推荐下一步：fused CUDA 前处理 kernel**（PIL.resize + normalize 上 GPU）—— 这是
  缓冲复用路线无法跨越的真正瓶颈。本 Stage 依约束未实现。

## 证据

`results/runtime_optimization/stage7a/`
- `set_image_audit.md` / `optimization_decisions.md` / `benchmark_comparison.md` / `nsight_comparison.md`
- `benchmark_comparison.json` / `quality_regression.json` / `tegrastats_comparison.json`
- `baseline/` `optimized/`（set_image.json / joint.json / *_nostream.json / tegrastats.log）
- `coco_baseline/` `coco_optimized/`（summary.json / per_instance.jsonl / failed_instances.json / run.log）
- `nsight_optimized/`（edge_perception.nsys-rep / .sqlite / *_sum.csv/txt / profile_analysis.json / profile_meta.json）

---

# JD Capability Mapping（基于本 Day 真实完成能力）

仅列出 Stage 7A 实际完成的能力，未涉及不列（无 custom CUDA kernel / TensorRT Plugin / C++ Runtime / INT8）。

- **Jetson 推理运行时优化**：在不改 upstream/engine 的前提下，重构 TensorRT 推理
  数据路径（预分配 GPU buffer / 静态 decoder 输入缓存 / point 模板复用）。
- **TensorRT 运行时集成**：复用 torch2trt TRTModule（owl FP16 / NanoSAM enc FP16 /
  dec FP32），保持 `execute_async_v3` 在 `current_stream` 上执行，不改 engine。
- **CUDA Stream 编排**：显式 `torch.cuda.Stream` + Event 构造真实依赖
  （`event.record`/`event.synchronize`/`s.synchronize`），区分有/无 stream 两变体，
  识别 torch2trt 内部用 current_stream —— 无需 monkey-patch，并记录 STREAM_LIMITATION。
- **Pinned host memory + non-blocking H2D**：`pin_memory=True` staging buffer（`is_pinned`
  校验），`copy_(non_blocking=True)` 跨流传输。
- **Buffer 复用 / 预分配**：消除每帧 `torch.zeros` 大块分配与静态数据逐帧重建。
- **Nsight Systems 系统级 Profiling**：`nsys profile --trace=cuda,osrt,nvtx` +
  `nsys stats` 多报告（cuda_api_sum / cuda_gpu_kern_sum / cuda_gpu_mem_size_sum /
  cuda_gpu_mem_time_sum / osrt_sum / nvtx_sum），NVTX range 划分 pipeline 阶段，
  NVTX_EVENTS + CUPTI 计算 per-stage GPU busy/idle。
- **CPU-GPU 协同分析**：定位主机串行段（PIL.resize / CPU normalize）为 GPU 空闲根因，
  区分"H2D churn"与"主机计算瓶颈"两类不同成因。
- **延迟百分位基准**：warmup≥30/iter≥200 的 set_image + joint micro benchmark，
  P50/P95/P99/max 与 baseline-vs-optimized 对照。
- **精度一致性回归**：COCO QUICK 100-instance，mean/max abs IoU delta 双向比对，
  断言数值语义零偏移。
- **资源-功耗权衡度量**：tegrastats 采样 RAM/GR3D/GPU temp/VDD_IN，量化优化带来的
  常驻内存与功耗代价。
- **负收益工程结论**：依验收策略对 joint 路径给出 REJECTED，保留负结果证据并
  指出真正瓶颈（fused CUDA 前处理 kernel）为下一步方向。
