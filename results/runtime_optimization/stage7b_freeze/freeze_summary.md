# Stage 7B-Freeze — Experimental Fused CUDA Preprocessing Prototype

## Why we started a CUDA kernel (Motivation)

Stage 7A confirmed the joint pipeline's GPU idle was **not** small-H2D-count (that
was reduced 44.7→4.0/iter with no joint benefit) but **host preprocessing**: PIL.resize
≈15.5 ms + normalize ≈4.9 ms run on CPU while the GPU sits idle, before the encoder's
~9 ms. Stage 7A's joint variant was REJECTED. The stated next step (Stage 7A
recommendation) was a **fused CUDA preprocessing kernel** to move resize + normalize +
layout/dtype onto the GPU. Stage 7B implemented exactly that prototype.

## What the kernel does

`csrc/cuda_preprocess/cuda_preprocess_kernel.cu` — a single `__global__` launch, one
thread per output pixel, fusing:
- **bilinear resize** (half-pixel center convention + boundary clamp)
- **per-channel normalize** (mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375])
- **HWC→NCHW layout**
- **uint8→FP32 dtype**

C++ binding `csrc/cuda_preprocess/cuda_preprocess.cpp`; Python wrapper
`edge_perception/runtime/cuda_preprocess.py` (CudaPreprocessor + CudaPreprocessPredictor,
reused pinned staging + reused GPU buffers + explicit stream, opt-in `--runtime cuda`).
Built via `setup_cuda_preprocess.py` (torch CUDAExtension, nvcc 12.6). Output `.so` is
gitignored.

## Performance potential (evaluated workload, owl_glove_small.jpg, warmup≥30/iters≥300)

| metric | baseline | cuda | speedup |
|---|---|---|---|
| standalone preprocess boundary | 26.8 ms | 3.12 ms | **8.58×** |
| set_image mean | 64.4 | 14.1 | **4.57×** |
| set_image P50 / P95 / P99 | 66.4 / 70.8 / 74.2 | 13.6 / 16.5 / 19.8 | — |
| joint mean | 106.4 | 46.7 | **2.28×** |
| joint P50 / P95 / P99 | 107.7 / 114.2 / 119.5 | 46.7 / 47.4 / 47.9 | — |

The PIL.resize host gap is eliminated and latency drops sharply; tail latency (P95/P99)
is strictly better on the CUDA path. **These are PROVISIONAL performance numbers.**

## Correctness result (the gate that decides)

**Status: EXPERIMENTAL_CUDA_PROTOTYPE_NUMERICAL_DRIFT** — the CUDA path runs and is
fast, but preprocessing numerical change causes **observable segmentation degradation**.

### Tensor numerical comparison (baseline PIL-BICUBIC vs CUDA bilinear)
| image | region mean_abs | region max_abs | region RMSE | region P95_abs |
|---|---|---|---|---|
| owl (756×499) | 1.667 | 4.464 | 1.980 | 3.799 |
| dogs (1180×760) | 2.154 | 4.464 | 2.429 | 4.041 |
(shape match True, dtype FP32, pad region exactly zero on CUDA.)

Root cause of drift: PIL default resample is **BICUBIC** (verified), the kernel uses
**bilinear**. The error is bounded (~max 4.5 in normalized units ≈ ±6 px) but it is a
*semantic* resampling difference, not floating-point noise.

### Owl regression (owl_glove_small.jpg)
- baseline / cuda: mask_nonzero 34065 / 51434; pred IoU 0.721 / 0.632
- **baseline-vs-CUDA mask IoU = 0.600**; mask area ratio 1.51 (CUDA mask 51% larger)
- owl_PASS (non-empty, no NaN/Inf): **true** — but mask drift is large.

### Dogs regression (dogs.jpg, box [100,100,850,759], slot 0)
- baseline / cuda: mask_nonzero 242157 / 239469; pred IoU 0.662 / 0.642
- **baseline-vs-CUDA mask IoU = 0.538**; mask area ratio 0.99
- dogs_PASS (non-empty, no NaN/Inf): **true** — but mask IoU 0.54 is poor.

### COCO QUICK 100-instance (Day-5 fixed manifest, slot 0, same engines/evaluator)
| | baseline | cuda |
|---|---|---|
| mIoU | **0.7213** | **0.4756** |
| small / medium / large mIoU | 0.6598 / 0.7279 / 0.7925 | 0.3904 / 0.4923 / 0.5674 |
| valid / failed | 100 / 0 | 100 / 0 |

- mean signed IoU delta = **−0.2458**
- mean abs IoU delta = **0.2562**, max abs IoU delta = **0.6575**
- 91/100 instances drift > 0.05, 81/100 drift > 0.10

This is a **severe, systematic quality drop** (−34% relative mIoU), not noise. Per the
freeze acceptance policy this is **NUMERICAL_DRIFT**, not VALIDATED.

## Is the CUDA path the V1 default runtime? — NO

Default runtime remains **baseline / officially-compatible preprocessing**. The CUDA
path stays **opt-in** (`--runtime cuda` / `--preprocess cuda`). It is **not** promoted:
despite the large latency win, the COCO quality regression disqualifies it for V1.

## Why we freeze now (not: why we stop optimizing because we failed)

We freeze because:
1. EdgePerception V1's goal is a **correct, stable** perception demo — the baseline
   runtime already passes COCO QUICK (mIoU 0.7213) and the joint pipeline works.
2. The prototype **proved the latency hypothesis** (host preprocessing was the
   bottleneck; fusing it on GPU removes idle) — that is a real, valuable finding.
3. The correctness gap is a **resampling-semantics** problem (bilinear ≠ PIL BICUBIC),
   a systematic CUDA-engineering task (a GPU bicubic/area-resize kernel, possibly with
   PIL-equivalent antialias prefilter), which belongs to V2, not V1 finalization.
4. Per the freeze directive: do not relax quality standards to preserve performance
   numbers; do not keep negative-drift code as default.

## What V2 can do (deferred, NOT done now)

- Replace bilinear with a PIL-faithful **BICUBIC** (or area-averaging for downscale)
  CUDA kernel; re-run COCO QUICK — target mean abs IoU delta ≈ 0.
- Then: Nsight Compute kernel profiling, shared-memory/texture/vectorized load, occupancy
  tuning, pinned zero-copy input, stream overlap with NanoOWL.
- INT8, C++ TensorRT runtime, memory optimization, camera/GStreamer zero-copy.

## Evidence

`results/runtime_optimization/stage7b_freeze/`:
- `prototype_contract.md` — input/output/transform/binding contract (from source)
- `tensor_numerical_comparison.json` — baseline-vs-CUDA tensor error
- `owl_regression.json` / `dogs_regression.json` — mask regression + baseline-vs-CUDA IoU
- `coco_quick_baseline.json` / `coco_quick_cuda.json` / `coco_quick_paired.json` — COCO gate
- `performance_summary.json` — set_image + joint + standalone-preprocess latency
- `freeze_summary.md` — this document

`results/runtime_optimization/stage7b/` — full benchmark JSON + preprocess benchmark +
nsys-eligible raw stats (Stage 7B run artifacts).
