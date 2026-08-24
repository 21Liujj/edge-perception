# NanoSAM INT8 Post-Training Quantization

**Module C — INT8 PTQ for Jetson Orin deployment**

This document explains INT8 quantization fundamentals and the engineering path for quantizing the self-trained NanoSAM ResNet18 encoder (50-epoch student from Module B) to INT8 via TensorRT post-training quantization (PTQ), bound to this project's actual constraints and artifacts.

---

## 1. Quantization Basics

### 1.1 FP32 / FP16 vs INT8

**Floating-point (FP32 / FP16)**: Each value is represented as sign + exponent + mantissa. Wide dynamic range, but compute-intensive and memory-intensive.

**Integer (INT8)**: Each value is represented as an 8-bit signed or unsigned integer. To map real-valued weights/activations to INT8, we use:

```
real_value ≈ scale * quantized_int8_value
```

Where `scale` is a floating-point factor that maps the INT8 range `[-128, 127]` (or `[0, 255]` unsigned) back to the real dynamic range. Some frameworks also use a `zero_point` offset:

```
real_value ≈ scale * (quantized_int8_value - zero_point)
```

**TensorRT 10.3** uses scale-based quantization for implicit PTQ (IInt8EntropyCalibrator2); explicit Q/DQ nodes can also carry zero-point if the ONNX graph includes them, but the primary path for this project is scale-only implicit quantization.

### 1.2 Weight Quantization vs Activation Quantization

- **Weight quantization**: Model parameters (Conv kernels, etc.) are quantized offline. Scales are computed from weight tensors directly (per-channel or per-tensor).
- **Activation quantization**: Intermediate feature maps are quantized at runtime. Their dynamic range is unknown until we run real data → **calibration** is needed.

### 1.3 Calibration

**Calibration** = running a representative dataset through the FP32/FP16 model to collect activation statistics (min/max or distribution), then deriving INT8 scales that minimize quantization error.

**Dynamic range**: The [min, max] of each activation tensor across the calibration set.

**Quantization error**: The difference between the FP32 activation and its INT8-dequantized approximation. Good calibration minimizes this error across the distribution, not just the extremes.

### 1.4 Dequantization

After INT8 compute, the result is **dequantized** back to FP16/FP32 for layers that cannot run in INT8 (e.g., Erf, some activations). TensorRT inserts these casts automatically.

### 1.5 Mixed Precision

Not all operations benefit from or support INT8. TensorRT can execute:
- **INT8**: Conv, ReLU, Add, Mul (if scales align), MaxPool
- **FP16 fallback**: Erf, Div (GELU components), ConvTranspose (uncertain), complex element-wise ops

Mixed precision = some layers INT8, others FP16. TensorRT decides automatically unless you override with layer precision constraints.

### 1.6 PTQ vs QAT

- **PTQ (Post-Training Quantization)**: Quantize a trained FP32/FP16 model without retraining. Fast, no gradient updates, but quality can degrade if the model is sensitive to quantization noise.
- **QAT (Quantization-Aware Training)**: Insert fake-quantization ops during training so the model learns weights robust to INT8. Higher quality, but requires retraining (OUT OF SCOPE for Module C — distillation is COMPLETE, STOP_LONG_TRAINING).

---

## 2. TensorRT INT8 Capability (TRT 10.3 on Jetson Orin)

**Hardware**: Jetson Orin (sm_87 Ampere) has `platform_has_fast_int8 = True` → DP4A / IMMA instructions for fast INT8 matmul/conv.

**Software**: TensorRT 10.3.0 (Python + trtexec).

### 2.1 Implicit Quantization (Calibration-Based PTQ)

**Path**: `trtexec --onnx=<model.onnx> --int8 --fp16 --calib=<calibration.cache> --saveEngine=<int8.engine>`

1. You provide a **calibrator** (IInt8EntropyCalibrator2) that feeds calibration images.
2. TensorRT builds the engine, runs the calibration set, collects activation ranges, computes INT8 scales, and writes a `calibration.cache` (binary, engine-specific, not portable).
3. The cache can be reused for future builds of the same ONNX graph.

**Calibrators**:
- `IInt8EntropyCalibrator2` (default, KL-divergence minimization for activation distributions)
- `IInt8MinMaxCalibrator` (simple min/max clipping; faster, less accurate)
- `IInt8LegacyCalibrator` (deprecated)

**Recommendation for NanoSAM**: `IInt8EntropyCalibrator2` (TensorRT's best practice for CNN PTQ).

### 2.2 Explicit Quantization (Q/DQ Nodes)

**Path**: Insert `QuantizeLinear` / `DequantizeLinear` (Q/DQ) nodes into the ONNX graph (via ONNX Runtime quantization tools or PyTorch FX quantization). Build with `NetworkDefinitionCreationFlag.STRONGLY_TYPED`. TensorRT respects the explicit scales.

**Advantage**: Per-channel weight quantization control, selective layer quantization (e.g., keep GELU cluster FP16, quantize Convs).

**Cost**: ONNX graph surgery, more engineering complexity.

**Fallback for this project**: If implicit PTQ (2.1) causes quality regression, explicit Q/DQ is the next step before considering QAT.

---

## 3. ONNX Graph Analysis

**Source**: `models/self_distilled/epoch50/nanosam_resnet18_epoch049.onnx` (93 nodes, FP32 weights, no Q/DQ).

**Operator breakdown** (INT8 likelihood = theoretical, actual precision determined by TensorRT layer fusion + builder decisions):

| Operator | Count | INT8 Likelihood | Notes |
|----------|-------|-----------------|-------|
| Conv | 25 | HIGH | Primary target for INT8 (bulk of compute) |
| ReLU | 17 | HIGH | Fuses with Conv, INT8-friendly |
| Add | 14 | HIGH | Residual connections; watch for scale mismatch across branches |
| Constant | 15 | N/A | Folded at build time |
| Mul | 10 | MEDIUM | INT8 or FP16 fallback (part of GELU) |
| Div | 5 | LOW | FP16 fallback (GELU denominator) |
| Erf | 5 | LOW | FP16 fallback or fused approximation (GELU core) |
| MaxPool | 1 | HIGH | INT8 |
| ConvTranspose | 1 | UNCERTAIN | Upsample in decoder projection; may fall back to FP16 |

**Sensitive tail**: 
- **GELU cluster**: 5× Erf + 5× Div + 10× Mul (likely FP16 or mixed-precision fused kernel)
- **ConvTranspose**: Output projection upsampler (1 instance, end of encoder)
- **14× residual Adds**: Scale alignment critical; TensorRT inserts requantization if branch scales differ

**Conclusion**: Conv-ReLU-Add backbone should quantize well. GELU and ConvTranspose are the quality-risk zones.

---

## 4. Calibration Dataset Design (Stage C0 → C1)

### 4.1 Source

**COCO 2017 val2017** (5000 images), already on Jetson device.

**Why val2017 not train2017?**
- val2017 is the same distribution NanoSAM evaluation sees (Module C baseline protocol uses val2017).
- train2017 (118k images, 19GB) is NOT on device; user would need to pull it. Marginal calibration benefit for an encoder already trained on train2017.
- val2017 is a valid, in-distribution, present source for PTQ activation statistics.

**No evaluation leakage**: Calibration uses raw images only (no masks, no boxes, no GT). STANDARD500 eval set is never touched during calibration.

### 4.2 Sample Size

**Primary recommendation: 512 images**

| Size | Wall Time (est.) | Coverage |
|------|------------------|----------|
| 100  | ~30s | Low |
| 256  | ~75s | Moderate |
| **512**  | **~150s** | **Good (recommended)** |
| 1024 | ~300s | High (diminishing return) |

**Rationale**: 512 balances coverage vs calibration wall-time. TensorRT rule-of-thumb: 500-1000 batches. If INT8 quality is borderline in C1, bump to 1024.

### 4.3 Sampling

- **Deterministic random**: `numpy.default_rng(seed=20260824).choice(val2017_list, size=512, replace=False)`
- **No stratification**: Calibration is image-level encoder statistics, not category/size metrics. Complex stratification expands scope without PTQ benefit.

### 4.4 Preprocessing (CRITICAL: must match deployment)

Calibration images MUST go through the EXACT same preprocessing as deployment:

```python
from nanosam.utils.predictor import preprocess_image

# RGB PIL image → [1, 3, 1024, 1024] torch.float32 cuda tensor
tensor = preprocess_image(image, size=1024)
```

**Steps**:
1. Aspect-preserving resize (long edge → 1024)
2. HWC → CHW
3. To float32
4. ImageNet mean/std normalization
5. Zero-pad to [1, 3, 1024, 1024]
6. To CUDA

**DO NOT** feed raw JPEG bytes or skip normalization — calibration would see the wrong input distribution, and INT8 scales would be garbage.

### 4.5 Cache Design

**Path**: `models/self_distilled/epoch50/int8/calibration.cache`

**Manifest**: `calibration_manifest.json` — image paths, SHA256 per image, seed, sample count, ONNX SHA, preprocessing version, TensorRT version.

**Reuse guard**: Cache is bound to ONNX SHA + preprocessing + TRT version. A drift in any of these invalidates the cache.

---

## 5. Quality & Performance Gates (Module C Engineering Heuristics)

### 5.1 FP16 Baseline (Frozen)

**Source**: Stage Jetson-6C-2 (50-epoch student FP16)

| Metric | Value |
|--------|-------|
| STANDARD500 mIoU | 0.715650 |
| FULL COCO mIoU | 0.699183 |
| Encoder latency (mean) | 11.08 ms |
| Encoder latency (P50) | 10.86 ms |
| Encoder latency (P95) | 11.48 ms |
| Encoder latency (P99) | 11.55 ms |

**Protocol**: `execute_async_v3`, encoder only, warmup 30, measure 100 (Stage 6C-1).

### 5.2 Quality Gate (Primary: STANDARD500)

**INT8 quality check runs STANDARD500 FIRST** (500 images, ~8min). Only if quality gate passes, run FULL COCO (~2.5h).

**Heuristic** (project-specific, not statistical):

```
delta = INT8_mIoU_STANDARD500 - FP16_mIoU_STANDARD500
```

| delta | Verdict |
|-------|---------|
| ≥ -0.005 | QUALITY_ACCEPTABLE |
| [-0.015, -0.005) | QUALITY_DEGRADED_BUT_USABLE |
| < -0.015 | QUALITY_REGRESSION |

**Rationale**: NanoSAM is already a distilled student (FP16 FULL 0.699 vs historical MobileSAM 0.703). Further degradation from INT8 must be bounded. -0.005 (0.5 pp) is acceptable PTQ noise; -0.015 crosses into deployment-blocking territory.

### 5.3 Performance Gate

**Metrics to compare (same-protocol FP16 vs INT8)**:
- Encoder latency: mean, P50, P95, P99
- Engine file size (MB)
- GPU memory allocation (MB)
- RSS (MB)
- (Optional) Power (mW)

**Expectation**: INT8 should show latency reduction (target: 20-40% faster) and/or memory reduction. If quality degrades AND performance does not improve, INT8 is not worth deploying.

**Caution**: Do NOT compare historical FP16 numbers from different protocols to INT8. Re-measure FP16 and INT8 in the same benchmark session with the same script.

---

## 6. Mixed Precision Fallback Plan (If Full INT8 Regresses)

**Fallback sequence** (Stage C0 design only, not executed):

1. **Full INT8** (implicit PTQ, IInt8EntropyCalibrator2) → quality gate
2. If QUALITY_REGRESSION:
   - Profile which layers have the highest quantization error (TensorRT layer precision inspection, or compare per-layer outputs FP16 vs INT8)
   - Identify suspects: ConvTranspose, GELU cluster (Erf/Div/Mul), residual Adds with scale mismatch
3. **Selective FP16 override**:
   - Option A: Keep ConvTranspose + GELU cluster FP16, quantize Convs → rebuild
   - Option B: Explicit Q/DQ ONNX (per-channel weight quant, skip sensitive ops) → rebuild
4. Re-gate
5. If still regressing: escalate to QAT (OUT OF SCOPE for Module C; would need user decision to resume training)

**Do NOT** jump to complex layer search without evidence. Start with full INT8; if it fails, profile first.

---

## 7. Stage Jetson-C0 Audit Summary

**Status**: INT8_PTQ_PATH_READY

**Selected path**: TensorRT implicit PTQ (IInt8EntropyCalibrator2)

**Next stage** (NOT executed): **Jetson-C1** — Build calibration dataset (512 val2017 images, seed 20260824) → generate `calibration.cache` → first INT8 engine build → STANDARD500 quality gate vs FP16 0.715650.

**Evidence**: `results/stage_jetson_c0/` — environment, source artifact audit, TRT INT8 capability, ONNX operator audit, quantization strategy, calibration design, preprocessing freeze, calibration cache design, FP16 baseline freeze, quality gate design, performance gate design, mixed precision plan, stage summary.

**Git**: Local commit (not pushed).

---

## 8. References

- TensorRT Developer Guide (INT8 calibration): https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8
- NVIDIA blog: "8-bit Inference with TensorRT": https://developer.nvidia.com/blog/8-bit-inference-with-tensorrt/
- Paper: "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Jacob et al., 2018)
- This project's frozen artifacts: `models/self_distilled/epoch50/nanosam_resnet18_epoch049.onnx` (SHA `906d8fd7…`)

**Document role**: Knowledge base + engineering path for NanoSAM INT8 PTQ. Update as Module C progresses (C1 calibration, C2 first INT8 build, C3 quality validation, C4 performance benchmark, C5 mixed-precision if needed).
