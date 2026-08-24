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

---

## 9. CALIBRATION SOURCE CORRECTION (C0 → C1A)

**Date**: 2026-08-24  
**Stage**: Jetson-C1A  
**Change**: Calibration dataset source

### 9.1 Original C0 Design (Commit 8ebd6ed)

**Proposed calibration source**: COCO 2017 val2017 (5000 images)

**Rationale (C0)**:
- val2017 already present on Jetson device (no download needed)
- val2017 is the distribution NanoSAM evaluation sees (Module C baseline protocol)
- PTQ calibration uses raw images only (no segmentation GT), so no direct GT leakage

**C0 excerpt** (§ 4.1):
> COCO 2017 val2017 (5000 images), already on Jetson device. val2017 is the same distribution NanoSAM evaluation sees.

### 9.2 Corrected Design (C1A Final)

**Required calibration source**: COCO 2017 train2017

**Rationale (Correction)**:

While PTQ calibration does not consume segmentation ground truth, there is still a **distribution overlap risk**:

1. **STANDARD500** evaluation set: sampled from val2017
2. **FULL COCO** evaluation set: entire val2017 (5000 images)
3. **Calibration** (C0 proposal): 512 images from val2017

This creates **image-level overlap** between calibration and evaluation:
- If calibration uses val2017, the INT8 scale ranges are tuned on **the same image distribution** we later evaluate on.
- This is not GT leakage (calibration sees no masks), but it is **evaluation set leakage** — the models quantization is optimized for the test distribution.


---

## 9. CALIBRATION SOURCE CORRECTION (C0 → C1A)

**Date**: 2026-08-24  
**Stage**: Jetson-C1A  
**Change**: Calibration dataset source

### 9.1 Original C0 Design (Commit 8ebd6ed)

**Proposed calibration source**: COCO 2017 val2017 (5000 images)

**Rationale (C0)**:
- val2017 already present on Jetson device (no download needed)
- val2017 is the distribution NanoSAM evaluation sees (Module C baseline protocol)
- PTQ calibration uses raw images only (no segmentation GT), so no direct GT leakage

**C0 excerpt** (§ 4.1):
> COCO 2017 val2017 (5000 images), already on Jetson device. val2017 is the same distribution NanoSAM evaluation sees.

### 9.2 Corrected Design (C1A Final)

**Required calibration source**: COCO 2017 train2017

**Rationale (Correction)**:

While PTQ calibration does not consume segmentation ground truth, there is still a distribution overlap risk.

STANDARD500 and FULL COCO evaluation both use val2017. If calibration also uses val2017, the INT8 scale ranges are tuned on the same image distribution we later evaluate on. This is not GT leakage, but it is evaluation set leakage.

**Strict isolation principle**: Calibration and evaluation must use disjoint image sets, even at the image level, even when calibration does not see task labels.

**Final protocol** (Module C):
```
train2017 → Knowledge distillation training + INT8 calibration
val2017   → STANDARD500 + FULL COCO evaluation
```

This ensures:
1. Zero image overlap between calibration and evaluation
2. Conservative validation: If INT8 quality holds on unseen val2017, it generalizes
3. Consistent with ML best practice: Tune on train, validate on val

### 9.3 Impact on C0 Evidence

C0 evidence files remain unchanged for historical record. C1A and all subsequent stages use train2017 for calibration. This correction is design evolution, not error concealment.

### 9.4 Practical Consequence (C1A)

**Blocker**: train2017 is NOT present on Jetson device.

**C1A status**: BLOCKED (TRAIN2017_NOT_LOCAL)

**Unblock**: Transfer 512 train2017 images from AutoDL 5090 to Jetson.

**Recommendation**: Transfer 512-image calibration subset:
- Deterministic sampling on 5090: seed 20260824, n=512
- Tar archive: ~50-100 MB, transfer time ~5-10 minutes
- Extract to: data/coco/train2017_calib512/

Once train2017 is available, C1A resumes with:
- Manifest: data/calibration/nanosam_int8_train2017_512.json
- Calibrator: IInt8EntropyCalibrator2
- Cache: models/self_distilled/epoch50/int8/calibration_train2017_512.cache

---

**Summary**: C0 proposed val2017, C1A requires train2017 for strict calibration/evaluation isolation.

---

## 10. Stage Jetson-C1A-R — Calibration Cache Generation (COMPLETE)

**Date**: 2026-08-24
**Status**: PASS

### 10.1 Blocker Resolution

**Prior blocker**: train2017 not present on Jetson device.

**Resolution**:
1. SSH to AutoDL 5090 (bjb2:33884)
2. Verified train2017 present: 118,287 images
3. Deterministic sampling: numpy.random.default_rng(20260824).choice(118287, size=512, replace=False)
4. Per-image SHA256 verification: 512/512 PASS
5. Bundle: nanosam_train2017_calib512_seed20260824.tar.gz (80 MB)
6. Transfer: 5090 -> local -> Jetson
7. Extract + verify on Jetson: 512/512 SHA256 PASS

### 10.2 Calibration Manifest

**Path**: data/calibration/nanosam_int8_train2017_512.json
**SHA256**: c32bc4b2d3e977d3fc1b705b39c48e40a09b20b630e8ba1150b567d999242faf
**Image count**: 512 train2017 images
**Seed**: 20260824

### 10.3 Calibrator Implementation

**Class**: IInt8EntropyCalibrator2
**Implementation**: scripts/int8/simple_calibrator.py::SimpleCalibrator
**Batch size**: 1
**Preprocessing**: nanosam.utils.predictor.preprocess_image(size=1024)

### 10.4 Calibration Cache (Generated + Verified)

**Path**: models/self_distilled/epoch50/int8/calibration_train2017_512.cache
**SHA256**: 9689497ac73606952abfc113162907655ce80f79642e77ffd2a33daa22ac1487
**Size**: 5062 bytes

**Generation**:
- Batches calibrated: 512/512
- Calibration duration: 473.883s
- Post-processing duration: 16.340s
- Exit code: 0

**Cache reuse verification**:
- Second run: Zero batches reprocessed
- Log: "Reading Calibration Cache for calibrator: EntropyCalibration2"
- Cache SHA256 unchanged
- Wall clock: ~3s (vs 473.9s)

### 10.5 Evidence

results/stage_jetson_c1a/: c1a_evidence.json, calibration_train2017_512.cache, calibration_run1_generate.log.gz, checksums.txt

### 10.6 Next Stage

Stage Jetson-C1B: Build formal deployment INT8 TensorRT engine.

## 11. Stage Jetson-C1B: Formal INT8 Engine Build & Audit

**Status**: COMPLETE
**Result**: INT8_DEPLOYMENT_ENGINE_READY

### 11.1 Deployment Engine (DEFAULT profiling)

**ONNX**: `models/self_distilled/epoch50/nanosam_resnet18_epoch049.onnx`
**Cache**: `models/self_distilled/epoch50/int8/calibration_train2017_512.cache` (5062 bytes, SHA `9689497a...`)
**Output**: `models/self_distilled/epoch50/int8/nanosam_resnet18_epoch049_int8.engine`

**Size**: 22,249,660 bytes (21.22MB)
**SHA256**: `d8dd7a2f61427e072baeaf06bb38e78d2dab79c95f41c3839883d5891497d650`
**Build time**: 284.03 seconds
**Flags**: `INT8 | FP16` (mixed precision, FP16 fallback enabled)
**ProfilingVerbosity**: DEFAULT (for production deployment)

**Cache reuse**: PASS — TensorRT build log confirms "Generated calibration scales using calibration cache", zero calibration images reprocessed.

### 11.2 Audit Engine (DETAILED profiling)

**Output**: `models/self_distilled/epoch50/int8/nanosam_resnet18_epoch049_int8_detailed.engine`

**Purpose**: Analysis artifact only (precision/layer audit)
**Size**: 17,626,436 bytes (16.81MB)
**SHA256**: `f3afcb3d93d00401e4cff8e321f746126f25ce07c788c90afa833d9c51f8136e`
**Build time**: 234.55 seconds
**ProfilingVerbosity**: DETAILED (required for per-layer precision visibility)

Deployment and audit engines kept separate per design — deployment engine stability unaffected by profiling verbosity.

### 11.3 Runtime Verification

- Deserialize: PASS
- I/O contract: `image [1,3,1024,1024]` → `image_embeddings [1,256,64,64]`
- Runtime smoke (30 warmup + 200 measured): output finite, no fatal errors
- Predictor compatibility: PASS — loaded via upstream `nanosam.utils.predictor.Predictor` unmodified

### 11.4 Precision/Layer Audit (DETAILED engine)

| Precision | Layer count | Percentage | Examples |
|---|---|---|---|
| **INT8** | 30 | **90.9%** | Backbone Conv/ReLU/MaxPool main path |
| **FP16** | 2 | 6.1% | `pos_embedding` addition Reformat + Scale |
| **FP32** | 1 | 3.0% | Final output Reformat |
| Total | 33 | 100% | — |

Visibility: **FULL** (requires `ProfilingVerbosity.DETAILED`; default verbosity exposes layer names only, not precision)

**Key observation**: 90.9% INT8 coverage on compute path; FP16 fallback in `pos_embedding` (SAM position encoding module) and FP32 output reformat for downstream compatibility.

### 11.5 Same-Protocol Latency (FP16 vs INT8, encoder only)

**Protocol**: warmup=30, measured=200 iterations, run order FP16→INT8→FP16 to observe run-order drift

| Engine | mean | P50 | P95 | P99 |
|---|---|---|---|---|
| FP16 (run1) | 11.106ms | 11.043ms | 11.624ms | 11.773ms |
| **INT8** | **6.020ms** | **5.984ms** | **6.150ms** | **6.164ms** |
| FP16 (run2) | 10.790ms | 10.754ms | 10.904ms | 10.917ms |

**Speedup: mean 1.845x, P50 1.845x**
**FP16 run-to-run drift: <3%** (confirms measurement stability)

### 11.6 Engine Size

FP16: 36.29MB → INT8: 21.22MB
**Reduction: -41.52%**

### 11.7 Resource Comparison

CUDA allocated (post-inference): FP16 16.0MB, INT8 16.0MB — **identical** (activation memory dominated by fixed I/O tensor sizes, not weights).
Marked `MEMORY_RESULT_DESCRIPTIVE_ONLY` (single-process sequential measurement, not isolated).

### 11.8 Evidence

`results/stage_jetson_c1b/`: artifact_provenance.json, cache_reuse.json, environment.json, predictor_compatibility.json, runtime_smoke.json, resource_comparison.json, precision_layer_audit.json, latency_fp16.json, latency_int8.json, performance_comparison.json, engine_size_comparison.json, trt_build_summary.json, trt_build_summary_detailed.json, stage_summary.md

### 11.9 Next Stage

Stage Jetson-C2: Quality gate evaluation (STANDARD500 + FULL COCO mIoU vs FP16 baseline).


## 12. Stage Jetson-C2: Quality Gate Evaluation

**Status**: COMPLETE
**Result**: QUALITY_GATE_FAIL (degradation exceeds tolerance)

### 12.1 Evaluation Protocol

**Dataset**: COCO val2017
**Subsets**: STANDARD500 (500 images, 1283 instances), FULL (5000 images, 12930 instances)
**Prompt**: Ground-truth bounding box → NanoSAM single-box prompt (fixed slot 0)
**Metric**: Instance-level mean IoU (mask IoU averaged across all instances)
**Baseline**: FP16 TensorRT engine (same epoch49 ONNX)

### 12.2 Results

#### STANDARD500 (500 images)

| Engine | mIoU | Δ (abs) | Δ (%) |
|---|---|---|---|
| FP16 baseline | 0.715650 | — | — |
| **INT8** | **0.700264** | **-0.015386** | **-2.15%** |

**Quality gate**: FAIL (threshold <0.5%)

#### FULL COCO (5000 images)

| Engine | mIoU | Δ (abs) | Δ (%) |
|---|---|---|---|
| FP16 baseline | 0.699183 | — | — |
| **INT8** | **0.681910** | **-0.017273** | **-2.47%** |

**Quality gate**: FAIL (threshold <0.5%)

**Consistency**: STANDARD500 and FULL COCO show aligned degradation (~2.2-2.5%), confirming subset sampling was representative.

### 12.3 Size-Stratified Breakdown (FULL COCO)

| Size category | FP16 | INT8 | Δ (%) | Instance count |
|---|---|---|---|---|
| Small (area <32²) | 0.6475 | 0.6257 | **-3.37%** | 4485 |
| Medium (32²≤area<96²) | 0.7601 | 0.7461 | -1.84% | 5110 |
| Large (area≥96²) | 0.8145 | 0.8034 | -1.36% | 3335 |

**Pattern**: Degradation inversely proportional to object size — small objects most affected (precision loss amplified at fine spatial scales).

### 12.4 Root Cause

**Layer audit (Section 11.4)**: 90.9% INT8, but `pos_embedding` module (SAM absolute positional encoding) fell back to FP16.

**Hypothesis**: INT8 quantization on conv/norm/relu backbone path accumulated numerical error → positional encoding mismatch → spatial localization degradation → mask boundary quality drop.

Small objects have fewer pixels → boundary precision matters more → higher relative IoU loss.

### 12.5 Engineering Conclusion

**Tradeoff achieved**:
- Performance: **1.845x speedup** (11.1ms → 6.0ms encoder latency)
- Compression: **41.5% size reduction** (36.3MB → 21.2MB)
- Quality cost: **-2.47% mIoU** (FULL COCO)

**Industrial assessment**:
- Quality gate FAIL (exceeds <0.5% tolerance)
- Acceptable for latency-critical applications (real-time robotics, edge video analytics) where 2.5% quality loss trades favorably for 1.8x throughput
- NOT acceptable for high-precision scenarios (medical imaging, fine-grained recognition)

**No further optimization pursued** — INT8 PTQ baseline documented as-is per project scope.

### 12.6 Evidence

`results/stage_jetson_c2/`: stage_summary.md (contains full evaluation logs, size-stratified metrics, and tradeoff analysis)

### 12.7 Module Closure

Module C (INT8 Deployment Study) complete. FP16 remains production deployment target for EdgePerception v1.
