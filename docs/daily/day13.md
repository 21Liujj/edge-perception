# Day 13: Stage Jetson-C0 — INT8 PTQ Design Audit

**Date**: 2026-08-24  
**Stage**: Jetson-C0 (Module C INT8 PTQ)  
**Status**: PASS → INT8_PTQ_PATH_READY

## WHAT

INT8 PTQ capability audit and calibration pipeline design for 50-epoch NanoSAM student (design-only, no INT8 build).

## WHY

Module B distillation COMPLETE (STANDARD500 0.71565, FULL 0.699183). Next optimization: INT8 quantization for latency/memory reduction while maintaining quality. Must design PTQ path before building INT8 engine.

## INPUT

**Frozen artifacts**:
- ONNX: `nanosam_resnet18_epoch049.onnx` (SHA `906d8fd7...`)
- FP16 engine: `nanosam_resnet18_epoch049_fp16.engine` (SHA `99147caf...`)
- FP16 STANDARD500: 0.715650
- FP16 FULL: 0.699183
- FP16 latency: 11.08ms mean

**Environment**: Jetson Orin, TRT 10.3.0, CUDA 12.6, torch 2.5.0a0, no upgrades

## AUDIT

1. **TRT INT8 capability**: `platform_has_fast_int8 = True`, IInt8EntropyCalibrator2 available, explicit Q/DQ supported
2. **ONNX graph**: 93 nodes (25 Conv, 17 ReLU, 14 Add, 5 Erf/Div GELU, 1 ConvTranspose), no Q/DQ → implicit PTQ fits
3. **Strategy**: Selected **A - TRT implicit PTQ** (no retraining, low cost). Fallback: explicit Q/DQ if regression.

## DESIGN

**Calibration**:
- Source: COCO val2017 (5000 images, already on device)
- Sample: 512 images, seed 20260824, deterministic random
- Preprocessing: `nanosam.utils.predictor.preprocess_image` (MUST match deployment)
- Cache: `models/self_distilled/epoch50/int8/calibration.cache` + manifest

**Quality gate** (STANDARD500 first):
- delta ≥ -0.005: ACCEPTABLE
- -0.015 < delta < -0.005: DEGRADED_BUT_USABLE  
- delta < -0.015: REGRESSION

**Performance gate**: Same-protocol FP16 vs INT8 (latency, engine size, GPU mem, RSS)

**Mixed precision fallback**: Full INT8 → profile → selective FP16 (ConvTranspose, GELU cluster)

## EVIDENCE

`results/stage_jetson_c0/`: environment, source_artifact_audit, trt_int8_capability, onnx_operator_audit, quantization_strategy, calibration_design, preprocessing_freeze, calibration_cache_design, fp16_baseline_freeze, quality_gate_design, performance_gate_design, mixed_precision_plan, stage_summary.md

`docs/int8_ptq.md`: Quantization fundamentals + NanoSAM PTQ engineering path

`scripts/stage_jetson_c0.py`: 496L audit script

## NEXT (NOT EXECUTED)

Stage Jetson-C1: Build 512-image calibration dataset → IInt8EntropyCalibrator2 cache → first INT8 engine → STANDARD500 quality gate

## DECISION

**INT8_PTQ_PATH_READY**  
**Path**: TensorRT implicit PTQ, IInt8EntropyCalibrator2, 512 val2017, seed 20260824  
**Risk**: GELU/ConvTranspose may regress; quality gate + fallback designed


---

## Stage Jetson-C1A: INT8 Calibration Cache (BLOCKED)

**Date**: 2026-08-24 (same day as C0)  
**Status**: BLOCKED (TRAIN2017_NOT_LOCAL)

### WHAT

Attempted to build INT8 calibration cache using COCO train2017. Discovered train2017 is not available on Jetson device.

### CALIBRATION SOURCE CORRECTION

**C0 design**: Proposed val2017 for calibration (already on device)  
**C1A correction**: Requires train2017 for strict calibration/evaluation isolation

**Rationale**: STANDARD500 and FULL eval both use val2017. Even though calibration does not see GT, using val2017 for calibration creates image-level overlap with evaluation set (evaluation set leakage). Conservative practice: tune on train, validate on val.

**Final protocol**:
- train2017 → distillation training + INT8 calibration
- val2017 → STANDARD500 + FULL evaluation

### BLOCKER

**train2017**: NOT FOUND on Jetson  
**Available**: val2017 only (5000 images, reserved for evaluation)

### TRANSFER PLAN

**Recommended**: Option A — Transfer 512-image calibration subset from AutoDL 5090

Steps:
1. On 5090: Sample 512 train2017 images (seed 20260824, deterministic random)
2. Create tar: train2017_calib512.tar.gz (~50-100 MB)
3. Transfer: scp 5090 → Jetson (~5-10 min)
4. Extract to: data/coco/train2017_calib512/
5. Resume C1A

Alternative: Transfer full train2017 (19 GB, 118k images) — more storage/time, but enables future experiments.

### EVIDENCE

results/stage_jetson_c1a/:
- source_dataset_audit.json (val2017 found, train2017 not found)
- transfer_plan.json (3 options with recommendations)
- stage_summary.md (blocked status, unblock requirements)

### DOCUMENTATION

docs/int8_ptq.md: Added § 9 CALIBRATION_SOURCE_CORRECTION (C0 historical design + C1A correction rationale + transfer plan)

### NEXT

Awaiting user decision: Execute transfer plan Option A, B, or C. Once train2017 available, resume C1A to:
1. Generate deterministic 512-image manifest
2. Implement IInt8EntropyCalibrator2
3. Run calibration with deployment preprocessing
4. Generate calibration_train2017_512.cache
5. Verify cache reuse

### STOP

C1A cannot proceed without train2017. No commit (no functional work completed).


---

## C1A-R CONTINUATION (train2017 transfer complete)

Transfer executed: AutoDL 5090 → local → Jetson (512 images, 80 MB bundle, seed 20260824)

### CALIBRATOR IMPLEMENTATION

scripts/int8/simple_calibrator.py: SimpleCalibrator(trt.IInt8EntropyCalibrator2)
- Batch size: 1
- Preprocessing: nanosam.utils.predictor.preprocess_image(size=1024)
- Device memory: torch CUDA tensor

scripts/int8/run_calibration.py: TensorRT builder with INT8+FP16 flags, calibrator assignment

### CALIBRATION RUN 1 (cache generation)

512 batches calibrated, 473.883s calibration + 16.340s post-processing
Cache generated: models/self_distilled/epoch50/int8/calibration_train2017_512.cache
SHA256: 9689497ac73606952abfc113162907655ce80f79642e77ffd2a33daa22ac1487
Size: 5062 bytes
Exit code: 0

### CALIBRATION RUN 2 (cache reuse verification)

Log: "Reading Calibration Cache for calibrator: EntropyCalibration2"
Zero batches reprocessed
Cache SHA256 unchanged
Wall clock: ~3s (vs 473.9s for generation)
Cache reuse: PASS

### EVIDENCE

results/stage_jetson_c1a/:
- c1a_evidence.json (full provenance + metrics)
- calibration_train2017_512.cache (copy of generated cache)
- calibration_run1_generate.log.gz (TensorRT build log)
- checksums.txt (SHA256 of all artifacts)
- day13_stage_report.md

### DOCUMENTATION

docs/int8_ptq.md: Added § 10 Stage Jetson-C1A-R (COMPLETE) — blocker resolution, calibration manifest, calibrator implementation, cache generation + reuse verification, evidence

### RESULT

Stage Jetson-C1A-R: PASS
INT8 calibration cache ready for formal engine build (C1B).

## STAGE JETSON-C1B: FORMAL INT8 ENGINE BUILD & AUDIT

### DEPLOYMENT ENGINE

Built with existing train2017-512 calibration cache. Cache reuse confirmed
(TensorRT log: "Generated calibration scales using calibration cache", zero
images reprocessed).

Path: models/self_distilled/epoch50/int8/nanosam_resnet18_epoch049_int8.engine
Size: 22,249,660 bytes (21.22MB)
SHA256: d8dd7a2f61427e072baeaf06bb38e78d2dab79c95f41c3839883d5891497d650
Build time: 284.03s
Flags: INT8 + FP16 (mixed precision fallback)

### AUDIT ENGINE (analysis only, DETAILED profiling)

Path: models/self_distilled/epoch50/int8/nanosam_resnet18_epoch049_int8_detailed.engine
Size: 17,626,436 bytes (16.81MB)
SHA256: f3afcb3d93d00401e4cff8e321f746126f25ce07c788c90afa833d9c51f8136e
Build time: 234.55s

Root cause investigated: default ProfilingVerbosity only exposes layer names,
not precision. Rebuilt with ProfilingVerbosity.DETAILED (same ONNX + same
cache, only addition) to get structured per-layer JSON with Weights/Datatype/
Tactic fields needed to infer actual precision.

### PRECISION AUDIT RESULT

33 layers total: 30 INT8 (90.9%), 2 FP16 (6.1%, pos_embedding add), 1 FP32
(3.0%, output reformat). Visibility: FULL.

### PREDICTOR COMPATIBILITY

torch2trt import failure root-caused to missing venv activation (not an
engine incompatibility) — torch2trt egg present at
.venv/lib/python3.10/site-packages/torch2trt-0.5.0-py3.10.egg, resolved once
venv properly activated. Loaded via upstream nanosam.utils.predictor.Predictor
unmodified: PASS. Embedding shape [1,256,64,64], finite.

### SAME-PROTOCOL LATENCY (warmup=30, measured=200, FP16→INT8→FP16)

FP16 run1: mean 11.106ms, P50 11.043ms, P95 11.624ms, P99 11.773ms
INT8:      mean  6.020ms, P50  5.984ms, P95  6.150ms, P99  6.164ms
FP16 run2: mean 10.790ms, P50 10.754ms, P95 10.904ms, P99 10.917ms

Speedup: mean 1.845x, P50 1.845x
Engine size: FP16 36.29MB -> INT8 21.22MB (-41.52%)

### EVIDENCE

results/stage_jetson_c1b/: artifact_provenance.json, cache_reuse.json,
environment.json, predictor_compatibility.json, runtime_smoke.json,
resource_comparison.json, precision_layer_audit.json, latency_fp16.json,
latency_int8.json, performance_comparison.json, engine_size_comparison.json,
trt_build_summary.json, trt_build_summary_detailed.json, stage_summary.md

### DOCUMENTATION

docs/int8_ptq.md: Added Section 11 Stage Jetson-C1B (COMPLETE)

### RESULT

Stage Jetson-C1B: INT8_DEPLOYMENT_ENGINE_READY
No quality claim made at this stage. Quality gate deferred to Stage C2.

## Stage Jetson-C2: Quality Gate Evaluation

**Status**: COMPLETE  
**Result**: QUALITY_GATE_FAIL

### Evaluation Protocol

- **Dataset**: COCO val2017  
- **Subsets**: STANDARD500 (500 images), FULL (5000 images)  
- **Prompt**: GT box → NanoSAM single-box prompt (slot 0)  
- **Metric**: Instance-level mean IoU  
- **Baseline**: FP16 TensorRT (epoch49 ONNX)

### Results

**STANDARD500**:
- FP16: 0.715650  
- INT8: 0.700264  
- Δ: -2.15%

**FULL COCO**:
- FP16: 0.699183  
- INT8: 0.681910  
- Δ: **-2.47%**

**Quality gate**: FAIL (threshold <0.5%)

### Size-Stratified Analysis (FULL COCO)

| Size | FP16 | INT8 | Δ (%) |
|---|---|---|---|
| Small (<32²) | 0.6475 | 0.6257 | **-3.37%** |
| Medium | 0.7601 | 0.7461 | -1.84% |
| Large (≥96²) | 0.8145 | 0.8034 | -1.36% |

Small objects most affected — precision loss amplified at fine spatial scales.

### Root Cause

- Layer audit: 90.9% INT8, but `pos_embedding` fell back to FP16  
- Hypothesis: INT8 quantization noise accumulated in backbone → positional encoding mismatch → spatial localization degradation

### Engineering Conclusion

**Tradeoff achieved**:
- ✅ Performance: **1.845x speedup** (11.1ms → 6.0ms)  
- ✅ Compression: **41.5% reduction** (36.3MB → 21.2MB)  
- ❌ Quality cost: **-2.47% mIoU**

**Assessment**:
- Acceptable for latency-critical apps (real-time robotics, edge video)  
- NOT acceptable for high-precision scenarios (medical, fine-grained recognition)

**No further optimization** — INT8 PTQ baseline documented as-is.

### Evidence

- `results/stage_jetson_c2/stage_summary.md`  
- Updated `docs/int8_ptq.md` Section 12

### Module Closure

Module C (INT8 Deployment Study) complete. FP16 remains production target for EdgePerception v1.

---

## Day 13 Summary

**Time**: 2026-08-24  
**Focus**: Module C (INT8 Deployment Study) — Stages C1A, C1B, C2

### Key Achievements

1. **C1A**: INT8 calibration cache generation (512 train2017 images, 473.9s)  
2. **C1B**: Formal INT8 engine build + precision audit (90.9% INT8, 1.845x speedup)  
3. **C2**: Quality evaluation on STANDARD500 + FULL COCO (-2.47% mIoU, quality gate FAIL)

### Technical Highlights

- TensorRT 10.3 IInt8EntropyCalibrator2 with cache reuse  
- ProfilingVerbosity.DETAILED for full precision visibility  
- Same-protocol latency benchmarking (FP16→INT8→FP16)  
- Size-stratified COCO analysis revealing small-object vulnerability

### Engineering Outcome

INT8 PTQ achieves **1.845x speedup** and **41.5% compression** at cost of **2.5% quality loss** — acceptable for latency-critical applications, documented as baseline tradeoff.

**Module C closed. No further INT8 optimization per project scope.**

### Next Phase

EdgePerception v2: NanoSAM + NanoOWL multi-modal edge perception pipeline.
