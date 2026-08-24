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
