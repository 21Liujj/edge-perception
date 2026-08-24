# Stage Jetson-C0 — INT8 PTQ Capability & Calibration Design Audit

## Final: INT8_PTQ_PATH_READY

## 1. Environment (frozen)
- Jetson Orin, L4T R36.4.3, CUDA 12.6, **TensorRT 10.3.0** (python + trtexec), torch 2.5.0a0+nv24.08, python 3.10.12
- GPU sm_87 Ampere, `platform_has_fast_int8 = True` (hardware INT8 fast path: DP4A / IMMA)
- No dependency upgrade performed.

## 2. Source Artifact Freeze (Module C baseline)
- ONNX `models/self_distilled/epoch50/nanosam_resnet18_epoch049.onnx` SHA `906d8fd752aa…` — match (True)
- FP16 engine `models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine` SHA `99147caf00f0…` — match (True)
- Both are the frozen INT8 source + FP16 quality/perf baseline.

## 3. Conceptual Quantization Contract
See `docs/int8_ptq.md` § "Quantization basics" for the full conceptual write-up (FP32/FP16 vs INT8, scale, weight vs activation, calibration, dynamic range, quantization error, dequantization, mixed precision, PTQ vs QAT) — bound to this NanoSAM project, not generic theory.

## 4. TensorRT INT8 Capability (live API, not docs)
- `platform_has_fast_int8 = True`
- Implicit PTQ: `config.set_flag(trt.BuilderFlag.INT8)` + `config.set_int8_calibrator(IInt8EntropyCalibrator2)`
- Explicit Q/DQ: `add_quantize`/`add_dequantize` + `STRONGLY_TYPED` network
- Calibrators available: IInt8EntropyCalibrator2 / MinMax / Legacy / base IInt8Calibrator
- trtexec: `--int8 --fp16 --calib=<cache> --builderOptimizationLevel=3 --memPoolSize=workspace:2048MiB --skipInference`

## 5. ONNX Operator Audit (93 nodes)
| op | count | INT8 likelihood |
|---|---|---|
| Conv | 25 | INT8 (primary target) |
| Relu | 17 | INT8 |
| Add | 14 | INT8 (residual — watch scales) |
| Constant | 15 | folded |
| Mul | 10 | INT8 or FP16 fallback |
| Div | 5 | FP16 fallback (GELU) |
| Erf | 5 | FP16 fallback / fused (GELU) |
| MaxPool | 1 | INT8 |
| ConvTranspose | 1 | INT8 or FP16 fallback (uncertain) |

- Graph is plain FP32 (no Q/DQ) → **implicit-quantization PTQ**.
- Sensitive tail: ConvTranspose output proj + Erf/Div/Mul GELU (5 each) + 14 residual Adds.
- "Likelihood" is theoretical; actual per-op precision confirmed only in C1 build profile.

## 6. Selected Path: A — TRT implicit-calibration PTQ
- No retraining (STOP_LONG_TRAINING honored), low cost, reuses 5B-1/6C-1 trtexec protocol.
- Fallback if regression: B explicit Q-DQ (per-channel + keep GELU/ConvTranspose FP16), then mixed-precision.

## 7-8. Calibration Dataset
- Source: COCO val2017 (5000 images, already on device). train2017 NOT present (would need user to pull 19GB; marginal benefit).
- Principle: image-level encoder-input stats only; no masks/boxes; never touch STANDARD500 GT set (no leakage).
- Seed 20260824, deterministic random, no stratification (calibration ≠ category metrics).
- **Primary size: 512** (coverage vs wall-time; bump to 1024 if C1 borderline).

## 9. Preprocessing Freeze
- MUST reuse `nanosam.utils.predictor.preprocess_image(image, size=1024)` — the exact deployment/eval path.
- RGB → aspect-resize long→1024 → CHW float → ImageNet mean/std → zero-pad [1,3,1024,1024] → cuda fp32.
- C1 imports the function directly (no reimplementation).

## 10. Calibration Cache
- `models/self_distilled/epoch50/int8/calibration.cache` + `calibration_manifest.json`
- Bound to ONNX SHA + preprocess version + manifest SHA + TRT 10.3; reuse-guard checked before any reuse.

## 11-12. FP16 Baselines
- STANDARD500 0.71565 | FULL 0.699183 (historical FULL 0.703201)
- INT8 runs STANDARD500 FIRST; FULL only if gate passes.
- FP16 latency (6C-1, encoder-only): mean 11.08ms / P50 10.86 / P95 11.48 / P99 11.55.
- Formal INT8 benchmark MUST re-measure FP16+INT8 same-session same-script.

## 13. Quality Gate (heuristic)
- delta = INT8_STD500 − FP16_STD500 (FP16 = 0.71565)
- ≥ −0.005 ACCEPTABLE | (−0.015, −0.005) DEGRADED_BUT_USABLE | < −0.015 REGRESSION

## 14. Performance Gate
- Same-protocol FP16 vs INT8: latency mean/P50/P95/P99, engine size, GPU alloc, RSS. No power-mode change.

## 15. Mixed Precision Plan (design only, not built)
- full INT8 → identify sensitive ops (ConvTranspose, GELU cluster, residual Adds) → keep those FP16 → re-gate.

## Evidence
`results/stage_jetson_c0/`: environment, source_artifact_audit, trt_int8_capability,
onnx_operator_audit, quantization_strategy, calibration_design, preprocessing_freeze,
calibration_cache_design, fp16_baseline_freeze, quality_gate_design,
performance_gate_design, mixed_precision_plan, this stage_summary.md.

## Out of scope (STOP — none done)
INT8 engine build, STANDARD500/FULL INT8 eval, QAT, model/decoder change, continued
distillation, NanoOWL, C++, power-mode/jetson_clocks.

## Next (NOT executed)
Stage Jetson-C1: build calibration dataset (512 val2017 images, seed 20260824) →
IInt8EntropyCalibrator2 cache → first INT8 engine build (trtexec --int8 --fp16 --calib)
→ STANDARD500 quality gate vs FP16 0.71565.
