# Day 12 — Jetson Deployment of Self-Trained NanoSAM Student (Stage Jetson-5B-1)

## WHAT

Built and smoke-tested the self-trained NanoSAM ResNet18 student image encoder
as a TensorRT FP16 engine on the Jetson Orin Nano target, from the ONNX
artifact trained on the RTX 5090.

## WHY

Close the loop from 5090 training to Jetson inference: prove the self-trained
student ONNX is deployable through the project's verified NanoSAM build/runtime
path without modifying upstream, and obtain a loadable, finite, numerically
sane encoder engine distinct from the historical baseline.

## INPUT

- ONNX: `models/self_distilled/epoch10/nanosam_resnet18_epoch009.onnx`
- ONNX SHA256: `aacbea870b8eca91e61fbb6325806c60a80e2f4a193f96597881e060b5c5cf15` (verified)
- Contract: `image [1,3,1024,1024] FP32` -> `image_embeddings [1,256,64,64] FP32`, opset 16, 93 nodes
- Historical baseline (protected, untouched): `third_party/nanosam/data/resnet18_image_encoder.engine`
  SHA256 `c277e5ca445dc29b06e8be41a481e74bd806c4dd82071cbfbd47fddcf7aaa6a8`

## BUILD

- Builder: `trtexec`, FP16, `--builderOptimizationLevel=3`, `--memPoolSize=workspace:2048MiB`, `--skipInference`
- Built on Jetson Orin (TRT 10.3.0, CUDA 12.6, SM 8.7)
- Wall time ~21000 ms
- Engine: `models/self_distilled/epoch10/nanosam_resnet18_epoch009_fp16.engine` (37136716 bytes)
- Engine SHA256: `0d63103cf91daf41bcaeebd86b4d1fc3b3bcce4c9c70dc28b1c76dd32a6d5397`
- Build log PASSED; no fatal parser/build errors

## RUNTIME

- Deserialize PASS; input `image [1,3,1024,1024]` -> output `image_embeddings [1,256,64,64]`
- Output finite, no NaN, no Inf (30 measured runs with seed=20260819)

## LATENCY

Self-trained encoder FP16 (warmup30, measured100):
mean 17.46ms | P50 17.30ms | P95 20.92ms | P99 21.48ms

Historical baseline FP16 same protocol (warmup30, measured100):
mean 15.19ms | P50 14.98ms | P95 16.55ms | P99 16.86ms

Note: these numbers (encoder `execute_async_v3` only) are far below the ~60-70ms
historical figures, which included full pipeline overhead; the comparison is only
valid within this identical script, not against prior protocols.

## NUMERICAL SANITY

TRT FP16 vs ONNX Runtime CPU FP32 (same deterministic input):
max_abs 0.0903 | mean_abs 0.000386 | rmse 0.00160 | relative_L2 0.01763 | cosine 0.99985

## PREDICTOR COMPATIBILITY

`nanosam.utils.predictor.load_image_encoder_engine` loads the new engine
unchanged; embedding `[1,256,64,64]`, finite.

## ARTIFACT

- New self-trained engine SHA256 `0d63103c...` != historical `c277e5ca...` -> DIFFERENT_ENGINE (expected)
- Historical baseline engine untouched, SHA still matches expected
- Evidence: `results/stage_jetson_5b1/` (environment, source_onnx_audit,
  baseline_engine_audit, trt_build_summary, runtime_smoke, numerical_sanity,
  predictor_compatibility, latency_smoke, latency_baseline_same_protocol,
  engine_identity, artifact_manifest, stage_summary)

## NEXT

STOP. No STANDARD500, COCO mIoU, further training, INT8, NanoOWL joint, CUDA
preprocessing optimization, or power-mode changes. Await authorization for
Stage Jetson-5B-2.

## Stage Jetson-5B-2 — Self-Trained STANDARD500 Quality Gate

### WHAT
Ran the frozen STANDARD500 segmentation quality gate on the 10-epoch
self-trained NanoSAM student encoder (Jetson FP16 engine) and compared
apples-to-apples against the historical deployed baseline.

### WHY
Answer the core question: does the 10-epoch distilled student match the deployed
encoder's segmentation quality, or is it undertrained? Separate distillation
loss (0.000535) from downstream mask quality.

### FROZEN PROTOCOL
- Manifest `data/eval_manifests/standard_manifest.json` SHA `19a00a83…78f015`
- 500 instances / 455 images / seed 2027 / fixed slot0 / GT-box prompt
- Evaluator `scripts/eval_nanosam_coco.py` (prompt-conditioned instance IoU)
- Decoder `mobile_sam_mask_decoder.engine` SHA `afb5ac49…ccb4f2` unchanged
- Historical baseline engine SHA `c277e5ca…a6a8` untouched
- Self-trained engine SHA `ab2d9962…0a140` (reconciled in 5B-1; DIFFERENT engine)

### SELF-TRAINED RESULT
- mIoU **0.707340** | median 0.776695 | P10 0.370367 | P90 0.917869
- IoU>=0.50 0.836 | IoU>=0.75 0.552 | valid 500 failed 0

### BASELINE
- Historical mIoU **0.718128** (source: HISTORICAL, no rerun; paired per-instance available)

### PAIRED ANALYSIS
- improved 196 / degraded 303 / equal 1 ; mean paired delta −0.010788

### SIZE BREAKDOWN (self / baseline / delta)
- small n=175: 0.586296 vs 0.605320 → −0.019024
- medium n=166: 0.751911 vs 0.762659 → −0.010748
- large n=159: 0.794032 vs 0.795796 → −0.001764
- small<medium<large ordering preserved.

### QUALITY GATE
- **UNDERTRAINED_BUT_HEALTHY** (|Δ|=0.0108, within 0.005–0.02 band)
- Pipeline healthy (no NaN/Inf/CUDA/TRT errors); gap concentrated on small objects.

### NEXT DECISION
Recommendation (advisory, NOT executed): **RESUME TO 50 epochs**, then re-validate
STANDARD500. STOP — no training started; awaiting authorization.

## Stage Jetson-5B-3 — FP32 vs FP16 Downstream Quality Control

### WHAT
Swapped only the image-encoder backend (TRT FP16 → ORT CPU FP32) for the same
self-trained 10-epoch student and re-ran the frozen STANDARD500, to isolate the
FP16 deployment-precision contribution to the Stage 5B-2 regression.

### WHY
Stage 5B-1 encoder-embedding parity (cosine 0.99985) does not prove downstream
mask-quality parity; determine whether the −0.010788 regression is training or
FP16.

### FROZEN PROTOCOL
- Same manifest/decoder/slot0/prompt/IoU as Stages 5B-1/5B-2; no upstream edit, no rebuild.
- ONNX `aacbea87…5cf15`; TRT engine `ab2d9962…0a140` (frozen); decoder `afb5ac49…ccb4f2`.
- Encoder backend: ORT CPUExecutionProvider FP32 (intra=8 threads); embedding moved .cuda() for the TRT decoder.

### RESULT — ORT FP32 STANDARD500
- mIoU **0.707379** | median 0.776226 | P10 0.370908 | P90 0.919480 | ≥0.5 0.834 | ≥0.75 0.552 | 500/500 valid
- small 0.586319 / medium 0.751945 / large 0.794094 | wall 578s (CPU ORT, quality control not perf)

### FP16 vs FP32
- FP16 mIoU 0.707340 − FP32 mIoU 0.707379 = **+0.000039** (FP16 negligibly lower)
- Paired: 176 higher / 176 lower / 148 equal; median 0; P95 abs 0.001928
- Size FP16−FP32: small −0.000023 / med −0.000034 / large −0.000062 (all ~0)

### QUALITY DECOMPOSITION
- total gap (baseline − FP16) = +0.010788
- FP16 precision gap (FP32 − FP16) = +0.000039 (0.36% of total)
- training/model gap (baseline − FP32) = +0.010749

### CONCLUSION
- FP16_QUALITY_IMPACT = **NEGLIGIBLE**; main conclusion = **TRAINING_GAP_DOMINANT**
- The small-object regression is training, not FP16.

### NEXT DECISION
Recommendation (advisory, NOT executed): **RESUME_TO_50**. FP16 path is sound;
effort should go to distillation, not precision. STOP.

## Stage Jetson-6C-1 — 50-Epoch Student FP16 Engine Build

### WHAT
Built the 50-epoch self-distilled NanoSAM ResNet18 student image encoder as a
TensorRT FP16 engine on Jetson Orin, from the ONNX produced by Stage 5090-6B.
This mirrors Stage 5B-1's protocol but on the epoch049 (50ep) artifact.

### INPUT
- ONNX: `models/self_distilled/epoch50/nanosam_resnet18_epoch049.onnx`
- ONNX SHA256: `906d8fd752aa2f4dc28d353adb20a462387c9414c4ddf01c1fca93ddf3f31dd7` (verified)
- Contract: `image [1,3,1024,1024] FP32` -> `image_embeddings [1,256,64,64] FP32`, opset 16, 93 nodes (same graph as 10ep)

### BUILD (reused Stage5B-1 protocol)
- Builder: `trtexec`, FP16, `--builderOptimizationLevel=3`, `--memPoolSize=workspace:2048MiB`, `--skipInference`
- Wall time ~145 s
- Engine: `models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine` (38,049,788 B)
- Engine SHA256: `99147caf00f0174b802848c93928232c02ac996e91e6660372bbc8923112baf4`
- Build protection: single builder, unique `.tmp` path, atomic rename to final only after all checks PASS (Stage5B-1 concurrent-trtexec race lesson applied). No overwrite of epoch10 or historical engine.
- Benign warning: `Some tactics do not have sufficient workspace memory` (same as 5B-1).

### RUNTIME / PREDICTOR
- Deserialize PASS; `image [1,3,1024,1024]` -> `image_embeddings [1,256,64,64]`; finite, no NaN/Inf.
- `nanosam.utils.predictor.load_image_encoder_engine` loads it unchanged; embedding [1,256,64,64] finite, upstream unmodified.

### LATENCY (encoder execute_async_v3 only; warmup30 measured100)
mean 11.077ms | P50 10.862 | P95 11.480 | P99 11.546 (NOT full segmentation / NOT system FPS).

### IDENTITY
| engine | sha256 (prefix) | size |
|---|---|---|
| 50ep (new) | `99147caf…` | 38,049,788 B |
| 10ep | `ab2d9962…` | 37,060,492 B |
| historical deployed | `c277e5ca…` | 34,766,700 B |
All DIFFERENT (expected). Historical and 10ep engines untouched.

### NEXT
STOP. No STANDARD500 / FULL COCO / INT8 / NanoOWL / C++ / training.
Stage Jetson-6C-2 ready: YES — frozen STANDARD500 mIoU of the 50ep FP16 engine (same 500 manifest as 10ep/baseline), to enable the 10ep / 50ep / historical三方 comparison.
