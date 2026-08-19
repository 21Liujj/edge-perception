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
