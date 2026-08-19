# Stage Jetson-5B-1 — Self-Trained NanoSAM FP16 Engine Build & Runtime Smoke

## Verdict: SELF_TRAINED_TRT_FP16_READY

## 1. Environment
- Host: nvidia-desktop (Ubuntu 22.04.5 LTS)
- JetPack/L4T: # R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: gener
- CUDA runtime: 12.6 | TensorRT: 10.3.0 (expected 10.3.0.30, drift=False)
- torch: 2.5.0a0+872d972e41.nv24.08 | torchvision: 0.20.0a0+afc54f7
- GPU: Orin cap [8, 7] | cuda_available=True

## 2. Source ONNX
- Path: `/home/nvidia/jetson-deploy-lab/models/self_distilled/epoch10/nanosam_resnet18_epoch009.onnx`
- SHA256: `aacbea870b8eca91e61fbb6325806c60a80e2f4a193f96597881e060b5c5cf15` (expected match: True)
- Opset [['', 16]] | nodes 93 | ir 8
- Contract: image [1, 3, 1024, 1024] -> image_embeddings [1, 256, 64, 64] | PASS=True

## 3. TensorRT FP16 Build
- Builder: trtexec, FP16, optimizationLevel=3, workspace:2048MiB
- Build wall time: ~21000 ms
- Engine: `/home/nvidia/jetson-deploy-lab/models/self_distilled/epoch10/nanosam_resnet18_epoch009_fp16.engine` (37136716 bytes)
- Engine SHA256: `0d63103cf91daf41bcaeebd86b4d1fc3b3bcce4c9c70dc28b1c76dd32a6d5397`
- Build passed: True | fatal errors: None

## 4. Runtime Smoke
- Deserialize: PASS | runtime: PASS
- Input `image` [1, 3, 1024, 1024] -> Output `image_embeddings` [1, 256, 64, 64]
- Output shape [1, 256, 64, 64] | finite=True | NaN=False | Inf=False

## 5. Numerical Sanity (TRT FP16 vs ORT CPU FP32)
- Reference backend: ONNXRuntime CPU FP32
- max_abs=0.090285 mean_abs=0.000386 rmse=0.001601
- relative_L2=0.017633 cosine=0.999847
- (FP16 vs FP32 — reasonable precision gap, parity not required)

## 6. Predictor Compatibility
- PASS via nanosam.utils.predictor.load_image_encoder_engine (TRTModule)
- embedding shape [1, 256, 64, 64] finite=True shape_ok=True

## 7. Self-Trained Encoder Latency (warmup30 measured100)
- mean=17.460ms P50=17.300 P95=20.915 P99=21.480

## 8. Historical Baseline Same-Protocol
- historical NanoSAM student encoder FP16 (Day3 baseline) SHA `c277e5ca445dc29b...`
- mean=15.186ms P50=14.984 P95=16.547 P99=16.858
- Note: both ~15-18ms here vs historical ~60-70ms because this measures ONLY encoder execute_async_v3 (no full pipeline overhead). Apples-to-apples within this protocol.

## 9. Engine Identity
- self-trained SHA `0d63103cf91daf41bcaeebd86b4d1fc3b3bcce4c9c70dc28b1c76dd32a6d5397`
- historical SHA `c277e5ca445dc29b06e8be41a481e74bd806c4dd82071cbfbd47fddcf7aaa6a8`
- Result: DIFFERENT_ENGINE (expected)

## 10. Baseline Protection
- Historical FP16 engine unchanged, SHA matches expected `c277e5ca445dc29b...` (True)
- Self-trained engine written to models/self_distilled/epoch10/ (distinct path)

## Out of Scope (enforced STOP)
STANDARD500, COCO mIoU, further training, INT8, NanoOWL joint, CUDA preprocessing optimization, power-mode changes — NOT performed.
