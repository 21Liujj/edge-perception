# Stage Jetson-6C-1 — 50-Epoch NanoSAM Student ONNX → TensorRT FP16

## Verdict: SELF_TRAINED_50EP_TRT_FP16_READY

## 1. Environment (frozen, matches Stage5B-1)
- Host: nvidia-desktop (Jetson Orin, L4T R36.4.3)
- Python 3.10.12 | torch 2.5.0a0+872d972e41.nv24.08 | TensorRT 10.3.0
- nanosam commit `653633614b2eb93b06ba3be9adb2aeffb117bd72` (unchanged)
- No power-mode change, no jetson_clocks, no dependency upgrade, no third_party edit.

## 2. Source ONNX (50ep)
- Path: `models/self_distilled/epoch50/nanosam_resnet18_epoch049.onnx`
- SHA256: `906d8fd752aa2f4dc28d353adb20a462387c9414c4ddf01c1fca93ddf3f31dd7` — match: True
- Opset 16 | IR 8 | producer pytorch | 93 nodes
- Contract: `image` [1,3,1024,1024] (fp32) → `image_embeddings` [1,256,64,64] (fp32) — PASS

## 3. TensorRT FP16 Build
- Builder: trtexec (`/usr/src/tensorrt/bin/trtexec`), FP16, builderOptimizationLevel=3, workspace:2048MiB, --skipInference
- Build wall time: 145.3 s (trtexec reported; reused across reruns, see build.log)
- Output: `models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine` (38,049,788 B)
- Engine SHA256: `99147caf00f0174b802848c93928232c02ac996e91e6660372bbc8923112baf4`
- Build protection: single builder, unique `.tmp` output path, atomic rename to final only after deserialize+runtime+predictor+latency all PASS. (Stage5B-1 concurrent-trtexec race lesson applied.) No overwrite of epoch10 or historical engine.
- Warnings: `Some tactics do not have sufficient workspace memory` (benign, same as Stage5B-1).

## 4. Deserialize / Runtime Smoke
- deserialize: PASS (engine loaded via `trt.Runtime`)
- `image` [1,3,1024,1024] fp32 → `image_embeddings` [1,256,64,64] fp32
- Output finite=True | NaN=False | Inf=False

## 5. Predictor Compatibility
- Loaded via `nanosam.utils.predictor.load_image_encoder_engine` (TRTModule) — PASS, upstream unmodified
- Embedding shape [1,256,64,64], finite=True, shape_ok=True

## 6. Latency Smoke (image-encoder TRT execute_async_v3 only)
- warmup=30, measured=100
- mean **11.077 ms** | P50 10.862 | P95 11.480 | P99 11.546
- This is encoder execute latency, NOT full segmentation, NOT system FPS. (For reference, Stage5B-1 10ep engine on same protocol: mean 17.460 ms; historical baseline 15.186 ms — encoder-only, see 5B-1 note.)

## 7. Artifact Identity (all DIFFERENT as expected)
| engine | sha256 (prefix) | size |
|---|---|---|
| 50ep (new) | `99147caf…` | 38,049,788 B |
| 10ep | `ab2d9962…` | 37,060,492 B |
| historical deployed | `c277e5ca…` | 34,766,700 B |

- 50ep vs 10ep: DIFFERENT | 50ep vs historical: DIFFERENT

## 8. Baseline Protection
- Historical `third_party/nanosam/data/resnet18_image_encoder.engine` unchanged (sha `c277e5ca…`).
- 10ep engine `models/self_distilled/epoch10/...` unchanged (sha `ab2d9962…`).
- New 50ep engine written to distinct path `models/self_distilled/epoch50/`.

## Out of Scope (enforced STOP)
STANDARD500, FULL COCO, INT8, NanoOWL, C++, training — NOT performed.

## Next
Stage Jetson-6C-2 ready: YES (frozen STANDARD500 mIoU eval of the 50ep FP16 engine, same 500 manifest as 10ep/baseline).
