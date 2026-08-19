# NanoSAM Training Path

This document traces the full lifecycle of the self-trained NanoSAM student
model, from teacher distillation on the RTX 5090 training host to TensorRT FP16
runtime on the Jetson Orin target.

## Overview

The deployed NanoSAM image encoder historically shipped as a fixed upstream
ResNet18 TensorRT engine. The self-trained path replaces only the student
weights, keeping the encoder architecture and the downstream mask decoder
untouched, so the field-proven predictor API continues to work.

```
RTX 5090 (training host)                    Jetson Orin (deploy target)
─────────────────────────                   ──────────────────────────
MobileSAM TinyViT (teacher)                 ONNX (FP32, static 1x3x1024x1024)
        │                                          │
        ▼                                          ▼
Feature distillation                        TensorRT FP16 engine build (local)
        │                                          │
        ▼                                          ▼
NanoSAM ResNet18 student (10 epochs)        Runtime deserialize + inference
        │                                          │
        ▼                                          ▼
checkpoint epoch009                         image_embeddings [1,256,64,64]
        │
        ▼
PyTorch -> ONNX (opset 16, 93 nodes)
        │
        ▼
ORT parity check (PASS, on 5090)
        │
        ▼
ONNX transferred to Jetson (SHA256 verified)
```

## Jetson Deployment of Self-Trained Student

The ONNX artifact produced on the 5090 is verified by SHA256 on the Jetson
(`aacbea870b8e...`), audited against the input/output contract
(`image [1,3,1024,1024] FP32 -> image_embeddings [1,256,64,64] FP32`,
opset 16, 93 nodes), and rebuilt into a TensorRT FP16 engine **on the Jetson
itself**.

Stage Jetson-5B-1 established this link end-to-end:

- ONNX -> TensorRT FP16 engine via `trtexec` on the Orin target.
- Deserialize + runtime smoke: output `[1,256,64,64]`, finite, no NaN/Inf.
- Numerical sanity vs ONNX Runtime CPU FP32: cosine `0.99985`, relative L2
  `0.0176` — a reasonable FP16-vs-FP32 gap, not a 1e-7 parity requirement.
- NanoSAM `predictor.load_image_encoder_engine` loads the new engine
  unchanged (no upstream modification); embedding shape correct.
- Encoder latency smoke and apples-to-apples comparison against the
  historical baseline engine.

### Why the TensorRT engine must be built on the Jetson target

A TensorRT engine is a **plan**, not a portable artifact. During build, the
TensorRT optimizer:

1. Selects and tunes kernels for the **specific GPU architecture and SM
   capability** present on the build machine (here, Orin, SM 8.7 / DLA-capable).
   A 5090 (Blackwell, different SM, different memory subsystem) would choose
   different kernels — some of which do not even exist on Orin — so a 5090
   engine cannot deserialize on Jetson and would silently use wrong/missing
   kernels at best.
2. Embeds the **TensorRT / CUDA driver version** it was built against (TRT
   10.3.0, CUDA 12.6 on Jetson). Engines are not forward/backward compatible
   across TensorRT minor releases; the build host and deploy host must match.
3. Bakes in **device memory layout and layer fusion** decisions driven by the
   target's free memory and builder heuristics.

Rebuilding on the Jetson guarantees the engine is valid for exactly the
silicon and software stack it will run on. The 5090 is therefore used only for
training and ONNX export; the ONNX (architecture- and version-neutral) is the
interchange format, and the Jetson performs the final architecture-specific
specialization. This is also why the historical baseline engine and the
self-trained engine coexist as separate `.engine` files — neither is portable
to the other's build environment.

See `results/stage_jetson_5b1/stage_summary.md` for the build and runtime
evidence, and `docs/daily/day12.md` for the day-12 record.
