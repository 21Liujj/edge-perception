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

## Ten-Epoch Student Quality Gate

The distilled loss and the downstream segmentation quality are different
quantities and must be kept distinct:

- **Training (RTX 5090)**: epoch9 feature-distillation **mean loss ≈ 0.000535**
  (encoder embedding-space distillation, MobileSAM TinyViT teacher). A very low
  distillation loss says the student reproduces the teacher's image embedding;
  it does not by itself certify downstream mask quality.
- **Deployment (Jetson)**: the student ONNX became a Jetson FP16 TensorRT engine
  (Stage 5B-1); encoder embedding parity vs ORT FP32 was cosine 0.99985 /
  relative L2 0.01763.
- **Quality (Jetson, Stage 5B-2)**: under the frozen STANDARD500 protocol (seed
  2027, 500 instances, GT-box prompt, fixed slot0, frozen decoder), the 10-epoch
  student reached **mIoU 0.707340** vs the historical deployed baseline
  **0.718128** — an absolute delta of **−0.010788** (−1.50% relative).

Paired over the same 500 instances: 196 improved, 303 degraded, 1 equal; the
gap concentrates on small objects (small Δ −0.019, medium −0.011, large −0.002),
and the historical `small < medium < large` failure ordering is preserved.

**Gate: UNDERTRAINED_BUT_HEALTHY** (project-internal heuristic:
0.005 < |Δ| ≤ 0.02 ⟹ resume training rather than ship as-is). The small-object
regression and the 303/500 degraded majority point to incomplete convergence,
not a pipeline or FP16-conversion fault (Stage 5B-1 embedding parity was
healthy; no NaN/Inf/CUDA/TRT errors in the 500-run). A PyTorch/ORT segmentation
control would be needed to fully separate "undertrained" from "FP16 error", but
within this stage the encoder parity evidence makes undertraining the most
parsimonious explanation.

Recommendation (advisory, not executed): resume distillation to ~50 epochs and
re-validate. No training was started here.
