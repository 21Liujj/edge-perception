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

## FP32 vs FP16 Downstream Quality Control

Encoder numerical parity is **not** a substitute for downstream segmentation
quality parity.

- **Stage 5B-1** measured encoder-embedding numerical parity: TRT FP16 vs ORT
  FP32 on a deterministic input gave relative L2 `0.01763` and cosine `0.99985`.
  That is an *embedding-space* check; it does not certify mask quality, because
  the mask decoder is nonlinear and amplifies or dampens embedding errors
  unevenly.
- **Stage 5B-3** measured the quantity that actually matters: downstream
  STANDARD500 mask mIoU with the same 10-epoch student, swapping **only** the
  image encoder backend (TRT FP16 → ORT CPU FP32) while keeping the frozen
  decoder, fixed slot0, GT-box prompt, manifest, and IoU code byte-identical.

Result:

| backend | mIoU |
|---|---|
| self-trained ORT FP32 (CPU) | `0.707379` |
| self-trained TRT FP16 | `0.707340` |
| **FP16 downstream delta (FP16 − FP32)** | **+0.000039** |

Paired over the same 500 instances: 176 FP16-higher / 176 FP16-lower / 148
equal; median delta `0.0`; P95 absolute delta `0.001928`.

Quality decomposition (engineering approximation; nonlinear decoder ⇒ not
strictly additive, additive residual = 0.0):

- total gap (baseline − FP16) = `+0.010788`
- FP16 precision gap (FP32 − FP16) = `+0.000039` (**0.36%** of the total gap)
- training/model gap (baseline − FP32) = `+0.010749`

**FP16_QUALITY_IMPACT = NEGLIGIBLE** (|delta| ≤ 0.002). The regression vs the
deployed baseline — including the small-object gap — is dominated by the
training/model gap, **not** by FP16 conversion. Therefore the Stage 5B-2
`UNDERTRAINED_BUT_HEALTHY` diagnosis holds, and the small-object regression is
a training symptom rather than a deployment-precision symptom: reducing the
FP16 deployment path would recover essentially nothing (~0.00004 mIoU);
resuming distillation is the lever.

## Fifty-Epoch Student Jetson FP16 Engine (Stage Jetson-6C-1)

After Stage 5090-6B exported the 50-epoch student ONNX (checkpoint epoch049,
sha `906d8fd7…`) and transferred it to Jetson, Stage Jetson-6C-1 built the
deployment TensorRT FP16 engine — the Jetson-side mirror of Stage 5B-1 (10ep).

- ONNX contract identical to 10ep: `image [1,3,1024,1024] FP32` -> `image_embeddings [1,256,64,64] FP32`, opset 16, 93 nodes.
- Engine: `models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine`, SHA256 `99147caf00f0174b802848c93928232c02ac996e91e6660372bbc8923112baf4`, 38,049,788 B.
- Built with the frozen Stage5B-1 protocol (trtexec FP16, builderOptimizationLevel=3, workspace:2048MiB, --skipInference), single builder, unique `.tmp` -> atomic rename (5B-1 race lesson applied).
- Deserialize + runtime smoke PASS (finite, no NaN/Inf); predictor compat PASS via `load_image_encoder_engine` (upstream unmodified).
- Engine identity: 50ep `99147caf…` != 10ep `ab2d9962…` != historical `c277e5ca…` (all DIFFERENT, expected).
- Latency (encoder execute only, warmup30 measured100): mean 11.08ms / P50 10.86 / P95 11.48 / P99 11.55.

**Out of scope (STOP):** STANDARD500, FULL COCO, INT8, NanoOWL, C++, training.
**Next (Stage Jetson-6C-2, ready):** frozen STANDARD500 mIoU of the 50ep FP16 engine on the same 500 manifest used by 10ep/historical, enabling the 10ep / 50ep / historical 三方 comparison (10ep TRT FP16 0.707340 / historical 0.718128 / 50ep pending).

## Fifty-Epoch Downstream Quality (Stage Jetson-6C-2)

Stage Jetson-6C-2 ran the frozen STANDARD500 gate on the 50ep FP16 engine and
produced the 10ep / 50ep / historical three-way comparison.

| model | loss | STANDARD500 mIoU | train wall |
|---|---|---|---|
| 10ep self-trained | 0.0005354 | 0.707340 | 4h41m (16876s) |
| **50ep self-trained** | **0.0004241** | **0.715650** | 23h53m (86123s) total; +19h12m for epoch10→49 |
| historical deployed | — | 0.718128 | — |

- 50ep − 10ep = **+0.008310** (relative +1.18%); remaining gap to historical = **+0.002478**.
- Additional 40 epochs: +0.0021 mIoU / 10 epochs, +0.00043 mIoU / hour (diminishing slope).
- Size: extra 40 epochs mainly lifted **small** (+0.0176, now ~historical) and medium (+0.0100); large −0.0094.
- Paired 10→50: 349 improved / 137 degraded / 14 equal, mean +0.0083.
- Quality gate: **NEAR_PARITY**; recommendation **STOP_LONG_TRAINING** (advisory, not executed).

Feature-space loss keeps dropping (−20.8% over epoch10→49) and still converts to
task-space gain, but the slope is shallow; 50ep is within 0.0025 of the deployed
baseline. Per the project heuristic, long training to 100/200ep is not justified.

**Out of scope (STOP):** FULL COCO, 100/200 epoch, INT8, NanoOWL, C++, rebuild.

## Fifty-Epoch FULL COCO Validation (Stage Jetson-6D)

Stage Jetson-6D validated the 50ep FP16 engine on the **full COCO val2017**
(36,335 legal non-crowd instances, 4,952 unique images) using the frozen B3
FULL evaluator — `eval_nanosam_full_b3.py` `evaluate()` + `analyze()` called
directly, bypassing B3's `main()` standard-gate (STD_MIOU check, would exit 10
with the 50ep encoder by design). Only the encoder was swapped; decoder
`afb5ac49…`, manifest `ad8843c7…`, preprocess, GT-box prompt [2,3], fixed slot0,
and IoU all unchanged.

| metric | 50ep FULL | historical FULL | delta |
|---|---|---|---|
| mean IoU | **0.699183** | 0.703201 | −0.004018 |
| median | 0.747113 | 0.751699 | −0.004586 |
| P10 | 0.408485 | 0.415992 | −0.007507 |
| P90 | 0.909632 | 0.910241 | −0.000610 |
| IoU≥0.50 | 0.836219 | 0.842136 | −0.005917 |
| IoU≥0.75 | 0.494647 | 0.503454 | −0.008807 |

- absolute delta **−0.004018** | relative **−0.571%** — within NEAR_PARITY (≤0.005).
- valid 36335 / failed **0**.

### Size breakdown
| size | n | 50ep | historical | delta |
|---|---|---|---|---|
| small | 15264 | 0.613713 | 0.621048 | −0.007335 |
| medium | 12392 | 0.741617 | 0.744661 | −0.003044 |
| large | 8679 | 0.788913 | 0.788489 | **+0.000424** |

- STANDARD500 size trend on FULL: **PARTIAL** — medium near-parity (✓), small
  drifts past 0.005 (✗, −0.0073), large slightly *up* not down (✗, +0.0004).
  The STANDARD500 observation (small/med ~par, large down) does **not** fully
  transfer to FULL: large is at parity, small is the weakest bucket.

### Paired 50ep vs historical (per-instance, n=36335)
- improved 16464 / degraded 19715 / equal-or-near 156.
- mean delta **−0.004018** | median −0.001626 | P10 −0.0484 | P90 +0.0369 | P95 abs 0.0963.
- Symmetric spread: ~45% of instances the 50ep encoder is *better* than historical,
  ~54% worse — a near-tie with a tiny aggregate edge to historical, not a
  systematic regression.

### Mask selection (fixed slot0 vs alternatives)
- fixed slot0 mIoU **0.699183** vs predicted-IoU-argmax 0.629860 (−0.0693) vs
  oracle-best-of-4 0.727538 (+0.0284 headroom).
- 99.98% of predictions land in slot 2 — the decoder's score model is
  degenerate (argmax≠oracle 82.7%); fixed slot0 remains the right policy.

### Categories
- best5: toaster(0.873,n9) · bear(0.870,n71) · microwave(0.854,n55) · stop sign(0.849,n75) · parking meter(0.848,n60)
- worst5: **skis(0.339,n241)** · fork(0.471,n215) · knife(0.503,n325) · spoon(0.528,n253) · dining table(0.544,n695)
- Thin/elongated structures (skis, fork, knife, spoon) remain the dominant
  failure mode — same regime as historical, not a 50ep-specific weakness.

### Runtime
- wall 1988s (33 min) | 18.28 inst/s | 2.49 img/s | encoder calls 4952 | reuse 31383
- peak RSS 1.01GB | CUDA alloc 0.036GB | reserved 0.061GB (evaluation throughput, NOT deployment FPS).

### Quality gate & distillation final decision
- **NEAR_PARITY** (abs gap 0.004018 ≤ 0.005) → **Distillation Module: COMPLETE**.
- recommendation (advisory, NOT executed): **STOP_LONG_TRAINING**.
- Evidence chain: feature loss 0.001122→0.0004241 (epoch0→49); STANDARD500
  0.707340→0.715650 (10→50ep); FULL COCO 0.699183 vs historical 0.703201.
- The 50ep self-distilled student reaches within 0.4 mIoU points of the
  historically-deployed NanoSAM encoder on full COCO — sufficient to close the
  knowledge-distillation module. 100/200ep not justified by the shallow slope.

**Out of scope (STOP):** 100/200ep, INT8, NanoOWL, C++, rebuild — NOT performed.
**Next (awaiting authorization):** INT8 PTQ module.
