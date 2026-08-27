# EdgePerception

**Lightweight Vision-Language Perception and TensorRT Deployment on Jetson Orin**

Open-vocabulary image segmentation on embedded GPU hardware: give it an image
and a text prompt like `"cup"`, get back a pixel mask — no fixed category
list, no retraining per object class.

**Status: `EDGE_PERCEPTION_V1_COMPLETE`**

```
Image + "a cup" ──► NanoOWL (OWL-ViT, TensorRT FP16) ──► bounding box
                                                              │
                                                              ▼
                     NanoSAM (ResNet18, distilled, TensorRT) ──► pixel mask
```

---

## 1. What this is

Two independently-engineered models chained into one open-vocabulary
perception pipeline, deployed on Jetson AGX Orin:

- **NanoOWL** — text-conditioned zero-shot object detection (OWL-ViT
  base-patch32). Given an image and free-text prompts, returns bounding boxes
  with scores. TensorRT FP16 image encoder; CLIP text tower stays in
  PyTorch.
- **NanoSAM** — promptable segmentation. A ResNet18 encoder distilled from a
  MobileSAM teacher, paired with the frozen upstream MobileSAM mask decoder.
  Given a box or point prompt, returns a pixel mask. TensorRT FP16
  (production) and TensorRT INT8 (documented, not deployed — see §5).

Chaining them gives "segment the X" with an open vocabulary, at the cost of
propagating NanoOWL's box imprecision into NanoSAM's mask quality — a cost
this project measures explicitly rather than assumes away (§4).

## 2. Why two models instead of one

NanoSAM alone has no notion of "find the cup" — it is a *promptable*
segmenter, not a detector. NanoOWL alone draws boxes but produces no
pixel-accurate mask. Neither substitutes for the other; chaining them is the
open-vocabulary instance segmentation solution that doesn't require training
a new joint model or fixing a category list.

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         EdgePerception v1                            │
│                                                                      │
│  Training (RTX 5090 / offline)         Deployment (Jetson AGX Orin) │
│                                                                      │
│  MobileSAM Teacher                                                   │
│        │ knowledge distillation                                     │
│        ▼                                                            │
│  NanoSAM ResNet18 Student (50 epoch) ──► ONNX ──► TensorRT FP16/INT8│
│                                                                      │
│  OWL-ViT base-patch32 (frozen)        ──► ONNX ──► TensorRT FP16    │
│  (image encoder only; text tower stays PyTorch)                     │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │              scripts/run_edge_perception.py                │    │
│  │                                                              │    │
│  │  image + text ──► NanoOWL.predict() ──► boxes (xyxy, orig   │    │
│  │  px) ──► clip_box() ──► bridge (no axis swap, no norm) ──►  │    │
│  │  NanoSAM.predict(points=[[x1,y1],[x2,y2]], labels=[2,3])    │    │
│  │  ──► mask (fixed slot 0) ──► detections.json / overlay.jpg /│    │
│  │  mask.png                                                    │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Python runtime (both models)   +   C++ TensorRT runtime (NanoSAM)  │
└──────────────────────────────────────────────────────────────────────┘
```

Full pipeline documentation: [`docs/edgeperception_vlm_pipeline.md`](docs/edgeperception_vlm_pipeline.md).

## 4. Results

### 4.1 Quality

| Component | Metric | Value | Protocol |
|---|---|---|---|
| NanoOWL | FULL COCO AP@0.50:0.95 | **0.2586** | zero-shot, 80 categories, 3418 instances |
| NanoSAM (FP16, 50ep) | FULL COCO mIoU | **0.6992** | GT-box prompt, fixed slot0, 36335 instances |
| NanoSAM (INT8) | FULL COCO mIoU | 0.7003 (STANDARD500: −2.15% vs FP16) | quality gate **FAIL** |
| Joint pipeline (E2E) | mask_iou vs GT-box ceiling | **0.384 vs 0.752** | 50-case stratified COCO, seed 2027 |
| Joint pipeline | detection success rate | **0.84** (42/50) | NanoOWL finds a matching box |

**Joint pipeline error decomposition** (n=50) — where quality is actually lost:

| Bucket | Count | Meaning |
|---|---|---|
| OK | 22 | detection + segmentation both succeeded |
| **B: localization propagation** | **17** | NanoOWL found the object; NanoSAM *could* segment it well given a perfect box, but the box wasn't tight enough |
| A: detection failure | 8 | NanoOWL found no matching detection above threshold |
| C: segmentation failure | 3 | NanoSAM's own mask quality failure, independent of the box |

**The dominant joint-pipeline failure mode is box-localization propagation
(B), not NanoSAM's own segmentation quality (C is rare).** Full breakdown:
[`results/stage_e4_joint_pipeline/`](results/stage_e4_joint_pipeline/).

### 4.2 Latency (Jetson AGX Orin, 25W, TensorRT FP16)

| Stage | Mean (ms) | Notes |
|---|---|---|
| NanoOWL, warm prompt (text cached) | 22.1 | core TRT inference only |
| NanoOWL, cold prompt (no cache) | 79.1 | includes PyTorch text encoding — **3.6x** warm |
| NanoOWL, VLM E2E (in joint pipeline) | 36.4 | preprocess + TRT + cached text + postprocess |
| NanoSAM FP16 encoder | 60.9 | dominant single cost in the joint pipeline |
| NanoSAM FP16 decoder | 15.9 | box prompt → mask |
| NanoSAM INT8 encoder | ~6.0 | 1.845x speedup vs FP16 — **not deployed**, quality gate FAIL |
| **Joint pipeline, steady-state E2E** | **113.7 (8.8 FPS)** | full "image + text → mask", n=300 |
| Joint pipeline, cached embedding | 33.9 | NanoSAM embedding reused; per-prompt cost only |
| C++ runtime, NanoSAM full inference | 68 | encoder 36 + decoder 32, TensorRT C++ |

Full latency evidence: [`results/stage_e5_final/final_latency.json`](results/stage_e5_final/final_latency.json),
[`results/joint_benchmark/`](results/joint_benchmark/).

### 4.3 Deployment footprint

| Engine | Size |
|---|---|
| NanoOWL image encoder (FP16) | 176 MB |
| NanoSAM encoder (FP16, production) | 37 MB |
| NanoSAM encoder (INT8, research) | 22 MB |
| NanoSAM mask decoder | 19 MB |

## 5. Production backend policy

NanoSAM's default deployment encoder is the **FP16 50-epoch self-distilled
model** (`models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine`,
SHA256 `99147caf…`, FULL COCO mIoU 0.6992, quality gate **PASS**).

INT8 (PTQ) is measured and documented but **not the default**: it clears a
1.845x encoder speedup but fails this project's own quality gate (−2.15%
mIoU vs FP16, threshold −0.5%). It is reachable only via an explicit CLI
override, never silently. Details: [`docs/int8_ptq.md`](docs/int8_ptq.md).

This policy exists because a pre-E4 version of the joint pipeline script
defaulted to an *untrained* upstream NanoSAM encoder (different SHA256) —
a real bug, caught by diffing the loaded engine's hash against the validated
model's hash, not by "the demo looked fine." Fixed in commit `a58ff03`.

## 6. Coordinate bridge

NanoOWL outputs `[x1,y1,x2,y2]` in **original-image pixel space**.
NanoSAM's box prompt expects `point_coords=[[x1,y1],[x2,y2]]`,
`point_labels=[2,3]`, also in original-image pixel space. Verified from
source (not memory, not docs) in both directions — **no axis swap, no manual
normalization needed**. Boxes are clipped to `[0,W]x[0,H]` before the
NanoSAM call, with the clip and any out-of-bounds flag recorded, never
silent. Full proof: [`results/stage_e4_joint_pipeline/bridge_validation.json`](results/stage_e4_joint_pipeline/bridge_validation.json).

## 7. Mask selection

NanoSAM's decoder emits 4 candidate masks with predicted IoU scores. This
project always selects **fixed slot 0**, not the IoU argmax — measured, not
assumed: on full COCO, fixed slot0 (mIoU 0.699) beats the argmax policy
(mIoU 0.630) because 99.98% of predictions land in one slot regardless of
instance (the decoder's own score head is degenerate; argmax agrees with the
oracle-best-of-4 only 82.7% of the time).

## 8. Repository layout

```
scripts/run_edge_perception.py       production joint pipeline CLI
scripts/eval_joint_pipeline_coco.py  50-case COCO joint evaluation harness
scripts/benchmark_edge_perception.py steady-state E2E latency benchmark
cpp_runtime/                          C++ TensorRT runtime (NanoSAM)
docs/                                 stage reports, algorithm notes, protocols
results/                              per-stage evidence (JSON/CSV/images)
third_party/nanoowl/, third_party/nanosam/   upstream repos (unmodified)
models/self_distilled/                trained NanoSAM checkpoints + engines
```

## 9. CLI usage

```bash
python scripts/run_edge_perception.py \
    --image data/test/kitchen.jpg \
    --text "cup" \
    --score-threshold 0.10 \
    --output-dir results/demo_cup
```

| Argument | Default | Meaning |
|---|---|---|
| `--image` | NanoOWL sample asset | input image path |
| `--text` | `"an owl,a glove,a finger"` | comma-separated open-vocabulary prompt(s) |
| `--score-threshold` | 0.10 | NanoOWL detection score threshold |
| `--top-k` | 1 | detections pushed through NanoSAM; `-1` = all above threshold |
| `--target-label` | None | restrict detections to this label before top-k |
| `--nanosam-image-encoder-engine` | production FP16 50-epoch | override to use INT8 or another encoder |
| `--output-dir` | `results/joint_pipeline/default` | output directory |

Output contract: `detections.json` (full structured result, status
`OK`/`NO_DETECTION`), `overlay.jpg` (boxes + masks drawn), `mask.png`
(rank-1 mask; `mask_k2.png`… for `--top-k`>1).

## 10. C++ runtime

A separate, production-oriented C++ TensorRT runtime exists for NanoSAM
(encoder + decoder, CMake build, image-mode and tensor-mode CLIs). It is
correctness-verified against the Python runtime two ways: tensor-mode parity
on identical frozen preprocessed tensors (encoder embedding max-abs-diff
<1e-5, decoder mask agreement >99.9%), and a 20-instance image-mode
regression (documented preprocessing delta from OpenCV vs PIL resize
filters, mean relative L2 5.1%, does not affect tensor-mode parity). Full
detail: [`cpp_runtime/README.md`](cpp_runtime/README.md). NanoOWL is not yet
ported to C++.

## 11. Evaluation methodology

- **NanoOWL**: FULL COCO val2017 zero-shot detection, all 80 categories.
- **NanoSAM**: FULL COCO val2017 (36,335 legal non-crowd instances) with
  GT-box prompts, isolating segmentation quality from detection quality.
- **Joint pipeline**: a deterministic 50-case stratified sample (seed 2027,
  3x3 grid of object size × scene crowding) comparing NanoOWL-box→NanoSAM
  (real end-to-end) against GT-box→NanoSAM (quality ceiling) on the *same*
  cached image embedding, so the only variable that differs is the box
  source. This isolates whether quality loss comes from detection or
  segmentation — see §4.1's error decomposition.
- **Latency**: all GPU timing taken after `torch.cuda.synchronize()`;
  steady-state numbers exclude model load, disk I/O, and visualization;
  every latency number in this README states its protocol (warm vs cold,
  cached-embedding vs full E2E) because these differ by 3-6x and conflating
  them would misrepresent real deployment cost.

## 12. Limitations

- Not a real-time camera system — offline image + box/point prompt
  inference only, no `/dev/video*` pipeline in the main line.
- Not an LLM-based VLM — vision-language grounding via CLIP-style text
  embeddings, not a generative language model; no TTFT/prefill/decode
  concepts apply.
- NanoOWL's text tower remains PyTorch; only its image encoder is TensorRT.
- No ROS2 integration — deferred to future work.
- INT8 uses PTQ only; no QAT attempted. The quality gate failure is a PTQ
  result, not necessarily a ceiling on INT8 feasibility.
- C++ runtime covers NanoSAM only; NanoOWL's image encoder is not yet
  ported.
- Out-of-distribution robustness (non-COCO-style imagery, e.g. fisheye or
  heavily rotated input) is unevaluated.

## 13. Interview / resume material

[`docs/interview_summary.md`](docs/interview_summary.md) — 30-second, 2-minute,
and 5-minute versions of this project, plus 20 likely interview questions
with answers grounded in the actual measurements in this repository (not
generic talking points).

## Stage history

| Stage | Result | Commit |
|---|---|---|
| E1 | NanoOWL VLM runtime architecture audit | `d05c84d` |
| E2 | NanoOWL zero-shot quality freeze (FULL COCO) | `7383efc` |
| E3 | NanoOWL VLM runtime benchmark (cold vs warm prompt) | `ea96c9c` |
| E4 | NanoOWL→NanoSAM joint pipeline integration + 50-case eval | `a58ff03` |
| E5 | Final consolidation, README, interview material | *(this stage)* |

Full stage-by-stage evidence tracker: `edgeperception_stage_tracker.md`
(session working notes) and `docs/daily/` (daily engineering logs).
