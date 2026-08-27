# EdgePerception VLM Pipeline: NanoOWL -> NanoSAM (Stage E4)

## Overview

This document describes the joint open-vocabulary perception pipeline that
chains NanoOWL (text-conditioned detection) into NanoSAM (promptable
segmentation), both deployed on Jetson AGX Orin with TensorRT.

    Image + Text Prompt --> NanoOWL --> Bounding Box(es) --> NanoSAM Box Prompt --> Mask(s)

The pipeline runs in a single process (`scripts/run_edge_perception.py`),
in-process reusing the already-validated `OwlPredictor` (NanoOWL) and
`Predictor` (NanoSAM) classes. It does not subprocess either model's CLI and
does not modify any `third_party/` code.

## Why this pipeline exists

NanoSAM alone is a *promptable* segmenter -- it requires a box or point
prompt and has no notion of "find the cup." NanoOWL alone is a *detector* --
it draws boxes around text-described objects but does not produce
pixel-accurate masks. Chaining them gives open-vocabulary instance
segmentation: "segment the cup" with no fixed category list and no manual
prompting.

## Coordinate bridge

The critical integration question is whether the two models agree on
coordinate convention. Verified directly from source (not from memory,
not from either project's documentation):

- **NanoOWL** (`third_party/nanosam/nanoowl/owl_predictor.py`,
  `OwlPredictor.predict()`): returns `output.boxes` as xyxy in **original
  image pixel space** when `pad_square=False` (this pipeline always uses
  `pad_square=False`). `_owl_box_roi_to_box_global` performs the
  normalized-ROI -> original-pixel mapping internally.
- **NanoSAM** (`third_party/nanosam/nanosam/utils/predictor.py`,
  `Predictor.predict(points, point_labels)`): expects `points` in **original
  image XY pixel space**; `preprocess_points()` rescales them internally by
  `image_encoder_size / max(image_width, image_height)`. A box prompt is two
  points, `[[x1,y1],[x2,y2]]`, with `point_labels=[2,3]` (2 = box top-left
  corner, 3 = box bottom-right corner) -- the same convention as the official
  MobileSAM/NanoSAM `examples/basic_usage.py`.

**Conclusion:** NanoOWL's `[x1,y1,x2,y2]` box maps *directly* onto NanoSAM's
`point_coords=[[x1,y1],[x2,y2]]`, `point_labels=[2,3]`. No axis swap, no
manual normalization needed in the bridge code -- both ends already speak
original-image pixel space.

Because NanoOWL boxes are not guaranteed to lie strictly inside the image
(the model can predict slightly past an edge), the bridge clips every box to
`[0,W]x[0,H]` before handing it to NanoSAM, and records both the clipped and
the original (unclipped) box plus an `out_of_bounds` flag in the output JSON
-- clipping is transparent, never silent.

Full validation evidence (source citations + empirical confirmation from the
smoke test and the 50-case COCO eval) is in
`results/stage_e4_joint_pipeline/bridge_validation.json`.

## Mask selection rule

NanoSAM's mask decoder emits 4 candidate masks with 4 predicted IoU scores.
Following the official `examples/basic_usage.py` convention (also used by
`scripts/run_nanosam_infer.py` and `scripts/eval_nanosam_coco.py` elsewhere
in this project), this pipeline always selects a **fixed slot 0**
(`hi_res_mask[0,0] > 0`), **not** an IoU argmax. This keeps the joint
pipeline's mask-selection behavior consistent with every other NanoSAM
evaluation in the project.

## Detection selection: top-k policy

NanoOWL can return multiple detections per prompt. The bridge selects the
top-k highest-score detections (optionally restricted to `--target-label`
first) and runs NanoSAM once per selected detection:

- **Default `--top-k 1`**: single best detection -> single mask. Matches the
  common single-object "find the X" robotics use case.
- **`--top-k -1`**: all detections above `--score-threshold` are segmented.
- NanoSAM's image embedding (`set_image`) is computed **once per input
  image** and reused across all top-k detections -- the expensive vision
  encoder call is not repeated per detection.
- If zero detections survive the threshold (and optional label filter), the
  pipeline exits cleanly with `status=NO_DETECTION` and does **not** call
  NanoSAM at all.

## Production backend

NanoSAM's default encoder is the **FP16 50-epoch self-distilled production
engine**
(`models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine`, SHA256
`99147caf00f0174b802848c93928232c02ac996e91e6660372bbc8923112baf4`,
quality-gate PASS, full-COCO mIoU 0.6992). This corrects a bug in the
pre-E4 version of this script, which defaulted to the *untrained* upstream
`third_party/nanosam/data/resnet18_image_encoder.engine`. The INT8 research
backend (quality-gate FAIL, -2.15% mIoU vs FP16) is available only via
explicit `--nanosam-image-encoder-engine` override, never as a default.

NanoOWL's encoder is the frozen E1/E2 TensorRT engine
(`third_party/nanoowl/data/owlvit_image_encoder_patch32.engine`).

No engine rebuild, no retrain, no power-mode / `jetson_clocks` change was
made for this stage.

## CLI usage

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
| `--text` | `"an owl,a glove,a finger"` | comma-separated open-vocabulary text prompt(s) |
| `--score-threshold` | 0.10 | NanoOWL detection score threshold (matches E2's primary reporting threshold) |
| `--top-k` | 1 | max detections (desc by score) pushed through NanoSAM; `-1` = all above threshold |
| `--target-label` | None | restrict detections to this label before top-k |
| `--nanoowl-engine` | frozen E1/E2 engine | NanoOWL TensorRT image encoder path |
| `--nanosam-image-encoder-engine` | production FP16 50-epoch engine | NanoSAM image encoder path |
| `--nanosam-mask-decoder-engine` | `third_party/nanosam/data/mobile_sam_mask_decoder.engine` | NanoSAM mask decoder path |
| `--output-dir` | `results/joint_pipeline/default` | output directory |

### Output contract

- `detections.json` -- full structured result: input/text/threshold/top_k,
  all NanoOWL detections, selected top-k detections, per-detection bridge +
  NanoSAM result (predicted IoUs, selected mask index, NaN/Inf guard),
  `status` in `{OK, NO_DETECTION}`.
- `overlay.jpg` -- single image with all selected boxes + masks drawn (one
  color per detection).
- `mask.png` -- binary mask (0/255) for the rank-1 (highest score) selected
  detection. Additional detections (`--top-k` > 1) are saved as
  `mask_k2.png`, `mask_k3.png`, ...
- (Diagnostic only, not part of the contract: `input.jpg`,
  `nanoowl_visualization.jpg`, `timing.json`.)

### Smoke test result

`--image third_party/nanoowl/assets/owl_glove_small.jpg --text "an owl,a
glove,a finger" --top-k 2`: 1 detection found above threshold (label="an
owl", score=0.3419, box=[275.19,82.35,463.64,399.35]); top-k correctly capped
at the available count. NanoSAM selected_iou=0.7139, nonzero_pixels=34122,
no NaN/Inf. All 3 contract files produced correctly.

## 50-case COCO quality evaluation

`scripts/eval_joint_pipeline_coco.py` runs the full joint pipeline's
detection-selection rule (NanoOWL top-1, text = GT category name) against 50
deterministically-sampled COCO instances (seed 2027), covering a 3x3 grid of
size (`small`/`medium`/`large`) x scene (`single`/`multi`/`crowded`). For
each instance, NanoSAM segments both the NanoOWL box and the GT box (the
latter is the same protocol `scripts/eval_nanosam_coco.py` uses to measure
NanoSAM-only quality, giving a fair "ceiling" reference), reusing one cached
image embedding per unique image.

### Results (n=50)

| Metric | Mean | Median |
|---|---|---|
| Detection success rate | 0.84 (42/50) | -- |
| box_iou (NanoOWL vs GT box), n=42 | 0.513 | 0.685 |
| mask_iou, NanoOWL box -> NanoSAM (end-to-end) | 0.384 | 0.365 |
| mask_iou, GT box -> NanoSAM (ceiling) | 0.752 | 0.783 |
| Degradation (ceiling - end-to-end) | 0.368 | 0.301 |

### Error decomposition

| Bucket | Count | Meaning |
|---|---|---|
| OK | 22 | detection + segmentation both succeeded |
| B_localization_propagation | 17 | detection succeeded and NanoSAM *can* segment well given a perfect box, but NanoOWL's box imprecision measurably hurt the mask |
| A_detection_failure | 8 | NanoOWL found no matching detection above threshold |
| C_segmentation_failure | 3 | NanoSAM's own quality failure even with a perfect GT box |

**The dominant joint-pipeline failure mode is box-localization propagation
(B), not NanoSAM's own segmentation quality (C is rare -- 47/50 GT-box
segmentations succeed).** An imperfect-but-present NanoOWL detection still
costs ~0.37 mean IoU relative to the GT-box ceiling. Small objects (mean
mask_iou 0.272) and crowded/multi-object scenes (mean 0.244-0.290 vs 0.598
for single-object scenes) are hit hardest -- consistent with E2's own
per-category failure-mode findings (small-object miss, crowded-scene
confusion) propagating directly through the joint pipeline.

Full per-instance data: `results/stage_e4_joint_pipeline/error_decomposition.csv`.
Aggregate JSON: `results/stage_e4_joint_pipeline/joint_quality.json`.
8 side-by-side demo visualizations: `results/stage_e4_joint_pipeline/demo/`.

## Deliverables

- `scripts/run_edge_perception.py` (updated) -- production joint pipeline CLI
- `scripts/eval_joint_pipeline_coco.py` (new) -- 50-case COCO evaluation harness
- `results/stage_e4_joint_pipeline/pipeline_contract.json`
- `results/stage_e4_joint_pipeline/bridge_validation.json`
- `results/stage_e4_joint_pipeline/eval_manifest.json`
- `results/stage_e4_joint_pipeline/joint_quality.json`
- `results/stage_e4_joint_pipeline/error_decomposition.csv`
- `results/stage_e4_joint_pipeline/demo/`
- `results/stage_e4_joint_pipeline/stage_summary.md`
- `docs/edgeperception_vlm_pipeline.md` (this file)

## Status

**JOINT_VLM_PIPELINE_INTEGRATED** -- Stage E4 PASS.
