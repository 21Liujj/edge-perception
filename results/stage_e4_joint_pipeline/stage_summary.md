# Stage E4: NanoOWL -> NanoSAM Pipeline Integration

**Status:** PASS -> JOINT_VLM_PIPELINE_INTEGRATED

## What changed

`scripts/run_edge_perception.py` (pre-existing Stage 4A script) had one critical
bug and several spec gaps, both fixed:

- **Critical bug fixed:** default NanoSAM encoder was
  `third_party/nanosam/data/resnet18_image_encoder.engine` (untrained upstream
  default, SHA `c277e5ca...`). Changed default to the production FP16 50-epoch
  self-distilled encoder
  `models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine`
  (SHA `99147caf...`, quality-gate PASS, full-COCO mIoU 0.6992). The INT8
  backend (quality-gate FAIL) remains available only via explicit
  `--nanosam-image-encoder-engine` override, never as a default.
- Added `--top-k` (default 1, documented policy: single best detection ->
  single mask, matches the "find the X" robotics use case; `-1` = all
  detections above threshold). NanoSAM's `set_image` embedding is computed
  once per input image and reused across all top-k detections.
- Renamed `--threshold` -> `--score-threshold`, `--prompts` -> `--text` to
  match the E4 CLI spec.
- Renamed outputs to the spec's 3-file contract: `detections.json`,
  `overlay.jpg`, `mask.png` (`mask_k2.png`, ... for top-k>1). Diagnostic-only
  files (`input.jpg`, `nanoowl_visualization.jpg`, `timing.json`) kept but not
  part of the contract.

## Coordinate bridge validation

Read from source (not memory): NanoOWL's `OwlPredictor.predict()` returns
xyxy boxes in original-image pixel space (pad_square=False); NanoSAM's
`Predictor.predict(points, point_labels)` expects points in the same
original-image pixel space and rescales internally. **Conclusion: NanoOWL's
box maps directly to NanoSAM's point_coords with no axis swap and no manual
normalization.** Confirmed empirically via smoke test (coherent mask, no
NaN/Inf) and via the 50-case COCO eval (box_iou mean 0.51 between NanoOWL and
GT boxes -- consistent with a correct, non-swapped bridge). Full detail in
`results/stage_e4_joint_pipeline/bridge_validation.json`.

## 50-case COCO evaluation

Deterministic sample (seed 2027) targeting a 3x3 grid of
{small,medium,large} x {single,multi,crowded}, 6 per cell before top-up to
50 from the full candidate pool (36,335 valid instances). For each instance:
NanoOWL detects using the GT category name as the text prompt (top-1), then
NanoSAM segments both (a) the NanoOWL box and (b) the GT box (quality
ceiling / NanoSAM-only reference), on the same cached image embedding.

### Headline metrics (n=50)

| Metric | Mean | Median |
|---|---|---|
| Detection success rate | 0.84 (42/50) | -- |
| box_iou (NanoOWL vs GT box), n=42 | 0.513 | 0.685 |
| mask_iou, NanoOWL box -> NanoSAM (end-to-end) | 0.384 | 0.365 |
| mask_iou, GT box -> NanoSAM (ceiling) | 0.752 | 0.783 |
| Degradation (ceiling - end-to-end) | 0.368 | 0.301 |

### Error decomposition (n=50)

| Bucket | Count | Meaning |
|---|---|---|
| OK | 22 | detection + segmentation both succeeded (box_iou>=0.50 or degradation<=0.15, and mask_iou_gtbox>=0.50) |
| B_localization_propagation | 17 | NanoOWL found the object, NanoSAM *could* segment it well given a perfect box (mask_iou_gtbox>=0.50), but NanoOWL's box imprecision measurably hurt the mask (box_iou<0.50 or degradation>0.15) |
| A_detection_failure | 8 | NanoOWL found no matching detection for the GT category above threshold=0.10 |
| C_segmentation_failure | 3 | even with a perfect GT box, NanoSAM's own mask_iou_gtbox<0.50 -- NanoSAM's failure, independent of detection |

**Interpretation:** the dominant failure mode in the joint pipeline is **B
(localization propagation)**, not C (NanoSAM's own segmentation quality,
which is fine 47/50 times given a good box) and not purely A (detection). Box
imprecision from NanoOWL is the single largest quality bottleneck once you
chain the two models -- an imperfect-but-present detection still costs ~0.37
mean IoU relative to the GT-box ceiling.

### By size bucket (mask_iou, end-to-end)

| Bucket | n | Mean | Median |
|---|---|---|---|
| small | 18 | 0.272 | 0.000 |
| medium | 18 | 0.440 | 0.610 |
| large | 14 | 0.457 | 0.499 |

Small objects are hit hardest -- consistent with E2's own finding that
small-object miss (area<32^2 px) is a top NanoOWL failure mode; that failure
propagates directly into the joint pipeline's mask quality.

### By scene type (mask_iou, end-to-end)

| Scene | n | Mean | Median |
|---|---|---|---|
| single | 18 | 0.598 | 0.727 |
| multi | 18 | 0.244 | 0.0002 |
| crowded | 14 | 0.290 | 0.000 |

Single-object scenes perform far better than multi/crowded -- crowded scenes
were also flagged in E2 as a NanoOWL failure mode; the joint pipeline
inherits this directly.

## Deliverables

- `results/stage_e4_joint_pipeline/pipeline_contract.json` -- CLI/output contract, production backend policy, smoke test result
- `results/stage_e4_joint_pipeline/bridge_validation.json` -- coordinate bridge proof (source-code + empirical)
- `results/stage_e4_joint_pipeline/eval_manifest.json` -- the 50 selected COCO instances (deterministic, seed 2027)
- `results/stage_e4_joint_pipeline/joint_quality.json` -- aggregate metrics (this file's tables, machine-readable)
- `results/stage_e4_joint_pipeline/error_decomposition.csv` -- per-instance row: box_iou, mask_iou_owlbox, mask_iou_gtbox, degradation, error_bucket
- `results/stage_e4_joint_pipeline/demo/` -- 8 side-by-side GT-vs-joint-prediction visualizations
- `scripts/run_edge_perception.py` (updated) -- the production joint pipeline CLI
- `scripts/eval_joint_pipeline_coco.py` (new) -- the 50-case evaluation harness
- `docs/edgeperception_vlm_pipeline.md` -- full pipeline documentation

## Status

**JOINT_VLM_PIPELINE_INTEGRATED** -- Stage E4 PASS.
