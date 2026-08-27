# NanoOWL Zero-Shot Detection Quality Freeze (Stage E2)

## Scope

This freezes the open-vocabulary detection quality of the deployed NanoOWL
(OWL-ViT base-patch32, TensorRT image encoder) on COCO val2017 FULL (5000
images). This is COCO **open-vocabulary / zero-shot** detection using
COCO category-name text prompts, not segmentation mIoU and not a closed-set
fine-tuned detector AP. Scores are not directly comparable to a
COCO-finetuned detector leaderboard entry.

## Protocol (reused, not re-designed)

Reused verbatim from `scripts/eval_nanoowl_coco.py` (Stage B2, unchanged):

- Model: `google/owlvit-base-patch32`, TensorRT image encoder engine
  `third_party/nanoowl/data/owlvit_image_encoder_patch32.engine`
- Dataset: full COCO val2017, `data/coco/annotations/instances_val2017.json`,
  no manifest subsampling (all 5000 images, `manifest: None`)
- Prompt template: `"a {category_name}"`, one prompt per COCO category
  (80 categories), text embeddings cached once
- Raw inference threshold 0.05, primary reporting threshold 0.10, top-k
  per image 100 (see `threshold_ablation.json` / `topk_ablation.json` for the
  0.05/0.10/0.20/0.30 and 10/50/100 sensitivity sweep already recorded in the
  Stage B2 run)
- Box conversion: predictor outputs xyxy, clipped to image bounds, converted
  to COCO xywh; degenerate/non-finite boxes dropped and counted
  (`invalid_raw_predictions`)
- No NMS; per-patch prompt argmax with sigmoid score
- Formal metrics via `pycocotools.COCOeval`, `iouType="bbox"`

## Evidence audit

The historical FULL run at `results/algorithm_b2_nanoowl/full/` was verified
in place (not re-run):

- Engine SHA256 (recomputed, matches `run_metadata.json`):
  `d3f723f97d5e466805105f8b25da83d898e215c09598fb2b41c06d5254d09037`
- Annotations SHA256 (recomputed, matches `run_metadata.json`):
  `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`
- `image_count=5000`, `gt_instance_count=36781` — this is the full val2017
  set, confirming "FULL" (not quick/standard/smoke subsets, which also exist
  alongside it under the same results tree)

The user-quoted historical summary numbers
(AP≈0.2586 / AP50≈0.4079 / AP75≈0.2729 / APsmall≈0.0953 / APmedium≈0.2641 /
APlarge≈0.4422) match the on-disk `coco_eval_summary.json` to full float
precision (not approximated) — see Metrics below.

## Metrics (FULL COCO val2017, 5000 images, threshold=0.10, top_k=100)

| Metric | Value |
|---|---|
| AP | 0.258616 |
| AP50 | 0.407885 |
| AP75 | 0.272948 |
| APsmall | 0.095263 |
| APmedium | 0.264089 |
| APlarge | 0.442186 |
| AR1 | 0.255193 |
| AR10 | 0.365058 |
| AR100 | 0.372312 |
| ARsmall | 0.132255 |
| ARmedium | 0.384414 |
| ARlarge | 0.624806 |

`prediction_count=44962`, `zero_prediction_images=82/5000`.

Full record: `results/stage_e2_nanoowl_quality/full_metrics.json` (copied
from the verified Stage B2 `full/` run, same numbers, with engine/annotation
SHA256 attached).

## Category breakdown

Full per-category table (80 rows): `results/stage_e2_nanoowl_quality/category_metrics.csv`.
Rankings below use `minimum_gt_count_for_rankings=5` (categories with fewer
than 5 GT instances excluded from best/worst ranking, per existing script
default).

**Best 10 by AP:**

| category | AP | AP50 | gt | pred |
|---|---|---|---|---|
| cat | 0.6151 | 0.8359 | 202 | 248 |
| train | 0.5922 | 0.7964 | 190 | 303 |
| airplane | 0.5754 | 0.8288 | 143 | 225 |
| fire hydrant | 0.5560 | 0.7457 | 101 | 90 |
| giraffe | 0.5402 | 0.7878 | 232 | 357 |
| dog | 0.5397 | 0.6958 | 218 | 254 |
| elephant | 0.5273 | 0.8016 | 255 | 510 |
| bus | 0.5183 | 0.7010 | 285 | 342 |
| bear | 0.5174 | 0.6619 | 71 | 103 |
| zebra | 0.5153 | 0.7715 | 268 | 438 |

**Worst 10 by AP:**

| category | AP | AP50 | gt | pred |
|---|---|---|---|---|
| mouse | 0.0000 | 0.0000 | 106 | 24 |
| knife | 0.0251 | 0.0414 | 326 | 157 |
| book | 0.0419 | 0.0705 | 1161 | 372 |
| handbag | 0.0527 | 0.1009 | 540 | 500 |
| backpack | 0.0593 | 0.1204 | 371 | 617 |
| spoon | 0.0699 | 0.1290 | 253 | 628 |
| potted plant | 0.0706 | 0.1449 | 343 | 473 |
| oven | 0.0977 | 0.1978 | 143 | 470 |
| skis | 0.1015 | 0.2543 | 241 | 934 |
| hair drier | 0.1069 | 0.1253 | 11 | 168 |

**Evidenced pattern:** the worst categories are dominated by (a) small
handheld/thin objects (knife, spoon, mouse — small silhouette, easily
confused with visually similar tools), (b) categories with heavy semantic
overlap in the 80-class vocabulary (backpack/handbag, spoon/knife/fork —
confirmed by `semantic_confusion_pairs` below), and (c) categories where
prediction_count far exceeds gt_count (oven 470 vs 143, skis 934 vs 241,
backpack 617 vs 371) — over-triggering, not under-detection, is the
prediction-count-side driver for several of these. `mouse` has the opposite
pattern (24 predictions vs 106 GT — under-triggering) and AP50=0 exactly,
the most severe failure in the class list.

The best categories are large, visually distinctive, low-ambiguity object
classes (vehicles, large animals) — consistent with the largest
`APlarge - APsmall` gaps also being dominated by categories such as
`stop sign` (gap 0.753), `cat` (0.662), `train` (0.628): these categories
are recognized well when large in the frame and much worse when small.

## Error analysis

Curated cases (small number, not a full sweep):
`results/stage_e2_nanoowl_quality/failure_cases.json`, matcher = greedy
class-aware IoU≥0.50 matching (analysis-only, independent of COCOeval).

- **Semantic confusion** (class B): top confused pairs by count —
  car↔truck (74/57), chair↔couch (58), knife→fork (50), knife→spoon (38),
  sheep↔cow (44/25), person→chair/backpack/surfboard/couch/bed (frequent
  person false-negatives re-attributed to nearby object classes). Full top-10
  list in `failure_cases.json`.
- **Small-object miss** (class C): 3 curated images from the
  `small_object_failure` group, each with `small_fn_iou50` (missed GT with
  area < 32²px) as the dominant component of `fn_iou50`.
- **Crowded scene** (class D): 3 curated images from `false_negative_heavy`
  group; e.g. image 435081 has gt_count=62, tp_iou50=4, fn_iou50=58,
  small_fn_iou50=41 — most misses in dense scenes are also small-object
  misses, i.e. crowding and small-object failure co-occur rather than being
  independent failure modes in this data.
- **Localization error** (class A): 3 curated images from
  `false_positive_heavy` group (high `fp_iou50` = predictions of a category
  present in the prompt set that fail to reach IoU≥0.50 against any
  same-class GT box, i.e. wrong location/size rather than wrong class).
- **Language ambiguity** (class E): not separately isolated as a distinct
  image-level failure group by the existing script (no case labeled purely
  "ambiguous wording" survived); the closest evidence is the prompt
  sensitivity study below, which shows wording changes altering match
  outcomes on identical images/GT.
- 15 pre-existing visualization images from the Stage B2 full run
  (`results/algorithm_b2_nanoowl/full/visualizations/`) are referenced as
  image evidence rather than regenerated; no new large-scale visualization
  sweep was produced for E2 per scope constraints.

## Prompt sensitivity mini study

Small, fixed-scope study (not prompt engineering): 5 categories
(person, car, chair, knife, cat), 4 images per category (20 images total,
each with ≥1 non-crowd GT instance of that category), 3 templates
(`"{c}"`, `"a {c}"`, `"photo of a {c}"`), raw threshold 0.05, single-category
text prompt per query (isolated from the 80-way multi-prompt full-eval
setting).

Data: `results/stage_e2_nanoowl_quality/prompt_sensitivity.csv` (60 rows) and
`prompt_sensitivity_raw.json`.

Match rate (best-IoU prediction reaches IoU≥0.50 against category GT) across
20 images:

| template | matched / 20 |
|---|---|
| `{c}` | 18 |
| `a {c}` | 17 |
| `photo of a {c}` | 18 |

**Conclusion**: on this small sample, all three wordings localize the target
object about equally often (17-18/20) — wording does not change *whether*
the object is found. What does change is the **raw candidate count and
score magnitude**: the bare category name (`"car"`, `"knife"`) consistently
produces more raw candidates above threshold 0.05 than `"a {c}"` or
`"photo of a {c}"` (e.g. image 724 "car": 23 raw candidates for `"car"` vs 0
for both longer templates — the object was not found at all under the
longer templates in that one case), and top scores are often higher for the
bare noun template. This is consistent with the deployed pipeline's
production choice of `"a {category_name}"` as a reasonable middle ground,
but the sample (5 categories, 20 images) is too small to generalize beyond
this dataset; this is not a prompt-engineering project and the study was not
expanded.

## Final freeze

**NANOOWL_ZERO_SHOT_QUALITY_FROZEN.** All metrics above trace to the
Stage B2 FULL COCO run's on-disk JSON, independently re-verified (engine and
annotation SHA256 recomputed and matched) rather than re-run. Category
breakdown and error/failure-case curation derive from the same run's
`category_analysis.json` and `failure_cases.json`. The prompt sensitivity
mini study is new, small-scope, and out of the COCOeval AP/AR computation
(it does not alter the frozen full_metrics numbers above).
