# Day 12 — Stage Jetson-6D: 50ep NanoSAM FULL COCO Validation

**Date:** 2026-08-24
**Stage:** Jetson-6D (knowledge-distillation module closure)
**Result:** ✅ **NEAR_PARITY — Distillation Module COMPLETE**

## Goal
Validate the 50-epoch self-distilled ResNet18 student (FP16 TRT) on the **full**
COCO val2017 (36,335 instances) against the historically-deployed NanoSAM
encoder FULL baseline (0.703201), and decide whether the distillation module
can be closed.

## Protocol (frozen)
- Evaluator: `scripts/eval_nanosam_full_b3.py` — `evaluate()` + `analyze()`
  called directly, bypassing `main()` standard-gate (would exit 10: 50ep !=
  STD_MIOU by design).
- Encoder swapped ONLY: 50ep `99147caf…` ← historical `c277e5ca…`.
- Unchanged: decoder `afb5ac49…`, FULL manifest `ad8843c7…` (36335 inst / 4952
  img), GT-box corner prompt points [2,3], fixed slot0, COCO area size split.
- Pre-eval smoke: 5 instances, 5/5 valid 0 fail, mean 0.674.
- Two bugs caught pre-run: B3 `evaluate()` takes Path not str (materialize does
  `image_root/file_name`); gate-bypass needed.

## Result

| | 50ep FULL | historical FULL | delta |
|---|---|---|---|
| **mean IoU** | **0.699183** | 0.703201 | **−0.004018** |
| median | 0.747113 | 0.751699 | −0.004586 |
| IoU≥0.50 | 0.836219 | 0.842136 | −0.005917 |
| IoU≥0.75 | 0.494647 | 0.503454 | −0.008807 |

36335/36335 valid, **0 failed**. Wall 1988s (33 min), 18.3 inst/s.

### Size
- small n15264: 0.6137 vs 0.6210 (−0.0073) ← weakest bucket
- medium n12392: 0.7416 vs 0.7447 (−0.0030) ← near-parity ✓
- large n8679: 0.7889 vs 0.7885 (**+0.0004**) ← at parity / slightly better

STANDARD500 trend (small/med ~par, large down) → **PARTIAL** on FULL: large is
at parity, small drifts. The subset didn't forecast the small-bucket weakness.

### Paired (n=36335)
improved 16464 / degraded 19715 / equal 156; mean −0.0040; P95 abs 0.0963.
~45% of instances 50ep beats historical — a near-tie, not systematic regression.

### Mask selection
fixed slot0 0.6992 >> predicted-argmax 0.6299; oracle 0.7275 (+0.028 headroom).
99.98% of preds land slot 2 (decoder score degenerate); fixed slot0 confirmed.

### Categories
- best: toaster 0.873, bear 0.870, microwave 0.854, stop sign 0.849
- worst: **skis 0.339**, fork 0.471, knife 0.503, spoon 0.528 — thin/elongated,
  same failure regime as historical (not 50ep-specific).

## Distillation final decision
- **GATE = NEAR_PARITY** (abs gap 0.004018 ≤ 0.005).
- **Distillation Module: COMPLETE.** recommendation: STOP_LONG_TRAINING.
- Evidence chain: feature loss 0.001122→0.0004241 (epoch0→49); STANDARD500
  0.707340→0.715650 (10→50ep); FULL 0.699183 vs historical 0.703201.
- 50ep student within 0.4 mIoU of deployed encoder on full COCO. The
  epoch10→49 slope is shallow (+0.0021/10ep); 100/200ep not justified.

## Evidence (`results/stage_jetson_6d/`)
full_metrics.json, full_summary.json, historical_comparison.json,
size_breakdown.json, size_bucket_analysis.json, paired_comparison.json,
quality_gate.json, distillation_final_decision.json, mask_selection_full.json,
category_analysis.json, category_size_analysis.json, predicted_iou_analysis.json,
iou_distribution.json(+png), failure_cases.json, runtime_summary.json,
standard_vs_full.json, per_instance_primary.csv (12MB, 36335 rows, sha
e3c47e29…), visualizations/ (12 PNGs), stage_summary.md.

## Out of scope (STOP — none performed)
100/200ep training, INT8 PTQ, NanoOWL joint, C++, CUDA optimization, power-mode
/jetson_clocks, engine/decoder rebuild, manifest modification, git push.

## Next
INT8 PTQ module — **awaiting user authorization** (Module C).
