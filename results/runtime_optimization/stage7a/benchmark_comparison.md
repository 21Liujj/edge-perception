# Stage 7A — Benchmark Comparison (baseline vs optimized)

Method: warmup ≥ 30 iters, measured ≥ 200 iters. Same owl image (756x499),
same FP16 owl + FP16 NanoSAM enc / FP32 NanoSAM dec engines. `--runtime optimized`
uses OptimizedPredictor (buffer reuse + pinned + static cache + point template;
stream variant unless `--no-stream`). Percentiles computed on measured iters.

## set_image (micro benchmark)

| metric (ms) | baseline | opt stream | opt no-stream |
|---|---|---|---|
| mean | 63.95 | 59.51 | 65.43 |
| min | 45.90 | 43.06 | 49.06 |
| **P50** | 66.14 | **61.53** | 67.04 |
| **P95** | 69.13 | **64.67** | 72.30 |
| **P99** | 70.55 | **66.21** | 73.32 |
| max | 72.92 | 67.51 | 73.85 |

- speedup stream: **1.0745x** (P95 −4.46ms)
- speedup no-stream: 0.9773x (P95 +3.16ms, worse)
- **Verdict: set_image STREAM variant ACCEPTED** (faster on mean + every percentile;
  P95/P99 strictly better). no-stream variant REJECTED (slightly worse — staging copy
  overhead with no overlap benefit).

## joint (full edge-perception pipeline: NanoOWL + NanoSAM)

| metric (ms) | baseline | opt stream | opt no-stream |
|---|---|---|---|
| mean | 105.16 | 110.72 | 108.79 |
| min | 89.22 | 95.34 | 90.91 |
| **P50** | 105.12 | 113.87 | 110.07 |
| **P95** | 114.39 | 115.19 | 116.08 |
| **P99** | 115.62 | 116.23 | 116.40 |
| max | 117.26 | 116.54 | 117.29 |

- speedup stream: **0.9498x** (mean +5.56ms slower)
- speedup no-stream: 0.9666x (mean +3.63ms slower)
- P95 stream +0.80ms (worse), P95 no-stream +1.68ms (worse)
- **Verdict: joint REJECTED (both variants)** — P95/P99 not better than baseline,
  negative mean speedup. The set_image gain does not survive the joint pipeline
  because the explicit-stream event/sync overhead + staging boundary crossing the
  default stream exceeds the per-frame copy-count benefit, and NanoOWL↔NanoSAM run
  sequentially with no overlap window.

## Noise threshold

Inter-run jitter on this board is ~2-3ms (see baseline min/max spread). The
set_image stream win (−4.5ms P95) exceeds noise; the joint deltas (+0.8 to +1.7ms
P95) are within/near noise but trend negative, so by the acceptance policy
(P95 not worse required for ACCEPT) joint is REJECTED.

## Correctness gates (baseline & optimized both)

- owl image: owl detected (mask_nonzero = 34065, matches baseline); no NaN/Inf.
- dogs image: detected (mask_nonzero = 375624, matches baseline); no NaN/Inf.

## COCO QUICK 100-instance regression (vs Day 5 baseline)

| metric | baseline | optimized |
|---|---|---|
| mIoU | 0.721328258289041 | 0.721328258289041 |
| median IoU | 0.7602281711661292 | 0.7602281711661292 |
| p10 / p90 | 0.4487 / 0.9208 | 0.4487 / 0.9208 |
| IoU≥0.50 ratio | 0.88 | 0.88 |
| IoU≥0.75 ratio | 0.53 | 0.53 |
| mIoU small | 0.6598302010892132 | 0.6598302010892132 |
| mIoU medium | 0.7278570899484904 | 0.7278570899484904 |
| mIoU large | 0.792488102086808 | 0.792488102086808 |
| valid / failed | 100 / 0 | 100 / 0 |

- **mean abs IoU delta: 0.0**  (across all 100 instances)
- **max abs IoU delta: 0.0**
- Numerical semantics preserved exactly (same PIL.resize BILINEAR, mean/std, pad).
- Note: optimized wall per COCO run is LOWER (9.89s vs 11.35s) because COCO eval
  reuses embeddings across instances in the same image; this is eval-harness
  reuse, not a per-frame pipeline win — not claimed as a Stage 7A benefit.

## Acceptance summary

| path | verdict | reason |
|---|---|---|
| set_image (stream) | ACCEPTED | 1.075x, P95/P99 better, correctness identical |
| joint (stream) | REJECTED | 0.950x slower, P95/P99 worse |
| joint (no-stream) | REJECTED | 0.967x slower, P95 worse |

The optimized runtime is NOT promoted to default; kept as `--runtime optimized`.
Honest negative result on the joint (production) path is recorded per acceptance policy.
