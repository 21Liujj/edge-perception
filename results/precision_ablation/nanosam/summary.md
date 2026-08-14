# NanoSAM Image Encoder FP32 vs FP16 Precision Ablation (Stage 5C)

Controlled ablation varying ONLY the NanoSAM image-encoder precision (FP16 vs FP32).
Same ONNX (resnet18_image_encoder.onnx, SHA256 d266fcdc...), same mask decoder (FP32,
mobile_sam_mask_decoder.engine, afb5ac49...), same COCO STANDARD manifest (seed=2027,
500 instances), same evaluator, same GT box prompt, same slot 0, same metric code.
No engine rebuilt for the FP16 baseline; FP32 encoder built fresh without --fp16.
Not a real-time FPS claim; repeated-encode latency characterization.

## Engine size

| Engine | Precision | Size (bytes) | SHA256 |
|---|---|---|---|
| resnet18_image_encoder.engine      | FP16 | 34,766,700 | c277e5ca...aaa6a8 |
| resnet18_image_encoder_fp32.engine | FP32 | 63,893,220 | 51d75af6...1cbbed |

FP32 -> FP16 engine size reduction: 45.6% (28.9 MB saved).

## COCO STANDARD paired quality (500 instances, GT box, slot 0)

| Variant | valid/failed | mIoU | median | P10 | P50 | P90 | IoU>=0.50 | IoU>=0.75 |
|---|---|---|---|---|---|---|---|---|
| A — FP16 enc | 500/0 | 0.71813 | 0.78387 | 0.37913 | 0.78387 | 0.91923 | 0.846 | 0.590 |
| B — FP32 enc | 500/0 | 0.71821 | 0.78387 | 0.37871 | 0.78387 | 0.91926 | 0.846 | 0.590 |

### mIoU by size bucket

| Variant | small | medium | large |
|---|---|---|---|
| A — FP16 | 0.60532 (175) | 0.76266 (166) | 0.79580 (159) |
| B — FP32 | 0.60542 (175) | 0.76275 (166) | 0.79584 (159) |

### Paired per-instance comparison (equal tolerance |delta| < 1e-4)

| metric | value |
|---|---|
| n paired | 500 |
| mean IoU delta (FP32 - FP16) | +8.07e-05 |
| median IoU delta | 0.0 |
| mean absolute IoU delta | 8.80e-04 |
| max absolute IoU delta | 0.01395 |
| FP32 better count | 176 |
| FP16 better count | 164 |
| approximately equal count | 160 |
| small-objects mIoU delta (FP32 - FP16) | +9.93e-05 |

The better/worse split is nearly symmetric (176 / 164 / 160): no systematic FP32 quality
advantage. Per-instance deltas are tiny (mean abs 8.8e-04). Small-object mIoU delta is
+9.9e-05 — negligible; only reported as fact, NOT attributed to FP16 precision.

## Paired latency (dogs.jpg, warmup>=20, iters>=100, cuda-sync)

| metric | FP16 | FP32 | FP32/FP16 |
|---|---|---|---|
| encoder P50 (ms) | 71.90 | 84.92 | 1.18x |
| encoder P95 (ms) | 79.36 | 92.95 | 1.17x |
| encoder mean (ms) | 69.95 | 82.98 | 1.19x |
| full segmentation P50 (ms) | 88.76 | 100.33 | 1.13x |
| full segmentation mean (ms) | 88.86 | 98.08 | 1.10x |

FP16 is faster: encoder P50 ~15% lower; full-segmentation P50 ~12% lower.

## Resource (tegrastats, steady-state encode burst, best-effort)

| metric | FP16 | FP32 |
|---|---|---|
| peak RAM (MB) | 2919 | 2951 |
| peak GR3D (%) | 99 | 99 |
| peak GPU temp (C) | 53 | 54 |
| peak VDD_IN (mW) | 7170 | 8160 |

FP32 uses slightly more RAM (+32 MB) and more power (+14% VDD_IN). GR3D saturated at 99%
for both (encoder-bound). Sampling is short; treated as indicative, not a stable mean.

## Trade-off table (FP32 -> FP16)

| Axis | FP32 -> FP16 effect |
|---|---|
| Latency speedup (encoder P50) | FP16 1.18x faster |
| Full segmentation P50 | FP16 1.13x faster |
| Engine size reduction | 45.6% smaller |
| Quality delta (mIoU) | -8.07e-05 (FP16 negligibly lower) |
| Small-object mIoU delta | -9.93e-05 (negligible) |
| paired mean abs IoU delta | 8.80e-04 |
| RAM / power | FP16 slightly lower |

## Conclusion

Under the evaluated COCO STANDARD subset (500 instances):
- FP16 is materially faster (encoder P50 ~15% lower) and ~46% smaller, with slightly
  lower RAM/power.
- The quality delta is negligible (mean mIoU delta 8e-05; mean abs per-instance delta
  9e-04; FP32-better vs FP16-better nearly symmetric 176/164).
- No systematic FP32 quality advantage, including on small objects.

Therefore FP16 is the preferred deployment precision for the NanoSAM image encoder under
the evaluated COCO subset: a clear latency/size win at negligible quality cost. This does
NOT generalize beyond this subset without a FULL run; small-object delta is reported as
fact, not attributed to precision.

## Default config restored

Default pipeline restored to: NanoSAM Image Encoder FP16 + Mask Decoder FP32 (FP32 engine
NOT set as default). dogs.jpg FP16 regression re-run and PASS.

## Evidence paths

- results/precision_ablation/nanosam/fp16/ (encoder_latencies.csv, full_segmentation_latencies.csv, latency_summary.json, tegrastats.log, standard/)
- results/precision_ablation/nanosam/fp32/ (same + box regression smoke: box_result.json, box_mask.npy, box_visualization.jpg)
- results/precision_ablation/nanosam/paired_quality.csv
- results/precision_ablation/nanosam/quality_comparison.json
- results/precision_ablation/nanosam/latency_comparison.json
- results/precision_ablation/nanosam/fp32_engine_build.log
- results/precision_ablation/nanosam/summary.md
