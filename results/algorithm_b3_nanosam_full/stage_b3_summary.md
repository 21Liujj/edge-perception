# Stage B3 Summary

Decision: `FULL_SEGMENTATION_BASELINE_ESTABLISHED`

## Protocol

This is prompt-conditioned instance mask quality evaluation, not semantic-segmentation mIoU and not COCO mask AP. Each legal non-crowd COCO annotation supplies a GT bounding-box prompt to NanoSAM; fixed mask slot 0 is compared with that annotation's GT mask.

B3 intentionally removes detector error. B2 measures NanoOWL text-prompted COCO bbox AP; future B5 will measure NanoOWL predicted box to NanoSAM mask error propagation.

## Legal Instances

- Total annotations: 36781
- Excluded crowd: 446
- Excluded invalid bbox: 0
- Excluded invalid mask: 0
- Other excluded: 0
- Legal instances: 36335 across 4952 images

## Full Metrics

- Fixed slot0 mIoU: 0.703201
- Median: 0.751699; P10/P90: 0.415992/0.910241
- IoU>=0.50: 0.842136; IoU>=0.75: 0.503454

## Size Analysis

| Size | Count | mIoU | Median | IoU>=0.5 | IoU>=0.75 |
|---|---:|---:|---:|---:|---:|
| small | 15264 | 0.621048 | 0.653846 | 0.753407 | 0.289505 |
| medium | 12392 | 0.744661 | 0.794296 | 0.899128 | 0.613541 |
| large | 8679 | 0.788489 | 0.850309 | 0.916811 | 0.722549 |

## Mask Selection at Scale

Fixed/predicted-argmax/oracle mIoU: 0.703201 / 0.636687 / 0.730897. Predicted argmax slot distribution: `{"0": {"count": 0, "ratio": 0.0}, "1": {"count": 0, "ratio": 0.0}, "2": {"count": 36330, "ratio": 0.9998623916334113}, "3": {"count": 5, "ratio": 0.0001376083665886886}}`.

## Predicted IoU Head

Candidate-level Pearson/Spearman/MAE: -0.026129 / -0.037662 / 0.202410. Argmax-to-oracle top-1 accuracy: 0.171047.

## Standard vs Full

FULL minus STANDARD mIoU: -0.014927; small/medium/large: 0.015728 / -0.017998 / -0.007307. These absolute deltas quantify how closely the balanced 500-instance subset approximates the COCO legal-instance population.

## Runtime and Stability

36335 instances, 4952 images, 4952 encoder calls, 36335 decoder calls, 1966.77 s wall time, 2.518 images/s, 18.474 instances/s, 0 failed. This is evaluation throughput, not deployment FPS.

## Implications for Training

Future B4 should retain this manifest, evaluator, and fixed-slot policy. Training priorities should follow the measured size gap, weak categories with adequate sample counts, boundary/thin-structure failures, and the predicted-IoU ranking mismatch. B3 does not train or modify the model.
