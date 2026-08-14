# NanoSAM COCO Quality Baseline — Stage 5B

Established the NanoSAM box-prompt segmentation quality baseline on COCO 2017 val.
Protocol frozen in `docs/evaluation_protocol.md` (Stage 5A). No protocol changed to
improve results; no engine rebuilt; no FULL inference run.

## Configuration (BASELINE LOCKED)

| Module | Precision |
|---|---|
| NanoSAM Image Encoder (ResNet18) | FP16 TensorRT |
| NanoSAM Mask Decoder (MobileSAM) | FP32 TensorRT |
| Prompt | COCO GT bounding box (xywh -> xyxy) |
| Mask selection | slot 0 (fixed, NOT IoU argmax) |
| Embedding reuse | same image -> single set_image |

Engines (SHA256, unchanged since Day 3 build):
- Image Encoder: `third_party/nanosam/data/resnet18_image_encoder.engine` (c277e5ca...aaa6a8, FP16)
- Mask Decoder:  `third_party/nanosam/data/mobile_sam_mask_decoder.engine` (afb5ac49...ccb4f2, FP32)

## Dataset

- COCO 2017 val2017: 5000 images, 80 categories, 36781 annotations (446 crowd, 36335 legal).
- pycocotools 2.0.11.

## Subset results (seed=2027, slot 0, GT box prompt)

| subset | valid | failed | mIoU | median | P10 | P90 | IoU>=0.50 | IoU>=0.75 |
|---|---|---|---|---|---|---|---|---|
| SMOKE   | 20  | 0 | 0.7555 | 0.7856 | 0.5753 | 0.8843 | 1.000 | 0.700 |
| QUICK   | 100 | 0 | 0.7213 | 0.7602 | 0.4487 | 0.9208 | 0.880 | 0.530 |
| STANDARD| 500 | 0 | 0.7181 | 0.7839 | 0.3791 | 0.9192 | 0.846 | 0.590 |

### mIoU by size bucket (STANDARD)

| bucket | count | mIoU |
|---|---|---|
| small  | 175 | 0.6053 |
| medium | 166 | 0.7627 |
| large  | 159 | 0.7958 |

### STANDARD throughput / reuse

- unique images: 455
- embedding reuse count: 45 (500 instances - 455 set_image calls)
- evaluation wall time: 52.06 s
- instances/sec (offline eval throughput, NOT real-time NanoSAM FPS): 9.604

> instances/sec is offline eval throughput (image load + set_image + predict + IoU),
> not the real-time NanoSAM inference FPS.

## Gating

- SMOKE PASS: execution errors=0, valid=20>0, no NaN/Inf IoU. Chain continued.

## Day 3 dogs.jpg Box Prompt regression (re-run)

- image: third_party/nanosam/assets/dogs.jpg (1180x760)
- box prompt: [100,100,850,759], labels [2,3], slot 0
- PASS: selected IoU 0.6619 (matches Day 3 baseline 0.662), mask non-empty
  (242155 nonzero pixels), no NaN/Inf.

## Evidence paths

- results/coco_nanosam_baseline/smoke/      (summary.json, per_instance.jsonl, failed_instances.json, run.log)
- results/coco_nanosam_baseline/quick/      (summary.json, per_instance.jsonl, failed_instances.json, run.log)
- results/coco_nanosam_baseline/standard/   (summary.json, per_instance.jsonl, failed_instances.json, run.log)
- results/coco_nanosam_baseline/baseline_summary.md
- data/eval_manifests/{smoke,quick,standard,full}_manifest.json
- results/nanosam_baseline/box_result.json  (dogs regression)

## Core environment (unchanged)

- torch 2.5.0a0+872d972e41.nv24.08
- tensorrt 10.3.0
- numpy 1.23.5
- torchvision 0.20.0a0+afc54f7
- pycocotools 2.0.11

## Constraints honored

- No engine rebuild; no FP32 ablation; no NanoOWL COCO AP; no INT8; no MP4; no ROS2;
  no stability run; no upstream modification; no remote; no push; no GitHub upload.
- No FULL inference run.
- Did not continue Stage 5C.
