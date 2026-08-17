# NanoSAM Algorithm Deep Dive

# Promptable Segmentation

Promptable segmentation predicts a mask conditioned on prompts such as points or boxes. In this project, each evaluation unit is one COCO annotation: its GT bounding box becomes a two-corner prompt and the predicted binary mask is compared with that annotation's GT mask. This is prompt-conditioned instance mask quality, not a full COCO semantic-segmentation benchmark.

Concept boundaries:

- Semantic segmentation assigns a class to each pixel without separating same-class instances.
- Instance segmentation predicts a separate mask and class for each object instance.
- Panoptic segmentation combines exhaustive semantic regions with countable object instances.
- Promptable segmentation predicts masks conditioned on a user or upstream-model prompt and may be class-agnostic.

# NanoSAM Architecture

The accepted deployment uses a NanoSAM ResNet18 image encoder engine and a MobileSAM-compatible FP32 mask decoder engine.

## Image Encoder

The encoder converts a normalized, padded image into a `[1,256,64,64]` embedding. Embeddings are reused for multiple annotation prompts on the same image. B1 and B3 use the accepted FP16-enabled student engine and do not rebuild or alter it.

## Prompt Encoder

A COCO box `[x,y,w,h]` is transformed into top-left and bottom-right point coordinates with labels `[2,3]`. NanoSAM scales coordinates into the encoder coordinate system. No detector output is used, which isolates mask quality from detection error.

## Mask Decoder

The decoder consumes image embeddings, prompt coordinates and labels, an optional previous mask, and the mask-input flag. Its actual engine outputs are FP32 `iou_predictions [1,4]` and `low_res_masks [1,4,256,256]`. The predictor bilinearly upsamples mask logits to the original image dimensions.

## IoU Prediction Head

The four scores are intended to estimate the quality of the corresponding four masks. They are regression outputs, not strict probabilities. B1 evaluates them against measured GT IoU rather than applying classification calibration metrics.

# Why Multiple Masks

A prompt can be ambiguous: a point or box may support different plausible extents, subparts, or object interpretations. The source configures three multimask outputs plus one single-mask output, for four exposed mask tokens. Multiple candidates create quality headroom only if a deployable selector can rank them reliably.

# Mask Selection

- Slot 0 preserves the official-compatible historical project baseline.
- Predicted argmax is deployable because the decoder already emits scores and GT is not used.
- Oracle best-of-4 uses GT IoU and is only a capacity upper bound.

The fixed 500-instance experiment found predicted argmax always selected slot 2. Oracle selected slots 0/1/2/3 for 39.8%/31.6%/18.2%/10.4% of instances. Slot 0 was the oracle choice most often, but only for 39.8%, so alternative candidates contain useful information that the current score ordering fails to expose.

# COCO Results

| Policy | mIoU | Median | P10 | P90 | IoU>=0.50 | IoU>=0.75 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed slot 0 | 0.718128 | 0.783875 | 0.379131 | 0.919229 | 0.846 | 0.590 |
| Predicted argmax | 0.661484 | 0.742365 | 0.248790 | 0.907244 | 0.786 | 0.490 |
| Oracle | 0.748094 | 0.812461 | 0.460853 | 0.925282 | 0.884 | 0.630 |

All policies have 500 valid and zero failed instances. Day 5's `0.718128` fixed-slot baseline remains unchanged. Predicted argmax loses `0.056644` mIoU; oracle provides `0.029966` headroom.

Only categories with at least five samples are summarized. Best fixed-slot categories include stop sign (n=6, 0.905), sandwich (n=6, 0.894), toilet (n=6, 0.881), toaster (n=6, 0.877), and bear (n=7, 0.871). Worst include skis (n=6, 0.330), dining table (n=6, 0.362), couch (n=6, 0.387), scissors (n=6, 0.412), and snowboard (n=6, 0.433). These fixed-subset counts do not justify population-level category claims.

# IoU Head Calibration

Across all 2,000 candidate masks:

- Pearson correlation: `-0.023302`
- Spearman rank correlation: `-0.031903`
- predicted/actual IoU MAE: `0.201619`
- argmax top-1 oracle accuracy: `18.2%` (91/500)

The current decoder scores rank this protocol poorly. Slot 2 has the highest predicted score in every instance but is the actual best in only 91. This is the direct reason predicted argmax is unsuitable here.

# Small Object Analysis

| Policy | Small mIoU | Delta vs fixed |
|---|---:|---:|
| Fixed slot 0 | 0.605320 | 0 |
| Predicted argmax | 0.492933 | -0.112387 |
| Oracle | 0.640204 | +0.034884 |

Oracle shows that selection contributes some small-object loss, but the upper bound remains well below medium (`0.786785`) and large (`0.826446`) oracle performance. The remaining gap may involve embedding resolution, mask capability, or prompt ambiguity. No single mechanism is proven by B1.

# Failure Cases

The saved groups cover fixed-slot worst cases, largest predicted-argmax improvements, largest regressions, and largest oracle headroom. Severe fixed failures include very small objects, thin structures, and prompts whose rectangular extent contains substantial background or competing structures. Predicted argmax's largest regressions are concentrated in examples where slot 0 is already strong but forced slot 2 changes extent sharply. Oracle-headroom cases demonstrate that a better candidate sometimes exists even when slot 0 and slot 2 are both poor.

Visualizations show original image, GT mask, fixed mask, predicted-argmax mask, and oracle mask with slot and score annotations. They are evidence for qualitative review, not additional samples.

# Deployment Decision

`KEEP_FIXED_SLOT0_BASELINE`

Predicted-IoU argmax is computationally simple and adds about `134 us` median selection/extraction overhead in the synchronized microbenchmark, negligible against the complete pipeline. However, it causes a large overall regression and an especially severe small-object regression. Runtime and implementation complexity cannot compensate for incorrect ranking.

The Day 5 fixed-slot result remains the official-compatible baseline. Oracle results identify mask-selection research headroom, but no GT-dependent policy is deployable. A future stage could investigate a validated selector or score-head mismatch, but B1 does not modify the model or deployment.

# Full COCO Evaluation

## Protocol

B3 is **prompt-conditioned instance mask quality evaluation**. It is not COCO semantic-segmentation mIoU and not COCO standard mask AP. The flow is:

```text
Ground-Truth Instance Bounding Box
        -> NanoSAM
        -> Predicted Instance Mask
        -> IoU with that instance's GT mask
```

The evaluation unit is one legal COCO annotation. The image embedding is computed once per image, each annotation's clipped GT box is encoded as corner points with labels `[2,3]`, high-resolution logits are thresholded at zero, and fixed slot 0 remains the primary deployment policy.

This separates the B2 and B3 algorithm boundaries. B2 evaluates `image + COCO category text -> NanoOWL -> bbox` with COCO bbox AP. B3 evaluates `image + GT instance bbox -> NanoSAM -> mask` with instance-mask IoU, intentionally removing detector error. A future B5 joint evaluation must use NanoOWL predicted boxes to measure error propagation.

## Legal Instances

COCO val2017 contains 5,000 images and 36,781 annotations. B3 excludes 446 `iscrowd=1` regions because a crowd region need not represent one unambiguous instance. No non-crowd annotation had an invalid bbox, undecodable/empty mask, invalid category, or missing image under the frozen protocol. The FULL manifest therefore contains 36,335 legal instances across 4,952 images.

Manifest: `data/eval_manifests/nanosam_coco_full_instances.json`
SHA256: `ad8843c78b3ea4c94c368bfbeb7ae26138c761a8adcabb277a4a63600c00a764`

The manifest, evaluator, and fixed-slot policy are frozen for future official/current student versus self-distilled student comparisons. Evaluation protocol must not change after training.

## Full Metrics

| Policy | mIoU | Median | P10 | P90 | IoU>=0.50 | IoU>=0.75 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed slot 0 | 0.703201 | 0.751699 | 0.415992 | 0.910241 | 0.842136 | 0.503454 |
| Predicted-IoU argmax | 0.636687 | 0.699531 | 0.261809 | 0.889355 | 0.743223 | 0.410733 |
| Oracle best-of-4 | 0.730897 | 0.775536 | 0.471877 | 0.917010 | 0.880804 | 0.556158 |

All 36,335 legal instances completed with zero failures, OOM events, or non-finite outputs. The fixed-slot result is the official FULL promptable-segmentation baseline.

## Size Analysis

| Size | Count | Fixed mIoU | Median | IoU>=0.50 | IoU>=0.75 | Oracle mIoU |
|---|---:|---:|---:|---:|---:|---:|
| Small | 15,264 | 0.621048 | 0.653846 | 0.753407 | 0.289505 | 0.647965 |
| Medium | 12,392 | 0.744661 | 0.794296 | 0.899128 | 0.613541 | 0.772387 |
| Large | 8,679 | 0.788489 | 0.850309 | 0.916811 | 0.722549 | 0.817512 |

Small-object mIoU trails medium by `0.123613` and large by `0.167441`. Oracle selection recovers only `0.026917` for small objects, so candidate selection alone does not close the size gap.

## Category Analysis

Best fixed-slot categories are bear (n=71, 0.871639), toaster (n=9, 0.870026), parking meter (n=60, 0.853351), microwave (n=55, 0.847845), and stop sign (n=75, 0.845746). The toaster count is small and should not support a strong population claim.

Worst categories are skis (n=241, 0.345456), fork (n=215, 0.483028), knife (n=325, 0.510497), spoon (n=253, 0.540393), and dining table (n=695, 0.542389). Size is an important confounder for skis (73.9% small), knife (66.5% small), and spoon (66.8% small), but not for dining table (16.4% small). Thin geometry and ambiguous rectangular extent remain observed patterns beyond size alone.

## IoU Distribution

The distribution is saved in `results/algorithm_b3_nanosam_full/iou_distribution.json` and `iou_distribution.png`. Median IoU is `0.751699`; 3.17% of instances are below 0.25, and 12.61% reach at least 0.90.

## Mask Selection at Scale

Predicted-IoU argmax selects slot 2 for 36,330/36,335 instances and slot 3 for five. It reduces mIoU by `0.066514` versus fixed slot 0. Oracle slots 0/1/2/3 occur in 39.90%/33.54%/17.11%/9.45% of instances and provide `0.027696` mean headroom. B1's conclusion is therefore not a small-subset artifact: `KEEP_FIXED_SLOT0_BASELINE` remains valid.

## Predicted IoU Head

Across 145,340 candidate masks, predicted versus actual IoU has Pearson `-0.026129`, Spearman `-0.037662`, MAE `0.202410`, and argmax-to-oracle top-1 accuracy `17.10%`. Ranking is not merely poorly calibrated; it is ineffective overall. Small-object correlation is negative (Pearson `-0.186672`), while large objects show weak positive association (Pearson `0.274039`) but only 20.29% top-1 selection accuracy. No size bucket supports replacing fixed slot 0 with predicted argmax.

## Failure Taxonomy

Mined evidence and 20 visualizations support these observations: small-object loss, thin or fragmented structures, boundary leakage or under-coverage, nearby-instance confusion, occlusion, and boxes containing substantial background. These are observed output patterns. Possible explanations include representation resolution, low-resolution mask logits, soft boundaries, and prompt ambiguity, but B3 does not establish architecture-level causes.

## Standard vs Full

FULL minus STANDARD fixed-slot mIoU is `-0.014927`. Size deltas are small `+0.015728`, medium `-0.017998`, and large `-0.007307`. However, FULL median is `-0.032176` and IoU>=0.75 ratio is `-0.086546` relative to STANDARD. The balanced 500-instance subset was useful for direction and reproduced the size ordering, but it is optimistic for high-quality-mask frequency and is not a sufficient substitute for the FULL population baseline.

## Implications for Training

Future distillation or training should preserve this manifest and evaluator. The primary measured targets are small-object representation, thin/fragmented foreground recovery, boundary and nearby-instance separation, weak high-count categories, and the IoU head's candidate ranking. Training must compare the official/current student and self-distilled student under the same fixed slot 0 protocol before any deployment-policy change.

## Evaluation Engineering

FULL evaluation used 4,952 encoder calls and 36,335 decoder calls, proving image-level embedding reuse. Wall time was 1,966.77 s, with 2.518 images/s and 18.474 instances/s. Peak RSS was 1.084 GB; PyTorch CUDA peak reserved memory was 65.0 MB. This is end-to-end evaluation throughput including mask transfer and metric computation, not deployment FPS.

# JD Capability Mapping

B1 and B3 cover image segmentation, instance-segmentation concepts, promptable segmentation, mask IoU, COCO data handling, size/category-aware evaluation, failure analysis, model-output interpretation, and large-scale evaluation engineering. They do not claim training, distillation, INT8, C++, or CUDA work.
