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

The encoder converts a normalized, padded image into a `[1,256,64,64]` embedding. Embeddings are reused for multiple annotation prompts on the same image. B1 uses the accepted FP16-enabled student engine and does not rebuild or alter it.

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

# JD Capability Mapping

B1 covers image segmentation, promptable segmentation, IoU measurement, mask selection, quality evaluation, failure analysis, object-size analysis, model-output interpretation, and deployment-oriented algorithm trade-offs. It does not claim training, distillation, INT8, or C++ work.
