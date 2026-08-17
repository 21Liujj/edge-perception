# Failure Taxonomy

This taxonomy is based on the mined FULL cases and saved visualizations. Visual inspection supports observations, not architecture-level causal claims.

## Small-object failures

**OBSERVED:** The small bucket has lower mIoU than medium and large, and mined small-object cases include near-zero masks or poor object coverage.

**POSSIBLE EXPLANATION:** Limited spatial representation, few object pixels after resizing, and box/background ambiguity may contribute. This experiment does not isolate those mechanisms.

## Thin and fragmented structures

**OBSERVED:** Several low-IoU cases have narrow, disconnected, or articulated foreground geometry inside a much larger rectangular prompt.

**POSSIBLE EXPLANATION:** Fine structures may be difficult to recover from the embedding and low-resolution mask logits. No architectural root cause is established here.

## Boundary leakage and under-coverage

**OBSERVED:** Saved cases show masks that extend into prompt-box background or omit valid GT regions.

**POSSIBLE EXPLANATION:** Similar local appearance, soft boundaries, occlusion, and rectangular prompt ambiguity can all be consistent with this pattern.

## Nearby-instance confusion

**OBSERVED:** Some boxes include neighboring objects or overlapping same-category instances, and the prediction may cover the wrong extent.

**POSSIBLE EXPLANATION:** The GT box may not uniquely identify boundaries where instances are adjacent or occluded. B3 does not include detector error.

## Mask-selection headroom

**OBSERVED:** Oracle best-of-4 improves mean IoU by 0.027696, while predicted-IoU argmax changes it by -0.066514 and reaches oracle top-1 only 0.171047 of the time.

**POSSIBLE EXPLANATION:** Alternative candidates contain useful masks, but the current predicted-IoU ordering is not a reliable selector under this protocol.
