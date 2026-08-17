#!/usr/bin/env python3
"""Generate Stage B3 human-readable evidence and documentation from FULL JSON outputs."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results/algorithm_b3_nanosam_full"
def load(n):return json.loads((OUT/n).read_text())
def f(x,n=6):return "n/a" if x is None else f"{x:.{n}f}"
def main():
    s=load("stage_b3_machine_summary.json"); m=s["full_metrics"]; z=s["size"]; sel=s["mask_selection"]; pi=s["predicted_iou"]; cmp=load("standard_vs_full.json"); cat=load("category_analysis.json"); csa=load("category_size_analysis.json"); rt=s["runtime"]; audit=s["protocol_audit"]
    worst=cat["worst_10"][:5]; best=cat["best_10"][:5]
    taxonomy=f'''# Failure Taxonomy

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

**OBSERVED:** Oracle best-of-4 improves mean IoU by {f(sel['oracle_headroom_vs_fixed'])}, while predicted-IoU argmax changes it by {f(sel['predicted_argmax_delta_vs_fixed'])} and reaches oracle top-1 only {f(pi['all']['argmax_top1_accuracy'])} of the time.

**POSSIBLE EXPLANATION:** Alternative candidates contain useful masks, but the current predicted-IoU ordering is not a reliable selector under this protocol.
'''
    (OUT/"failure_taxonomy.md").write_text(taxonomy)
    summary=f'''# Stage B3 Summary

Decision: `FULL_SEGMENTATION_BASELINE_ESTABLISHED`

## Protocol

This is prompt-conditioned instance mask quality evaluation, not semantic-segmentation mIoU and not COCO mask AP. Each legal non-crowd COCO annotation supplies a GT bounding-box prompt to NanoSAM; fixed mask slot 0 is compared with that annotation's GT mask.

B3 intentionally removes detector error. B2 measures NanoOWL text-prompted COCO bbox AP; future B5 will measure NanoOWL predicted box to NanoSAM mask error propagation.

## Legal Instances

- Total annotations: {audit['total_annotations']}
- Excluded crowd: {audit['excluded_crowd']}
- Excluded invalid bbox: {audit['excluded_invalid_bbox']}
- Excluded invalid mask: {audit['excluded_invalid_mask']}
- Other excluded: {audit['other_excluded']}
- Legal instances: {audit['legal_instances']} across {audit['legal_unique_images']} images

## Full Metrics

- Fixed slot0 mIoU: {f(m['mean_iou'])}
- Median: {f(m['median_iou'])}; P10/P90: {f(m['p10'])}/{f(m['p90'])}
- IoU>=0.50: {f(m['iou_ge_0_50_ratio'])}; IoU>=0.75: {f(m['iou_ge_0_75_ratio'])}

## Size Analysis

| Size | Count | mIoU | Median | IoU>=0.5 | IoU>=0.75 |
|---|---:|---:|---:|---:|---:|
'''+"\n".join(f"| {b} | {z[b]['count']} | {f(z[b]['mIoU'])} | {f(z[b]['median'])} | {f(z[b]['iou_ge_0_5'])} | {f(z[b]['iou_ge_0_75'])} |" for b in ("small","medium","large"))+f'''

## Mask Selection at Scale

Fixed/predicted-argmax/oracle mIoU: {f(sel['fixed_slot0']['mean_iou'])} / {f(sel['predicted_iou_argmax']['mean_iou'])} / {f(sel['oracle_best_of_4']['mean_iou'])}. Predicted argmax slot distribution: `{json.dumps(sel['predicted_argmax_slot_distribution'],sort_keys=True)}`.

## Predicted IoU Head

Candidate-level Pearson/Spearman/MAE: {f(pi['all']['pearson'])} / {f(pi['all']['spearman'])} / {f(pi['all']['mae'])}. Argmax-to-oracle top-1 accuracy: {f(pi['all']['argmax_top1_accuracy'])}.

## Standard vs Full

FULL minus STANDARD mIoU: {f(cmp['full_minus_standard']['mIoU'])}; small/medium/large: {f(cmp['full_minus_standard']['small_mIoU'])} / {f(cmp['full_minus_standard']['medium_mIoU'])} / {f(cmp['full_minus_standard']['large_mIoU'])}. These absolute deltas quantify how closely the balanced 500-instance subset approximates the COCO legal-instance population.

## Runtime and Stability

{rt['legal_instances']} instances, {rt['unique_images']} images, {rt['encoder_calls']} encoder calls, {rt['decoder_calls']} decoder calls, {f(rt['wall_time_seconds'],2)} s wall time, {f(rt['images_per_second'],3)} images/s, {f(rt['instances_per_second'],3)} instances/s, {rt['failed_count']} failed. This is evaluation throughput, not deployment FPS.

## Implications for Training

Future B4 should retain this manifest, evaluator, and fixed-slot policy. Training priorities should follow the measured size gap, weak categories with adequate sample counts, boundary/thin-structure failures, and the predicted-IoU ranking mismatch. B3 does not train or modify the model.
'''
    (OUT/"stage_b3_summary.md").write_text(summary)
    print(json.dumps({"summary":"written","best5":[[x['category_name'],x['count'],x['slot0_mIoU']] for x in best],"worst5":[[x['category_name'],x['count'],x['slot0_mIoU']] for x in worst]},indent=2))
if __name__=="__main__":main()
