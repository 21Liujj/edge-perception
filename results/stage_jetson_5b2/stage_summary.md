# Stage Jetson-5B-2 — Self-Trained NanoSAM STANDARD500 Quality Gate

## Verdict: UNDERTRAINED_BUT_HEALTHY

## Frozen Protocol
- Manifest: `data/eval_manifests/standard_manifest.json` SHA `19a00a836c2f1d1f...` (match=True)
- 500 instances / 455 unique images / seed 2027 / fixed slot0 confirmed
- Evaluator: `scripts/eval_nanosam_coco.py` (prompt-conditioned instance mask IoU)
- Decoder: `/home/nvidia/jetson-deploy-lab/third_party/nanosam/data/mobile_sam_mask_decoder.engine` SHA `afb5ac4958e6bab2...` unchanged
- Historical baseline engine: SHA `c277e5ca445dc29b...` untouched

## Self-Trained STANDARD500 Result
- mIoU = **0.707340274**
- median=0.776695 P10=0.370367 P90=0.917869
- IoU>=0.50: 0.836 | IoU>=0.75: 0.552
- valid=500 failed=0

## vs Historical Baseline = 0.718127977
- absolute delta (self - baseline) = **-0.010787703** (-1.502%)
- baseline source: HISTORICAL (no rerun; per_instance pairing available)

## Size Breakdown (self / historical / delta)
- small (n=175): 0.586296 vs 0.605320 → -0.019024
- medium (n=166): 0.751911 vs 0.762659 → -0.010748
- large (n=159): 0.794032 vs 0.795796 → -0.001764
- Failure ordering: **small < medium < large** (historical small<medium<large preserved)

## Paired Comparison (500 matched)
- improved=196 degraded=303 equal=1
- mean paired delta=-0.010788 median paired delta=-0.004717
- max improvement +0.2355 / max degradation -0.3751

## Categories (best5 / worst5)
- best: sandwich(6)=0.902, stop sign(6)=0.899, bus(7)=0.871, bear(7)=0.863, toaster(6)=0.857
- worst: skis(6)=0.348, scissors(6)=0.361, dining table(6)=0.390, snowboard(6)=0.392, couch(6)=0.432

## Runtime
- wall 53.2s | encoder set_image calls 455 | reuse 45 | decoder calls 500 | 9.41 inst/s (eval throughput, NOT deployment FPS)

## Quality Gate / Recommendation
- Gate: **UNDERTRAINED_BUT_HEALTHY** | Recommendation (advisory, NOT executed): **RESUME_TO_50**
- Decision threshold is PROJECT-INTERNAL heuristic, not a significance test.

## STOP
No training resume / 50-epoch / FULL COCO / INT8 / NanoOWL / C++ / stability performed.
