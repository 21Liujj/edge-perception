# Stage Jetson-5B-3 — FP32 vs FP16 Downstream Quality Control

## Purpose
Separate the 10-epoch **training gap** from the **FP16 deployment-precision gap**.
Same model, frozen STANDARD500 protocol; only the image encoder backend swapped
(TRT FP16 -> ORT CPU FP32). No upstream edit, no rebuild, no training.

## Artifacts (frozen, read-only)
- ONNX `aacbea870b8eca91...` (match=True)
- TRT engine `ab2d9962b4cce724...` (match=True)
- decoder `afb5ac4958e6bab2...` (match=True)
- Engine SHA process note: 5B-1 `0d63103c` -> 5B-2 `ab2d9962` (concurrent trtexec builds; future build system should use temp-output->validate->atomic-rename). Frozen here = `ab2d9962`.

## ORT FP32 STANDARD500
- mIoU = **0.707379324** | median 0.776226 P10 0.370908 P90 0.919480
- >=0.50 0.834 | >=0.75 0.552 | valid 500 failed 0
- wall 578.0s (0.87 inst/s; CPU ORT, quality control not perf bench)

## FP16 vs FP32
- FP16 (TRT) mIoU = **0.707340274** | FP32 (ORT) mIoU = **0.707379324**
- FP16_DOWNSTREAM_DELTA (FP16 - FP32) = **-0.000039049**
- Paired (n=500): FP16 higher 176 / FP16 lower 176 / equal 148
  mean delta -0.000039 | median +0.000000
  | max abs 0.009950 | P95 abs 0.001928

## Size Sensitivity (FP32 / FP16 / delta FP16-FP32)
- small n=175: FP32 0.586319 | FP16 0.586296 | Δ -0.000023
- medium n=166: FP32 0.751945 | FP16 0.751911 | Δ -0.000034
- large n=159: FP32 0.794094 | FP16 0.794032 | Δ -0.000062

## Quality Decomposition (engineering approximation, nonlinear decoder)
- total gap (baseline - FP16) = +0.010788
- FP16 precision gap (FP32 - FP16) = +0.000039
- training/model gap (baseline - FP32) = +0.010749
- fraction of total gap attributable to FP16 = 0.4% (approx)

## Gate
- FP16_QUALITY_IMPACT = **NEGLIGIBLE** (|delta|=0.000039)
- Main conclusion = **TRAINING_GAP_DOMINANT** | recommendation (advisory) = **RESUME_TO_50**

## Key takeaway
Encoder embedding numerical parity (5B-1 cosine 0.99985) does NOT by itself prove
downstream mask-quality parity; this stage measures that mask quality directly.
Result: FP16 cost on downstream mIoU is negligible (+0.000039); the self-trained
regression vs baseline is dominated by the training/model gap, not FP16.
