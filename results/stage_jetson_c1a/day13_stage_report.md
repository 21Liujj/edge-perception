# Day 13 — Stage Jetson-C1A-R (INT8 Calibration Cache)

## Goal
Generate a reusable TensorRT INT8 calibration cache for the NanoSAM ResNet18
epoch50 self-distilled model, using train2017 (not val2017) to keep the
evaluation set clean for later Stage C1B/C2 accuracy comparisons.

## Why train2017, not val2017
C0 originally proposed val2017 for calibration. That would leak evaluation
data into the quantization process — any INT8 vs FP16 mAP comparison run
later against val2017 would be biased in INT8's favor. Switched calibration
source to train2017 (117287+ images) with a disjoint sample from anything
used for eval.

## Steps performed
1. SSH to 5090 (`connect.westd.seetacloud.com:38718`), audited train2017
   (118287 images).
2. Deterministic sample of 512 images, seed 20260824, no replacement.
3. Verified SHA256 of every sampled file, packed into
   `nanosam_train2017_calib512_seed20260824.tar.gz` (80MB).
4. Transferred 5090 -> local -> Jetson (no sshpass installed on Jetson, so
   direct pull wasn't possible).
5. Verified all 512 images against manifest SHA256 on Jetson: 512/512 pass.
6. Implemented `SimpleCalibrator(trt.IInt8EntropyCalibrator2)` using the real
   deployment preprocessing path (`nanosam.utils.predictor.preprocess_image`,
   size=1024).
7. Ran full INT8 calibration build (`run_calibration.py`): 512/512 batches,
   473.9s calibration + 16.3s post-processing, exit code 0.
8. Re-ran the same build a second time to confirm cache reuse: log showed
   "Reading Calibration Cache" / "Generated calibration scales using
   calibration cache", zero batches reprocessed, ~3s wall clock.
9. Confirmed cache SHA256 identical before/after reuse run
   (`9689497ac73606952abfc113162907655ce80f79642e77ffd2a33daa22ac1487`).

## Explicitly out of scope this stage
- STANDARD500 evaluation
- Full COCO evaluation
- Formal FP16 vs INT8 benchmark
- Mixed precision
- C++ implementation

## Result: PASS

Artifacts: `results/stage_jetson_c1a/` on Jetson
(`c1a_evidence.json`, `calibration_train2017_512.cache`,
`calibration_run1_generate.log.gz`, `checksums.txt`, plus the pre-existing
`calibration_design.json`, `environment.json`, `source_dataset_audit.json`,
`transfer_plan.json`, `stage_summary.md`).
