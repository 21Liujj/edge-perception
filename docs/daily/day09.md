# Day 9 — Perception Algorithm Deep Dive

## Stage B1 — NanoSAM Mask Selection Ablation

### WHAT

Evaluated fixed slot 0, predicted-IoU argmax, and GT oracle best-of-4 over the unchanged Day 5 STANDARD 500-instance manifest. Each instance ran one decoder call; all four masks and scores were retained for offline policy, calibration, size, category, failure-case, and runtime analysis.

### WHY

Determine whether the historical slot-0 baseline limits segmentation quality and whether the decoder's predicted IoU scores provide a safe deployable selector.

### INPUT

- Manifest: `data/eval_manifests/standard_manifest.json`
- SHA256: `19a00a836c2f1d1f236ea34175d12c2afc670b9cb5136b30e6b39bf7ca78f015`
- 500 instances, 455 unique images
- COCO GT box prompts
- Accepted NanoSAM FP16 image encoder and FP32 mask decoder

### OUTPUT

Evidence is under `results/algorithm_b1/`, including 500-row CSV, policy and calibration summaries, category/size analyses, failure mining, runtime measurements, and 20 representative visualizations.

### RESULT

Fixed slot 0 reproduced Day 5 exactly at mIoU `0.718128`. Predicted argmax fell to `0.661484` (`-0.056644`) and small-object mIoU fell from `0.605320` to `0.492933`. Oracle reached `0.748094`, giving `+0.029966` overall headroom and `+0.034884` for small objects.

Predicted argmax selected slot 2 for all 500 instances and matched the oracle only 91 times (18.2%). Across 2,000 candidates, predicted/actual IoU Pearson correlation was `-0.023302` and MAE was `0.201619`.

### DEPLOYMENT DECISION

`KEEP_FIXED_SLOT0_BASELINE`

Predicted argmax overhead is small relative to the full pipeline, but quality is unstable and materially worse: 126 instances improve and 374 degrade. Historical Day 5 evidence is not overwritten.

### NEXT STEP

Stop after B1. A separately authorized stage may investigate why the deployed IoU head is slot-biased or evaluate a validated selector. No training, TensorRT rebuild, INT8, CUDA optimization, or NanoOWL experiment is started.
