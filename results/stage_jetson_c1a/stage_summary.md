# Stage Jetson-C1A — INT8 Calibration Cache Generation

## Final: BLOCKED (TRAIN2017_NOT_LOCAL)

## 1. Environment (verified)
- Jetson Orin, Python 3.10.12, PyTorch 2.5.0a0+872d972e41.nv24.08, TensorRT 10.3.0, CUDA available
- Git status: clean working tree (commit 8ebd6ed)
- NanoSAM upstream: unchanged

## 2. Source ONNX (verified)
- Path: `models/self_distilled/epoch50/nanosam_resnet18_epoch049.onnx`
- Expected SHA256: `906d8fd752aa2f4dc28d353adb20a462387c9414c4ddf01c1fca93ddf3f31dd7`
- Status: To be verified in actual calibration run

## 3. COCO Dataset Audit

**Searched paths**:
- `/home/nvidia/jetson-deploy-lab/data`
- `/home/nvidia/data` (does not exist)
- `/data` (does not exist)
- `~/data` (does not exist)

**Found**:
- ✅ val2017: `data/coco/val2017/` (5000 images)
- ❌ train2017: NOT FOUND

**Blocker**: train2017 is NOT available on local Jetson device.

## 4. Calibration Source Correction (C0 → C1A)

**Original C0 design** (commit 8ebd6ed):
- Calibration source: COCO val2017
- Rationale: "already on device, in-distribution for encoder"

**Corrected design** (C1A):
- Calibration source: COCO train2017
- Rationale: **Strict evaluation isolation** — STANDARD500 and FULL COCO eval both use val2017. Even though PTQ calibration does not use segmentation GT, to avoid any calibration/evaluation distribution overlap, Module C must use train2017 for calibration.

**Final protocol**:
```
train2017 → knowledge distillation training + INT8 calibration
val2017   → STANDARD500 + FULL evaluation
```

This ensures:
1. No image-level overlap between calibration and evaluation sets
2. Clean separation of tuning (calibration) and validation (evaluation) distributions
3. Conservative deployment validation (if INT8 works on unseen val2017, it generalizes)

## 5. Transfer Plan (REQUIRED)

To unblock C1A, one of the following must happen:

### Option A: Transfer 512 train2017 images from AutoDL 5090

**Source**: AutoDL 5090 server (bjb2:33884, already has full COCO train2017 for distillation training)

**Steps**:
1. On 5090: Sample 512 train2017 images (seed 20260824, deterministic random)
2. Create manifest: `train2017_calib_512_manifest.json` (image IDs, SHA256s, seed)
3. Tar archive: `coco_train2017_calib512.tar.gz` (~50-100 MB for 512 JPEGs)
4. Transfer to Jetson: `scp` or `rsync` from 5090 → Jetson
5. Extract to: `data/coco/train2017_calib512/`
6. Verify manifest SHA256s
7. Resume C1A calibration

**Pros**: Minimal transfer (512 images vs 118k full train2017), deterministic sampling on 5090 ensures reproducibility
**Cons**: Requires 5090 access and ~5-10min transfer time

### Option B: Transfer full train2017 to Jetson

**Source**: AutoDL 5090 or COCO downloads

**Steps**:
1. Download COCO train2017 (19 GB, 118k images)
2. Transfer to Jetson: `data/coco/train2017/`
3. Resume C1A with local sampling

**Pros**: Full dataset available for future experiments
**Cons**: 19 GB transfer (~20-60min depending on bandwidth), 19 GB Jetson storage

### Option C: Use pre-sampled calibration subset (if available)

If a canonical 512-image train2017 calibration subset exists in project history, reuse it.

**Recommendation**: **Option A** (transfer 512 images only) — minimal transfer, unblocks C1A immediately, deterministic.

## 6. Calibration Design (Ready, Pending Dataset)

Once train2017 (512 or full) is available:

**Manifest**: `data/calibration/nanosam_int8_train2017_512.json`
- Seed: 20260824
- Sample count: 512
- Source: train2017 (deterministic random, no stratification)
- ONNX SHA: `906d8fd7...`
- Preprocessing: `nanosam.utils.predictor.preprocess_image`

**Calibrator**: IInt8EntropyCalibrator2 (TRT 10.3 Python API)
- Batch size: 1 or 4 (to be determined based on Jetson 8GB memory)
- Preprocessing: Deployment-identical `preprocess_image(image, size=1024)` → [1,3,1024,1024] fp32 CUDA

**Cache**: `models/self_distilled/epoch50/int8/calibration_train2017_512.cache`
- Metadata: `calibration_metadata.json` (ONNX SHA, manifest SHA, TRT version, preprocessing ID)
- Provenance guard: Cache reuse requires exact match on ONNX + manifest + preprocessing + TRT 10.3

## 7. Out of Scope (Honored)

- ❌ STANDARD500 evaluation
- ❌ FULL COCO evaluation
- ❌ INT8 quality comparison
- ❌ Mixed precision search
- ❌ C++ implementation
- ❌ Power mode tuning
- ❌ Using val2017 for calibration (corrected)

## Next Steps (After Unblock)

1. **User decision**: Execute Option A, B, or C from transfer plan
2. **Resume C1A**: With train2017 available, re-run Stage C1A script
3. **C1B**: Build deployment INT8 engine using calibration cache
4. **C2**: STANDARD500 quality gate (INT8 vs FP16 0.715650)

## STOP

Stage C1A cannot proceed without train2017. Awaiting user action.
