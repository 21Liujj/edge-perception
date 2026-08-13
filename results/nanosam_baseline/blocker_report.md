# Stage 3B — NanoSAM TensorRT Baseline: BLOCKED (OFFLINE_TRANSFER_REQUIRED)

Date: 2026-08-13
NanoSAM upstream: third_party/nanosam @ 653633614b2eb93b06ba3be9adb2aeffb117bd72 (main), clean

## Status: OFFLINE_TRANSFER_REQUIRED (no Engine build, no inference)

## Part 一 — Prepare NanoSAM package: PASS

- Read-only audit of README.md / setup.py / examples / export scripts completed (Stage 3A).
- nanosam made importable via `python setup.py develop` (in-place editable egg-link into venv,
  no source modification, no new deps; install_requires=[] per setup.py).
  - `from nanosam.utils.predictor import Predictor` -> OK
- Core packages unchanged (see version table below). timm NOT installed (per rule).
- Upstream source NOT modified; upstream `git status` clean (egg-info is upstream-gitignored).

## Part 二 — Image Encoder ONNX: OFFLINE_TRANSFER_REQUIRED

- resnet18_image_encoder.onnx official source = Google Drive, file ID
  14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR
  (https://drive.google.com/file/d/14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR/view?usp=drive_link)
- Google Drive NOT reachable from Jetson:
    curl https://drive.google.com      -> http 000 (Connection timed out, port 443)
    curl https://docs.google.com       -> http 000 (Connection timed out)
    curl https://drive.usercontent.google.com -> http 000 (Connection timed out)
  (DNS resolves, but TCP 443 times out. GitHub and Hugging Face are reachable; Google Drive is not.)
- No download performed, no mirror used.
- Target Jetson path: third_party/nanosam/data/resnet18_image_encoder.onnx

## Part 三 — Mask Decoder ONNX: BLOCKED (local export needs timm; Drive fallback unreachable)

- Local export (official script) attempted:
    python -m nanosam.tools.export_sam_mask_decoder_onnx --model-type=vit_t \
        --checkpoint=assets/mobile_sam.pt --output=data/mobile_sam_mask_decoder.onnx
- FAILED with:
    ModuleNotFoundError: No module named 'timm'
  Cause: NanoSAM vit_t image-encoder backbone = TinyViT
  (nanosam/mobile_sam/modeling/tiny_vit_sam.py:15 imports `from timm.models.layers import ...`).
  The export path constructs the full Sam (incl. TinyViT image encoder) from assets/mobile_sam.pt.
- Per Stage 3B rule 8, NOT installing timm or any other package.
- README official alternative = download ONNX from Google Drive, file ID
  1jYNvnseTL49SNRx9PDcbkZ9DwsY8up7n
  (https://drive.google.com/file/d/1jYNvnseTL49SNRx9PDcbkZ9DwsY8up7n/view?usp=drive_link)
  - Google Drive unreachable (same as Part 二).
  - Target Jetson path: third_party/nanosam/data/mobile_sam_mask_decoder.onnx

## Part 四 — Engine builds: NOT STARTED
  (blocked on ONNX availability per rule: "不继续 Engine Build")

## Part 五 — Inference (Box / Point prompt): NOT STARTED

## Part 六 — Legality checks: N/A (no inference)
  Box Prompt: NOT RUN. Point Prompt: NOT RUN. No NaN/Inf to report (no inference).

## Part 七 — Records

### Required offline transfers (official files, no mirrors)
| # | Artifact | Official source | Drive file ID | Jetson target path | Purpose |
|---|----------|-----------------|---------------|--------------------|---------|
| 1 | resnet18_image_encoder.onnx | NanoSAM README -> Google Drive | 14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR | third_party/nanosam/data/resnet18_image_encoder.onnx | image encoder ONNX (trtexec --fp16 input) |
| 2 | mobile_sam_mask_decoder.onnx | NanoSAM README -> Google Drive | 1jYNvnseTL49SNRx9PDcbkZ9DwsY8up7n | third_party/nanosam/data/mobile_sam_mask_decoder.onnx | mask decoder ONNX (trtexec FP32 input); needed because local export requires timm |

Notes:
- assets/mobile_sam.pt is already in the repo (offline, ~40MB) — but cannot be used locally to
  produce the mask decoder ONNX without timm. If timm were permitted, item 2 could be produced
  locally without any network.
- After both ONNX files are offline-transferred to data/, Stage 3B can resume:
    image decoder:   trtexec --onnx=data/resnet18_image_encoder.onnx --saveEngine=data/resnet18_image_encoder.engine --fp16
    mask decoder:    trtexec --onnx=data/mobile_sam_mask_decoder.onnx --saveEngine=data/mobile_sam_mask_decoder.engine \
                       --minShapes=point_coords:1x1x2,point_labels:1x1 \
                       --optShapes=point_coords:1x1x2,point_labels:1x1 \
                       --maxShapes=point_coords:1x10x2,point_labels:1x10
  then run examples/basic_usage.py (box [100,100,850,759] -> points [[100,100],[850,759]] labels [2,3]) on assets/dogs.jpg.

### Core package versions (unchanged)
| Package | Version |
|---|---|
| torch | 2.5.0a0+872d972e41.nv24.08 |
| torchvision | 0.20.0a0+afc54f7 |
| numpy | 1.23.5 |
| tensorrt | 10.3.0 |
| torch2trt | importable (Day 1 install, unchanged) |
| transformers | 4.46.3 |
| Pillow | 12.3.0 |
| OpenCV | 4.10.0 |
| timm | MISSING (not installed this stage; install prohibited) |

### Constraints honored
- No NanoOWL+NanoSAM joint run, no benchmark, no INT8, no ROS2, no MP4, no training.
- Upstream NanoSAM NOT modified (git status clean).
- No remote / push / GitHub upload.
- No core package upgraded/downgraded; no new deps installed (timm not installed).
- Did NOT continue to Stage 3C.

### Engine / artifact state
- No *.engine, *.onnx, mask outputs produced.
- third_party/nanosam/data/ is empty.
- results/nanosam_baseline/ holds this report.

## Outcome
Stage 3B cannot complete on-Jetson because both required ONNX files are sourced from Google Drive,
which is unreachable from this Jetson, and the local mask-decoder export requires timm (install prohibited).
Provide the two official ONNX files offline to resume.
