# Stage 7B-Freeze — Experimental Fused CUDA Preprocess Prototype Contract

Read from actual current source (NOT memory):
- `csrc/cuda_preprocess/cuda_preprocess_kernel.cu` (the .cu kernel)
- `edge_perception/runtime/cuda_preprocess.py` (CudaPreprocessor + CudaPreprocessPredictor)
- TensorRT engine binding introspection (`resnet18_image_encoder.engine`)

## Input

- **Type**: `PIL.Image.Image` RGB (or `np.ndarray HxWx3 uint8 RGB`).
- **dtype**: uint8 [0,255].
- **color order**: RGB (no BGR swap).
- **max source dim**: ≤ 2048 (`max_src`, staging/src buffers are 2048×2048×3 uint8).

## Host-side work (unavoidable minimum, in `prepare_source`)

- PIL/ndarray → `np.ascontiguousarray(np.asarray(image, uint8))` → uint8 HWC numpy.
  **No resize, no normalize, no layout change on host.**
- uint8 HWC → pinned staging buffer (reused) → `copy_(non_blocking=True)` into a reused
  device uint8 HWC buffer (`_src_gpu`), on an explicit `torch.cuda.Stream`.

## CUDA kernel — fused operations (single launch, one thread per output pixel)

The kernel `fused_preprocess_kernel` fuses, **in one pass**:
1. **resize**: bilinear, **half-pixel center** coordinate convention + boundary clamp.
   - `src_x = (ox+0.5)*src_w/rw - 0.5`, clamped to [0, src_w-1]; same for y.
   - NOT PIL default (PIL default == **BICUBIC**). This is a deliberate, measured
     deviation (see tensor_numerical_comparison.json).
2. **per-channel normalize**: `(pixel - mean[c]) / std[c]`
   mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375] — SAME constants as upstream.
3. **HWC → NCHW layout**: reads HWC uint8, writes NCHW float32 (channel stride = size²).
4. **dtype conversion**: uint8 → float32, fused into the sample+normalize arithmetic.
- Resize target: aspect-preserved `rh×rw` placed top-left in (size,size); pad region
  outside [0:rh,0:rw] kept zero (cleared by wrapper when the region shrinks vs prev frame).
- grid/block: block(32,8)=256 threads, grid over (rw,rh); one thread = one output pixel.
  No shared-memory tiling / texture / Tensor Core (intentionally simple v1).

## Output (TensorRT-ready device tensor)

- **dtype**: float32 (FP32) — matches engine binding (see below).
- **shape**: `(1, 3, size, size)`, `size = 1024` → `(1,3,1024,1024)`.
- **layout**: NCHW, RGB channel order.
- **device**: cuda; reused preallocated tensor `image_tensor_gpu`.

## TensorRT encoder binding requirement

`third_party/nanosam/data/resnet18_image_encoder.engine` (FP16 engine, FP32 input binding):
| tensor | mode | shape | dtype |
|---|---|---|---|
| `image` | INPUT | (1,3,1024,1024) | FLOAT (FP32) |
| `image_embeddings` | OUTPUT | (1,256,64,64) | FLOAT |

Kernel output (FP32 (1,3,1024,1024)) is fed **directly** to the encoder on the same
stream — no intermediate copy / dtype conversion.

## Build env (confirmed, NOT blocked)

nvcc 12.6 (`/usr/local/cuda/bin/nvcc`), g++ 11.4, torch 2.5.0a0+nv24.08,
`torch.utils.cpp_extension.BuildExtension`; built via `setup_cuda_preprocess.py`.
Output `.so`: `edge_perception/_cuda_preprocess.cpython-310-aarch64-linux-gnu.so`
(GITIGNORED — not committed).
