# Stage 7B — NanoSAM Preprocessing Contract

Frozen from actual source, NOT from memory:
- `third_party/nanosam/nanosam/utils/predictor.py:preprocess_image`
- `edge_perception/runtime/nanosam_optimized.py` (Stage 7A wrapper)
- TensorRT engine binding introspection (`resnet18_image_encoder.engine`)

## Input Contract

- **Type**: `PIL.Image.Image` (RGB) — `set_image` also accepts `np.ndarray HxWx3 uint8 RGB`
  (converted via `PIL.Image.fromarray`).
- **Color order**: RGB. No BGR<->RGB swap exists in the pipeline (input is RGB end to end).
- **Source pixel dtype**: uint8 [0,255].
- **Aspect handling**: aspect = width/height.
  - if aspect >= 1: rw = size, rh = int(size / aspect)
  - else: rh = size, rw = int(size * aspect)
  (integer truncation, matches upstream `int(size/aspect)` / `int(size*aspect)`.)

## Transformation Contract (upstream baseline)

1. **Resize**: `image.resize((rw, rh))` — PIL `Image.resize` with **NO resample arg**.
   Empirically verified on this board (Pillow 12.3.0): the default == **BICUBIC**
   (==3), NOT bilinear. Output uint8 RGB, shape (rh, rw, 3).
2. **HWC→CHW + dtype**: `np.asarray(pil_resized)` → `torch.from_numpy(...).permute(2,0,1)`
   → `.float()`  (uint8 HWC → float32 CHW).
3. **Normalize** (per-channel, ImageNet-style but SAM values):
   mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375]
   `(x - mean[c]) / std[c]`
4. **Pad**: `torch.zeros((1,3,size,size))`; `image_tensor[0,:,:rh,:rw] = normalized`.
   Resize region placed top-left; rest zero. (1,3,size,size) float32.
5. **H2D**: `.cuda()` → device float32.

## Output Contract (TensorRT-ready)

- shape `(1, 3, size, size)`, `size = 1024`
- dtype **float32** (FP32) — confirmed from engine binding below.
- layout NCHW, RGB channel order.
- device: cuda.

## TensorRT Binding Contract

Encoder engine `third_party/nanosam/data/resnet18_image_encoder.engine`:
| tensor | mode | shape | dtype |
|---|---|---|---|
| `image` | INPUT | (1,3,1024,1024) | FLOAT (FP32) |
| `image_embeddings` | OUTPUT | (1,256,64,64) | FLOAT (FP32) |

NB: encoder engine input is **FP32**, not FP16 (the *engine weights/compute* are FP16;
the input tensor binding is FP32). So the fused kernel output dtype = FP32.

## Numerical-semantics decisions for the fused CUDA kernel

- **Resize sampling**: PIL default is BICUBIC. Matching PIL's exact bicubic+antialias
  filter is complex. For the v1 kernel I implement **bilinear resize with the
  half-pixel center coordinate convention + boundary clamp**, and document the exact
  sampling formula. This is a *deliberate, measured* deviation: the kernel is NOT
  bit-exact to PIL, which is allowed ("不能要求像素级 bit-exact"). Correctness is
  judged by tensor error stats + COCO QUICK mIoU delta (§5/§10). If COCO delta is
  non-negligible the verdict becomes REJECTED_NUMERICAL_MISMATCH and a bicubic kernel
  would be the next step.

  Sampling formula (bilinear, half-pixel centers):
  ```
  scale_x = src_w / dst_w ; scale_y = src_h / dst_h
  src_x = (dst_x + 0.5) * scale_x - 0.5   ; clamped to [0, src_w-1]
  src_y = (dst_y + 0.5) * scale_y - 0.5   ; clamped to [0, src_h-1]
  x0=floor(src_x), x1=min(x0+1,src_w-1); wx=src_x-x0
  y0=floor(src_y), y1=min(y0+1,src_h-1); wy=src_y-y0
  out = (1-wy)*((1-wx)*I(y0,x0)+wx*I(y0,x1)) + wy*((1-wx)*I(y1,x0)+wx*I(y1,x1))
  ```
  applied per RGB channel, in [0,255] uint8 → float during sampling.

- **Normalize**: exact same mean/std/operation, fused into the kernel (no separate pass).
- **Pad**: kernel writes only [0:rh,0:rw]; output buffer pre-zeroed once and reused
  (zero-pad region stays zero across frames since the written region is `rh*rw` and
  the trailing region is never touched — but to be safe vs prior frame's larger image,
  the wrapper zero-fills the pad region each frame OR the kernel writes a full clear.
  Decision: wrapper does `image_tensor_gpu.zero_()` only when the previous frame's
  rh*rw > current; simplest correct choice = kernel handles a `clear` flag. Final:
  wrapper zeros the (1,3,size,size) buffer before kernel only if needed; v1 zeroes
  every frame for correctness, measured in benchmark.)
- **dtype**: uint8 src → float32 mid (in-kernel) → float32 out (FP32 binding). No FP16
  conversion needed (binding is FP32).

## Build environment (confirmed, NOT blocked)

- nvcc 12.6 at `/usr/local/cuda/bin/nvcc` (CUDA_HOME=/usr/local/cuda)
- g++ 11.4
- torch 2.5.0a0+nv24.08, `torch.utils.cpp_extension.BuildExtension` available
- ncu (Nsight Compute 2024.3.1) at `/usr/local/cuda-12.6/bin/ncu` — available for §13
- PIL Pillow 12.3.0 (old API, `Image.BILINEAR`/`Image.BICUBIC` int consts; default=BICUBIC)
