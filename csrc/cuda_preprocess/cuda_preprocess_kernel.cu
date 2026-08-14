// SPDX-License-Identifier: Apache-2.0
// Stage 7B — Fused CUDA preprocessing kernel for NanoSAM image encoder.
//
// Fuses (single kernel launch, one thread per output pixel):
//   uint8 HWC source (on device) -> bilinear resize (half-pixel centers)
//   -> per-channel normalize (mean/std) -> HWC->NCHW layout -> float32 output.
//
// Output dst is (1,3,size,size) float32 device, channel stride = size*size.
// The resized region [0:rh,0:rw] is placed top-left (pad region pre-zeroed by caller,
// or cleared by an inline clear when rh*rw < size*size).
// NOT bit-exact vs PIL (PIL default = BICUBIC); correctness judged by tensor error
// stats + COCO QUICK mIoU delta. Sampling formula in preprocess_contract.md.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

__global__ void fused_preprocess_kernel(
    const uint8_t* __restrict__ src,  // (src_h, src_w, 3) HWC uint8, device
    int src_h, int src_w,
    float* __restrict__ dst,          // (1,3,size,size) NCHW float32, device
    int size, int rh, int rw,
    float mean0, float mean1, float mean2,
    float std0, float std1, float std2)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= rw || oy >= rh) return;

    // Bilinear, half-pixel centers, boundary clamp.
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(rw);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(rh);
    float src_x = (static_cast<float>(ox) + 0.5f) * scale_x - 0.5f;
    float src_y = (static_cast<float>(oy) + 0.5f) * scale_y - 0.5f;
    if (src_x < 0.0f) src_x = 0.0f;
    if (src_y < 0.0f) src_y = 0.0f;
    if (src_x > src_w - 1.0f) src_x = static_cast<float>(src_w - 1);
    if (src_y > src_h - 1.0f) src_y = static_cast<float>(src_h - 1);

    int x0 = static_cast<int>(src_x);
    int y0 = static_cast<int>(src_y);
    int x1 = x0 + 1; if (x1 > src_w - 1) x1 = src_w - 1;
    int y1 = y0 + 1; if (y1 > src_h - 1) y1 = src_h - 1;
    float wx = src_x - static_cast<float>(x0);
    float wy = src_y - static_cast<float>(y0);

    const int b00 = (y0 * src_w + x0) * 3;
    const int b01 = (y0 * src_w + x1) * 3;
    const int b10 = (y1 * src_w + x0) * 3;
    const int b11 = (y1 * src_w + x1) * 3;
    const float w00 = (1.0f - wy) * (1.0f - wx);
    const float w01 = (1.0f - wy) * wx;
    const float w10 = wy * (1.0f - wx);
    const float w11 = wy * wx;

    float v0 = w00 * src[b00 + 0] + w01 * src[b01 + 0] + w10 * src[b10 + 0] + w11 * src[b11 + 0];
    float v1 = w00 * src[b00 + 1] + w01 * src[b01 + 1] + w10 * src[b10 + 1] + w11 * src[b11 + 1];
    float v2 = w00 * src[b00 + 2] + w01 * src[b01 + 2] + w10 * src[b10 + 2] + w11 * src[b11 + 2];

    const int chan_stride = size * size;          // channel offset in (3,size,size)
    const int pos = oy * size + ox;               // (row oy, col ox) within channel plane
    dst[0 * chan_stride + pos] = (v0 - mean0) / std0;
    dst[1 * chan_stride + pos] = (v1 - mean1) / std1;
    dst[2 * chan_stride + pos] = (v2 - mean2) / std2;
}

// dst: (1,3,size,size) float32 cuda. src: (src_h,src_w,3) uint8 cuda (HWC RGB).
// rh,rw: resized region dims (aspect-preserved). Caller is responsible for the
// zero-pad region (dst must be zero outside [0:rh,0:rw]).
void fused_preprocess_resize_launch(
    torch::Tensor dst, torch::Tensor src,
    int64_t rh, int64_t rw,
    double mean0, double mean1, double mean2,
    double std0, double std1, double std2)
{
    TORCH_CHECK(dst.is_cuda(), "dst must be cuda");
    TORCH_CHECK(src.is_cuda(), "src must be cuda");
    TORCH_CHECK(src.scalar_type() == torch::kUInt8, "src must be uint8");
    TORCH_CHECK(dst.scalar_type() == torch::kFloat32, "dst must be float32");
    TORCH_CHECK(dst.dim() == 4 && dst.size(1) == 3, "dst must be (1,3,H,W)");
    TORCH_CHECK(src.dim() == 3 && src.size(2) == 3, "src must be (H,W,3)");
    TORCH_CHECK(dst.size(2) == dst.size(3), "dst must be square (size,size)");

    int src_h = static_cast<int>(src.size(0));
    int src_w = static_cast<int>(src.size(1));
    int size = static_cast<int>(dst.size(2));
    int rh_i = static_cast<int>(rh);
    int rw_i = static_cast<int>(rw);

    dim3 block(32, 8);
    dim3 grid((rw_i + block.x - 1) / block.x, (rh_i + block.y - 1) / block.y);
    fused_preprocess_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint8_t*>(src.data_ptr()),
        src_h, src_w, dst.data_ptr<float>(), size, rh_i, rw_i,
        (float)mean0, (float)mean1, (float)mean2,
        (float)std0, (float)std1, (float)std2);
}
