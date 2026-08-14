// SPDX-License-Identifier: Apache-2.0
// Stage 7B — C++ binding for the fused CUDA preprocessing kernel.
// Pybind module registration only; the kernel lives in cuda_preprocess_kernel.cu.
#include <torch/extension.h>

void fused_preprocess_resize_launch(
    torch::Tensor dst, torch::Tensor src,
    int64_t rh, int64_t rw,
    double mean0, double mean1, double mean2,
    double std0, double std1, double std2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_preprocess_resize", &fused_preprocess_resize_launch,
          "Fused uint8 HWC -> bilinear resize -> normalize -> NCHW float32 (CUDA)");
}
