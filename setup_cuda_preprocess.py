# SPDX-License-Identifier: Apache-2.0
"""Build script for the Stage 7B fused CUDA preprocessing extension.

Builds: edge_perception._cuda_preprocess  (a PyTorch C++ extension with a .cu kernel)

Usage:
  .venv/bin/python setup_cuda_preprocess.py build_ext --inplace
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "csrc", "cuda_preprocess")

setup(
    name="edge_perception._cuda_preprocess",
    ext_modules=[
        CUDAExtension(
            name="edge_perception._cuda_preprocess",
            sources=[
                os.path.join(CSRC, "cuda_preprocess.cpp"),
                os.path.join(CSRC, "cuda_preprocess_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
