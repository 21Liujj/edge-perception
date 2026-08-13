# -*- coding: utf-8 -*-
"""Stage 1I - minimal torch2trt (PyTorch -> TensorRT) smoke test, FP32 only.

Network: Conv2d -> ReLU (fixed seed, CUDA).
Flow:    PyTorch forward -> torch2trt convert (FP32) -> TensorRT forward -> compare.
No plugins, no install, no engine persistence, no network.
"""
import json
import os
import random
import sys

import torch
import torch.nn as nn

import torch2trt


def main() -> int:
    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
    ).to(device).eval()

    # Fixed input.
    x = torch.randn(1, 3, 64, 64, device=device, generator=torch.Generator(device).manual_seed(seed))
    input_shape = tuple(x.shape)

    # PyTorch forward.
    with torch.no_grad():
        y_pt = model(x)
    pytorch_out_shape = tuple(y_pt.shape)

    # torch2trt conversion, FP32 only (use_pool disabled, fp16_mode off).
    model_trt = torch2trt.torch2trt(
        model,
        [x],
        fp16_mode=False,
        int8_mode=False,
        use_python=True,
        max_batch_size=1,
    )

    # TensorRT forward.
    with torch.no_grad():
        y_trt = model_trt(x)
    trt_out_shape = tuple(y_trt.shape)

    max_abs_error = float(torch.max(torch.abs(y_pt - y_trt)).item())

    conversion_pass = (
        pytorch_out_shape == trt_out_shape
        and torch.isfinite(y_trt).all().item()
    )

    result = {
        "stage": "1I",
        "network": "Conv2d->ReLU",
        "precision": "FP32",
        "seed": seed,
        "input_shape": input_shape,
        "pytorch_output_shape": pytorch_out_shape,
        "tensorrt_output_shape": trt_out_shape,
        "max_abs_error": max_abs_error,
        "conversion": "PASS" if conversion_pass else "FAIL",
        "torch2trt_commit": os.environ.get("TORCH2TRT_COMMIT", "4e820ae31b4e35d59685935223b05b2e11d47b03"),
    }

    print(json.dumps(result, indent=2))

    out_dir = os.path.join("results", "torch2trt_smoke")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "smoke_torch2trt.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "smoke_torch2trt.txt"), "w") as f:
        f.write(json.dumps(result, indent=2) + "\n")

    return 0 if conversion_pass else 1


if __name__ == "__main__":
    sys.exit(main())
