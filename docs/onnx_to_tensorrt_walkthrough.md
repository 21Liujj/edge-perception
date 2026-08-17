# ONNX to TensorRT Walkthrough

# Why TensorRT

TensorRT converts a framework-neutral model representation into a serialized execution artifact optimized for a target NVIDIA GPU and runtime environment. Stage A4 closes the correctness path from PyTorch through ONNX and ONNX Runtime to TensorRT on Jetson Orin Nano. It deliberately excludes FP16, INT8, and performance optimization.

# ONNX Contract

The Stage A2 MobileSAM teacher image encoder is ONNX IR 8, opset 16, with 2,036 nodes. Its SHA256 is `f8d9ac5da0f9521aaf067b35dbe1eac78c50c9d7a0eef72dc3e07db5eb6f3ec7`.

- `image`: FP32 `[batch_size,3,1024,1024]`
- `image_embeddings`: FP32 `[Addimage_embeddings_dim_0,256,64,64]`

The exact Stage A3 input is reused without preprocessing or regeneration.

# Dynamic Shape and Optimization Profile

The real graph uses symbolic batch dimensions. TensorRT needs an optimization profile to define legal input ranges and provide shapes for tactic selection. Profile 0 fixes min, opt, and max to `1x3x1024x1024`, preserving the dynamic ONNX contract while limiting this engine to the batch-1 deployment case under test.

# ONNX Parser

The parser reads ONNX operators, tensors, initializers, attributes, and shape information into a TensorRT network definition. The verbose log confirms IR 8 and opset 16 parsing with no parser error or unsupported-operator warning. Parser success alone does not prove runtime or numerical correctness, so later stages remain mandatory.

# Builder

The builder combines the parsed network, optimization profile, precision policy, memory limits, and target device capabilities. The A4 engine uses TensorRT 10.3, FP32, and `--noTF32`. Neither `--fp16` nor INT8 was enabled. The build completed in 154.511 seconds.

# Tactics

A tactic is a candidate implementation for a layer or fused region. The verbose log records timing runners, multiple candidate tactics, and fastest-tactic selection. These choices depend on the target GPU, TensorRT version, shapes, precision, and available libraries. They are not equivalent to ONNX operators.

# Engine

The serialized engine is `results/toolchain_trt_a4/artifacts/mobile_sam_teacher_fp32.engine`, 32,403,988 bytes, SHA256 `e8d4b5c0a1175f41e721a1e1124b40bdd8759b4a1c35f5c31d42818a53751269`.

It exposes two FP32 IO tensors and one profile:

- input `image`: `[1,3,1024,1024]`
- output `image_embeddings`: `[1,256,64,64]`

The source ONNX has 2,036 nodes; the deserialized engine reports 327 layers. The build log shows pointwise fusion, convolution/residual fusion, Myelin `ForeignNode` regions, reformatting, and tactic selection. `ONNX Operator != TensorRT Layer != CUDA Kernel`.

# Runtime

The project script uses `tensorrt.Runtime`, `deserialize_cuda_engine`, and `ICudaEngine`. Deserialization passed for the independently generated engine. `trtexec` benchmark output is not used as correctness inference.

# Execution Context

The script creates an `IExecutionContext`, sets the batch-1 input shape, binds Torch CUDA tensor addresses with the TensorRT tensor API, and calls `execute_async_v3` on a CUDA stream. The output is copied back only after stream synchronization. It contains no NaN or Inf.

# PyTorch / ORT / TRT Parity

All runtimes read the same FP32 tensor contract.

| Pair | Max abs | Mean abs | RMSE | Relative L2 | Cosine |
|---|---:|---:|---:|---:|---:|
| PyTorch CUDA vs ORT CPU | `1.1568367e-3` | `5.1479038e-5` | `7.3261547e-5` | `7.1655962e-4` | `0.999999743641` |
| PyTorch CUDA vs TensorRT | `1.1569858e-3` | `5.1490759e-5` | `7.3277486e-5` | `7.1671552e-4` | `0.999999743527` |
| ORT CPU vs TensorRT | `1.5497208e-6` | `1.0029438e-7` | `1.3172473e-7` | `1.2884142e-6` | `0.999999999999` |

PyTorch CUDA versus TensorRT matches the already established GPU/backend drift scale. ORT CPU and TensorRT are near numerical precision despite crossing runtimes. This supports `TRT_CONVERSION_PASS`.

# Engine Portability

ONNX stores a portable graph and weights. A TensorRT engine stores an optimized execution artifact tied to the build's target GPU architecture, TensorRT environment, precision policy, and optimization profiles. It should not be treated as universally portable. A new target or runtime version should rebuild and revalidate the engine.

# Common Failure Layers

A customer conversion failure should be isolated in order:

```text
PyTorch reference
        |
        v
ONNX export?
        |
        v
onnx.checker?
        |
        v
ORT parity?
        |
        v
TRT parser?
        |
        v
Unsupported operator?
        |
        v
Shape/profile?
        |
        v
Builder/tactic?
        |
        v
Engine load?
        |
        v
Runtime inference?
        |
        v
Numerical parity?
```

Each boundary has different evidence: framework output, graph validity, reference-runtime output, parser diagnostics, profile legality, builder logs, deserialization status, execution status, and numerical metrics. Skipping boundaries makes root-cause claims unreliable.

```text
PyTorch
   |
   v
ONNX
   |
   v
ONNX Runtime
   |
   v
TensorRT Parser
   |
   v
Builder
   |
   v
Engine
   |
   v
Runtime
   |
   v
GPU
```
