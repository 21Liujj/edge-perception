# TensorRT Conversion Audit

## Conversion path

The 2,036-node ONNX graph was read by the TensorRT ONNX parser and converted into a TensorRT network definition. The builder applied profile 0, evaluated tactics using the default tactic sources and local timing cache, serialized an engine, and the TensorRT Python Runtime deserialized it into an `ICudaEngine`. An `IExecutionContext` then bound CUDA tensor addresses and executed inference with `execute_async_v3`.

## Observed optimization evidence

The verbose log directly shows:

- pointwise fusion sequences such as Div, Erf, Add, and Mul represented as fused `PWN(...)` layers;
- convolution plus residual Add fusion in named layers;
- `ForeignNode`/Myelin regions spanning multiple ONNX operations;
- reformatting copy nodes and layout changes;
- multiple measured tactics and a selected fastest tactic;
- an engine with 327 TensorRT layers, not 2,036 layers;
- 29,656,320 bytes of weights memory, 107,243,008 bytes of scratch memory, and 191,129,088 bytes of activation memory reported by the builder.

These observations prove that the source graph was transformed and fused. They do not imply a one-to-one relation among graph operators, engine layers, and launches.

## Operator, layer, and kernel distinction

`ONNX Operator != TensorRT Layer != CUDA Kernel`.

An ONNX operator is a portable graph semantic. A TensorRT layer is an optimized engine-level representation after parsing and graph transformation. A CUDA kernel is an execution implementation selected by a tactic or generated for a fused region. TensorRT can perform constant folding, fusion, layout optimization, and tactic selection. Only the fusion, reformatting, Myelin regions, tactic timing, and layer count above are claimed as observed for this build; other transformations are general TensorRT capabilities.

## Serialization and portability

The ONNX file is a cross-framework graph plus weights. The 32,403,988-byte engine is an optimized serialized execution artifact for the current GPU architecture, TensorRT 10.3 environment, FP32 precision policy, and batch-1 profile. It is not universally portable. The engine reports standard capability and no hardware compatibility level.

## Decision

`TRT_CONVERSION_PASS`

Build, deserialization, execution context creation, inference, finite output checks, shape checks, and numerical parity all passed. PyTorch CUDA versus TensorRT relative L2 is `7.1671552e-4`, matching the known cross-backend scale. ORT CPU versus TensorRT is substantially closer at `1.2884142e-6` relative L2.
