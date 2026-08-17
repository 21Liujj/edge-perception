# ONNX to TensorRT Build Contract

## Source artifact

- ONNX: `results/toolchain_export_a2/artifacts/mobile_sam_image_encoder_exported.onnx`
- Size: 28,106,961 bytes
- SHA256: `f8d9ac5da0f9521aaf067b35dbe1eac78c50c9d7a0eef72dc3e07db5eb6f3ec7`
- IR version: 8
- Opset: 16
- Nodes: 2,036

## Tensor contract

The actual ONNX graph declares:

- input `image`: `float32 [batch_size, 3, 1024, 1024]`
- output `image_embeddings`: `float32 [Addimage_embeddings_dim_0, 256, 64, 64]`

The input and output batch dimensions are symbolic. TensorRT therefore requires an optimization profile that bounds the runtime input shape. Stage A4 validates only the Jetson deployment case `batch=1`, so profile 0 is deliberately fixed:

- min: `image=1x3x1024x1024`
- opt: `image=1x3x1024x1024`
- max: `image=1x3x1024x1024`

This does not benchmark or promise other batch sizes.

## Build precision

TensorRT 10.3 `trtexec --help` reports that TF32 is enabled by default and supports `--noTF32`. The parity engine was built with `--noTF32`, without `--fp16`, and without INT8. Its intended precision is FP32 with TF32 disabled.

## Build command

```text
/usr/src/tensorrt/bin/trtexec   --onnx=results/toolchain_export_a2/artifacts/mobile_sam_image_encoder_exported.onnx   --saveEngine=results/toolchain_trt_a4/artifacts/mobile_sam_teacher_fp32.engine   --minShapes=image:1x3x1024x1024   --optShapes=image:1x3x1024x1024   --maxShapes=image:1x3x1024x1024   --noTF32   --verbose
```

The full output is retained in `trtexec_build.log`.
