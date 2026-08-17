# Stage A4 Summary

## Decision

`TRT_CONVERSION_PASS`

TensorRT 10.3 parsed and built the Stage A2 MobileSAM teacher ONNX into an FP32 engine with TF32 disabled and a batch-1 optimization profile. TensorRT Python Runtime deserialization and inference passed using the exact Stage A3 input tensor.

## Artifacts

- ONNX: 28,106,961 bytes
- Engine: 32,403,988 bytes
- Engine SHA256: `e8d4b5c0a1175f41e721a1e1124b40bdd8759b4a1c35f5c31d42818a53751269`
- Input: `[1,3,1024,1024] float32`
- Output: `[1,256,64,64] float32`

## Parity

| Pair | Max abs | Mean abs | RMSE | Relative L2 | Cosine |
|---|---:|---:|---:|---:|---:|
| PyTorch CUDA vs ORT CPU | 1.1568367e-3 | 5.1479038e-5 | 7.3261547e-5 | 7.1655962e-4 | 0.999999743641 |
| PyTorch CUDA vs TRT FP32 | 1.1569858e-3 | 5.1490759e-5 | 7.3277486e-5 | 7.1671552e-4 | 0.999999743527 |
| ORT CPU vs TRT FP32 | 1.5497208e-6 | 1.0029438e-7 | 1.3172473e-7 | 1.2884142e-6 | 0.999999999999 |

No pair has an element above `1e-2`; ORT CPU versus TensorRT has no element above `1e-3`. Outputs have the expected shape and dtype and contain no NaN or Inf.

## Runtime baseline

20 warmups and 100 measured TensorRT Python Runtime iterations:

- mean: 111.0050 ms
- P50: 110.9705 ms
- P95: 111.2115 ms
- P99: 111.3970 ms

This is a correctness baseline, not an optimization result.
