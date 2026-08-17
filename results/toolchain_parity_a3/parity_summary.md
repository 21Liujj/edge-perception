# Stage A3 Numerical Parity Summary

## Decision

**PARITY_PASS**

The controlled comparison isolates ONNX export semantics from CPU/CUDA backend effects. PyTorch CPU FP32 and ONNX Runtime CPU FP32 agree to approximately `1e-6` relative L2, while comparisons that cross from CPU to CUDA show a small, consistent drift of approximately `7.16e-4` relative L2. The evidence attributes the measurable difference to backend-dependent numerical execution rather than an ONNX graph semantic error.

## Controlled comparisons

| Comparison | Max absolute error | Mean absolute error | RMSE | Relative L2 | Cosine similarity | Pearson correlation |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch CPU vs PyTorch CUDA | 1.1559427e-3 | 5.1467617e-5 | 7.3245054e-5 | 7.1641795e-4 | 0.999999743755 | 0.999999743549 |
| PyTorch CPU vs ORT CPU | 1.1622906e-6 | 8.8386128e-8 | 1.1833108e-7 | 1.1574093e-6 | 0.999999999999 | 0.999999999999 |
| PyTorch CUDA vs ORT CPU | 1.1568367e-3 | 5.1479038e-5 | 7.3261547e-5 | 7.1655962e-4 | 0.999999743641 | 0.999999743435 |

The fixed input is `results/toolchain_parity_a3/input_reference.npy`, shape `[1, 3, 1024, 1024]`, dtype `float32`, with raw SHA256 `fda4c3d2ad945c3441a7bd79de1711655d5324d2921831549003a769892aba93`. Every backend loaded this same artifact.

## Scale and thresholds

The PyTorch CPU output standard deviation is `0.1021883908`; its absolute-value P50, P90, P95, and P99 are `0.0424494`, `0.1618136`, `0.2225315`, and `0.3707185`. CPU-versus-ORT mean absolute error is `8.65e-7` of output standard deviation. No CPU-versus-ORT element exceeds `1e-3`. CPU-versus-CUDA has one element per 1,048,576 above `1e-3` and none above `1e-2`.

## Audit status

- Export warning audit: `NO_CRITICAL_EXPORT_WARNING_FOUND`
- Intermediate localization: `INTERMEDIATE_LOCALIZATION_NOT_REQUIRED`
- Downstream mask check: `DOWNSTREAM_CHECK_NOT_RUN`
- Operator audit: standard ONNX domain only; zero ATen fallback and zero custom-domain nodes

Stage A2's previously recorded large CUDA-versus-ORT errors were not reproduced despite using an input whose raw SHA256 matches the A2 record and reproducing the A2 CUDA output statistics. Those old metrics are therefore not valid evidence of current export divergence; the exact stale or mismatched temporary pairing that produced them cannot be reconstructed from retained A2 artifacts.
