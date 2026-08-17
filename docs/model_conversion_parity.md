# MobileSAM Model Conversion Numerical Parity

## Why numerical parity matters

A successful ONNX export must preserve model behavior, not only produce a graph that passes structural validation. Numerical comparison also has to separate conversion error from expected implementation differences between CPU and CUDA kernels. This audit uses controlled backend pairs so that a cross-device discrepancy is not incorrectly assigned to ONNX export semantics.

## Controlled Input

All three executions load the same immutable artifact: `results/toolchain_parity_a3/input_reference.npy`.

| Property | Value |
|---|---|
| Shape | `[1, 3, 1024, 1024]` |
| Dtype | `float32` |
| Generation | `torch.randn`, CUDA generator seed `20260817` |
| Raw SHA256 | `fda4c3d2ad945c3441a7bd79de1711655d5324d2921831549003a769892aba93` |
| Minimum / maximum | `-4.9204912` / `4.7816596` |
| Mean / standard deviation | `-0.00146395` / `0.99975181` |

The raw hash equals the Stage A2 input hash. No backend regenerated random data. PyTorch models used the same checkpoint, `eval()` mode, and `torch.no_grad()`; ONNX Runtime used `CPUExecutionProvider`.

## PyTorch CPU vs CUDA

| Metric | Result |
|---|---:|
| Max absolute error | `1.1559427e-3` |
| Mean absolute error | `5.1467617e-5` |
| RMSE | `7.3245054e-5` |
| Relative L2 error | `7.1641795e-4` |
| Cosine similarity | `0.999999743755` |
| Pearson correlation | `0.999999743549` |
| P50 / P90 / P95 absolute error | `3.6091e-5` / `1.1820e-4` / `1.5134e-4` |
| P99 / P99.9 absolute error | `2.3405e-4` / `3.9478e-4` |

No element exceeds `1e-2`; one of 1,048,576 elements exceeds `1e-3`. This is small CPU/CUDA backend drift, consistent with different kernel implementations and reduction order.

## PyTorch CPU vs ONNX Runtime CPU

| Metric | Result |
|---|---:|
| Max absolute error | `1.1622906e-6` |
| Mean absolute error | `8.8386128e-8` |
| RMSE | `1.1833108e-7` |
| Relative L2 error | `1.1574093e-6` |
| Cosine similarity | `0.999999999999` |
| Pearson correlation | `0.999999999999` |
| P50 / P90 / P95 absolute error | `5.9605e-8` / `1.7881e-7` / `2.3842e-7` |
| P99 / P99.9 absolute error | `3.5763e-7` / `5.3644e-7` |

No element exceeds `1e-3`. Holding the device class constant reduces relative L2 error by approximately 619 times compared with PyTorch CPU versus CUDA. This controlled pair is the direct export-semantic test and shows near-numerical-precision agreement.

## PyTorch CUDA vs ONNX Runtime CPU

| Metric | Result |
|---|---:|
| Max absolute error | `1.1568367e-3` |
| Mean absolute error | `5.1479038e-5` |
| RMSE | `7.3261547e-5` |
| Relative L2 error | `7.1655962e-4` |
| Cosine similarity | `0.999999743641` |
| Pearson correlation | `0.999999743435` |

This cross-backend result closely matches PyTorch CPU versus CUDA. ONNX Runtime does not introduce a second, larger error signature.

The Stage A2 file recorded max absolute error `0.2841012`, mean absolute error `0.0217554`, and RMSE `0.0318601`. A3 does not reproduce those values: the fixed input has the same recorded raw hash, and A3 reproduces the Stage A2 PyTorch CUDA output statistics. The retained A2 evidence does not identify the exact stale or mismatched temporary tensor/output pairing, so its historical comparison is classified as non-reproducible rather than assigned an unsupported root cause. It is not evidence that the current ONNX graph has a semantic defect.

## Output scale

The PyTorch CPU reference spans `-0.5274308` to `0.5370337`, with standard deviation `0.1021884`.

| Absolute output percentile | Magnitude |
|---|---:|
| P50 | `0.0424494` |
| P90 | `0.1618136` |
| P95 | `0.2225315` |
| P99 | `0.3707185` |
| Maximum | `0.5370337` |

CPU-versus-CUDA mean absolute error is `5.04e-4` of output standard deviation. CPU-versus-ORT CPU mean absolute error is `8.65e-7` of output standard deviation. A large elementwise relative error near a zero-valued reference would be unstable and is retained for audit only; it is not used as the pass/fail decision metric.

## Export warnings

The retained Stage A2 transcript contains timm registry-name overwrite warnings, tracer warnings for `assert L == H * W` and `B = len(x)`, and a Slice constant-folding limitation for non-unit steps. The export contract fixes the input spatial dimensions at 1024 by 1024, and the resulting graph retains a symbolic batch axis.

There is no observed unsupported-operator warning, ATen fallback, custom-domain operator, precision-conversion warning, or successful-run exporter exception. The graph has 2,036 standard-domain nodes, passes `onnx.checker`, and receives `NO_CRITICAL_EXPORT_WARNING_FOUND`.

## Operator risk

The graph contains 29 Conv, 60 MatMul, 10 Softmax, and 23 Erf nodes. Layer normalization is decomposed into ReduceMean, Sub, Pow, Add, Sqrt, and Div; GELU uses Erf. Conv and MatMul reductions, decomposed normalization, Erf/GELU, and Softmax are plausible CPU/CUDA drift candidates because backend implementations can differ in accumulation order or transcendental approximations.

These are candidates, not localized root causes. Gather, Reshape, and Transpose are indexing or layout operations and have low intrinsic numerical risk. All nodes use the standard ONNX domain, with no ATen fallback or custom operator.

## Intermediate localization

Status: `INTERMEDIATE_LOCALIZATION_NOT_REQUIRED`.

The direct PyTorch CPU versus ORT CPU pair has relative L2 error `1.1574e-6`, P99 absolute error `3.5763e-7`, and no value above `1e-3`. Dumping and exposing intermediate graph outputs would expand the audit without a material divergence to localize.

## Final conclusion

**PARITY_PASS**

The ONNX graph is numerically faithful for the fixed FP32 contract. The measurable difference is small CPU/CUDA backend drift, while the controlled CPU-to-CPU export comparison agrees near numerical precision. The evidence does not support `PARITY_ACCEPTABLE_WITH_BACKEND_DRIFT` as a weaker result because backend drift is both expected and small, and export-semantic parity itself passes strongly.

Downstream mask validation is `DOWNSTREAM_CHECK_NOT_RUN`. Injecting ORT CPU embeddings into the CUDA predictor would require a new mixed-runtime path and is not proportionate after embedding parity passed. Intermediate localization is `NOT_REQUIRED`. A separately authorized TensorRT stage is numerically justified, but no TensorRT build or next-stage work is part of A3.

## Troubleshooting decision chain

1. Reuse one serialized FP32 input and verify its raw SHA256 before every execution.
2. Recreate the MobileSAM `vit_t` image encoder from the same checkpoint and enforce clean state-dict loading, `eval()`, and `no_grad()`.
3. Compare PyTorch CPU with PyTorch CUDA to establish device-dependent baseline drift.
4. Compare PyTorch CPU with ONNX Runtime CPU to isolate ONNX export semantics while holding the device class constant.
5. Compare PyTorch CUDA with ONNX Runtime CPU to determine whether the cross-backend signature is explained by the baseline drift.
6. Contextualize absolute errors against output distribution and stable norm-based metrics; do not fail on maximum relative error near zero.
7. Inspect actual ONNX operators, domains, checker status, and export warnings before attributing a discrepancy to an operator.
8. Localize intermediate activations only when the controlled same-device pair has a material divergence.
9. Run downstream semantic checks only when embedding metrics leave a decision-relevant ambiguity and the check does not require an unrelated runtime pipeline.
10. Record non-reproducible historical metrics as unresolved evidence rather than inventing a root cause.
