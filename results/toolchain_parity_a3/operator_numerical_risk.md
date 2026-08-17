# Operator Numerical Risk Audit

Source graph: `results/toolchain_export_a2/artifacts/mobile_sam_image_encoder_exported.onnx`, SHA256 `f8d9ac5da0f9521aaf067b35dbe1eac78c50c9d7a0eef72dc3e07db5eb6f3ec7`.

| Operator | Actual count | Numerical-risk candidate assessment |
|---|---:|---|
| Conv | 29 | Candidate for CPU/CUDA drift because kernels may use different accumulation order and implementations. |
| MatMul | 60 | Candidate for CPU/CUDA drift for the same reduction-order reason, especially in attention projections. |
| Softmax | 10 | Candidate because exponentiation and reduction order can amplify small upstream differences; not a proven root cause. |
| LayerNormalization | 0 | Not fused. Its semantics are decomposed into ReduceMean, Sub, Pow, Add, Sqrt, and Div. |
| Erf | 23 | Candidate because CPU and CUDA transcendental implementations can differ slightly; used by decomposed GELU. |
| Resize | 0 | Not present and cannot explain this graph's result. |
| Gather | 52 | Indexing, not intrinsically approximate; low numerical risk. |
| Reshape | 126 | Layout-only and not a direct numerical-error source. |
| Transpose | 96 | Layout-only and not a direct numerical-error source. |
| Mul | 98 | May propagate existing error but is not independently identified. |
| Add | 139 | May propagate or cancel upstream drift; not independently identified. |
| Div | 45 | Candidate near small denominators and in decomposed normalization. |

All 2,036 nodes use the standard ONNX domain; there is no ATen fallback or custom-domain operator. PyTorch CPU versus ORT CPU has relative L2 `1.1574e-6`, while PyTorch CPU versus CUDA has relative L2 `7.1642e-4`. Conv, MatMul, normalization reductions, Erf/GELU, and Softmax are therefore backend-drift candidates, not proven root causes. Export semantics are not implicated by the controlled CPU comparison.
