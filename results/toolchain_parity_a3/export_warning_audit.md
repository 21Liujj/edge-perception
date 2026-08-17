# ONNX Export Warning Audit

Stage A2 did not retain a standalone export log. This audit uses warnings captured in the Stage A2 execution transcript and the exported graph.

Observed warnings:

- TinyViT timm registry overwrite warnings for local `tiny_vit_*` registrations; these concern naming, not graph semantics.
- `TracerWarning` for `assert L == H * W`; input spatial dimensions are fixed at 1024x1024 in this contract.
- `TracerWarning` for `B = len(x)`; batch extraction is traced, while the graph retains a symbolic input batch axis.
- Constant-folding warning for Slice with steps other than 1; the exporter retained the operation instead of folding it.

Not observed: unsupported operator warning, ATen fallback, custom operator, precision-conversion warning, or exporter exception in the successful run. Graph inspection confirms 2,036 standard-domain nodes, zero ATen nodes, and zero custom-domain nodes. `onnx.checker` passes and controlled CPU-to-CPU parity is near numerical precision.

**NO_CRITICAL_EXPORT_WARNING_FOUND**
