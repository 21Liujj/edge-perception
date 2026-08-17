# Day 8 — Model Toolchain Deep Dive

## Stage A1

PyTorch / ONNX / TensorRT Conversion Audit

### WHAT

Audited the fixed NanoSAM ResNet18 image encoder model source, local checkpoints, training/distillation path, official ONNX exporter, deployed ONNX graph, TensorRT bindings, external preprocessing, and exact missing conversion links. No model artifact was modified.

### WHY

Establish a reproducible, source-grounded starting contract before independently executing PyTorch -> ONNX -> ONNX Runtime -> TensorRT.

### INPUT

- NanoSAM upstream 653633614b2eb93b06ba3be9adb2aeffb117bd72
- Official offline-transferred resnet18_image_encoder.onnx
- Existing FP16-enabled resnet18_image_encoder.engine
- Local README, Python source, Git history, model caches, and prior project evidence

### OUTPUT

- docs/model_conversion_contract.md
- results/toolchain_audit/nanosam_encoder_onnx_summary.json
- results/toolchain_audit/nanosam_encoder_trt_contract.json
- results/toolchain_audit/pytorch_model_source.md
- results/toolchain_audit/checkpoint_inventory.txt
- results/toolchain_audit/export_path_audit.md
- results/toolchain_audit/audit_summary.md

### PASS criteria

- Identify exact PyTorch class/factory and source-defined architecture.
- Inventory local checkpoint state without downloading.
- Find and document exporter arguments and static contract.
- Run onnx.checker and record real graph statistics.
- Deserialize the current engine read-only and match bindings to ONNX.
- Define a source-grounded A2 starting point without executing A2.

### Actual audit result

Model class and official exporter are available. ONNX checker passed: image FLOAT [1,3,1024,1024] -> image_embeddings FLOAT [1,256,64,64], opset 16, 93 nodes. TensorRT 10.3.0 engine bindings match the ONNX contract and retain FLOAT boundaries while the build enabled FP16 tactics. The local ONNX came from the official README artifact by verified offline transfer; it is structurally consistent with the exporter, but its exact absent training checkpoint cannot be reproduced locally.

Environment remained unchanged: Python 3.10.12; torch 2.5.0a0+872d972e41.nv24.08; torchvision 0.20.0a0+afc54f7; CUDA available; torch CUDA 12.6. Main repository has no remote; upstream was not modified. No packages were installed or changed.

### Blocker

PYTORCH_CHECKPOINT_NOT_LOCAL is the critical blocker. The expected NanoSAM ResNet18 student checkpoint data/models/resnet18/checkpoint.pth is absent. timm is also missing, so model construction currently cannot run. Stage A1 status is BLOCKED only for autonomous PyTorch-to-ONNX reproduction; the audit itself completed.

### Stage A2 starting point

After explicit authorization: recover a fixed official ResNet18 student checkpoint and SHA256, provide pinned compatible timm, instantiate create_model("resnet18"), load checkpoint["model"], eval, establish PyTorch output, export through the official script or a minimal project-local wrapper, validate ONNX and ONNX Runtime parity, and only then build TensorRT. No A2 action was started.

## Stage A2 — PyTorch → ONNX Export Rehearsal

### WHAT

Loaded the local MobileSAM `vit_t` teacher checkpoint into its TinyViT image encoder, generated a deterministic CUDA FP32 PyTorch reference, exported a new teacher-only ONNX graph, ran ONNX checker, inspected real nodes and initializers, and executed the same input in ONNX Runtime CPU.

### WHY

Rehearse a complete, source-grounded checkpoint-to-ONNX workflow while the NanoSAM ResNet18 student checkpoint remains unavailable. The teacher result is kept explicitly separate from the student deployment chain.

### INPUT

- `third_party/nanosam/assets/mobile_sam.pt`, SHA256 `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`
- MobileSAM registry model `vit_t`
- seed 20260817, CUDA FP32 `[1,3,1024,1024]`
- isolated `.venv-toolchain` with inherited Jetson torch/CUDA stack

### OUTPUT

- `docs/pytorch_to_onnx_walkthrough.md`
- `results/toolchain_export_a2/checkpoint_info.json`
- `results/toolchain_export_a2/pytorch_reference.json`
- `results/toolchain_export_a2/exported_onnx_summary.json`
- `results/toolchain_export_a2/graph_operator_inspection.json`
- `results/toolchain_export_a2/pytorch_vs_onnx.json`
- `results/toolchain_export_a2/mobile_sam_export_audit.md`
- `results/toolchain_export_a2/export_summary.md`
- generated, Git-ignored ONNX under `results/toolchain_export_a2/artifacts/`

### RESULT

Checkpoint load passed with no missing or unexpected keys. PyTorch produced finite FP32 `[1,256,64,64]`. ONNX checker passed for a 28,106,961-byte graph with IR 8, opset 16, 2,036 nodes, and 180 initializers. ONNX Runtime 1.20.0 CPU produced the same shape and dtype. CUDA PyTorch versus CPU ORT metrics were max absolute error 0.2841012, mean absolute error 0.0217554, and RMSE 0.0318601; backend differences and near-zero relative-error instability are explicitly documented.

### JD Capability Mapping

- PyTorch checkpoint and state-dict validation
- deterministic GPU reference inference
- ONNX export contract design
- ONNX graph and initializer inspection
- ONNX Runtime parity measurement
- dependency isolation and version control hygiene
- Linux, Python, Git, and deployment toolchain debugging

No existing ONNX or engine changed, no upstream source changed, no TensorRT build occurred, and Stage A3 was not started.
## Stage A3 — Numerical Parity Audit

### WHAT

Ran three controlled FP32 comparisons for the MobileSAM `vit_t` image encoder using one serialized input: PyTorch CPU versus PyTorch CUDA, PyTorch CPU versus ONNX Runtime CPU, and PyTorch CUDA versus ONNX Runtime CPU. Recorded full error distributions, output scale, graph operator risk, export warnings, and decision evidence under `results/toolchain_parity_a3/`.

### WHY

The Stage A2 cross-backend comparison reported unexpectedly large differences but did not isolate device effects from export effects. A same-device CPU comparison was required to determine whether the exported graph changed model semantics.

### CONTROLLED VARIABLES

- Same `float32` input artifact and raw SHA256 for all executions
- Same MobileSAM checkpoint SHA256 and `vit_t` image encoder
- PyTorch `eval()` and `torch.no_grad()`
- No precision conversion, training, distillation, TensorRT, or INT8
- ONNX Runtime `CPUExecutionProvider`
- Existing Stage A2 ONNX used read-only

### RESULT

`PARITY_PASS`

PyTorch CPU versus ONNX Runtime CPU reached relative L2 `1.1574093e-6`, max absolute error `1.1622906e-6`, and cosine similarity `0.999999999999`. PyTorch CPU versus CUDA reached relative L2 `7.1641795e-4`; CUDA versus ONNX Runtime CPU reached `7.1655962e-4`. No CPU-versus-ORT element exceeded `1e-3`.

### ROOT-CAUSE STATUS

The controlled CPU pair shows that ONNX export semantics are numerically faithful. The larger comparisons both cross CPU/CUDA and share nearly the same error signature, so the measurable difference is backend drift. Stage A2's much larger recorded pair did not reproduce with the hash-matched input and reproduced CUDA output statistics; the exact stale or mismatched temporary pairing cannot be reconstructed and is not assigned a speculative cause.

Intermediate localization: `NOT_REQUIRED`. Downstream mask check: `NOT_RUN`. Export warning audit: `NO_CRITICAL_EXPORT_WARNING_FOUND`.

### NEXT STEP

A separately authorized TensorRT stage is numerically justified. A3 stops here and does not build TensorRT artifacts.

### JD CAPABILITY MAPPING

Demonstrates reproducible model-conversion validation, cross-framework numerical debugging, backend isolation, ONNX graph inspection, evidence-based root-cause analysis, and deployment-readiness documentation on Jetson.
