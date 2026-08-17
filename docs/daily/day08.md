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
