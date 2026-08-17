# MobileSAM Teacher Export Audit

## Scope

Stage A2 rehearses the local MobileSAM teacher image encoder conversion only. It does not export, train, alter, or infer NanoSAM student weights, and it does not build TensorRT.

## Fixed Inputs

- Checkpoint: `third_party/nanosam/assets/mobile_sam.pt`
- Model registry key: `vit_t`
- Encoder: `nanosam.mobile_sam.modeling.tiny_vit_sam.TinyViT`
- Checkpoint SHA256: `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`
- Checkpoint structure: `collections.OrderedDict`, 439 tensor entries
- Load contract: no missing keys and no unexpected keys
- Reference seed: `20260817`
- Reference tensor: CUDA FP32 `[1,3,1024,1024]`

The full model was set to evaluation mode. The image encoder forward ran under `torch.no_grad()`. The reference output is FP32 `[1,256,64,64]` with no NaN or Inf. Input and output statistics are in `pytorch_reference.json`; the full input tensor was retained only in `/tmp` for parity and is not a repository artifact.

## Isolated Toolchain

`.venv-toolchain` was created with `--system-site-packages`. It inherited Python 3.10.12, torch `2.5.0a0+872d972e41.nv24.08`, torchvision `0.20.0a0+afc54f7`, NumPy 1.23.5, and CUDA 12.6 support. The isolated venv received only pinned, no-dependency installs needed by this path: timm 0.9.12, ONNX 1.17.0, and protobuf 3.20.3. Torch, torchvision, NumPy, Pillow, CUDA, and TensorRT were not installed or upgraded.

## Export

The export mirrors the local official teacher exporter:

- input `image`, FP32 `[batch_size,3,1024,1024]`
- output `image_embeddings`, FP32 `[batch_size,256,64,64]`
- opset 16
- input batch axis declared dynamic
- model in evaluation mode

Generated artifact: `results/toolchain_export_a2/artifacts/mobile_sam_image_encoder_exported.onnx`. It is intentionally Git-ignored.

`onnx.checker.check_model` passed. The graph has 2,036 nodes, 180 initializers, and 23 operator types. Representative real operators include Conv, MatMul, Add, Mul, and Softmax. Detailed counts are in `exported_onnx_summary.json` and examples are in `graph_operator_inspection.json`.

## Parity

ONNX Runtime 1.20.0 executed the same saved input with `CPUExecutionProvider`. Output shape and dtype matched PyTorch. PyTorch ran on CUDA while ORT ran on CPU, so the recorded numerical differences include backend-specific FP32 accumulation behavior. The maximum relative error is dominated by near-zero reference values and is not used alone as an acceptance criterion. Exact metrics are in `pytorch_vs_onnx.json`.

## Boundaries

No existing ONNX or TensorRT engine was overwritten. No file under `third_party/nanosam` was modified. TensorRT conversion and Stage A3 were not started.
