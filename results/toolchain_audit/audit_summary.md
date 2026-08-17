# Stage A1 Audit Summary

Status: BLOCKED for autonomous PyTorch-to-ONNX reproduction; all requested audit work passed.

1. PyTorch model class found: YES
   - third_party/nanosam/nanosam/models/timm_image_encoder.py
   - TimmImageEncoder; registered resnet18() factory

2. PyTorch student checkpoint local: NO
   - PYTORCH_CHECKPOINT_NOT_LOCAL
   - assets/mobile_sam.pt is the teacher/full MobileSAM checkpoint, not the NanoSAM ResNet18 student checkpoint

3. ONNX exporter found: YES
   - third_party/nanosam/nanosam/tools/export_image_encoder_onnx.py

4. ONNX contract
   - image FLOAT [1,3,1024,1024]
   - image_embeddings FLOAT [1,256,64,64]
   - opset 16; nodes 93; initializers 53
   - Conv25 Relu17 Constant15 Add14 Mul10 Div5 Erf5 ConvTranspose1 MaxPool1

5. TensorRT contract
   - image INPUT FLOAT [1,3,1024,1024] DEVICE
   - image_embeddings OUTPUT FLOAT [1,256,64,64] DEVICE
   - TensorRT 10.3.0; engine built with --fp16; contract matches ONNX

6. Unique critical missing link
   - Recoverable NanoSAM ResNet18 student checkpoint matching the model/export path
   - timm is also absent and must be provided before model construction

7. Stage A2 recommendation
   - With authorization, recover the official/fixed data/models/resnet18/checkpoint.pth and verify SHA256.
   - Provide a pinned compatible timm dependency without modifying upstream.
   - Instantiate TimmImageEncoder through create_model, load checkpoint, eval, run PyTorch reference, export with the official script or a minimal project-local wrapper, run onnx.checker and ONNX Runtime parity, then build and compare TensorRT.
   - No Stage A2 action was executed.
