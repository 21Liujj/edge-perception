# Stage A2 Export Summary

Status: COMPLETE WITH DOCUMENTED CROSS-BACKEND NUMERICAL DIFFERENCE

- Teacher checkpoint load: PASS, no missing or unexpected keys
- PyTorch CUDA reference: PASS, FP32 `[1,256,64,64]`, finite
- ONNX export: PASS
- ONNX checker: PASS
- ONNX artifact size: 28,106,961 bytes
- ONNX artifact SHA256: `f8d9ac5da0f9521aaf067b35dbe1eac78c50c9d7a0eef72dc3e07db5eb6f3ec7`
- Graph: IR 8, opset 16, 2,036 nodes, 180 initializers
- ONNX Runtime CPU output: FP32 `[1,256,64,64]`
- Max absolute error: 0.2841012179851532
- Mean absolute error: 0.02175542408853448
- RMSE: 0.03186013262714965
- Max relative error: 328639.0, unstable at near-zero reference values

The run proves the local MobileSAM teacher checkpoint can be loaded, evaluated, exported, structurally validated, and executed by ONNX Runtime. It does not establish a NanoSAM student checkpoint path and does not include TensorRT.
