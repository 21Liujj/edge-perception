
- Upstream: third_party/nanosam @ 653633614b2eb93b06ba3be9adb2aeffb117bd72.
- Class: TimmImageEncoder in nanosam/models/timm_image_encoder.py.
- Factory: resnet18() -> TimmImageEncoder("resnet18", pretrained=True); create_model uses the registry.
- Backbone: timm.create_model("resnet18", pretrained=True, features_only=True), not torchvision ResNet18 and not a locally defined ResNet implementation.
- For 1024 input, final feature is 512x32x32; NanoSAM consumes x[-1].
- Custom up_1: Conv2d/GELU stack from 512 to 256, then ConvTranspose2d 256->256 with stride 2, yielding 256x64x64.
- Custom proj: 3x3 Conv2d/GELU followed by 1x1 Conv2d to feature_dim 256.
- Learned pos_embedding has shape [1,256,64,64] and is added to the result.
- Final output is therefore [1,256,64,64]; neck/upsampling and projection layers both exist.
- Input size 1024 is explicit in train.py, export_image_encoder_onnx.py, README training command, and runtime Predictor.

## External preprocessing

The graph starts at normalized NCHW FLOAT. Runtime preprocess_image uses RGB input, longest-side resize to 1024 with top-left zero padding, then mean [123.675,116.28,103.53] and std [58.395,57.12,57.375]. PIL.Image.resize is called without an explicit resampling mode. Training ImageFolder instead uses RandomResizedCrop(1024), ToTensor, and equivalent normalization.

## Limitations

The environment lacks timm, and the trained NanoSAM ResNet18 student checkpoint is not local. The supplied ONNX is consistent with this exporter/model contract, but the absent checkpoint prevents exact PyTorch reconstruction.
