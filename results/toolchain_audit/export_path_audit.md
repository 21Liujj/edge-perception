# NanoSAM Export Path Audit

## Official local path

Exporter: third_party/nanosam/nanosam/tools/export_image_encoder_onnx.py
Model: nanosam.models.create_model(model_name), default resnet18
Checkpoint: required --checkpoint; loads torch.load(path)["model"]
Dummy input: torch.randn(1,3,input_size,input_size), default input_size=1024
Input name: image
Output name: image_embeddings
Default opset: 16
Dynamic axes: none (static batch and spatial dimensions)
do_constant_folding: not explicitly set; torch.onnx.export default applies
Output filename: required --output
Model state: moved to CUDA and eval() before export

## Current ONNX provenance

The deployed ONNX is the official NanoSAM README artifact (Google Drive file ID 14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR), recovered by user offline transfer on Day 3. Its names, static shapes, opset, PyTorch producer, parameter names, and graph structure are consistent with the local exporter. The absent student checkpoint means byte-for-byte reproduction from a specific PyTorch state cannot be independently proven locally.

## Training and distillation path

Teacher: MobileSAM vit_t image encoder exported from assets/mobile_sam.pt and built as a TensorRT engine.
Student: create_model("resnet18"), the custom TimmImageEncoder.
Images: unlabeled RGB JPG/PNG; README specifies COCO 2017 train images.
Target: teacher image feature embeddings, not masks or labels.
Loss: Huber by default; L1 and MSE are selectable.
Checkpoint: output_dir/checkpoint.pth containing model, optimizer, and epoch.
Flow: train.py checkpoint -> export_image_encoder_onnx.py -> ONNX -> README trtexec --fp16 -> serialized engine.

## Can it run now?

No. The primary blocker is the missing NanoSAM ResNet18 student checkpoint. The secondary dependency blocker is missing timm. PyTorch, ONNX, ONNX Runtime, TensorRT, torch2trt, matplotlib, and tqdm are importable. No export was executed in Stage A1.

Exporter implementation risks: create_model requests pretrained=True and may attempt to fetch uncached timm weights; model.cuda() is hard-coded despite selecting a device variable; torch.load has no map_location; output is static shape; current PyTorch 2.5 differs from ONNX producer PyTorch 1.14.0.
