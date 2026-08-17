# NanoSAM Image Encoder Conversion Contract

## Source Model

PyTorch class: nanosam.models.timm_image_encoder.TimmImageEncoder
Factory: create_model("resnet18") -> registered resnet18() -> TimmImageEncoder("resnet18", pretrained=True)
Backbone: timm ResNet18 features_only, plus NanoSAM up_1 neck, proj head, and learned positional embedding
Checkpoint: PYTORCH_CHECKPOINT_NOT_LOCAL; expected training output data/models/resnet18/checkpoint.pth

## Input

### Model Input Contract

name: image
shape: [1,3,1024,1024] (static NCHW)
dtype: FLOAT (FP32 binding)
value range: normalized real values; no clipping is defined
color order: RGB before external normalization
normalization: external, not in encoder graph
resize: external, not in encoder graph

### External Preprocessing Contract

Runtime source preserves aspect ratio, resizes the longest side to 1024, places normalized data at the top-left of a zero tensor, and returns [1,3,1024,1024]. It uses RGB values nominally in [0,255], mean [123.675,116.28,103.53], and std [58.395,57.12,57.375]. PIL resize mode is not explicitly pinned in source. Training uses RandomResizedCrop(1024) and equivalent normalization, so training and runtime geometry must not be conflated.

## Output

name: image_embeddings
shape: [1,256,64,64]
dtype: FLOAT
semantic meaning: distilled MobileSAM-compatible image feature embedding consumed by the shared prompt/mask decoder

## ONNX

path: third_party/nanosam/data/resnet18_image_encoder.onnx
SHA256: d266fcdc9e4f0182b59946cd3cf1be331641f1d0d1de2415c4fd94e7a1c9cf0a
opset: 16
IR version: 8
producer: pytorch 1.14.0
input: image FLOAT [1,3,1024,1024]
output: image_embeddings FLOAT [1,256,64,64]
initializer count: 53
node count: 93
operator distribution: Conv25, Relu17, Constant15, Add14, Mul10, Div5, Erf5, ConvTranspose1, MaxPool1
onnx.checker: PASS

## TensorRT

engine input: image INPUT FLOAT [1,3,1024,1024] DEVICE
engine output: image_embeddings OUTPUT FLOAT [1,256,64,64] DEVICE
TensorRT: 10.3.0
precision: built with trtexec --fp16 (FP32+FP16 tactics); network boundary tensors remain FLOAT
ONNX/TRT contract: MATCH

## Expected Conversion

PyTorch -> ONNX -> ONNX Runtime -> TensorRT

## Current Toolchain Status

PyTorch Model Class        AVAILABLE
PyTorch Student Checkpoint MISSING
timm Dependency            MISSING
Official Local Exporter    AVAILABLE
ONNX Graph                 AVAILABLE / CHECKED
ONNX Runtime               AVAILABLE / NOT EXECUTED IN A1
TensorRT Engine            AVAILABLE / CHECKED

PyTorch Model Class [AVAILABLE]
        +
Checkpoint [MISSING]
        |
        v
state_dict load [KNOWN / BLOCKED]
        |
        v
model.eval() [KNOWN / BLOCKED]
        |
        v
dummy input [KNOWN: 1x3x1024x1024]
        |
        v
torch.onnx.export [AVAILABLE / BLOCKED]
        |
        v
ONNX Graph [AVAILABLE]
        |
        v
onnx.checker [AVAILABLE / PASS]
        |
        v
ONNX Runtime [AVAILABLE / A1 NOT EXECUTED]
        |
        v
TensorRT Builder [KNOWN / A1 NOT EXECUTED]
        |
        v
Serialized Engine [AVAILABLE]
        |
        v
TensorRT Runtime [AVAILABLE / CONTRACT CHECKED]
        |
        v
Jetson GPU [AVAILABLE]

## Known Risks

- Student checkpoint availability and provenance are unresolved.
- timm is missing; the factory also requests pretrained=True and may attempt network access if its base weights are not cached.
- Preprocessing is outside the graph; resize, RGB, normalization, or padding mismatch changes embeddings.
- ONNX is static batch 1 and static 1024x1024.
- Exporter hard-codes CUDA and uses torch.load without map_location.
- Exporter does not explicitly set do_constant_folding or dynamic_axes.
- Existing graph was produced by PyTorch 1.14.0; current environment uses PyTorch 2.5.0a0, so graph differences are possible.
- FP16 TensorRT tactics may introduce numerical differences despite FLOAT bindings.

## Stage A2 — MobileSAM Teacher Export Rehearsal

Stage A2 used the local, fixed `third_party/nanosam/assets/mobile_sam.pt` MobileSAM teacher checkpoint with registry key `vit_t`. It did not substitute for the missing NanoSAM ResNet18 student checkpoint.

Teacher contract:

- source: MobileSAM `Sam.image_encoder`, TinyViT
- checkpoint SHA256: `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`
- strict state-dict contract: PASS, no missing or unexpected keys
- PyTorch reference: CUDA FP32, eval, no_grad, deterministic seed 20260817
- input: `image` FLOAT `[1,3,1024,1024]`
- output: `image_embeddings` FLOAT `[1,256,64,64]`
- exported ONNX: opset 16, dynamic input batch axis
- ONNX checker: PASS
- graph: 2,036 nodes, 180 initializers
- ORT: CPUExecutionProvider executed the identical raw input and matched output shape/dtype; cross-backend numerical metrics are preserved in evidence

Teacher chain status:

```text
mobile_sam.pt [AVAILABLE / VERIFIED]
        -> MobileSAM vit_t + TinyViT [LOAD PASS]
        -> PyTorch CUDA reference [PASS]
        -> torch.onnx.export opset 16 [PASS]
        -> new teacher ONNX [AVAILABLE / GIT-IGNORED]
        -> onnx.checker [PASS]
        -> ONNX Runtime CPU [EXECUTED]
        -> TensorRT [NOT STARTED; OUT OF A2]
```

Student chain status remains separate:

```text
TimmImageEncoder ResNet18 class [AVAILABLE]
        +
NanoSAM student checkpoint [MISSING]
        -> PyTorch student reference [BLOCKED]
        -> student ONNX reproduction [BLOCKED]
        -> existing official student ONNX [AVAILABLE / CHECKED IN A1]
        -> existing TensorRT engine [AVAILABLE / CONTRACT CHECKED IN A1]
```

Future factual distillation chain, not executed in A2:

```text
unlabeled RGB images
        -> fixed MobileSAM teacher image embeddings
        -> NanoSAM ResNet18 student optimization (Huber/L1/MSE)
        -> checkpoint.pth containing student model state
        -> student PyTorch reference
        -> student ONNX
        -> ONNX Runtime parity
        -> TensorRT build and runtime validation
```

Stage A2 did not download a student checkpoint, infer weights from ONNX, modify the student model, train, distill, or build TensorRT.

# JD Capability Mapping

- PyTorch model structure audit
- ONNX graph inspection
- TensorRT binding inspection
- model conversion contract
- Linux
- Python
- Git
- toolchain debugging mindset
