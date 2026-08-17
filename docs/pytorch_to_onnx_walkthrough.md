# PyTorch to ONNX Walkthrough: MobileSAM Teacher

## 1. Load the Fixed Checkpoint

`sam_model_registry["vit_t"]` constructs the local MobileSAM `Sam` model with a TinyViT image encoder. The checkpoint is a 439-entry `OrderedDict`. Loading it produced no missing or unexpected keys. This is a MobileSAM teacher checkpoint, not a NanoSAM ResNet18 student checkpoint.

## 2. Establish a PyTorch Reference

A fixed seed (`20260817`) creates one FP32 CUDA tensor with shape `[1,3,1024,1024]`. The encoder is in `eval()` mode and executes under `torch.no_grad()`. The output contract is FP32 `[1,256,64,64]`. JSON evidence records tensor statistics, finite checks, and a SHA256 of the raw input bytes without committing the tensor.

## 3. Export the Image Encoder

`torch.onnx.export` receives the TinyViT encoder and the reference-shaped tensor. Names and opset follow the repository exporter:

```python
torch.onnx.export(
    encoder,
    (image,),
    output_path,
    input_names=["image"],
    output_names=["image_embeddings"],
    dynamic_axes={"image": {0: "batch_size"}},
    opset_version=16,
)
```

The resulting graph input is `image` FLOAT `[batch_size,3,1024,1024]`; the output is `image_embeddings` FLOAT `[symbolic_batch,256,64,64]`. Although only the input batch axis was explicitly named, shape propagation preserves a symbolic output batch dimension.

## 4. Map PyTorch Layers to ONNX

The first TinyViT patch-embedding convolution becomes `/patch_embed/seq/seq.0/c/Conv`. Its image activation input is `image`; its learned weight and bias are serialized as initializers `onnx::Conv_2786` and `onnx::Conv_2787`. The ONNX node therefore separates executable Conv semantics from the checkpoint-derived parameter tensors.

A TinyViT attention projection becomes `/layers.1/blocks.0/attn/qkv/MatMul`. The activation input comes from the preceding normalization subgraph, while learned projection weights are initializer `onnx::MatMul_2878`. ONNX represents the surrounding tensor layout explicitly with Reshape and Transpose nodes.

Five operators verified in the actual graph are Conv (29), MatMul (60), Add (139), Mul (98), and Softmax (10). These are inspected graph contents, not an expected operator list.

## 5. Validate Structure and Runtime

`onnx.checker.check_model` passes. The graph contains 2,036 nodes and 180 initializers. ONNX Runtime 1.20.0 loads the graph with `CPUExecutionProvider` and produces FP32 `[1,256,64,64]` for the same raw input.

The PyTorch reference used CUDA and ORT used CPU. Reported parity metrics therefore include different FP32 kernels and accumulation order: max absolute error 0.2841012, mean absolute error 0.0217554, and RMSE 0.0318601. The large max relative error occurs where reference values approach zero; it is retained as evidence rather than hidden.

## 6. Artifact Policy

The 28 MB ONNX file is generated under `results/toolchain_export_a2/artifacts/` and ignored by Git. Only Markdown, JSON, and small contract evidence are committed. No existing ONNX, engine, checkpoint, or upstream source is changed.
