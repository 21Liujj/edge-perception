# NanoOWL Runtime Architecture Audit (Stage E1)

## Frozen conclusion

NanoOWL deploys a TensorRT-accelerated image encoder and detector head. Text encoding, normalized cross-modal image-text matching, and decode remain CUDA PyTorch/Python.

## Scope

This audit uses only the checked-out `third_party/nanoowl` source, current TensorRT plan, local Hugging Face cache, and existing repository evidence. It does not modify third_party, rebuild the engine, or rerun large evaluations.

- NanoOWL commit: `fb553dee4c58f8c53ec2bf01f355a54648e67dbd`
- Model: `google/owlvit-base-patch32`
- HF snapshot: `cbc355fb364588351c5d51c7f74465e8e7ec6f72`
- Architecture: `OwlViTForObjectDetection`; image 768; patch 32; 24x24 = 576 patches; shared projection 512.

## Image branch

`OwlPredictor.predict` preprocesses a PIL image with `ImagePreprocessor` on CUDA (float32, OWL-ViT/CLIP normalization), creates a full-image ROI, and uses torchvision ROI Align to produce `[1,3,768,768]`. With the current engine, `encode_image` invokes TensorRT.

The exported `encode_image_torch` graph covers `vision_model`, vision `post_layernorm`, class-token removal and modulation, detector layer norm, `box_head`, box bias, box sigmoid, center-to-corners conversion, `class_head.dense0`, `logit_shift`, and `logit_scale` with ELU+1. It produces per-patch visual representations and candidate boxes.

## Text branch

Text is encoded outside TensorRT by `OwlPredictor.encode_text`:

`prompt strings -> OwlViTProcessor / CLIP tokenizer -> input_ids + attention_mask -> OwlViTTextTransformer -> pooled output -> text_projection (Linear) -> 512-dimensional text embedding`

The live runtime confirmed `OwlViTTextTransformer`, a Linear projection, CUDA float32 model parameters, PyTorch `2.5.0a0+872d972e41.nv24.08`, Transformers `4.46.3`, and CUDA device `Orin`. `encode_text` does not normalize; normalization is done during decode.

## TensorRT contract and boundary

Current engine: `third_party/nanoowl/data/owlvit_image_encoder_patch32.engine`; size 183507980 bytes; SHA256 `d3f723f97d5e466805105f8b25da83d898e215c09598fb2b41c06d5254d09037`. TensorRT 10.3.0 built it with `trtexec --fp16` and fixed shape `image:1x3x768x768`. External I/O is FP32 despite internally mixed FP32+FP16 execution.

`image INPUT FLOAT [1,3,768,768]`

`image_embeds OUTPUT FLOAT [1,576,768]`

`image_class_embeds OUTPUT FLOAT [1,576,512]`

`logit_shift OUTPUT FLOAT [1,576,1]`

`logit_scale OUTPUT FLOAT [1,576,1]`

`pred_boxes OUTPUT FLOAT [1,576,4]`

The engine contains no tokenizer, text transformer, text projection, image/text normalization, cross-modal similarity, prompt argmax, threshold filtering, Python packaging, or NMS. Image preprocessing and ROI Align also remain outside the plan.

## VLM matching and boxes

Vision and language interact in `OwlPredictor.decode` after both branches: normalized `image_class_embeds` and normalized `query_embeds` are compared by `torch.einsum(\"...pd,...qd->...pq\", ...)`, then per-patch `logit_shift` and `logit_scale` are applied before sigmoid. The prompt maximum chooses one label per patch and thresholds filter results.

The engine has already decoded box geometry, but upstream decode performs no NMS. With `pad_square=False`, project call sites treat boxes as original-image xyxy and may pass the selected box to NanoSAM.

This is a VLM because it computes image-region and text representations in a learned shared 512-dimensional space and performs cross-modal similarity/grounding. NanoSAM geometric segmentation prompting is not this operation.

## Current deployed architecture and cache

PIL image -> PyTorch preprocessing/ROI Align -> TensorRT image encoder/head; prompt -> processor/tokenizer -> PyTorch text transformer/projection; image embedding + text embedding -> PyTorch normalization/einsum/score/decode.

Text embeddings are reusable for the same ordered prompts, tokenizer/model revision and weights, device, and dtype. Upstream OwlPredictor has no prompt-keyed automatic cache; callers pass `text_encodings` explicitly. Existing video code encodes once before its frame loop, and existing COCO evaluation reuses the 80-prompt encoding. Image outputs are image-dependent.

## Historical evidence boundaries

Existing narrow NanoOWL evidence: mean 22.49 ms, about 44.46 FPS, one loaded predictor, one image, three prompts, 40 warmups, 300 timed iterations; this is not system FPS. Existing COCO evidence: AP 0.258616 and 30.26 images/s in the evaluation image loop; this is evaluation throughput. Existing joint pipeline evidence: 8.54 FPS end to end and 38.6133 ms NanoOWL module time; the cached-image scenario is not full pipeline FPS.

## Frozen conclusion

This repository deploys the TensorRT image side of NanoOWL plus a PyTorch/Transformers text side. Vision-language grounding and postprocessing remain in PyTorch/Python at `OwlPredictor.decode`.
