# NanoSAM Decoder Output Contract

## Runtime evidence

The accepted deployment is unchanged:

- Image encoder: `third_party/nanosam/data/resnet18_image_encoder.engine`, FP16-enabled, SHA256 `c277e5ca445dc29b06e8be41a481e74bd806c4dd82071cbfbd47fddcf7aaa6a8`
- Mask decoder: `third_party/nanosam/data/mobile_sam_mask_decoder.engine`, FP32, SHA256 `afb5ac4958e6bab21ed8aa0cc65798f53a71406fe6124cf6b2e008da6fccb4f2`

TensorRT 10.3 engine inspection reports seven IO tensors. The two outputs are:

- `iou_predictions`: FP32 `[1,4]`
- `low_res_masks`: FP32 `[1,4,256,256]`

For every one of the 500 B1 instances, `Predictor.predict` returned:

- upscaled mask logits: `[1,4,H,W]`, where H and W are the original image dimensions;
- predicted IoU scores: `[1,4]`;
- low-resolution mask logits: `[1,4,256,256]`.

There are four mask slots, and score element `i` corresponds to mask slot `i`. This correspondence follows the engine output tensor shapes and the project/upstream evaluation indexing (`mask[0, iou_preds.argmax()]`).

## Threshold semantics

`nanosam.utils.predictor.Predictor` bilinearly upscales `low_res_masks` and returns logits without thresholding. The accepted Day 5 evaluator defines a binary mask as `hi_res_mask[0, slot] > 0`. B1 retains exactly that threshold for all four candidates.

## Selection policies

- `FIXED_SLOT_0`: always slot 0; deployable and historical baseline.
- `PREDICTED_IOU_ARGMAX`: `argmax(iou_predictions[0])`; deployable and uses no GT.
- `ORACLE_BEST_OF_4`: argmax over four measured GT IoUs; analysis-only, not deployable.

The mask decoder source configures `num_multimask_outputs=3` and `num_mask_tokens=4`: one single-mask token plus three multimask tokens. The deployed ONNX/TensorRT contract exposes all four together.
