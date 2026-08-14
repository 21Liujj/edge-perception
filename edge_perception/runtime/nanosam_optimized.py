# -*- coding: utf-8 -*-
"""Stage 7A — CUDA-aware optimized NanoSAM runtime wrapper.

Reuses upstream TensorRT engines via the SAME loaders as
nanosam.utils.predictor (no upstream edit). Re-engineers the set_image /
predict DATA PATH only.

Numerical semantics IDENTICAL to upstream:
  PIL.resize (BILINEAR), mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375],
  zero pad, fp32 input to TRT. Does NOT change model/engine/algorithm/mask slot.

Optimizations (each ACCEPTED/REJECTED recorded in optimization_decisions.md):
  OPT1 pinned host staging buffer for the normalized image, reused.
  OPT2 preallocated GPU input tensor (1,3,1024,1024) reused; preprocess writes
      into it via a pinned staging tensor + (optional) non_blocking H2D.
  OPT3 GPU-resident mean/std constants cached (not rebuilt per frame).
  OPT4 static decoder inputs (mask_input zero, has_mask_input) cached on device.
  OPT6 point template built once; only the 4 point coords updated per frame.
  OPT5 explicit CUDA stream (use_stream=True) for set_image H2D + encoder.
      A no-stream variant (use_stream=False) keeps OPT1-4,6 on default stream.

Interface matches nanosam.utils.predictor.Predictor:
  set_image(image) -> None  (image: PIL.Image or np.ndarray HxWx3 uint8 RGB)
  predict(points, point_labels, mask_input=None) -> (hi_res_mask, mask_iou, low_res_mask)
"""
import numpy as np
import PIL.Image
import torch

from nanosam.utils.predictor import (
    load_image_encoder_engine, load_mask_decoder_engine,
    preprocess_points, upscale_mask,
)


class OptimizedPredictor:
    def __init__(self, image_encoder_engine, mask_decoder_engine,
                 image_encoder_size=1024, orig_image_encoder_size=1024,
                 use_stream=True):
        self.image_encoder_engine = load_image_encoder_engine(image_encoder_engine)
        self.mask_decoder_engine = load_mask_decoder_engine(mask_decoder_engine)
        self.image_encoder_size = image_encoder_size
        self.orig_image_encoder_size = orig_image_encoder_size

        self._device = torch.device("cuda")
        self._size = image_encoder_size
        self._use_stream = use_stream

        # OPT3: GPU-resident mean/std constants cached once.
        mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(1, 3, 1, 1)
        self._mean_gpu = mean.to(self._device)
        self._std_gpu = std.to(self._device)

        # OPT2: preallocated GPU input tensor reused every frame.
        self._image_tensor_gpu = torch.zeros((1, 3, self._size, self._size),
                                             dtype=torch.float32, device=self._device)

        # OPT1: pinned host staging buffer.
        self._staging_pinned = torch.zeros((1, 3, self._size, self._size),
                                           dtype=torch.float32, pin_memory=True)
        assert self._staging_pinned.is_pinned(), "pinned staging buffer not pinned"

        # OPT5: explicit stream + event.
        self._stream = torch.cuda.Stream()
        self._event = torch.cuda.Event()

        # OPT4: static decoder inputs cached on device.
        self._mask_input_zero = torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=self._device)
        self._has_mask_input_zero = torch.tensor([0.0], dtype=torch.float32, device=self._device)
        self._has_mask_input_one = torch.tensor([1.0], dtype=torch.float32, device=self._device)

        # OPT6: point/label templates on device; only 4 point coords updated per frame.
        self._points_gpu = torch.zeros((1, 2, 2), dtype=torch.float32, device=self._device)
        self._labels_gpu = torch.tensor([[2.0, 3.0]], dtype=torch.float32, device=self._device)
        self._points_pinned = torch.zeros((1, 2, 2), dtype=torch.float32, pin_memory=True)
        assert self._points_pinned.is_pinned()

        self._image = None
        self._img_h = 0
        self._img_w = 0
        self.features = None

    def set_image(self, image):
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
        self._image = image
        self._img_h = image.height
        self._img_w = image.width

        size = self._size
        aspect = image.width / image.height
        if aspect >= 1:
            rw = size; rh = int(size / aspect)
        else:
            rh = size; rw = int(size * aspect)

        pil_resized = image.resize((rw, rh))
        arr = np.asarray(pil_resized)  # uint8 HxWx3 RGB

        # normalized CHW in a numpy buffer (in-place arithmetic; no torch.zeros per frame)
        chwx = np.ascontiguousarray(arr.transpose(2, 0, 1)).astype(np.float32)  # (3,rh,rw)
        chwx[0] = (chwx[0] - 123.675) / 58.395
        chwx[1] = (chwx[1] - 116.28) / 57.12
        chwx[2] = (chwx[2] - 103.53) / 57.375

        stage = self._staging_pinned  # (1,3,size,size) pinned fp32
        stage.zero_()  # reset pad region
        stage[0, :, :rh, :rw] = torch.from_numpy(chwx)

        if self._use_stream:
            s = self._stream
            with torch.cuda.stream(s):
                self._image_tensor_gpu.copy_(stage, non_blocking=True)
                self.features = self.image_encoder_engine(self._image_tensor_gpu)
            self._event.record(s)
        else:
            self._image_tensor_gpu.copy_(stage, non_blocking=False)
            self.features = self.image_encoder_engine(self._image_tensor_gpu)

    def predict(self, points, point_labels, mask_input=None):
        pts = np.asarray(points, dtype=np.float32).reshape(1, 2, 2)
        self._points_pinned.copy_(torch.from_numpy(pts))
        scale = self.orig_image_encoder_size / max(self._img_h, self._img_w)
        if self._use_stream:
            s = self._stream
            with torch.cuda.stream(s):
                self._event.synchronize()
                self._points_gpu.copy_(self._points_pinned, non_blocking=True)
                points_scaled = self._points_gpu * scale
                if mask_input is None:
                    mask_input = self._mask_input_zero; has_mask_input = self._has_mask_input_zero
                else:
                    has_mask_input = self._has_mask_input_one
                mask_iou, low_res_mask = self.mask_decoder_engine(
                    self.features, points_scaled, self._labels_gpu, mask_input, has_mask_input,
                )
            s.synchronize()
        else:
            self._points_gpu.copy_(self._points_pinned, non_blocking=False)
            points_scaled = self._points_gpu * scale
            if mask_input is None:
                mask_input = self._mask_input_zero; has_mask_input = self._has_mask_input_zero
            else:
                has_mask_input = self._has_mask_input_one
            mask_iou, low_res_mask = self.mask_decoder_engine(
                self.features, points_scaled, self._labels_gpu, mask_input, has_mask_input,
            )
        hi_res_mask = upscale_mask(low_res_mask, (self._img_h, self._img_w))
        return hi_res_mask, mask_iou, low_res_mask
