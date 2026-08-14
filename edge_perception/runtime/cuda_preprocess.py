# -*- coding: utf-8 -*-
"""Stage 7B — Python wrapper for the fused CUDA preprocessing extension.

Exposes CudaPreprocessPredictor with the same interface as
nanosam.utils.predictor.Predictor / Stage 7A OptimizedPredictor.

Host work kept to the unavoidable minimum:
  - PIL/np source -> uint8 HWC numpy (NO resize, NO normalize on host)
  - uint8 HWC -> pinned staging -> non_blocking H2D into a reused device uint8 buffer
  - launch the fused .cu kernel: bilinear resize (half-pixel) + per-channel normalize
    + HWC->NCHW + uint8->float32, writing into a reused (1,3,size,size) fp32 device tensor

Numerical semantics: resize is BILINEAR half-pixel (NOT PIL BICUBIC). Documented in
preprocess_contract.md. Correctness judged by tensor error + COCO QUICK mIoU delta.
"""
import numpy as np
import torch

from nanosam.utils.predictor import preprocess_points, upscale_mask

_mean = (123.675, 116.28, 103.53)
_std = (58.395, 57.12, 57.375)

_mod = None
def _module():
    global _mod
    if _mod is None:
        import edge_perception._cuda_preprocess as m
        _mod = m
    return _mod


class CudaPreprocessor:
    """Owns reusable device + pinned buffers; one .cu kernel launch per set_image.

    preprocess() issues work on self._stream and returns WITHOUT synchronizing
    (the encoder launch runs on the same stream in the predictor). Callers that
    need the result on the host must sync.
    """

    def __init__(self, size=1024, max_src=2048):
        self._size = size
        self._max_src = max_src
        self._device = torch.device("cuda")
        self.image_tensor_gpu = torch.zeros((1, 3, size, size), dtype=torch.float32, device=self._device)
        self._src_gpu = torch.zeros((max_src, max_src, 3), dtype=torch.uint8, device=self._device)
        self._src_pinned = torch.zeros((max_src, max_src, 3), dtype=torch.uint8, pin_memory=True)
        self._prev_rh = None
        self._prev_rw = None
        self._stream = torch.cuda.Stream()

    def prepare_source(self, image):
        """PIL/ndarray -> (src_h, src_w). Stages uint8 HWC into the device buffer
        on self._stream (non_blocking). Returns (src_h, src_w)."""
        import PIL.Image
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
        arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))  # (H,W,3)
        src_h, src_w = arr.shape[0], arr.shape[1]
        assert src_h <= self._max_src and src_w <= self._max_src, \
            f"source {src_h}x{src_w} exceeds max_src {self._max_src}; enlarge max_src"
        pin = self._src_pinned
        pin[:src_h, :src_w, :].copy_(torch.from_numpy(arr))
        s = self._stream
        with torch.cuda.stream(s):
            self._src_gpu[:src_h, :src_w, :].copy_(pin[:src_h, :src_w, :], non_blocking=True)
        return src_h, src_w

    def launch_fused(self, src_h, src_w, rh, rw):
        """Launch the fused kernel on self._stream; zero pad bands as needed."""
        s = self._stream
        with torch.cuda.stream(s):
            # Clear stale pad region when the resized region shrank vs last frame.
            if self._prev_rh is not None:
                if rh < self._prev_rh:
                    self.image_tensor_gpu[:, :, rh:self._prev_rh, :].zero_()
                if rw < self._prev_rw:
                    self.image_tensor_gpu[:, :, :, rw:self._prev_rw].zero_()
            self._prev_rh = rh
            self._prev_rw = rw
            _module().fused_preprocess_resize(
                self.image_tensor_gpu, self._src_gpu,
                rh, rw,
                _mean[0], _mean[1], _mean[2],
                _std[0], _std[1], _std[2],
            )


class CudaPreprocessPredictor:
    """Predictor using the fused CUDA preprocessing kernel."""

    def __init__(self, image_encoder_engine, mask_decoder_engine,
                 image_encoder_size=1024, orig_image_encoder_size=1024):
        from nanosam.utils.predictor import (
            load_image_encoder_engine, load_mask_decoder_engine,
        )
        self.image_encoder_engine = load_image_encoder_engine(image_encoder_engine)
        self.mask_decoder_engine = load_mask_decoder_engine(mask_decoder_engine)
        self.image_encoder_size = image_encoder_size
        self.orig_image_encoder_size = orig_image_encoder_size
        self._size = image_encoder_size
        self._device = torch.device("cuda")
        self._prep = CudaPreprocessor(size=image_encoder_size)

        self._mask_input_zero = torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=self._device)
        self._has_mask_input_zero = torch.tensor([0.0], dtype=torch.float32, device=self._device)
        self._has_mask_input_one = torch.tensor([1.0], dtype=torch.float32, device=self._device)
        self._points_gpu = torch.zeros((1, 2, 2), dtype=torch.float32, device=self._device)
        self._labels_gpu = torch.tensor([[2.0, 3.0]], dtype=torch.float32, device=self._device)

        self._image = None
        self._img_h = 0
        self._img_w = 0
        self.features = None

    def _resized_dims(self, image):
        W, H = image.width, image.height
        aspect = W / H
        size = self._size
        if aspect >= 1:
            rw = size; rh = int(size / aspect)
        else:
            rh = size; rw = int(size * aspect)
        return rh, rw

    def set_image(self, image):
        import PIL.Image
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
        self._image = image
        self._img_h = image.height
        self._img_w = image.width
        rh, rw = self._resized_dims(image)
        src_h, src_w = self._prep.prepare_source(image)
        self._prep.launch_fused(src_h, src_w, rh, rw)
        # Encoder on the SAME stream -> kernel output is ready; no cross-stream sync.
        s = self._prep._stream
        with torch.cuda.stream(s):
            self.features = self.image_encoder_engine(self._prep.image_tensor_gpu)

    def predict(self, points, point_labels, mask_input=None):
        pts = np.asarray(points, dtype=np.float32).reshape(1, 2, 2)
        self._points_gpu.copy_(torch.from_numpy(pts))
        scale = self.orig_image_encoder_size / max(self._img_h, self._img_w)
        # Ensure features + kernel output are ready before the decoder reads them.
        self._prep._stream.synchronize()
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
