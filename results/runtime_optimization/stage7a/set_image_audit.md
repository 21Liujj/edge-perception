# Stage 7A — NanoSAM set_image Data-Path Audit

## Current data flow (upstream `nanosam.utils.predictor.Predictor.set_image`)

```
input PIL.Image
  -> preprocess_image(image, size=1024)
       PIL.resize((rw,rh))            [BILINEAR, on host]
       np.asarray(pil_resized)        [uint8 HxWx3]
       torch.from_numpy(arr).permute(2,0,1)   [uint8 -> view]
       .float()                        [fp32 3xrhxrw]   *** NEW tensor + 3M elem cast
       (t - mean[:,None,None]) / std   [fp32]  *** NEW tensor + 3M elem arith, on CPU
       torch.zeros((1,3,1024,1024))    *** NEW 12.5MB alloc per frame (pad frame)
       it[0,:,:rh,:rw] = normalized    [copy into pad frame]
       .cuda()                         *** H2D fp32 8MB
  -> image_encoder_engine(image_tensor)   [TRT, default stream]
```

`predict()` path (run_mask_decoder) per call:
```
torch.tensor([points]).float().cuda()    *** NEW H2D each frame (dynamic, small)
torch.tensor([labels]).float().cuda()    *** NEW H2D each frame (static 2 floats!)
torch.zeros(1,1,256,256).float().cuda()  *** NEW H2D each frame (STATIC mask placeholder)
torch.tensor([0]).float().cuda()         *** NEW H2D each frame (STATIC has_mask flag)
```

## Per-stage timing (baseline path, mean ms over 100 iters, owl image 756x499)

| stage | mean ms |
|---|---|
| PIL.resize (to 1024x~672) | **15.547** |
| to_np (np.asarray) | 0.860 |
| to_torch_chw (permute) | 0.007 |
| .float() | 0.325 |
| normalize (mean/std arith) | **4.938** |
| torch.zeros(1,3,1024,1024) | 0.456 |
| pad copy | 1.610 |
| h2d (.cuda()) | 2.083 |
| encoder (GPU TRT) | 8.950 |
| **FULL set_image** | **27.501** |

(isolated stage sum < full-path mean because stage isolation omits cross-stage
overhead; Stage 6C end-to-end set_image wall ~55.7ms under nsys.)

## Allocation audit (per iteration)

| item | baseline | issue |
|---|---|---|
| np.asarray(pil_resized) | new uint8 each frame | unavoidable (PIL output) |
| torch.from_numpy(arr).permute | view (no copy) | OK |
| .float() | NEW fp32 tensor each frame | repeated allocation |
| normalize result | NEW fp32 tensor each frame | repeated allocation |
| torch.zeros(1,3,1024,1024) pad | NEW 12.5MB each frame | **repeated large alloc** |
| it.cuda() H2D | NEW fp32 GPU tensor each frame | repeated H2D 8MB |
| torch.tensor([points]).float().cuda() | new | repeated small H2D |
| torch.tensor([labels]).float().cuda() | new | **static data rebuilt/H2D each frame** |
| torch.zeros(1,1,256,256).cuda() mask_input | new | **static data rebuilt/H2D each frame** |
| torch.tensor([0]).cuda() has_mask | new | **static data rebuilt/H2D each frame** |

## Root cause of Stage 6C set_image GPU idle (~36.7ms)

The GPU encoder takes only ~9ms, but ~15.5ms PIL.resize + ~4.9ms normalize +
~2.1ms pad + ~2.1ms H2D all run on the **CPU/host**, serially before the encoder,
while the GPU sits idle. Buffer reuse removes the pad-frame allocation (~0.5ms)
and the static H2D churn (mask/has_mask/labels), but **cannot remove PIL.resize
(CPU, 15.5ms)** or the normalize arithmetic (CPU, 4.9ms). These remain the
dominant host gap and would require a **fused CUDA preprocessing kernel** to
eliminate (recommended next step, not done this stage).
