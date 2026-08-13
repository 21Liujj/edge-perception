# NanoSAM Module Benchmark (Jetson Orin Nano)

Note: this benchmark repeatedly encodes the SAME image to characterize encoder/decoder latency. It is not a production caching strategy. All GPU timing is taken after `torch.cuda.synchronize()`.

## Module latencies (ms)
- model_load: **626.954** ms
- set_image (image encoder, n=100): mean **66.3219** | min 48.951 | max 75.6008 | P50 68.3064 | P95 74.7322 | P99 75.2667
- box_decode (mask decoder, n=300): mean **11.0126** | min 9.4854 | max 15.6441 | P50 10.5491 | P95 13.8645 | P99 15.5847
  - estimated prompt FPS = 1000 / mean = **90.81**
- point_decode (n=100): mean **10.2938** | P50 10.3027 | P95 10.4284 | P99 10.5383
  - estimated prompt FPS = **97.15**
- full_segmentation (encoder + first box prompt): **52.186** ms
  - real encoder + first box prompt path; NOT prompt-only FPS expressed as pipeline FPS

## tegrastats (peak during sampling)
- RAM peak: N/A MB (last N/A)
- GR3D peak: N/A% (last N/A)
- temperature peak: N/A C
- VDD_IN peak: N/A mW (last N/A)

## Prompt FPS caveat
Prompt FPS = 1000 / mean decoder latency reflects reuse of a cached image embedding only. It is NOT the NanoSAM end-to-end pipeline FPS; full pipeline latency includes the encoder path (see full_segmentation_ms).
