# Joint EdgePerception Pipeline Benchmark (Jetson Orin Nano)

One process; models/engines loaded once. Fixed input owl_glove_small.jpg, prompts=an owl,a glove,a finger, target=an owl, thr=0.1. Mask selection = official slot 0 (not IoU argmax). All GPU timing after torch.cuda.synchronize(). Excludes load/I/O/viz.

## A. Model load latency
- total model_load_ms: **2298.288** (nanoowl 1668.42 + nanosam 629.873)
- warmup: 30 (not counted) | test iterations: 300

## B. Steady-state complete joint inference latency (pipeline_e2e, ms)
- mean **113.6504** | min 96.9864 | max 135.5122 | P50 114.8659 | P95 125.0567 | P99 131.3147
- steady-state E2E FPS = 1000/mean = **8.8** (inference pipeline FPS, excludes disk I/O / visualization)

## C. Module latency breakdown (mean ms, n=300)
- nanoowl: mean **36.4022** | P50 37.5464 | P95 44.5328 | P99 44.8078
- bridge : mean **0.1014** | P50 0.0995 | P95 0.1193 | P99 0.1418
- nanosam_set_image: mean **60.8995** | P50 60.8522 | P95 66.8714 | P99 69.7716
- nanosam_decode   : mean **15.9491** | P50 16.3334 | P95 19.4291 | P99 19.8038
- module sum (mean): 113.3522 | pipeline_e2e mean: 113.6504 | unaccounted_overhead_ms: 0.2982

## D. Cached-image / reused-embedding latency (cached_embedding, ms)
- scenario: set_image ONCE; per iter = NanoOWL box -> bridge -> decoder
- mean **33.8785** | P50 33.6075 | P95 36.267 | P99 36.6775
- NOTE: cached-image scenario, NOT full pipeline FPS.

## E. System resource metrics (tegrastats during E2E load, n=385)
- RAM used MB: mean 3556.164 | max 4154.0
- GR3D FREQ %: mean 52.665 | max 99.0
- GPU temp C  : mean 56.838 | max 58.062
- CPU temp C  : mean 55.033 | max 56.375
- VDD_IN avg mW: mean 9804.174 | max 13067.0
- EMC FREQ %  : N/A (N/A if not reported)

## Correctness
- Every tested iteration: selected label = an owl, mask non-empty, no NaN/Inf.
- errors recorded: 0
