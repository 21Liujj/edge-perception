# Joint EdgePerception Pipeline Benchmark (Jetson Orin Nano)

One process; models/engines loaded once. Fixed input owl_glove_small.jpg, prompts=an owl,a glove,a finger, target=an owl, thr=0.1. Mask selection = official slot 0 (not IoU argmax). All GPU timing after torch.cuda.synchronize(). Excludes load/I/O/viz.

## A. Model load latency
- total model_load_ms: **2220.798** (nanoowl 1594.971 + nanosam 625.831)
- warmup: 30 (not counted) | test iterations: 300

## B. Steady-state complete joint inference latency (pipeline_e2e, ms)
- mean **117.0351** | min 87.8657 | max 139.8605 | P50 115.8755 | P95 132.4133 | P99 136.7383
- steady-state E2E FPS = 1000/mean = **8.54** (inference pipeline FPS, excludes disk I/O / visualization)

## C. Module latency breakdown (mean ms, n=300)
- nanoowl: mean **38.6133** | P50 37.7168 | P95 44.9096 | P99 45.1861
- bridge : mean **0.1046** | P50 0.1035 | P95 0.1148 | P99 0.1457
- nanosam_set_image: mean **60.7769** | P50 60.2287 | P95 70.437 | P99 72.9323
- nanosam_decode   : mean **17.2227** | P50 16.6307 | P95 20.0158 | P99 20.6568
- module sum (mean): 116.7175 | pipeline_e2e mean: 117.0351 | unaccounted_overhead_ms: 0.3176

## D. Cached-image / reused-embedding latency (cached_embedding, ms)
- scenario: set_image ONCE; per iter = NanoOWL box -> bridge -> decoder
- mean **34.0178** | P50 33.5994 | P95 36.3087 | P99 40.0005
- NOTE: cached-image scenario, NOT full pipeline FPS.

## E. System resource metrics (tegrastats during E2E load, n=190)
- RAM used MB: mean 4139.795 | max 4154.0
- GR3D FREQ %: mean 51.542 | max 99.0
- GPU temp C  : mean 56.204 | max 57.031
- CPU temp C  : mean 54.41 | max 55.375
- VDD_IN avg mW: mean 9749.558 | max 13067.0
- EMC FREQ %  : N/A (N/A if not reported)

## Correctness
- Every tested iteration: selected label = an owl, mask non-empty, no NaN/Inf.
- errors recorded: 0
