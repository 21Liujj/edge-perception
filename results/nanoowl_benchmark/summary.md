# NanoOWL TensorRT Baseline — Steady-state Benchmark (Stage 2C)

Real measurements only. No optimization, no bottleneck interpretation.

## Setup
- Platform: Jetson Orin (L4T R36.4.3 / JetPack 6.2 family), no jetson_clocks, no power-mode change.
- Model: google/owlvit-base-patch32
- Engine: third_party/nanoowl/data/owlvit_image_encoder_patch32.engine (FP16, fixed from Day 1)
- Input: third_party/nanoowl/assets/owl_glove_small.jpg
- Prompts: ["an owl", "a glove", "a finger"], threshold 0.10
- Model loaded once; warmup then timed inferences (no per-iter reload).

## Timing
- model_load_ms: 1860.80
- warmup iters: 40
- benchmark iters: 300

## Latency (ms, end-to-end per inference incl. image preprocess + TRT + decode)
- mean: 22.49
- min: 20.58
- max: 26.07
- p50: 22.47
- p95: 22.85
- p99: 24.38
- estimated FPS = 1000/mean: 44.46

## Validity (per-iter during benchmark)
- inference_errors: 0
- NaN detected in outputs: false
- "an owl" detection present every iter: true

## System resources (tegrastats, sampled 500ms during timed window)
- Samples: 14
- RAM used: min 3116 / max 3124 / mean 3121 MB (of 7620 MB)
- GR3D (GPU) freq%: min 67 / max 94 / mean 79.57
- GPU temp: min 54.81 / max 57.19 / mean 56.26 C
- CPU temp: min 52.03 / max 54.03 / mean 53.24 C
- VDD_IN power: min 14720 / max 16830 / mean 16148.71 mW
- EMC: not reported by tegrastats on this L4T build (recorded as unavailable)

## Post-benchmark regression
- owl baseline re-run: PASS — detected "an owl", score 0.3419235944747925 (matches Day 1 baseline).

## Artifacts
- benchmark.json — full structured results
- latencies.csv — 300 per-iter latency samples
- tegrastats.log — raw tegrastats output
- tegrastats_stats.json — parsed resource stats
- post_bench_regression/ — post-benchmark owl regression evidence
