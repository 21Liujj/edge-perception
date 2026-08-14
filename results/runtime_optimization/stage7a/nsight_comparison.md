# Stage 7A — Nsight Systems Re-profile (baseline vs optimized)

nsys profile `--trace=cuda,osrt,nvtx`, warmup 20 / measured 15 iters, owl image
756x499, prompts "an owl,a glove,a finger", target "an owl". Stage 6C baseline
profile reused for comparison; Stage 7A profile is the optimized runtime.

## Per-stage GPU busy (nsight, NVTX + CUPTI)

| stage | baseline wall ms | baseline GPU busy % | opt wall ms | opt GPU busy % |
|---|---|---|---|---|
| preprocess | 3.4 | 0.0 | 2.18 | 0.0 |
| nanoowl | 35.5 | ~85 | 36.4 | 77.8 |
| nanoowl_postprocess | 0.02 | 0.0 | 0.02 | 0.0 |
| bridge | 0.06 | 0.0 | 0.07 | 0.0 |
| **nanosam_set_image** | 55.7 | 34.1 | 57.92 | 29.5 |
| nanosam_decode | 16.2 | ~82 | 15.71 | 82.0 |
| pipeline_total | ~112 | — | 115.89 | — |

| aggregate | baseline | optimized |
|---|---|---|
| overall GPU busy | 53.9% | 50.5% |
| overall GPU idle | 46.1% | 49.5% |
| set_image GPU busy ms | 19.0 | 17.08 |
| set_image GPU idle ms | ~36.7 | ~40.8 |

## H2D copies (cuda_gpu_mem_size_sum, host→device)

| | baseline | optimized |
|---|---|---|
| H2D copies per iter | **44.7** | **4.0** |
| H2D total count in profile | ~670 | 60 |
| H2D total bytes in profile | — | 205,720,140 (~206MB) |

The OPT4 static decoder-input cache (mask_input_zero, has_mask_input, labels)
dropped per-iter H2D copies from 44.7 to 4.0 — the clearest measurable win.
However this did NOT translate to lower set_image wall or higher overall GPU busy,
because the removed copies were small/frequent (latency-amortized) and the dominant
host gap (PIL.resize 15.5ms CPU + normalize 4.9ms CPU) was untouched.

## Key finding

- **H2D churn reduced ~11x** (44.7 → 4.0 /iter) — objective win on copy count.
- **set_image GPU idle went UP slightly** (36.7 → 40.8ms): the optimized wrapper's
  staging pinned copy + stream boundary adds host-side work that lands in the GPU-idle
  window, so GPU busy% on set_image DROPPED 34.1% → 29.5%.
- **Overall GPU busy went DOWN** 53.9% → 50.5%: consistent with the joint benchmark
  being slower (0.950x). The optimization's own overhead offset the copy-count
  reduction in the sequential offline benchmark.

## Conclusion

Re-profiling confirms the audit conclusion: the Stage 6C GPU-idle root cause is
host-side PIL.resize + CPU normalize, not H2D churn. Cutting H2D copies 11x is a
real improvement in data-path efficiency but does not improve wall latency or GPU
utilization on this single-stream benchmark. The bottleneck would only move with a
**fused CUDA preprocessing kernel** (PIL.resize + normalize on GPU) — recommended
next stage, NOT implemented this stage (per constraint: no custom CUDA kernel).

## Evidence

- profile: `nsight_optimized/edge_perception.nsys-rep`, `edge_perception.sqlite`
- stats CSV: `cuda_api_sum`, `cuda_gpu_kern_sum`, `cuda_gpu_mem_size_sum`,
  `cuda_gpu_mem_time_sum`, `osrt_sum`, `nvtx_sum` (all in nsight_optimized/)
- analysis: `nsight_optimized/profile_analysis.json`, `profile_meta.json`
- Stage 6C baseline profile preserved (per "preserve Stage 6C evidence" requirement).
