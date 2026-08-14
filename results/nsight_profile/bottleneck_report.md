# EdgePerception Nsight Systems Profiling — Bottleneck Report

Stage 6C profiling evidence (Nsight Systems 2024.5.4, JetPack R36.4.3, CUDA 12.6).
Workload: `scripts/profile_edge_perception.py`, fixed owl_glove_small.jpg, prompts
"an owl,a glove,a finger", target "an owl", mask slot 0, warmup 20, profile 15 iters.
Mean pipeline_total = 116.84 ms/iter. NO image/video/JSON-heavy I/O during profile.

Source data:
- `cuda_api_summary_clean.json`, `cuda_gpu_kern_sum.csv/txt`, `cuda_gpu_mem_*_sum.txt`
- `per_stage_gpu_busy.json`, `gpu_idle_analysis.json`, `sync_analysis.json`
- `edge_perception.nsys-rep`

---

# Bottleneck 1

Observed evidence:
- `nanosam_set_image` is the largest single stage: wall 55.70 ms/iter (47.7% of
  pipeline_total), but GPU is busy only 34.1% of that window (gpu_busy 18.97 ms).
  => ~36.7 ms/iter of GPU idle inside this stage.
- GPU-wide: profile phase GPU busy 53.9%, idle 46.1% (808.6 ms idle over
  1755 ms). Idle gaps exist; top5 gaps 35-40 ms align with the cudaDeviceSynchronize
  wait points that flush set_image work, followed by CPU host-side work.
- `cudaDeviceSynchronize`: 110 calls, avg 9.12 ms, max 107.7 ms. Of these ~77 are
  measurement sync added by the profiling script (2/iter x (20+15) + load); the
  ~33 upstream device syncs plus the large per-call wait indicate CPU is blocked
  waiting on the device then doing host work while GPU idles.

Likely cause:
- NanoSAM `Predictor.set_image` does CPU-side host work around the TensorRT image
  encoder enqueue: image preprocessing on host + enqueue + synchronization to
  collect the image embedding, plus Python/Torch dispatch overhead. The host path
  between GPU bursts is not overlapped with GPU compute, so the GPU sits idle
  ~36 ms inside set_image. This is the dominant source of GPU idle and the largest
  single latency contributor.

Candidate optimization:
- Overlap host preprocessing of the NEXT frame's image with the CURRENT frame's
  NanoSAM decode (producer/consumer or pipeline overlap), and/or precompute the
  NanoSAM image embedding asynchronously while NanoOWL runs. Also remove the
  measurement-only device sync — but the bulk idle is host-work-induced, not
  sync-call-induced.

Expected metric affected:
- E2E latency (P50/P95), GPU utilization (raise idle 46% -> higher), live_e2e
  latency in video runtime modes.

Confidence: high

---

# Bottleneck 2

Observed evidence:
- `cudaMemcpyAsync` is the top CUDA API by time: 33.4%, 1229 calls, total 1.117 s.
  MemOps: H2D = 670 copies, total 264 ms, total 1337 MB; median copy only 0.003 MB
  while max 174 MB => a long tail of many SMALL host->device copies interleaved
  with few large ones.
- H2D count (670) over 15 profile iters (~45 H2D/iter) suggests repeated small
  per-op tensor uploads (prompt/image/scratch), many of which stream-serialize
  on default stream and cannot overlap kernels efficiently.

Likely cause:
- Many small H2D copies are issued per inference (host tensor staging for prompt
  tokens, point prompts, intermediate buffers). Frequent small copies add
  serialization points and prevent memcpy/kernel overlap on the default stream.

Candidate optimization:
- Pin host buffers and reuse/preallocate staging tensors; batch small uploads;
  keep prompt encodings persistent on device (text_encodings already cached, but
  per-frame point-prompt tensors are recreated each call). Avoid repeated
  allocation/upload of constant-shaped small inputs.

Expected metric affected:
- E2E latency (P95/P99, tail), memcpy total time, GPU idle between bursts.

Confidence: medium

---

# Bottleneck 3

Observed evidence:
- `cudaStreamSynchronize`: 1027 calls, avg 213 us, total 0.219 s (6.5%). These
  are ALL upstream (the profiling script never calls cudaStreamSynchronize) and
  correlate with the TRT `Using default stream in enqueueV3()` warning.
- Multi-stream is actually used by TRT internally (memcpy on 6 streams, kernels
  on 8 streams), so the workload is NOT fully serialized — but enqueueV3's default
  stream usage inserts ~1027 lightweight stream syncs that fragment the timeline
  and are a source of micro-gaps (p50 idle gap 5 us, but many of them).

Likely cause:
- TensorRT enqueueV3 invoked on the default stream forces internal
  cudaStreamSynchronize guards for correctness. Each is individually cheap (213 us)
  but collectively fragments the schedule and prevents some overlap. NOTE: per the
  requirement, this is flagged by timeline evidence (1027 upstream stream syncs),
  NOT by the warning text alone. It is a secondary effect, not the primary
  bottleneck (Bottleneck 1 dominates).

Candidate optimization:
- Pass an explicit non-default CUDA stream to the TensorRT execution contexts
  (requires an integration change in our own wrapper, NOT upstream edits) so
  enqueueV3 runs on a managed stream and the guarding stream syncs are removed,
  enabling dequeue/stream overlap.

Expected metric affected:
- GPU utilization (micro-gap reduction), P95 latency, pipeline overlap potential.

Confidence: medium

---

## Bottlenecks NOT supported by evidence (reported for completeness)

- CPU preprocessing as a top bottleneck: REJECTED. preprocess wall 1.76 ms/iter,
  GPU busy 0% there but the absolute cost is tiny (1.5% of pipeline). nanoowl_postprocess
  + bridge = ~0.1 ms combined. Not a material bottleneck.
- "default stream IS the bottleneck": PARTIALLY SUPPORTED but secondary. The
  stream syncs are real (1027 calls) but cheap (213 us avg, 6.5% total). The
  dominant idle is host-work inside nanosam_set_image (Bottleneck 1), not the
  stream syncs themselves. We do NOT conclude "default stream is the bottleneck".
- Python overhead: present (inter-stage gap only 3.4% of pipeline, so Python
  dispatch between NVTX stages is small); not a top-3 bottleneck by itself.

---

## Day 7 Optimization Candidates (profile-driven, NOT to be implemented this stage)

Based on the evidence, the 1-2 most worthwhile candidates for actual construction
in Day 7 (pending explicit authorization):

1. (Primary) Overlap NanoSAM image-encoder host path / next-frame preprocessing
   with current-frame GPU work (pipeline overlap / producer-consumer with async
   embedding). Targets Bottleneck 1 — the dominant 46% GPU idle. Highest expected
   payoff: raise GPU utilization, cut E2E latency toward the GPU-busy bound.

2. (Secondary) Explicit non-default CUDA stream for TensorRT enqueue (our wrapper
   only, no upstream edits) + pinned/reused host staging buffers for small H2D
   copies. Targets Bottlenecks 2 & 3.

Selection is driven by profiling (Bottleneck 1 high-confidence, dominant), not by
convenience. These are candidates only — Day 7 is NOT started this stage.
