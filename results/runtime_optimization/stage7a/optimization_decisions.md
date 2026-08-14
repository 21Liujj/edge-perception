# Stage 7A — Optimization Decisions (ACCEPTED / REJECTED)

Stage 6C bottleneck context: overall GPU idle 46.1%, set_image GPU idle ~36.7ms,
H2D ~44.7 copies/iter.

## OPT1 — Pinned host staging buffer (ACCEPTED, narrow)
- `torch.zeros(..., pin_memory=True)`, verified `is_pinned()==True`.
- Reused each frame instead of `torch.zeros(1,3,1024,1024)` per frame.
- Evidence: removes per-frame 12.5MB pad allocation; small RAM cost +27MB peak.
- Caveat: benefit only materializes with non_blocking H2D + overlap; in the
  sequential joint path the staging copy adds overhead vs baseline direct .cuda().

## OPT2 — Preallocated GPU input tensor reused (ACCEPTED)
- `self._image_tensor_gpu` reused; no per-frame GPU input alloc.
- Correctness: identical fp32 values (COCO mIoU delta = 0.0).

## OPT3 — GPU-resident mean/std constants cached (ACCEPTED)
- Built once; not rebuilt per frame (baseline rebuilt `torch.tensor([...])[:,None,None]` each call).
- Note: in the chosen no-stream-deferred normalize variant we still normalize on
  host in-place (to keep semantics identical); mean/std cache is used in set_image
  path where applicable. Net correctness preserved.

## OPT4 — Static decoder inputs cached on device (ACCEPTED, strong)
- `mask_input_zero`, `has_mask_input_zero/one`, `labels_gpu` cached on device.
- **This the largest measurable win**: H2D copies/iter 44.7 -> 4.0 (nsight).
- Correctness: shapes verified (point_coords (1,-1,2), point_labels (1,-1),
  mask_input (1,1,256,256), has_mask_input (1,)); mask_nonzero identical.

## OPT5 — Explicit CUDA stream for set_image (REJECTED for joint; mixed)
- Implemented with `torch.cuda.Stream` + Event, proper dependency (event.synchronize
  before decoder reads features; s.synchronize before upscale) — NO fake async.
- set_image isolated benchmark: speedup 1.075x (P95 -4.5ms) — modest improvement.
- **Joint benchmark: 0.950x (slower), P95 +0.8ms worse** — the stream's
  event.synchronize()/s.synchronize() overhead plus staging-copy boundary crossing
  the default stream exceeds benefit when there is no concurrent work to overlap
  (sequential joint: NanoOWL runs, then NanoSAM; no overlap window).
- STREAM_LIMITATION note: torch2trt TRTModule uses `torch.cuda.current_stream()`
  internally, so an explicit stream IS honored (no monkey-patch needed). The
  limitation is that overlapping only helps if producer work runs concurrently;
  in this offline single-stream benchmark it does not.
- Decision: stream variant retained for set_image-isolated use (ACCEPTED there);
  REJECTED as default for the joint pipeline. A `use_stream=False` variant is
  the safe default for joint. (Buffer reuse OPT1-4,6 kept either way.)

## OPT6 — Point template reuse (ACCEPTED)
- `_points_gpu`/`_points_pinned` templates; only 4 point coords updated per frame
  via pinned copy. Labels static (OPT4).

## Outcome

| optimization | set_image | joint |
|---|---|---|
| OPT1 pinned | part of accepted set | — |
| OPT2 GPU input reuse | ACCEPTED | ACCEPTED |
| OPT3 mean/std cache | ACCEPTED | ACCEPTED |
| OPT4 static decoder cache | STRONG (H2D 44.7->4.0) | STRONG |
| OPT5 explicit stream | ACCEPTED (1.075x) | REJECTED (0.950x, P95 worse) |
| OPT6 point template | ACCEPTED | ACCEPTED |

Net effect on the JOINT pipeline (the production path): **negative (no meaningful
speedup)** despite cutting H2D copies ~11x. Reason: the set_image GPU idle is
dominated by PIL.resize (CPU, 15.5ms) + normalize (CPU, 4.9ms) which no buffer/stream
optimization removes. Cutting small static H2D copies does not open a usable
overlap window because NanoOWL and NanoSAM run sequentially with no concurrent
producer. The honest conclusion: **buffer reuse + pinned + static cache + stream is
insufficient to solve Stage 6C bottleneck 1; only a fused CUDA preprocessing kernel
(next stage) or true producer/consumer cross-frame overlap (Stage 6C candidate 1)
can reduce the 46% GPU idle.**

Per the acceptance policy, this stage's optimized wrapper is therefore NOT made
the default runtime; it is kept as an instrumented, correctness-verified option
(`--runtime optimized`) and the negative joint result is recorded honestly.
