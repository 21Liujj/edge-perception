# Realtime Video Runtime Strategy Comparison

Input: `results/video_pipeline/baseline/input.mp4` (synthetic Stage 6A video, **runtime-controlled benchmark; NOT an algorithm benchmark**).

Source FPS: 10.0, source frames: 300 (60 frames/loop x 5 loops), resolution 756x498.

Realtime replay: producer releases each frame at source-FPS pace (100 ms apart at 10 FPS), **not disk-decode speed**; source timestamps monotonic across loops.

Output video writing DISABLED during benchmark (VideoWriter I/O would pollute runtime timing).

Processing throughput is offline pipeline throughput under realtime-paced input; it is **not** a claim of real-time camera FPS.

## Summary table

| Mode | Processed FPS | Processed | Dropped | Drop Rate | Live Lat P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Queue Wait P95 (ms) | Frame Age P95 (ms) | Max Q Depth | Producer Blocked (ms) | RAM Peak (MB) | GR3D Mean (%) | GPU Temp Peak (C) | CPU Temp Peak (C) | Power Mean (mW) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sequential | 7.87 | 300 | 0 | 0.0 | 4130.4721 | 7868.4417 | 8172.9415 | 8235.8598 | N/A | 0.013 | N/A | N/A | 3502.0 | 54.54 | 55.75 | 54.06 | 8140.86 |
| fifo | 8.51 | 300 | 0 | 0.0 | 2763.5627 | 5027.5754 | 5316.0689 | 5367.1083 | 1100.0805 | 1100.1024 | 8 | 28.295 | 3629.0 | 51.53 | 57.06 | 55.34 | 8340.84 |
| latest | 8.59 | 260 | 40 | 0.1333 | 289.006 | 334.726 | 343.9827 | 353.2929 | 95.7433 | 95.7972 | N/A | N/A | 3683.0 | 55.17 | 58.19 | 56.25 | 8417.07 |

## Processing time (ms)

| Mode | mean | P50 | P95 | P99 | max |
|---|---|---|---|---|---|
| sequential | 126.5256 | 125.6984 | 142.7638 | 149.678 | 267.9454 |
| fifo | 117.0597 | 119.1918 | 124.2192 | 135.5678 | 151.7561 |
| latest | 115.8565 | 117.641 | 123.2128 | 133.6644 | 139.0189 |

## Model load (single init, identical across modes)

- NanoOWL load: 1459.486 ms
- NanoSAM load: 627.364 ms
- Total: 2086.85 ms

## Correctness checks (all modes)

- sequential: detection_frames=300, nan_inf_frames=0, drop_reason recorded=n/a
- fifo: detection_frames=300, nan_inf_frames=0, drop_reason recorded=n/a
- latest: detection_frames=260, nan_inf_frames=0, drop_reason recorded=superseded_by_newer_frame
