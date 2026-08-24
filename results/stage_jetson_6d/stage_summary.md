# Stage Jetson-6D — 50-Epoch NanoSAM FULL COCO Formal Validation

## Verdict: NEAR_PARITY — Distillation Module: COMPLETE

## 1. Frozen FULL Protocol
- Manifest `nanosam_coco_full_instances.json` SHA `ad8843c78b3e…` — match (legal 36335, images 4952)
- Evaluator = B3 frozen `evaluate()` + `analyze()`, fixed slot0, clipped GT-box prompt [2,3], same decoder/preprocess/IoU
- Encoder swapped ONLY to 50ep FP16; decoder `afb5ac49…` unchanged; no rebuild/retrain/upstream edit
- B3 `main()` standard-gate bypassed (would exit 10: 50ep != STD_MIOU by design); `analyze()` called directly

## 2. 50ep FULL Result
- mean IoU **0.699183** | median 0.747113 | P10 0.408485 | P90 0.909632
- IoU>=0.50 0.836219 | IoU>=0.75 0.494647
- valid 36335 / failed 0

## 3. Historical FULL Comparison
| metric | 50ep | historical | delta |
|---|---|---|---|
| mean IoU | 0.699183 | 0.703201 | -0.004018 |
| median | 0.747113 | 0.751699 | -0.004586 |
| P10 | 0.408485 | 0.415992 | -0.007507 |
| P90 | 0.909632 | 0.910241 | -0.000610 |
| >=0.5 | 0.836219 | 0.842136 | -0.005917 |
| >=0.75 | 0.494647 | 0.503454 | -0.008807 |

- absolute delta -0.004018 | relative -0.5714%

## 4. Size Breakdown (50ep / historical / delta)
| size | n | 50ep | historical | delta |
|---|---|---|---|---|
| small | 15264 | 0.613713 | 0.621048 | -0.007335 |
| medium | 12392 | 0.741617 | 0.744661 | -0.003044 |
| large | 8679 | 0.788913 | 0.788489 | +0.000424 |

- STANDARD500 size trend on FULL: **PARTIAL**
  (small_near_par=False, medium_near_par=True, large_down=False)

## 5. Paired 50ep vs Historical (per-instance)
- n 36335 | improved 16464 | degraded 19715 | equal 156
- mean delta -0.004018 | median -0.001626 | P95 abs 0.096318

## 6. Runtime
- wall 1988.2s | 18.275 inst/s | 2.491 img/s
- encoder calls 4952 | decoder calls 36335 | reuse 31383
- peak RSS 1.09GB | CUDA alloc 0.04GB | reserved 0.07GB
- evaluation throughput, NOT deployment FPS

## 7. Quality Gate
- **NEAR_PARITY** (abs gap 0.004018) → Distillation Module **COMPLETE**
- recommendation (advisory, NOT executed): **STOP_LONG_TRAINING**

## Out of Scope (STOP)
100/200ep, INT8, NanoOWL, C++, rebuild — NOT performed.
