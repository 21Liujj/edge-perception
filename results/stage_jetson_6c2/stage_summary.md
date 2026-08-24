# Stage Jetson-6C-2 — 50-Epoch NanoSAM STANDARD500 Quality Evaluation

## Verdict: NEAR_PARITY

## 1. Frozen Protocol
- Manifest SHA `19a00a83…f015` (n=500, unique=455, seed=2027) — match
- Evaluator `scripts/eval_nanosam_coco.py`, fixed slot0, GT-box prompt, same decoder/preprocess/IoU as 5B-2
- Decoder `afb5ac49…` unchanged; no rebuild/retrain/upstream edit

## 2. 50ep STANDARD500 Result
- mean IoU **0.715650** | median 0.782134 | P10 0.382154 | P90 0.918736
- IoU>=0.50 0.852 | IoU>=0.75 0.582
- valid 500 / failed 0

## 3. Three-Way Comparison
| model | mean IoU |
|---|---|
| 10ep self-trained | 0.707340 |
| **50ep self-trained** | **0.715650** |
| historical deployed | 0.718128 |

- 50ep − 10ep = **+0.008310** (relative +1.175%)
- 50ep − historical = **-0.002478**
- remaining gap (historical − 50ep) = **+0.002478**

## 4. Training Efficiency (additional 40 epochs, epoch10→49)
- additional wall ≈ 19.21 h
- absolute mIoU gain = +0.008310 | relative +1.175%
- gain per 10 epochs = +0.002078 | gain per hour = +0.000433
- feature loss 0.0005354 → 0.0004241 (−20.8%)

## 5. Size Breakdown (50ep / 10ep / historical, delta 50−10)
| size | n | 50ep | 10ep | historical | 50−10 |
|---|---|---|---|---|---|
| small | 175 | 0.597306 | 0.586296 | 0.605320 | +0.011010 |
| medium | 166 | 0.762776 | 0.751911 | 0.762659 | +0.010865 |
| large | 159 | 0.796703 | 0.794032 | 0.795796 | +0.002671 |

## 6. Paired 10→50 (same 500 instances)
- n 500 | improved 298 | degraded 199 | equal 3
- mean delta +0.008310 | median +0.003742 | P95 abs 0.102624

## 7. Paired 50 vs Historical
- n 500 | improved 235 | degraded 262 | equal 3
- mean delta -0.002478 | median -0.000506

## 8. Runtime
- wall 53.4s | 9.370 inst/s | encoder set-calls 455 | reuse 45
- evaluation throughput, NOT deployment FPS.

## 9. Quality Gate
- **NEAR_PARITY** → recommendation **STOP_LONG_TRAINING**
- gap to historical +0.002478 | gain vs 10ep +0.008310

## Out of Scope (enforced STOP)
FULL COCO, 100/200 epoch training, INT8, NanoOWL, C++, engine/decoder rebuild — NOT performed.

## Next
Recommendation (advisory, NOT executed): **STOP_LONG_TRAINING**.
