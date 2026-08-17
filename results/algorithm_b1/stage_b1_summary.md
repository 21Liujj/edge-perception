# Stage B1 Summary

## Decision

`KEEP_FIXED_SLOT0_BASELINE`

On the fixed Day 5 STANDARD manifest, predicted-IoU argmax reduced mIoU from `0.718128` to `0.661484` (`-0.056644`). It selected slot 2 for all 500 instances, improved 126, and degraded 374. The historical slot-0 policy remains the deployment baseline.

## Policy results

| Policy | mIoU | Small | Medium | Large |
|---|---:|---:|---:|---:|
| Fixed slot 0 | 0.718128 | 0.605320 | 0.762659 | 0.795796 |
| Predicted-IoU argmax | 0.661484 | 0.492933 | 0.740791 | 0.764198 |
| Oracle best-of-4 | 0.748094 | 0.640204 | 0.786785 | 0.826446 |

Oracle headroom over slot 0 is `+0.029966` mIoU. Candidate selection can recover some quality, but the deployed predicted-IoU scores do not identify those candidates reliably.

## IoU head findings

Across 2,000 candidate masks, predicted versus actual IoU has Pearson `-0.023302`, Spearman `-0.031903`, and MAE `0.201619`. Argmax agrees with the oracle on 91/500 instances (`18.2%`). Scores systematically make slot 2 the maximum for every instance, while the oracle distribution is slot 0/1/2/3 = 199/158/91/52.

This result applies to the accepted NanoSAM student encoder plus MobileSAM FP32 mask decoder contract and the fixed COCO GT-box protocol. It does not establish that all SAM IoU heads are uninformative.

## Small objects

Predicted argmax reduces small-object mIoU by `0.112387`. Oracle improves it by only `0.034884`, from `0.605320` to `0.640204`, which remains below medium and large. Mask selection explains a limited part of the small-object gap. Candidate explanations for the remaining gap include embedding spatial resolution, decoder mask capacity, and box-prompt ambiguity; B1 does not isolate one root cause.

## Failure patterns

The worst categories in this fixed subset are skis (n=6, 0.330), dining table (n=6, 0.362), couch (n=6, 0.387), scissors (n=6, 0.412), snowboard (n=6, 0.433). The mined examples show thin/elongated objects, very small objects, and ambiguous or composite box regions among severe failures. Large oracle gains also appear when one alternative slot captures a substantially different object extent, including couch, pizza, chair, tennis racket, and dining table cases. These are descriptive patterns from the fixed sample, not causal proof.

## Runtime

Selection-only benchmark with 100 warmups and 1,000 synchronized iterations measured median `127.001 us` for fixed slot 0 and `261.895 us` for predicted argmax, a median delta of `134.315 us`. The overhead is measurable in this microbenchmark but negligible relative to the full image encoder/decoder pipeline. Runtime cost is not the rejection reason; the quality regression is.
