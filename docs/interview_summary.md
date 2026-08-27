# EdgePerception — Interview Summary

## 30-second version

I built EdgePerception, an open-vocabulary vision-language perception pipeline on Jetson Orin: NanoOWL (text-conditioned detection, OWL-ViT) chained into NanoSAM (promptable segmentation, MobileSAM-distilled ResNet18), both deployed with TensorRT FP16. Given an image and a text prompt like "cup," it returns a pixel mask — no fixed category list. I trained the segmentation student via knowledge distillation, quantified the FP16/INT8 quality-latency tradeoff, built a C++ TensorRT runtime, and ran a 50-case COCO evaluation that decomposes exactly where the joint pipeline loses quality: mostly detection-box imprecision propagating into segmentation, not the segmenter itself.

## 2-minute version

The project has two halves that had to be engineered independently before being joined. NanoSAM: I distilled a MobileSAM teacher into a ResNet18 student over 50 epochs, converted to ONNX then TensorRT, and validated FP16 quality on the full COCO val2017 set (36,335 instances, mIoU 0.699 vs 0.703 historical — near-parity). I also ran an INT8 PTQ study: 1.845x encoder speedup, but a −2.15% mIoU regression that fails my own quality gate (<0.5% threshold), so production stays on FP16 and INT8 is documented as a latency-critical opt-in, not a default.

NanoOWL: I froze its zero-shot detection quality on full COCO (AP 0.259) and then did a runtime audit that caught a benchmarking artifact — a historical "22.5ms" number was actually a warm-cache measurement with the text branch already encoded, not true cold-prompt latency (79.1ms, 3.6x higher). That distinction matters operationally: a robot repeatedly searching for the same object class benefits enormously from prompt caching (72% latency reduction), but a robot switching prompts every frame does not.

Joining them required verifying, from source, that NanoOWL's output box convention (xyxy, original-image pixels) and NanoSAM's input point convention (also original-image pixels) agree with no axis swap — I didn't assume this, I traced both `predict()` implementations. Then I built the production pipeline script, fixed a real bug where it defaulted to an untrained NanoSAM encoder instead of my trained one, and ran a 50-case stratified COCO evaluation (by object size and scene crowding) comparing the full pipeline against a GT-box "ceiling." The pipeline gets a mask most of the time (84% detection rate) but the mask quality drops from a 0.75 ceiling to 0.38 end-to-end — and the error decomposition shows that's overwhelmingly because NanoOWL's box isn't tight enough, not because NanoSAM fails given a good box (only 3/50 are pure segmentation failures). Steady-state joint latency is ~114ms (8.8 FPS) on Jetson, dominated by NanoSAM's image encoder.

## 5-minute version

(Builds on the 2-minute version — walk through in order if asked to go deeper)

**Model/training layer**: MobileSAM teacher, ResNet18 student, feature-distillation loss (not just mask loss), 10-epoch checkpoint as a fast baseline then 50-epoch as the deployment candidate. Measured the actual training curve, not just the final number: feature loss dropped from 0.001122 to 0.0004241, STANDARD500 mIoU from 0.707 to 0.716 between 10 and 50 epochs — a documented diminishing-returns slope that justified stopping at 50 rather than assuming "more epochs = better" without evidence.

**Conversion/graph layer**: PyTorch → ONNX → TensorRT for both models. I didn't just trust the conversion — I ran numerical parity audits at each hop (PyTorch vs ONNX Runtime, ONNX vs TensorRT) and reported max-abs-diff, not just "it works." NanoOWL's image encoder is TensorRT; its text side deliberately stays in PyTorch because that's the CLIP text tower — the project doesn't force everything into TensorRT if it doesn't buy anything (the text encoder is where prompt caching, not engine conversion, is the real lever).

**Runtime layer**: Python production pipeline plus a separate C++ TensorRT runtime for NanoSAM (TensorRT 10.x, enqueueV3, RAII CUDA buffer management, CMake build). The C++ runtime has both an image-mode CLI and a tensor-mode CLI that bypasses preprocessing entirely for correctness verification — I found and documented a real precision delta (OpenCV bilinear resize vs PIL resize, 5.1% mean relative L2) rather than papering over it, and showed that delta doesn't affect tensor-mode parity because both runtimes then consume identical frozen tensors.

**Quantization layer**: INT8 PTQ with a calibration cache bound to ONNX SHA + preprocessing version + TensorRT version (so a silent calibration/model mismatch can't happen), gated by a two-tier quality heuristic (≥−0.5% acceptable, −0.5% to −1.5% degraded-but-usable, <−1.5% blocking). INT8 failed that gate here, and I kept that result rather than only reporting the speedup — the honest engineering answer is "1.8x faster, not deployed by default, here's why."

**System integration layer**: the joint pipeline, its coordinate-bridge proof, the 50-case evaluation with error decomposition, and a caught production bug (wrong default encoder) that a purely "does it run" smoke test would have missed — it only surfaced because I checked which SHA the pipeline was actually loading against the SHA of the model I'd validated.

**What I'd do next**: ROS2 node wrapping the same pipeline for message-based robot integration, porting NanoOWL's image encoder into the C++ runtime so the full VLM pipeline (not just NanoSAM) has a C++ path, and QAT instead of PTQ to see if the INT8 quality gate can actually be passed.

## 20 likely interview questions and answers

**1. Why chain two separate models instead of one end-to-end model?**
NanoSAM alone can't do "find the X" — it's a promptable segmenter with no language understanding. NanoOWL alone can't produce pixel-accurate masks — it's a box detector. Chaining gives open-vocabulary instance segmentation without training a new joint model, and each half can be validated, versioned, and swapped independently.

**2. How do you know the coordinate bridge between the two models is correct, not just "it looks right"?**
I read `OwlPredictor.predict()` and `Predictor.predict()` from source and confirmed both operate in original-image pixel space with no swap needed. Then I confirmed it empirically two ways: a smoke test producing a coherent non-degenerate mask at the right location, and the 50-case COCO eval where box_iou between NanoOWL and GT boxes averages 0.51 — consistent with correct placement, not a mirrored/transposed bug (which would show near-zero IoU with a location pattern, not scattered misses).

**3. What's the actual bottleneck in the joint pipeline?**
NanoSAM's image encoder, at ~61ms of a ~114ms total — more than half. NanoOWL's detection is ~36ms, decode is ~16ms. If I were optimizing next, that's where I'd start (INT8 for NanoSAM's encoder specifically, or resolution reduction).

**4. Why does INT8 fail your quality gate — what would you do about it?**
PTQ calibration on a distilled student compounds two lossy steps (distillation, then quantization). −2.15% crossed my −0.5%/−1.5% thresholds. I'd try QAT next, or calibrate on a larger/more representative image set, before concluding INT8 is unusable here — right now it's "not validated for production," not "impossible."

**5. Explain the "22ms vs 79ms" discrepancy you found.**
A historical benchmark measured NanoOWL's core inference loop with the text prompt already encoded and cached before timing started — so it measured only the vision branch. Real cold-prompt latency, including PyTorch text encoding, is 79.1ms — 3.6x higher. This matters because whether caching helps depends entirely on whether the robot's use case repeats prompts.

**6. What's prompt caching and when does it help?**
NanoOWL's text embedding is deterministic for a fixed prompt, so I cache it and skip re-encoding text on every frame. It cuts latency 72% (79.1ms → 22.1ms) for repeated-prompt use cases (a robot searching for the same object over many frames) but doesn't help if the prompt changes every call.

**7. Why is mask quality only 0.38 IoU end-to-end when NanoSAM alone gets 0.70+?**
Different regime: NanoSAM's 0.70 mIoU is measured with ground-truth boxes. The joint pipeline uses NanoOWL's own (imperfect) boxes. My error decomposition shows 17/50 cases are "localization propagation" — NanoSAM could segment well given a good box, but NanoOWL's box wasn't tight enough — versus only 3/50 pure segmentation failures. The bottleneck is detection localization, not segmentation quality.

**8. Why fixed slot-0 mask selection instead of the decoder's own predicted-IoU argmax?**
I measured it: predicted-IoU argmax scores are degenerate (99.98% of predictions land in the same slot regardless of instance, and argmax agrees with the oracle-best-of-4 only 82.7% of the time). Fixed slot-0 mIoU (0.699) actually beats the argmax policy (0.630) on full COCO. I didn't assume the "smarter-sounding" policy was better — I measured both.

**9. How did you validate the C++ runtime against the Python one?**
Two checks: tensor-mode parity (both runtimes consume an identical frozen preprocessed tensor — encoder embedding max-abs-diff <1e-5, decoder mask agreement >99.9%) isolates TensorRT-engine correctness from preprocessing. Image-mode regression on 20 instances shows a small IoU delta (~0.71 vs ~0.70) attributable to a documented preprocessing difference (OpenCV vs PIL resize filter), not an engine bug.

**10. What would break if you fed this a fisheye/rotated camera image instead of COCO-style photos?**
Detection quality would likely drop — NanoOWL was zero-shot evaluated on COCO-distribution images, and I have no evidence about out-of-distribution robustness. That's a real limitation I'd flag, not something the current evaluation covers.

**11. Why is your NanoOWL AP (0.26) much lower than a trained detector like YOLO?**
It's a zero-shot open-vocabulary detector — no COCO-specific fine-tuning — trading category flexibility for accuracy. Best categories (cat 0.62, train 0.59) do reasonably; worst (mouse 0.00, knife 0.03) reflect genuine language-grounding and small-object limitations, not a training bug.

**12. What's the biggest engineering mistake you caught in this project?**
The production pipeline script defaulted to an untrained upstream NanoSAM encoder instead of my trained 50-epoch model — different SHA256, and it would have silently shipped wrong quality. I caught it by explicitly diffing the loaded engine's SHA against the validated model's SHA, not by "the demo looked fine."

**13. How do you know your 50-case evaluation sample isn't cherry-picked?**
Deterministic seed (2027), stratified 3x3 across object size and scene crowding sampled from the full valid-instance pool, not hand-picked images. The sampling manifest is saved so it's re-derivable and auditable.

**14. What's NOT done — where would a rigorous reviewer find gaps?**
No ROS2 integration (no message-passing robot interface yet), no camera real-time pipeline (offline images/video only), no QAT, NanoOWL's text encoder isn't in the C++ runtime, and out-of-distribution robustness beyond COCO-style images is unevaluated.

**15. Why measure both a "ceiling" (GT-box → NanoSAM) and the real end-to-end pipeline?**
To isolate which model in the chain is responsible for quality loss. Without the ceiling, a low end-to-end IoU is ambiguous — is NanoOWL or NanoSAM at fault? With it, I can attribute: NanoSAM given a good box performs close to its standalone numbers; the delta is overwhelmingly NanoOWL's box imprecision.

**16. How do you handle NanoOWL predicting a box that extends outside the image?**
`clip_box()` clamps to `[0,W]x[0,H]` before handing to NanoSAM, and the output JSON records both the clipped and original box plus an `out_of_bounds` flag — the clipping is auditable, not silent.

**17. Why does NanoSAM's own worst category involve thin objects like skis and forks?**
Box/point prompts and mask decoders both struggle with elongated, thin structures — consistent across both the historical baseline and my distilled student, so it's a model-family characteristic, not something introduced by distillation.

**18. What's the actual production latency budget and does 8.8 FPS meet it?**
Depends entirely on the target use case — I report the number with its protocol (single-object, steady-state, one image at a time, no batching) rather than claim a use-case-specific verdict. For a robot doing "look, then act" rather than continuous tracking, ~114ms per query may be acceptable; for continuous multi-object tracking it would not be, and the cached-embedding number (34ms) shows the achievable floor if the vision embedding is reused.

**19. Why not fine-tune NanoOWL on COCO to raise its AP?**
That would defeat the open-vocabulary point — fine-tuning on COCO's 80 fixed categories reintroduces the fixed-category-list limitation this pipeline exists to avoid. The zero-shot AP is the honest number for the capability being deployed.

**20. If you had one more week, what would you build?**
Port NanoOWL's image encoder into the C++ runtime so the full VLM pipeline (not just NanoSAM) has a C++ path, and try QAT on NanoSAM to see if INT8 can actually pass the quality gate rather than staying a documented-but-unused option.
