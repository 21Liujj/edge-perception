# Stage E3 Summary: NanoOWL VLM Runtime Benchmark

- Cold prompt (full pipeline per new prompt) mean: 79.115 ms (n=50, 5 prompts x 10 reps, fixed image)
- Warm prompt (cached text embedding, steady state) mean: 22.097 ms, P50=22.003, P95=22.671, P99=24.044 (n=300, fps=45.25)
- Language branch (tokenize+text_encode) mean: 42.012 ms
- Vision TRT (encode_image on TRT engine) mean: 33.965 ms
- Matching+postprocess (decode()) mean: 3.341 ms
- Prompt cache absolute saving: 57.017 ms (72.1% relative)

## CORE_INFERENCE_ONLY vs VLM_END_TO_END

Historical narrow benchmark (`results/nanoowl_benchmark/benchmark.json`, if present): mean=22.491 ms, fps=44.46. That number already reuses cached text_encodings (encoded once before the timed loop in scripts/benchmark_nanoowl.py) and times predictor.predict() as a whole black box (image preprocess + roi_extract + vision TRT + decode, NOT including text tokenize/encode). This is CORE_INFERENCE_ONLY / per-frame-with-precomputed-text, structurally the same boundary as this stage's WARM_PROMPT mode.
This stage's WARM_PROMPT mean (22.097 ms) is the directly comparable steady-state number to the historical narrow benchmark; any difference (see historical mean above) comes from stage-separated instrumentation overhead (extra Python calls / boundary crossing / sync() calls at each measured seam) and fixed-image rotation vs a single repeated image, not a different pipeline.
COLD_PROMPT (79.115 ms mean) is a genuinely different, wider boundary: it additionally includes tokenize+text_encode (~42.012 ms) that the historical narrow benchmark and this stage's WARM_PROMPT mode both exclude via caching.

## Answers

**Is the historical ~22.5ms really system latency?** No — it is CORE_INFERENCE_ONLY with the text prompt already cached (scripts/benchmark_nanoowl.py encodes text once before the timed loop). It does not include tokenize/text-encode (42.012 ms) — a genuinely cold prompt (this stage's COLD_PROMPT mode) costs 79.115 ms end-to-end.
**How much does the language branch cost?** ~42.012 ms (tokenize 0.909 ms + text encode 41.103 ms) per genuinely new prompt, versus effectively 0 ms per-frame once cached.
**What is the prompt cache effect?** Caching the text embedding removes the language branch from the per-frame path, saving 57.017 ms (72.1%) per inference relative to re-encoding the prompt every call.
**Final steady per-frame latency?** 22.097 ms mean (P50=22.003, P95=22.671, P99=24.044) with a cached prompt, steady-state, 300 measured iterations over 20 rotating fixed images.
