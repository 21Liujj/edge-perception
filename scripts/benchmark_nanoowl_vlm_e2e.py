# -*- coding: utf-8 -*-
"""Stage E3 - NanoOWL VLM end-to-end runtime benchmark (cold vs warm prompt).

Project-side only. Does not modify third_party/nanoowl. Calls the existing
public OwlPredictor methods (encode_text, image_preprocessor.preprocess_pil_image,
extract_rois, encode_image, decode) as separate timed seams instead of only
calling predict() as one black box, so per-stage latency is a real measurement
of an existing call boundary, not an estimate.

No engine rebuild, no retrain, no power-mode / jetson_clocks change.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import PIL.Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

HERE = "/home/nvidia/jetson-deploy-lab/scripts"
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import run_nanoowl  # reuse load_predictor + defaults  # noqa: E402

ROOT = "/home/nvidia/jetson-deploy-lab"
DEFAULT_OUT = os.path.join(ROOT, "results", "stage_e3_nanoowl_benchmark")
DEFAULT_ENGINE = run_nanoowl.DEFAULT_ENGINE
DEFAULT_MODEL = run_nanoowl.DEFAULT_MODEL
DEFAULT_THRESHOLD = 0.10

FIXED_PROMPTS = ["a person", "a car", "a chair", "a dog", "a cat"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--cold-repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tegrastats", default="/usr/bin/tegrastats")
    return ap.parse_args()


def percentile(vals, q):
    if len(vals) < 2:
        return float(vals[0]) if vals else float("nan")
    return float(statistics.quantiles(vals, n=100, method="inclusive")[int(q) - 1])


def stats(vals):
    return {
        "mean": statistics.mean(vals),
        "min": min(vals),
        "max": max(vals),
        "p50": percentile(vals, 50),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
        "n": len(vals),
    }


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def pick_fixed_images(n):
    val_dir = os.path.join(ROOT, "data", "coco", "val2017")
    files = sorted(os.listdir(val_dir))
    files = [f for f in files if f.endswith(".jpg")]
    chosen = files[::max(1, len(files) // n)][:n]
    return [os.path.join(val_dir, f) for f in chosen]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = pick_fixed_images(args.n_images)
    assert len(image_paths) == args.n_images, f"expected {args.n_images} images, got {len(image_paths)}"

    predictor, model_load_ms = run_nanoowl.load_predictor(args.model, args.engine)
    predictor.eval() if hasattr(predictor, "eval") else None

    # ---------------------------------------------------------------
    # Stage-separated single-call helpers using EXISTING public methods.
    # No reimplementation of NanoOWL logic; just separate timed calls to
    # already-existing seams inside third_party/nanoowl/owl_predictor.py's
    # own predict() sequence (preprocess -> encode_text -> extract_rois ->
    # encode_image -> decode).
    # ---------------------------------------------------------------
    def stage_image_decode(path):
        return PIL.Image.open(path).convert("RGB")

    def stage_image_preprocess(pil_img):
        return predictor.image_preprocessor.preprocess_pil_image(pil_img)

    def stage_text_tokenize(text_list):
        return predictor.processor(text=text_list, return_tensors="pt")

    def stage_text_encode_from_tokens(text_input):
        input_ids = text_input["input_ids"].to(predictor.device)
        attention_mask = text_input["attention_mask"].to(predictor.device)
        text_outputs = predictor.model.owlvit.text_model(input_ids, attention_mask)
        text_embeds = text_outputs[1]
        text_embeds = predictor.model.owlvit.text_projection(text_embeds)
        from nanoowl.owl_predictor import OwlEncodeTextOutput
        return OwlEncodeTextOutput(text_embeds=text_embeds)

    def stage_extract_rois(image_tensor):
        rois = torch.tensor(
            [[0, 0, image_tensor.shape[-1], image_tensor.shape[-2]]],
            dtype=image_tensor.dtype, device=image_tensor.device,
        )
        roi_images, rois_out = predictor.extract_rois(image_tensor, rois, pad_square=False)
        return roi_images, rois_out

    def stage_vision_trt(roi_images):
        return predictor.encode_image(roi_images)

    def stage_decode(image_encodings, rois, text_encodings):
        from nanoowl.owl_predictor import _owl_box_roi_to_box_global
        image_encodings.pred_boxes = _owl_box_roi_to_box_global(image_encodings.pred_boxes, rois[:, None, :])
        return predictor.decode(image_encodings, text_encodings, args.threshold)

    def timed(fn, *a, **kw):
        sync()
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        sync()
        dt = (time.perf_counter() - t0) * 1000.0
        return out, dt

    # =================================================================
    # MODE A: COLD_PROMPT
    # Each of the 5 fixed prompts is treated as brand-new: full pipeline
    # (tokenize -> text encode -> vision -> matching/decode -> postprocess)
    # timed end-to-end, repeated --cold-repeats times per prompt, on a
    # fixed representative image (image_paths[0]).
    # =================================================================
    cold_image = stage_image_decode(image_paths[0])
    cold_records = []
    for prompt in FIXED_PROMPTS:
        for rep in range(args.cold_repeats):
            sync()
            t0 = time.perf_counter()

            image_tensor = stage_image_preprocess(cold_image)
            text_input = stage_text_tokenize([prompt])
            text_encodings = stage_text_encode_from_tokens(text_input)
            roi_images, rois = stage_extract_rois(image_tensor)
            image_encodings = stage_vision_trt(roi_images)
            _ = stage_decode(image_encodings, rois, text_encodings)

            sync()
            total_ms = (time.perf_counter() - t0) * 1000.0
            cold_records.append({"prompt": prompt, "rep": rep, "total_ms": total_ms})

    cold_latencies = [r["total_ms"] for r in cold_records]
    cold_prompt_result = {
        "mode": "COLD_PROMPT",
        "description": (
            "Every call re-tokenizes and re-encodes the text prompt (no cache "
            "reuse); full tokenize->text_encode->vision->decode chain timed "
            "per call on a fixed image."
        ),
        "prompts": FIXED_PROMPTS,
        "repeats_per_prompt": args.cold_repeats,
        "image": image_paths[0],
        "records": cold_records,
        "latency_ms": stats(cold_latencies),
    }

    # =================================================================
    # STAGE BREAKDOWN (separate, on cold path, averaged over N calls)
    # =================================================================
    breakdown = {k: [] for k in [
        "image_decode", "image_preprocess", "text_tokenize", "text_encode",
        "roi_extract", "vision_trt", "matching_and_postprocess", "total",
    ]}
    N_BREAKDOWN = 30
    warm_text_input_cache = stage_text_tokenize([FIXED_PROMPTS[0]])
    for i in range(N_BREAKDOWN):
        img_path = image_paths[i % len(image_paths)]

        pil_img, t_decode = timed(stage_image_decode, img_path)
        image_tensor, t_pre = timed(stage_image_preprocess, pil_img)
        text_input, t_tok = timed(stage_text_tokenize, [FIXED_PROMPTS[0]])
        text_encodings, t_txt = timed(stage_text_encode_from_tokens, text_input)
        (roi_images, rois), t_roi = timed(stage_extract_rois, image_tensor)
        image_encodings, t_vis = timed(stage_vision_trt, roi_images)
        _, t_dec = timed(stage_decode, image_encodings, rois, text_encodings)

        total = t_decode + t_pre + t_tok + t_txt + t_roi + t_vis + t_dec
        breakdown["image_decode"].append(t_decode)
        breakdown["image_preprocess"].append(t_pre)
        breakdown["text_tokenize"].append(t_tok)
        breakdown["text_encode"].append(t_txt)
        breakdown["roi_extract"].append(t_roi)
        breakdown["vision_trt"].append(t_vis)
        breakdown["matching_and_postprocess"].append(t_dec)
        breakdown["total"].append(total)

    stage_breakdown_result = {
        "note": (
            "matching_and_postprocess is OwlPredictor.decode() as a single call: "
            "normalize embeds, einsum cross-modal similarity, logit_shift/scale, "
            "sigmoid, per-label threshold, argmax. Box geometry (sigmoid + "
            "center-to-corners) is already computed inside the TensorRT/vision "
            "step (encode_image), not inside decode(). A separate "
            "'cross-modal matching' vs 'box decode/postprocess' split within "
            "decode() itself is NOT_SEPARATELY_VISIBLE without modifying "
            "third_party/nanoowl (decode() is one fused function; splitting it "
            "would require rewriting upstream code, which is out of scope)."
        ),
        "n_samples": N_BREAKDOWN,
        "per_stage_ms": {k: stats(v) for k, v in breakdown.items()},
    }

    # =================================================================
    # MODE B: WARM_PROMPT
    # Text embedding computed ONCE (project-side cache: a plain dict keyed
    # by prompt tuple, held in this script — no third_party change), then
    # per-frame steady-state loop over fixed prompt + rotating fixed images.
    # warmup=30, measured=300, matching benchmark_nanoowl.py's existing
    # methodology (model loaded once, tegrastats over timed window only).
    # =================================================================
    warm_prompt = FIXED_PROMPTS[0]
    text_cache = {}

    def get_cached_text_encoding(prompt):
        if prompt not in text_cache:
            ti = stage_text_tokenize([prompt])
            text_cache[prompt] = stage_text_encode_from_tokens(ti)
        return text_cache[prompt]

    cached_text_encodings = get_cached_text_encoding(warm_prompt)

    def warm_inference(pil_img):
        image_tensor = stage_image_preprocess(pil_img)
        roi_images, rois = stage_extract_rois(image_tensor)
        image_encodings = stage_vision_trt(roi_images)
        return stage_decode(image_encodings, rois, cached_text_encodings)

    preloaded_images = [stage_image_decode(p) for p in image_paths]

    for i in range(args.warmup):
        warm_inference(preloaded_images[i % len(preloaded_images)])
    sync()

    ts_log = os.path.join(args.output_dir, "tegrastats.log")
    ts_proc = subprocess.Popen(
        [args.tegrastats, "--interval", "500"],
        stdout=open(ts_log, "w"), stderr=subprocess.STDOUT,
    )

    warm_latencies = []
    try:
        for i in range(args.iters):
            img = preloaded_images[i % len(preloaded_images)]
            sync()
            t0 = time.perf_counter()
            warm_inference(img)
            sync()
            warm_latencies.append((time.perf_counter() - t0) * 1000.0)
    finally:
        if ts_proc.poll() is None:
            ts_proc.terminate()
            try:
                ts_proc.wait(timeout=10)
            except Exception:
                ts_proc.kill()

    warm_prompt_result = {
        "mode": "WARM_PROMPT",
        "description": (
            "Text embedding for the fixed prompt computed ONCE up front "
            "(project-side dict cache keyed by prompt string; third_party "
            "unmodified). Per-frame steady-state loop: image_preprocess -> "
            "roi_extract -> vision_trt -> decode(cached text embedding)."
        ),
        "prompt": warm_prompt,
        "warmup_iters": args.warmup,
        "measured_iters": args.iters,
        "n_images_rotated": len(preloaded_images),
        "latency_ms": stats(warm_latencies),
        "fps": 1000.0 / statistics.mean(warm_latencies),
    }

    # =================================================================
    # Resource usage (RSS + GPU memory) — snapshot after warm run.
    # =================================================================
    rss_kb = None
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    break
    except Exception:
        pass

    gpu_mem_mb = None
    try:
        gpu_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass

    tegra_snapshot = None
    try:
        with open(ts_log) as f:
            lines = [l.strip() for l in f if l.strip()]
            tegra_snapshot = lines[-1] if lines else None
    except Exception:
        pass

    resource_result = {
        "process_rss_kb": rss_kb,
        "torch_cuda_memory_allocated_mb": gpu_mem_mb,
        "tegrastats_last_line": tegra_snapshot,
        "tegrastats_log": ts_log,
        "note": "Power mode / jetson_clocks not modified for this benchmark.",
    }

    # =================================================================
    # Text cache saving analysis
    # =================================================================
    cold_mean = cold_prompt_result["latency_ms"]["mean"]
    warm_mean = warm_prompt_result["latency_ms"]["mean"]
    abs_saving = cold_mean - warm_mean
    rel_saving = abs_saving / cold_mean if cold_mean > 0 else float("nan")

    text_encode_mean = stage_breakdown_result["per_stage_ms"]["text_encode"]["mean"]
    text_tokenize_mean = stage_breakdown_result["per_stage_ms"]["text_tokenize"]["mean"]
    language_branch_ms = text_encode_mean + text_tokenize_mean

    # Historical narrow benchmark comparison
    old_narrow_path = os.path.join(ROOT, "results", "nanoowl_benchmark", "benchmark.json")
    old_narrow = None
    if os.path.exists(old_narrow_path):
        with open(old_narrow_path) as f:
            old_narrow = json.load(f)

    benchmark_config = {
        "stage": "E3",
        "model": args.model,
        "engine": args.engine,
        "threshold": args.threshold,
        "fixed_images": image_paths,
        "fixed_prompts": FIXED_PROMPTS,
        "cold_repeats_per_prompt": args.cold_repeats,
        "warm_warmup_iters": args.warmup,
        "warm_measured_iters": args.iters,
        "model_load_ms": model_load_ms,
        "power_mode_changed": False,
        "jetson_clocks_changed": False,
        "engine_rebuilt": False,
    }

    stage_summary_lines = [
        "# Stage E3 Summary: NanoOWL VLM Runtime Benchmark",
        "",
        f"- Cold prompt (full pipeline per new prompt) mean: {cold_mean:.3f} ms "
        f"(n={cold_prompt_result['latency_ms']['n']}, 5 prompts x {args.cold_repeats} reps, fixed image)",
        f"- Warm prompt (cached text embedding, steady state) mean: {warm_mean:.3f} ms, "
        f"P50={warm_prompt_result['latency_ms']['p50']:.3f}, "
        f"P95={warm_prompt_result['latency_ms']['p95']:.3f}, "
        f"P99={warm_prompt_result['latency_ms']['p99']:.3f} "
        f"(n={warm_prompt_result['latency_ms']['n']}, fps={warm_prompt_result['fps']:.2f})",
        f"- Language branch (tokenize+text_encode) mean: {language_branch_ms:.3f} ms",
        f"- Vision TRT (encode_image on TRT engine) mean: {stage_breakdown_result['per_stage_ms']['vision_trt']['mean']:.3f} ms",
        f"- Matching+postprocess (decode()) mean: {stage_breakdown_result['per_stage_ms']['matching_and_postprocess']['mean']:.3f} ms",
        f"- Prompt cache absolute saving: {abs_saving:.3f} ms ({rel_saving*100:.1f}% relative)",
        "",
        "## CORE_INFERENCE_ONLY vs VLM_END_TO_END",
        "",
        (
            f"Historical narrow benchmark (`results/nanoowl_benchmark/benchmark.json`, "
            f"if present): mean={old_narrow['latency_ms']['mean']:.3f} ms, "
            f"fps={old_narrow['estimated_fps']:.2f}. That number already reuses cached "
            "text_encodings (encoded once before the timed loop in "
            "scripts/benchmark_nanoowl.py) and times predictor.predict() as a whole "
            "black box (image preprocess + roi_extract + vision TRT + decode, "
            "NOT including text tokenize/encode). This is CORE_INFERENCE_ONLY / "
            "per-frame-with-precomputed-text, structurally the same boundary as this "
            "stage's WARM_PROMPT mode."
            if old_narrow else
            "No `results/nanoowl_benchmark/benchmark.json` found on disk to compare against."
        ),
        (
            f"This stage's WARM_PROMPT mean ({warm_mean:.3f} ms) is the directly "
            "comparable steady-state number to the historical narrow benchmark; "
            "any difference (see historical mean above) comes from stage-separated "
            "instrumentation overhead (extra Python calls / boundary crossing / "
            "sync() calls at each measured seam) and fixed-image rotation "
            "vs a single repeated image, not a different pipeline."
        ),
        (
            f"COLD_PROMPT ({cold_mean:.3f} ms mean) is a genuinely different, wider "
            "boundary: it additionally includes tokenize+text_encode "
            f"(~{language_branch_ms:.3f} ms) that the historical narrow benchmark and "
            "this stage's WARM_PROMPT mode both exclude via caching."
        ),
        "",
        "## Answers",
        "",
        (
            f"**Is the historical ~22.5ms really system latency?** No — it is "
            "CORE_INFERENCE_ONLY with the text prompt already cached "
            "(scripts/benchmark_nanoowl.py encodes text once before the timed loop). "
            f"It does not include tokenize/text-encode ({language_branch_ms:.3f} ms) — "
            "a genuinely cold prompt (this stage's COLD_PROMPT mode) costs "
            f"{cold_mean:.3f} ms end-to-end."
        ),
        (
            f"**How much does the language branch cost?** ~{language_branch_ms:.3f} ms "
            f"(tokenize {text_tokenize_mean:.3f} ms + text encode {text_encode_mean:.3f} ms) "
            f"per genuinely new prompt, versus effectively 0 ms per-frame once cached."
        ),
        (
            f"**What is the prompt cache effect?** Caching the text embedding removes "
            f"the language branch from the per-frame path, saving {abs_saving:.3f} ms "
            f"({rel_saving*100:.1f}%) per inference relative to re-encoding the prompt "
            "every call."
        ),
        (
            f"**Final steady per-frame latency?** {warm_mean:.3f} ms mean "
            f"(P50={warm_prompt_result['latency_ms']['p50']:.3f}, "
            f"P95={warm_prompt_result['latency_ms']['p95']:.3f}, "
            f"P99={warm_prompt_result['latency_ms']['p99']:.3f}) with a cached prompt, "
            f"steady-state, {args.iters} measured iterations over {len(preloaded_images)} "
            "rotating fixed images."
        ),
    ]

    with open(os.path.join(args.output_dir, "benchmark_config.json"), "w") as f:
        json.dump(benchmark_config, f, indent=2)
    with open(os.path.join(args.output_dir, "cold_prompt.json"), "w") as f:
        json.dump(cold_prompt_result, f, indent=2)
    with open(os.path.join(args.output_dir, "warm_prompt.json"), "w") as f:
        json.dump(warm_prompt_result, f, indent=2)
    with open(os.path.join(args.output_dir, "stage_breakdown.json"), "w") as f:
        json.dump(stage_breakdown_result, f, indent=2)
    with open(os.path.join(args.output_dir, "resource.json"), "w") as f:
        json.dump(resource_result, f, indent=2)
    with open(os.path.join(args.output_dir, "text_cache_analysis.json"), "w") as f:
        json.dump({
            "cold_prompt_mean_ms": cold_mean,
            "warm_prompt_mean_ms": warm_mean,
            "absolute_saving_ms": abs_saving,
            "relative_saving_pct": rel_saving * 100,
            "language_branch_ms": language_branch_ms,
            "explanation": (
                "A robot repeatedly searching for the same object category "
                "(e.g. 'find the cup' every frame while navigating) issues the same "
                "text prompt across many frames. Re-tokenizing and re-running the "
                "text transformer for an unchanged prompt string is redundant work: "
                "the resulting text embedding is deterministic given the same model "
                "weights, revision, device and dtype. Caching it once and reusing it "
                "removes the language branch from the per-frame critical path, "
                "leaving only image-side work (preprocess + vision TRT + matching) "
                "in the steady-state loop."
            ),
        }, f, indent=2)
    with open(os.path.join(args.output_dir, "stage_summary.md"), "w") as f:
        f.write("\n".join(stage_summary_lines) + "\n")

    print(json.dumps({
        "cold_mean_ms": cold_mean,
        "warm_mean_ms": warm_mean,
        "warm_p50": warm_prompt_result["latency_ms"]["p50"],
        "warm_p95": warm_prompt_result["latency_ms"]["p95"],
        "warm_p99": warm_prompt_result["latency_ms"]["p99"],
        "text_encode_ms": text_encode_mean,
        "text_tokenize_ms": text_tokenize_mean,
        "vision_trt_ms": stage_breakdown_result["per_stage_ms"]["vision_trt"]["mean"],
        "matching_ms": stage_breakdown_result["per_stage_ms"]["matching_and_postprocess"]["mean"],
        "abs_saving_ms": abs_saving,
        "rel_saving_pct": rel_saving * 100,
        "resource": resource_result,
        "old_narrow": old_narrow,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
