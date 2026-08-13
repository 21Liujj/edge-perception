# -*- coding: utf-8 -*-
"""Stage 4B - Stable-state E2E benchmark of the NanoOWL -> NanoSAM joint pipeline.

One process, models loaded ONCE, engines loaded ONCE. Fixed input image + fixed
prompts + fixed target label. Measures steady-state E2E latency and a
module-level breakdown (nanoowl / bridge / nanosam_set_image / nanosam_decode),
plus a cached-image (reused-embedding) experiment.

Reuses the validated inference logic + bridge + selection from
scripts/run_edge_perception.py (no subprocess, no copy of third_party core).

Timing boundaries (per iteration), all GPU work followed by torch.cuda.synchronize():
  nanoowl_ms          : image preprocess + NanoOWL image encode(TRT) + decode +
                        postprocess -> selected box (text encodings reused/cached)
  bridge_ms           : selected box -> clip/coord convert -> NanoSAM box prompt
  nanosam_set_image_ms: NanoSAM image preprocess + image encoder -> embedding ready
  nanosam_decode_ms   : (embedding + box prompt) -> selected mask ready (slot 0)
  pipeline_e2e_ms     : from this iteration's ready input -> final selected mask
Excludes: model load, disk I/O, JSON write, visualization.

Mask selection: official fixed slot 0 (NOT IoU argmax). INT8/optimization off.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import PIL.Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from run_edge_perception import select_detection, clip_box  # noqa: E402 (validated logic)
from nanoowl.owl_predictor import OwlPredictor  # noqa: E402
from nanosam.utils.predictor import Predictor as NanosamPredictor  # noqa: E402

DEFAULT_IMAGE = os.path.join(ROOT, "third_party/nanoowl/assets/owl_glove_small.jpg")
DEFAULT_OWL_ENGINE = os.path.join(ROOT, "third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
DEFAULT_NANOSAM_ENC = os.path.join(ROOT, "third_party/nanosam/data/resnet18_image_encoder.engine")
DEFAULT_NANOSAM_DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
MODEL = "google/owlvit-base-patch32"

PROMPTS = "an owl,a glove,a finger"
TARGET_LABEL = "an owl"
THRESHOLD = 0.10

WARMUP = 30
ITERS = 300
CACHED_ITERS = 100


def percentiles(lat):
    arr = np.asarray(lat, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": round(float(arr.mean()), 4),
        "min_ms": round(float(arr.min()), 4),
        "max_ms": round(float(arr.max()), 4),
        "P50_ms": round(float(np.percentile(arr, 50)), 4),
        "P95_ms": round(float(np.percentile(arr, 95)), 4),
        "P99_ms": round(float(np.percentile(arr, 99)), 4),
    }


def syc():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---- tegrastats (JetPack6 --start --interval ms --logfile) ----
def tegrastats_start(logpath, interval_ms=100):
    subprocess.run(["/usr/bin/tegrastats", "--start", "--interval", str(interval_ms),
                    "--logfile", logpath],
                   capture_output=True, text=True, timeout=30)


def tegrastats_stop():
    subprocess.run(["/usr/bin/tegrastats", "--stop"], capture_output=True, text=True, timeout=30)


def parse_tegrastats(raw):
    """Best-effort parse; missing fields -> N/A."""
    ram, gru, gr3d, gput, cput, vdd, emc = [], [], [], [], [], [], []
    for line in raw.splitlines():
        m = re.search(r"RAM\s+(\d+)/\d+MB", line)
        if m:
            ram.append(int(m.group(1)))
        m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if m:
            gr3d.append(int(m.group(1)))
        m = re.search(r"gpu@([\d.]+)C", line)
        if m:
            gput.append(float(m.group(1)))
        m = re.search(r"cpu@([\d.]+)C", line)
        if m:
            cput.append(float(m.group(1)))
        m = re.search(r"VDD_IN\s+(\d+)mW/(\d+)mW", line)
        if m:
            vdd.append(int(m.group(2)))  # average column
        m = re.search(r"EMC_FREQ\s+(\d+)%", line)
        if m:
            emc.append(int(m.group(1)))

    def stat(a):
        if not a:
            return {"mean": "N/A", "min": "N/A", "max": "N/A", "count": 0}
        a = np.asarray(a, dtype=float)
        return {"mean": round(float(a.mean()), 3), "min": round(float(a.min()), 3),
                "max": round(float(a.max()), 3), "count": int(a.size)}

    return {
        "RAM_used_MB": stat(ram),
        "GR3D_FREQ_pct": stat(gr3d),
        "GPU_temp_C": stat(gput),
        "CPU_temp_C": stat(cput),
        "VDD_IN_avg_mW": stat(vdd),
        "EMC_FREQ_pct": stat(emc),
    }


def detect_once(owl, image, text, text_enc, threshold, target_label):
    """NanoOWL detect + postprocess-select -> (selected dict|None, num_dets, label)."""
    output = owl.predict(image=image, text=text, text_encodings=text_enc,
                         threshold=threshold, pad_square=False)
    syc()
    labels = output.labels.detach().cpu().tolist()
    scores = output.scores.detach().cpu().tolist()
    boxes = [[float(v) for v in b] for b in output.boxes.detach().cpu().tolist()]
    dets = []
    for li, sc, bx in zip(labels, scores, boxes):
        li = int(li)
        dets.append({"label_index": li,
                     "label": text[li] if 0 <= li < len(text) else str(li),
                     "score": float(sc), "box": bx})
    dets.sort(key=lambda d: d["score"], reverse=True)
    sel, _ = select_detection(dets, target_label)
    return sel, len(dets), dets


def main():
    outdir = os.path.join(ROOT, "results/joint_benchmark")
    os.makedirs(outdir, exist_ok=True)

    # ---- load models ONCE (timed) ----
    syc()
    t0 = time.perf_counter()
    owl = OwlPredictor(MODEL, image_encoder_engine=DEFAULT_OWL_ENGINE)
    syc()
    t1 = time.perf_counter()
    nsam = NanosamPredictor(DEFAULT_NANOSAM_ENC, DEFAULT_NANOSAM_DEC)
    syc()
    model_load_ms = (time.perf_counter() - t0) * 1000.0
    owl_load_ms = (t1 - t0) * 1000.0
    nanosam_load_ms = (time.perf_counter() - t1) * 1000.0

    image = PIL.Image.open(DEFAULT_IMAGE).convert("RGB")
    W, H = image.size
    text = [t.strip() for t in PROMPTS.split(",") if t.strip()]
    # text encodings: encode once (prompts fixed) -> reused for steady-state
    text_enc = owl.encode_text(text)
    syc()

    print(f"loaded: owl={owl_load_ms:.1f}ms nanosam={nanosam_load_ms:.1f}ms "
          f"total={model_load_ms:.1f}ms img={W}x{H}", flush=True)

    # correctness sentinel from a known run
    sel0, n0, _ = detect_once(owl, image, text, text_enc, THRESHOLD, TARGET_LABEL)
    print(f"sentinel: n_dets={n0} sel={sel0['label'] if sel0 else None} "
          f"score={sel0['score'] if sel0 else None:.4f}", flush=True) if sel0 else \
        print(f"sentinel: NO_DETECTION n_dets={n0}", flush=True)

    # ============ A. full E2E pipeline (warmup + test) ============
    print(f"warmup {WARMUP} ...", flush=True)
    e2e_rows = []
    mod_rows = []
    nanowl_lat, bridge_lat, ns_set_lat, ns_dec_lat, e2e_lat = [], [], [], [], []
    errors = []

    for i in range(-(WARMUP), ITERS):
        syc()
        t_start = time.perf_counter()

        # nanoowl: ready input -> selected box
        syc()
        t_a = time.perf_counter()
        sel, ndet, _dets = detect_once(owl, image, text, text_enc, THRESHOLD, TARGET_LABEL)
        syc()
        t_b = time.perf_counter()

        if sel is None:
            errors.append({"iter": i, "error": "NO_DETECTION during benchmark"})
            if i >= 0:
                print(f"FAIL iter {i}: NO_DETECTION", flush=True)
                raise SystemExit("Stage FAIL: NO_DETECTION in benchmark iteration")
            continue

        # bridge: box -> clip/conv -> box prompt
        t_c = time.perf_counter()
        clipped, oob, _orig = clip_box(sel["box"], W, H)
        pts = np.array([[clipped[0], clipped[1]], [clipped[2], clipped[3]]])
        labs = np.array([2, 3])
        t_d = time.perf_counter()

        # nanosam set_image: image preprocess + image encoder -> embedding
        syc()
        t_e = time.perf_counter()
        nsam.set_image(image)
        syc()
        t_f = time.perf_counter()

        # nanosam decode: embedding + prompt -> mask (slot 0)
        syc()
        t_g = time.perf_counter()
        hi, iou, low = nsam.predict(pts, labs)
        syc()
        t_h = time.perf_counter()

        syc()
        t_end = time.perf_counter()

        # correctness checks (only fatal -> FAIL)
        mask_bool = (hi[0, 0] > 0).detach().cpu().numpy()
        nz = int(mask_bool.sum())
        hi_f = hi.detach().float().cpu().numpy()
        has_nan = bool(np.isnan(hi_f).any())
        has_inf = bool(np.isinf(hi_f).any())
        if sel["label"] != TARGET_LABEL or nz == 0 or has_nan or has_inf:
            errors.append({"iter": i, "label": sel["label"], "nonzero": nz,
                           "nan": has_nan, "inf": has_inf})
            if i >= 0:
                print(f"FAIL iter {i}: label={sel['label']} nz={nz} nan={has_nan} inf={has_inf}",
                      flush=True)
                raise SystemExit("Stage FAIL: correctness check failed in benchmark iteration")

        if i < 0:
            continue  # warmup, not recorded

        n_ms = (t_b - t_a) * 1000.0
        br_ms = (t_d - t_c) * 1000.0
        nse_ms = (t_f - t_e) * 1000.0
        nsd_ms = (t_h - t_g) * 1000.0
        e_ms = (t_end - t_start) * 1000.0
        nanowl_lat.append(n_ms); bridge_lat.append(br_ms)
        ns_set_lat.append(nse_ms); ns_dec_lat.append(nsd_ms); e2e_lat.append(e_ms)
        e2e_rows.append([i, round(e_ms, 4)])
        mod_rows.append([i, round(n_ms, 4), round(br_ms, 4),
                         round(nse_ms, 4), round(nsd_ms, 4)])

    print(f"tested {len(e2e_lat)} iters, all checks passed", flush=True)

    nw = percentiles(nanowl_lat)
    bw = percentiles(bridge_lat)
    nse = percentiles(ns_set_lat)
    nsd = percentiles(ns_dec_lat)
    e2e = percentiles(e2e_lat)
    e2e_fps = round(1000.0 / max(e2e["mean_ms"], 1e-6), 2)
    sum_mean = nw["mean_ms"] + bw["mean_ms"] + nse["mean_ms"] + nsd["mean_ms"]
    unaccounted = round(e2e["mean_ms"] - sum_mean, 4)

    # ============ B. cached-embedding experiment ============
    # set_image ONCE -> reuse embedding; per iter: NanoOWL box -> bridge -> decode
    print(f"cached-embedding experiment {CACHED_ITERS} ...", flush=True)
    # warmup
    for _ in range(20):
        sel, _, _ = detect_once(owl, image, text, text_enc, THRESHOLD, TARGET_LABEL)
        syc()
        clipped, _, _ = clip_box(sel["box"], W, H)
        _pts = np.array([[clipped[0], clipped[1]], [clipped[2], clipped[3]]])
        nsam.predict(_pts, np.array([2, 3]))
        syc()
    # cache embedding once
    syc()
    nsam.set_image(image)
    syc()
    cached_lat = []
    for i in range(CACHED_ITERS):
        syc()
        t0 = time.perf_counter()
        sel, _, _ = detect_once(owl, image, text, text_enc, THRESHOLD, TARGET_LABEL)
        clipped, _, _ = clip_box(sel["box"], W, H)
        _pts = np.array([[clipped[0], clipped[1]], [clipped[2], clipped[3]]])
        nsam.predict(_pts, np.array([2, 3]))
        syc()
        cached_lat.append((time.perf_counter() - t0) * 1000.0)
    cached = percentiles(cached_lat)

    # ============ C. tegrastats during full 300-iter E2E ============
    # Run a fresh 300-iter pass under the tegrastats logger to capture load.
    ts_log = os.path.join(outdir, "tegrastats.log")
    tegrastats_start(ts_log, 100)
    time.sleep(0.3)
    # brief load run (~ enough samples, ~120 samples => ~12s at 100ms)
    LOAD_ITERS = 200
    for _ in range(LOAD_ITERS):
        sel, _, _ = detect_once(owl, image, text, text_enc, THRESHOLD, TARGET_LABEL)
        clipped, _, _ = clip_box(sel["box"], W, H)
        _pts = np.array([[clipped[0], clipped[1]], [clipped[2], clipped[3]]])
        nsam.set_image(image)
        nsam.predict(_pts, np.array([2, 3]))
    syc()
    time.sleep(0.5)
    tegrastats_stop()
    with open(ts_log) as f:
        raw_ts = f.read()
    ts_parsed = parse_tegrastats(raw_ts)
    with open(os.path.join(outdir, "tegrastats_parsed.json"), "w") as f:
        json.dump(ts_parsed, f, indent=2)

    # ============ evidence files ============
    write_csv(os.path.join(outdir, "e2e_latencies.csv"),
              ["iter", "pipeline_e2e_ms"], e2e_rows)
    write_csv(os.path.join(outdir, "module_latencies.csv"),
              ["iter", "nanoowl_ms", "bridge_ms", "nanosam_set_image_ms", "nanosam_decode_ms"],
              mod_rows)
    write_csv(os.path.join(outdir, "cached_embedding_latencies.csv"),
              ["iter", "cached_pipeline_ms"],
              [[i, round(v, 4)] for i, v in enumerate(cached_lat)])

    bench = {
        "model_load_ms": round(model_load_ms, 3),
        "nanoowl_load_ms": round(owl_load_ms, 3),
        "nanosam_load_ms": round(nanosam_load_ms, 3),
        "warmup": WARMUP,
        "iterations": ITERS,
        "input_image": DEFAULT_IMAGE,
        "prompts": text,
        "target_label": TARGET_LABEL,
        "threshold": THRESHOLD,
        "mask_selection_rule": "fixed slot 0 (basic_usage.py), not IoU argmax",
        "timing_units": "ms; all GPU timing after torch.cuda.synchronize()",
        "nanoowl": nw,
        "bridge": bw,
        "nanosam_set_image": nse,
        "nanosam_decode": nsd,
        "pipeline_e2e": e2e,
        "steady_state_e2e_fps": e2e_fps,
        "module_sum_mean_ms": round(sum_mean, 4),
        "unaccounted_overhead_ms": unaccounted,
        "cached_embedding_experiment": {
            "note": "cached-image / reused-embedding scenario (set_image ONCE); NOT full pipeline FPS",
            "iterations": CACHED_ITERS,
            **cached,
        },
        "tegrastats": ts_parsed,
        "errors": errors,
    }
    with open(os.path.join(outdir, "benchmark.json"), "w") as f:
        json.dump(bench, f, indent=2)

    # ---- summary.md ----
    md = [
        "# Joint EdgePerception Pipeline Benchmark (Jetson Orin Nano)",
        "",
        "One process; models/engines loaded once. Fixed input owl_glove_small.jpg, "
        f"prompts={PROMPTS}, target={TARGET_LABEL}, thr={THRESHOLD}. "
        "Mask selection = official slot 0 (not IoU argmax). "
        "All GPU timing after torch.cuda.synchronize(). Excludes load/I/O/viz.",
        "",
        "## A. Model load latency",
        f"- total model_load_ms: **{bench['model_load_ms']}** "
        f"(nanoowl {bench['nanoowl_load_ms']} + nanosam {bench['nanosam_load_ms']})",
        f"- warmup: {WARMUP} (not counted) | test iterations: {ITERS}",
        "",
        "## B. Steady-state complete joint inference latency (pipeline_e2e, ms)",
        f"- mean **{e2e['mean_ms']}** | min {e2e['min_ms']} | max {e2e['max_ms']} | "
        f"P50 {e2e['P50_ms']} | P95 {e2e['P95_ms']} | P99 {e2e['P99_ms']}",
        f"- steady-state E2E FPS = 1000/mean = **{e2e_fps}** "
        "(inference pipeline FPS, excludes disk I/O / visualization)",
        "",
        "## C. Module latency breakdown (mean ms, n=%d)" % ITERS,
        f"- nanoowl: mean **{nw['mean_ms']}** | P50 {nw['P50_ms']} | P95 {nw['P95_ms']} | P99 {nw['P99_ms']}",
        f"- bridge : mean **{bw['mean_ms']}** | P50 {bw['P50_ms']} | P95 {bw['P95_ms']} | P99 {bw['P99_ms']}",
        f"- nanosam_set_image: mean **{nse['mean_ms']}** | P50 {nse['P50_ms']} | P95 {nse['P95_ms']} | P99 {nse['P99_ms']}",
        f"- nanosam_decode   : mean **{nsd['mean_ms']}** | P50 {nsd['P50_ms']} | P95 {nsd['P95_ms']} | P99 {nsd['P99_ms']}",
        f"- module sum (mean): {round(sum_mean,4)} | pipeline_e2e mean: {e2e['mean_ms']} | "
        f"unaccounted_overhead_ms: {unaccounted}",
        "",
        "## D. Cached-image / reused-embedding latency (cached_embedding, ms)",
        f"- scenario: set_image ONCE; per iter = NanoOWL box -> bridge -> decoder",
        f"- mean **{cached['mean_ms']}** | P50 {cached['P50_ms']} | P95 {cached['P95_ms']} | P99 {cached['P99_ms']}",
        "- NOTE: cached-image scenario, NOT full pipeline FPS.",
        "",
        "## E. System resource metrics (tegrastats during E2E load, n=%d)" % ts_parsed["GR3D_FREQ_pct"]["count"],
        f"- RAM used MB: mean {ts_parsed['RAM_used_MB']['mean']} | max {ts_parsed['RAM_used_MB']['max']}",
        f"- GR3D FREQ %: mean {ts_parsed['GR3D_FREQ_pct']['mean']} | max {ts_parsed['GR3D_FREQ_pct']['max']}",
        f"- GPU temp C  : mean {ts_parsed['GPU_temp_C']['mean']} | max {ts_parsed['GPU_temp_C']['max']}",
        f"- CPU temp C  : mean {ts_parsed['CPU_temp_C']['mean']} | max {ts_parsed['CPU_temp_C']['max']}",
        f"- VDD_IN avg mW: mean {ts_parsed['VDD_IN_avg_mW']['mean']} | max {ts_parsed['VDD_IN_avg_mW']['max']}",
        f"- EMC FREQ %  : {ts_parsed['EMC_FREQ_pct']['mean']} (N/A if not reported)",
        "",
        "## Correctness",
        "- Every tested iteration: selected label = an owl, mask non-empty, no NaN/Inf.",
        f"- errors recorded: {len(errors)}",
    ]
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print("BENCH_DONE", flush=True)
    print(json.dumps({
        "model_load_ms": bench["model_load_ms"],
        "e2e": e2e, "fps": e2e_fps, "unaccounted": unaccounted,
        "cached_mean": cached["mean_ms"],
        "GR3D_max": ts_parsed["GR3D_FREQ_pct"]["max"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
