#!/usr/bin/env python3
"""NanoSAM module-level benchmark on Jetson Orin Nano.

Splits latency into model_load / set_image (image encoder) / prompt_decode
(mask decoder) and reports full_segmentation = set_image + first prompt.
All GPU work is followed by torch.cuda.synchronize() before recording time, so
latencies reflect real device execution rather than async kernel launch time.

This is a "repeatedly encode the SAME image" performance experiment (for
characterizing encoder/decoder latency), NOT a production caching strategy.
"""
import os
import csv
import json
import time
import subprocess
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from nanosam.utils.predictor import Predictor

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
ASSETS = os.path.join(ROOT, "third_party/nanosam/assets")
IMG = os.path.join(ASSETS, "dogs.jpg")
ENC = os.path.join(ROOT, "third_party/nanosam/data/resnet18_image_encoder.engine")
DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
OUT = os.path.join(ROOT, "results/nanosam_benchmark")

os.makedirs(OUT, exist_ok=True)

import PIL.Image  # noqa: E402

# Fixed Box Prompt (official basic_usage.py)
BBOX = [100, 100, 850, 759]
BOX_PTS = np.array([[BBOX[0], BBOX[1]], [BBOX[2], BBOX[3]]])
BOX_LABELS = np.array([2, 3])
# Fixed Point Prompt (foreground)
POINT_PTS = np.array([[475, 430]])
POINT_LABELS = np.array([1])

WARMUP_ENC = 20
TEST_ENC = 100
WARMUP_DEC = 30
TEST_BOX = 300
TEST_POINT = 100


def percentiles(lat):
    arr = np.array(lat, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": round(float(arr.mean()), 4),
        "min_ms": round(float(arr.min()), 4),
        "max_ms": round(float(arr.max()), 4),
        "P50_ms": round(float(np.percentile(arr, 50)), 4),
        "P95_ms": round(float(np.percentile(arr, 95)), 4),
        "P99_ms": round(float(np.percentile(arr, 99)), 4),
    }


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def sample_tegrastats(path, interval_us=100000, count=30):
    """Sample tegrastats for ~ (count*interval) ; interval in microseconds."""
    try:
        proc = subprocess.run(
            ["/usr/bin/tegrastats", str(interval_us), str(count)],
            capture_output=True, text=True, timeout=60,
        )
        raw = proc.stdout
    except Exception as e:
        raw = f"TEGRASTATS_ERROR: {e}"
    with open(path, "w") as f:
        f.write(raw)
    return raw


def parse_tegrastats(raw):
    """Best-effort parse of L4T tegrastats lines. Missing fields -> N/A."""
    ram_max = gr3d_max = temp_max = vddin_max = None
    ram_last = gr3d_last = temp_last = vddin_last = None
    import re
    for line in raw.splitlines():
        m = re.search(r"RAM\s+(\d+)/(?:(\d+))?[Mm]", line)
        if m:
            ram_last = m.group(1)
            ram_max = max(int(ram_max or 0), int(ram_last)) or None
        m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if m:
            gr3d_last = int(m.group(1))
            gr3d_max = max(int(gr3d_max or 0), gr3d_last) if gr3d_max is not None or gr3d_last else None
            gr3d_max = gr3d_last if gr3d_max is None else max(gr3d_max, gr3d_last)
        m = re.search(r"(?:CPU_TEMP|GPU_TEMP|temp@)(\d+)", line) or re.search(r"@(\d+)C", line)
        if m:
            temp_last = int(m.group(1))
            temp_max = max(int(temp_max or 0), temp_last)
        m = re.search(r"VDD_IN\s+(\d+)", line)
        if m:
            vddin_last = int(m.group(1))
            vddin_max = max(int(vddin_max or 0), vddin_last)
    return {
        "peak_RAM_MB": ram_max if ram_max else "N/A",
        "last_RAM_MB": ram_last if ram_last else "N/A",
        "peak_GR3D_pct": gr3d_max if gr3d_max is not None else "N/A",
        "last_GR3D_pct": gr3d_last if gr3d_last is not None else "N/A",
        "peak_temperature_C": temp_max if temp_max else "N/A",
        "peak_VDD_IN_mW": vddin_max if vddin_max else "N/A",
        "last_VDD_IN_mW": vddin_last if vddin_last else "N/A",
    }


def main():
    result = {"platform_note": "repeatedly-encoded same image performance experiment"}

    # ---- model load ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    p = Predictor(ENC, DEC)
    torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - t0) * 1000.0
    result["model_load_ms"] = round(model_load_ms, 3)
    print(f"model_load_ms={model_load_ms:.3f}", flush=True)

    image = PIL.Image.open(IMG).convert("RGB")

    # ---- A. Image Encoder / set_image (warmup + test, same image) ----
    for _ in range(WARMUP_ENC):
        p.set_image(image)
        torch.cuda.synchronize()
    enc_lat = []
    for _ in range(TEST_ENC):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        p.set_image(image)
        torch.cuda.synchronize()
        enc_lat.append((time.perf_counter() - t0) * 1000.0)
    result["set_image"] = percentiles(enc_lat)
    print(f"encoder mean={result['set_image']['mean_ms']} P95={result['set_image']['P95_ms']}", flush=True)
    write_csv(os.path.join(OUT, "encoder_latencies.csv"),
              ["iter", "set_image_ms"],
              [[i, round(v, 4)] for i, v in enumerate(enc_lat)])

    # cache embedding once: set_image already executed last in loop; features ready.
    cached_features = p.features
    # expose cached embedding to the decoder without re-encoding by reusing the
    # engine on the stored tensor directly via Predictor (predict uses self.features).
    # We avoid re-running set_image in the decoder loop by NOT calling set_image again.

    # ---- B. Prompt Decoder (Box) : warmup + test, fixed cached embedding ----
    for _ in range(WARMUP_DEC):
        p.predict(BOX_PTS, BOX_LABELS)
        torch.cuda.synchronize()
    box_lat = []
    for _ in range(TEST_BOX):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        p.predict(BOX_PTS, BOX_LABELS)
        torch.cuda.synchronize()
        box_lat.append((time.perf_counter() - t0) * 1000.0)
    result["box_decode"] = percentiles(box_lat)
    result["box_decode"]["estimated_prompt_fps"] = round(
        1000.0 / max(result["box_decode"]["mean_ms"], 1e-6), 2)
    print(f"box decode mean={result['box_decode']['mean_ms']} "
          f"fps={result['box_decode']['estimated_prompt_fps']}", flush=True)
    write_csv(os.path.join(OUT, "decoder_latencies.csv"),
              ["iter", "box_decode_ms"],
              [[i, round(v, 4)] for i, v in enumerate(box_lat)])

    # ---- C. Point Prompt smoke (reuses cached embedding) ----
    for _ in range(WARMUP_DEC):
        p.predict(POINT_PTS, POINT_LABELS)
        torch.cuda.synchronize()
    pt_lat = []
    for _ in range(TEST_POINT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        p.predict(POINT_PTS, POINT_LABELS)
        torch.cuda.synchronize()
        pt_lat.append((time.perf_counter() - t0) * 1000.0)
    result["point_decode"] = percentiles(pt_lat)
    result["point_decode"]["estimated_prompt_fps"] = round(
        1000.0 / max(result["point_decode"]["mean_ms"], 1e-6), 2)
    print(f"point decode mean={result['point_decode']['mean_ms']}", flush=True)
    # append point latencies to the same decoder csv
    with open(os.path.join(OUT, "decoder_latencies.csv"), "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "point_decode_ms"])
        w.writerows([[i, round(v, 4)] for i, v in enumerate(pt_lat)])

    # ---- full_segmentation: real encoder + first prompt path ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    p.set_image(image)
    p.predict(BOX_PTS, BOX_LABELS)
    torch.cuda.synchronize()
    full_ms = (time.perf_counter() - t0) * 1000.0
    result["full_segmentation_ms"] = round(full_ms, 3)
    result["full_segmentation_note"] = (
        "real encoder + first box prompt path; NOT prompt-only FPS expressed as pipeline FPS")
    print(f"full_segmentation_ms={full_ms:.3f}", flush=True)

    # ---- tegrastats ----
    print("sampling tegrastats ...", flush=True)
    raw_ts = sample_tegrastats(os.path.join(OUT, "tegrastats.log"), 100000, 30)
    result["tegrastats"] = parse_tegrastats(raw_ts)

    with open(os.path.join(OUT, "benchmark.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ---- summary.md ----
    s = result
    md = [
        "# NanoSAM Module Benchmark (Jetson Orin Nano)",
        "",
        "Note: this benchmark repeatedly encodes the SAME image to characterize "
        "encoder/decoder latency. It is not a production caching strategy. "
        "All GPU timing is taken after `torch.cuda.synchronize()`.",
        "",
        "## Module latencies (ms)",
        f"- model_load: **{s['model_load_ms']}** ms",
        f"- set_image (image encoder, n={s['set_image']['count']}): "
        f"mean **{s['set_image']['mean_ms']}** | min {s['set_image']['min_ms']} | "
        f"max {s['set_image']['max_ms']} | P50 {s['set_image']['P50_ms']} | "
        f"P95 {s['set_image']['P95_ms']} | P99 {s['set_image']['P99_ms']}",
        f"- box_decode (mask decoder, n={s['box_decode']['count']}): "
        f"mean **{s['box_decode']['mean_ms']}** | min {s['box_decode']['min_ms']} | "
        f"max {s['box_decode']['max_ms']} | P50 {s['box_decode']['P50_ms']} | "
        f"P95 {s['box_decode']['P95_ms']} | P99 {s['box_decode']['P99_ms']}",
        f"  - estimated prompt FPS = 1000 / mean = **{s['box_decode']['estimated_prompt_fps']}**",
        f"- point_decode (n={s['point_decode']['count']}): "
        f"mean **{s['point_decode']['mean_ms']}** | P50 {s['point_decode']['P50_ms']} | "
        f"P95 {s['point_decode']['P95_ms']} | P99 {s['point_decode']['P99_ms']}",
        f"  - estimated prompt FPS = **{s['point_decode']['estimated_prompt_fps']}**",
        f"- full_segmentation (encoder + first box prompt): **{s['full_segmentation_ms']}** ms",
        "  - {note}".format(note=s["full_segmentation_note"]),
        "",
        "## tegrastats (peak during sampling)",
        f"- RAM peak: {s['tegrastats']['peak_RAM_MB']} MB (last {s['tegrastats']['last_RAM_MB']})",
        f"- GR3D peak: {s['tegrastats']['peak_GR3D_pct']}% (last {s['tegrastats']['last_GR3D_pct']})",
        f"- temperature peak: {s['tegrastats']['peak_temperature_C']} C",
        f"- VDD_IN peak: {s['tegrastats']['peak_VDD_IN_mW']} mW (last {s['tegrastats']['last_VDD_IN_mW']})",
        "",
        "## Prompt FPS caveat",
        "Prompt FPS = 1000 / mean decoder latency reflects reuse of a cached image "
        "embedding only. It is NOT the NanoSAM end-to-end pipeline FPS; "
        "full pipeline latency includes the encoder path (see full_segmentation_ms).",
    ]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
