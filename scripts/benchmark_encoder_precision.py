#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired FP16 vs FP32 NanoSAM image-encoder latency benchmark (Stage 5C).

Same timing boundary / same methodology for both precisions on dogs.jpg:
  - encoder latency: set_image (image encoder only), warmup>=20, iters>=100, cuda-sync
  - full segmentation latency: set_image + first box decode, >=100 iters, cuda-sync
  - capture tegrastats during each precision's steady-state run

Not a real-time FPS claim; repeated-encode latency characterization.
Does NOT rebuild engines, modify upstream, or change the eval protocol.
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
IMG = os.path.join(ROOT, "third_party/nanosam/assets/dogs.jpg")
DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
OUT = os.path.join(ROOT, "results/precision_ablation/nanosam")

ENC_FP16 = os.path.join(ROOT, "third_party/nanosam/data/resnet18_image_encoder.engine")
ENC_FP32 = os.path.join(ROOT, "third_party/nanosam/data/resnet18_image_encoder_fp32.engine")

BBOX = [100, 100, 850, 759]
BOX_PTS = np.array([[BBOX[0], BBOX[1]], [BBOX[2], BBOX[3]]])
BOX_LABELS = np.array([2, 3])

WARMUP = 20
ITERS = 100
FULL_ITERS = 100

import PIL.Image  # noqa: E402


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


def sample_tegrastats(path, interval_ms=100, count=60):
    try:
        proc = subprocess.run(
            ["/usr/bin/tegrastats", "--interval", str(interval_ms)],
            capture_output=True, text=True, timeout=15,
        )
        raw = proc.stdout
    except Exception as e:
        raw = "TEGRASTATS_ERROR: %s" % e
    with open(path, "w") as f:
        f.write(raw)
    return raw


def parse_tegrastats(raw):
    import re
    ram_max = gr3d_max = temp_max = vddin_max = None
    for line in raw.splitlines():
        m = re.search(r"RAM\s+(\d+)/(\d+)", line)
        if m:
            ram_max = max(int(ram_max or 0), int(m.group(1))) or None
        m = re.search(r"GR3D_FREQ\s+(\d+)%@(\d+)", line) or re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if m:
            gr3d_max = max(int(gr3d_max or 0), int(m.group(1))) if gr3d_max is not None else int(m.group(1))
            gr3d_max = max(gr3d_max, int(m.group(1)))
        m = re.search(r"GPU@(-?\d+)C", line) or re.search(r"temp@(-?\d+)C", line) or re.search(r"(@|-)(\d+)C", line)
        if m:
            temp_max = max(int(temp_max or -999), int(m.group(m.lastindex))) if temp_max is not None else int(m.group(m.lastindex))
            temp_max = max(temp_max, int(m.group(m.lastindex)))
        m = re.search(r"VDD_IN\s+(\d+)", line)
        if m:
            vddin_max = max(int(vddin_max or 0), int(m.group(1)))
    return {
        "peak_RAM_MB": ram_max if ram_max else "N/A",
        "peak_GR3D_pct": gr3d_max if gr3d_max is not None else "N/A",
        "peak_temperature_C": temp_max if temp_max else "N/A",
        "peak_VDD_IN_mW": vddin_max if vddin_max else "N/A",
    }


def bench_precision(name, enc_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    image = PIL.Image.open(IMG).convert("RGB")
    p = Predictor(enc_path, DEC)

    # ----- encoder latency (set_image only) -----
    for _ in range(WARMUP):
        p.set_image(image)
        torch.cuda.synchronize()
    enc_lat = []
    for i in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        p.set_image(image)
        torch.cuda.synchronize()
        enc_lat.append((time.perf_counter() - t0) * 1000.0)
    write_csv(os.path.join(outdir, "encoder_latencies.csv"),
              ["iter", "encoder_ms"],
              [[i, round(v, 4)] for i, v in enumerate(enc_lat)])

    # ----- full segmentation latency (set_image + first box decode) -----
    full_lat = []
    for _ in range(WARMUP):
        p.set_image(image)
        torch.cuda.synchronize()
        p.predict(BOX_PTS, BOX_LABELS)
        torch.cuda.synchronize()
    for i in range(FULL_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        p.set_image(image)
        torch.cuda.synchronize()
        p.predict(BOX_PTS, BOX_LABELS)
        torch.cuda.synchronize()
        full_lat.append((time.perf_counter() - t0) * 1000.0)
    write_csv(os.path.join(outdir, "full_segmentation_latencies.csv"),
              ["iter", "full_seg_ms"],
              [[i, round(v, 4)] for i, v in enumerate(full_lat)])

    # ----- tegrastats during a steady encode burst -----
    tegra_raw = sample_tegrastats(os.path.join(outdir, "tegrastats.log"))
    tegra = parse_tegrastats(tegra_raw)
    # run a short burst while stats already captured above; do a fresh one synchronized
    proc = subprocess.Popen(["/usr/bin/tegrastats", "--interval", "100"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    burst = 40
    for _ in range(burst):
        p.set_image(image)
        torch.cuda.synchronize()
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=3)
    except Exception:
        out = ""
    tegra2 = parse_tegrastats(out)
    with open(os.path.join(outdir, "tegrastats.log"), "w") as f:
        f.write(out)

    result = {
        "precision": name,
        "encoder_engine": enc_path,
        "warmup": WARMUP,
        "encoder_iterations": ITERS,
        "full_segmentation_iterations": FULL_ITERS,
        "encoder": percentiles(enc_lat),
        "full_segmentation": percentiles(full_lat),
        "tegrastats": {k: max_t(tegra.get(k), tegra2.get(k)) for k in
                       ["peak_RAM_MB", "peak_GR3D_pct", "peak_temperature_C", "peak_VDD_IN_mW"]},
    }
    with open(os.path.join(outdir, "latency_summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("%s encoder mean=%.3f P50=%.3f P95=%.3f | full_seg mean=%.3f P50=%.3f" % (
        name, result["encoder"]["mean_ms"], result["encoder"]["P50_ms"],
        result["encoder"]["P95_ms"], result["full_segmentation"]["mean_ms"],
        result["full_segmentation"]["P50_ms"]), flush=True)
    return result


def max_t(a, b):
    vals = [v for v in [a, b] if isinstance(v, (int, float))]
    if vals:
        return max(vals)
    return "N/A"


def main():
    fp16 = bench_precision("fp16", ENC_FP16, os.path.join(OUT, "fp16"))
    fp32 = bench_precision("fp32", ENC_FP32, os.path.join(OUT, "fp32"))

    def speedup(a, b):
        if a and b and isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > 0:
            return round(a / b, 4)
        return None

    lat = {
        "fp16_encoder_P50_ms": fp16["encoder"]["P50_ms"],
        "fp16_encoder_P95_ms": fp16["encoder"]["P95_ms"],
        "fp32_encoder_P50_ms": fp32["encoder"]["P50_ms"],
        "fp32_encoder_P95_ms": fp32["encoder"]["P95_ms"],
        "fp16_encoder_mean_ms": fp16["encoder"]["mean_ms"],
        "fp32_encoder_mean_ms": fp32["encoder"]["mean_ms"],
        "fp16_full_segmentation_P50_ms": fp16["full_segmentation"]["P50_ms"],
        "fp32_full_segmentation_P50_ms": fp32["full_segmentation"]["P50_ms"],
        "fp16_full_segmentation_mean_ms": fp16["full_segmentation"]["mean_ms"],
        "fp32_full_segmentation_mean_ms": fp32["full_segmentation"]["mean_ms"],
        "encoder_P50_speedup_fp32_over_fp16": speedup(fp32["encoder"]["P50_ms"], fp16["encoder"]["P50_ms"]),
        "encoder_mean_speedup_fp32_over_fp16": speedup(fp32["encoder"]["mean_ms"], fp16["encoder"]["mean_ms"]),
        "full_seg_P50_speedup_fp32_over_fp16": speedup(fp32["full_segmentation"]["P50_ms"], fp16["full_segmentation"]["P50_ms"]),
        "fp16_tegrastats": fp16["tegrastats"],
        "fp32_tegrastats": fp32["tegrastats"],
    }
    with open(os.path.join(OUT, "latency_comparison.json"), "w") as f:
        json.dump(lat, f, indent=2)
    print(json.dumps(lat, indent=2), flush=True)


if __name__ == "__main__":
    main()
