# -*- coding: utf-8 -*-
"""Stage 7A benchmark: baseline vs optimized NanoSAM runtime.

Reuses validated inference conventions. One model init. Two micro-benchmarks:
  (A) set_image only  : warmup>=30, iters>=200, P50/P95/P99/max
  (B) joint pipeline  : NanoOWL + NanoSAM, warmup>=30, iters>=200

Correctness gates printed: owl joint case + dogs box-prompt case.

Usage:
  python benchmark_stage7a.py --runtime baseline|optimized --warmup 30 --iters 200 \
      --micro set_image|joint --out <path>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import cv2
import PIL.Image
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, ROOT)  # for "from edge_perception.runtime..." import

from nanoowl.owl_predictor import OwlPredictor  # noqa: E402
from nanosam.utils.predictor import Predictor as BaselinePredictor  # noqa: E402
from edge_perception.runtime.nanosam_optimized import OptimizedPredictor  # noqa: E402

DEFAULT_OWL_ENGINE = os.path.join(ROOT, "third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
DEFAULT_NANOSAM_ENC = os.path.join(ROOT, "third_party/nanosam/data/resnet18_image_encoder.engine")
DEFAULT_NANOSAM_DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
DEFAULT_MODEL = "google/owlvit-base-patch32"
OWL_IMG = os.path.join(ROOT, "third_party/nanoowl/assets/owl_glove_small.jpg")
DOGS_IMG = os.path.join(ROOT, "third_party/nanoowl/assets/owl_glove.jpg")  # placeholder, see --dogs-img
SELECTED_MASK_INDEX = 0


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    cx1 = min(max(float(x1), 0.0), float(w)); cy1 = min(max(float(y1), 0.0), float(h))
    cx2 = min(max(float(x2), 0.0), float(w)); cy2 = min(max(float(y2), 0.0), float(h))
    return [min(cx1, cx2), min(cy1, cy2), max(cx1, cx2), max(cy1, cy2)]


def detect_on_image(owl, image_pil, text, text_encodings, threshold):
    out = owl.predict(image=image_pil, text=text, text_encodings=text_encodings,
                      threshold=threshold, pad_square=False)
    cuda_sync()
    labels = out.labels.detach().cpu().tolist()
    scores = out.scores.detach().cpu().tolist()
    boxes = out.boxes.detach().cpu().tolist()
    dets = [{"label": text[int(li)] if 0 <= int(li) < len(text) else str(li),
             "score": float(sc), "box": [float(v) for v in bx]}
            for li, sc, bx in zip(labels, scores, boxes)]
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets


def select_detection(dets, target):
    if not dets: return None
    if target:
        m = [d for d in dets if d["label"] == target]
        return m[0] if m else None
    return dets[0]


def make_predictor(runtime, args):
    if runtime == "baseline":
        return BaselinePredictor(args.nanosam_enc, args.nanosam_dec)
    return OptimizedPredictor(args.nanosam_enc, args.nanosam_dec, use_stream=not args.no_stream)


def percentiles(vals):
    if not vals:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    a = np.array(vals, dtype=np.float64)
    return {"mean": round(float(a.mean()), 4), "min": round(float(a.min()), 4),
            "p50": round(float(np.percentile(a, 50)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "p99": round(float(np.percentile(a, 99)), 4),
            "max": round(float(a.max()), 4)}


def correctness_owl(owl, nsam, args):
    bgr = cv2.imread(OWL_IMG)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = PIL.Image.fromarray(rgb).convert("RGB")
    H, W = pil.height, pil.width
    text = [t.strip() for t in args.prompts.split(",") if t.strip()]
    enc = owl.encode_text(text); cuda_sync()
    dets = detect_on_image(owl, pil, text, enc, args.threshold)
    sel = select_detection(dets, args.target_label)
    if sel is None:
        return {"status": "NO_DETECTION"}
    box = clip_box(sel["box"], W, H)
    pts = np.array([[box[0], box[1]], [box[2], box[3]]], dtype=np.float32)
    labs = np.array([2, 3])
    nsam.set_image(pil); cuda_sync()
    hi, ioup, low = nsam.predict(pts, labs); cuda_sync()
    hi_f = hi.detach().float()
    nan = bool(torch.isnan(hi_f).any().item()); inf = bool(torch.isinf(hi_f).any().item())
    mask = (hi[0, SELECTED_MASK_INDEX] > 0).detach().cpu().numpy()
    return {"status": "DETECTED" if not (nan or inf) else "NAN_INF",
            "label": sel["label"], "score": round(sel["score"], 6),
            "box": box, "mask_nonzero": int(mask.sum()),
            "has_nan": nan, "has_inf": inf}


def correctness_dogs(owl, nsam, args):
    """Box-prompt NanoSAM regression on dogs image (box provided manually)."""
    if not os.path.exists(args.dogs_img):
        return {"status": "DOGS_IMG_MISSING", "path": args.dogs_img}
    bgr = cv2.imread(args.dogs_img)
    if bgr is None:
        return {"status": "DOGS_IMG_UNREADABLE"}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = PIL.Image.fromarray(rgb).convert("RGB")
    H, W = pil.height, pil.width
    # use OwlPrompt-free path: just NanoSAM on a manual dog box if provided via --dogs-box
    if not args.dogs_box or len(args.dogs_box) != 4:
        # try Owl detect "a dog"
        text = [t.strip() for t in "a dog,a cat,a person".split(",")]
        enc = owl.encode_text(text); cuda_sync()
        dets = detect_on_image(owl, pil, text, enc, 0.10)
        sel = select_detection(dets, "a dog")
        if sel is None:
            return {"status": "NO_DOG_DETECTION", "hint": "provide --dogs-box x1 y1 x2 y2"}
        box = clip_box(sel["box"], W, H)
    else:
        box = clip_box([float(v) for v in args.dogs_box], W, H)
    pts = np.array([[box[0], box[1]], [box[2], box[3]]], dtype=np.float32)
    labs = np.array([2, 3])
    nsam.set_image(pil); cuda_sync()
    hi, ioup, low = nsam.predict(pts, labs); cuda_sync()
    hi_f = hi.detach().float()
    nan = bool(torch.isnan(hi_f).any().item()); inf = bool(torch.isinf(hi_f).any().item())
    mask = (hi[0, SELECTED_MASK_INDEX] > 0).detach().cpu().numpy()
    return {"status": "DETECTED" if not (nan or inf) else "NAN_INF",
            "box": box, "mask_nonzero": int(mask.sum()),
            "has_nan": nan, "has_inf": inf}


def micro_set_image(nsam, pil, warmup, iters):
    # warmup
    for _ in range(warmup):
        nsam.set_image(pil); cuda_sync()
    cuda_sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        nsam.set_image(pil)
        cuda_sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return percentiles(times), times


def micro_joint(owl, nsam, pil, args, warmup, iters):
    W, H = pil.width, pil.height
    text = [t.strip() for t in args.prompts.split(",") if t.strip()]
    enc = owl.encode_text(text); cuda_sync()
    # warmup full joint
    for _ in range(warmup):
        dets = detect_on_image(owl, pil, text, enc, args.threshold)
        sel = select_detection(dets, args.target_label)
        if sel is not None:
            box = clip_box(sel["box"], W, H)
            pts = np.array([[box[0], box[1]], [box[2], box[3]]], dtype=np.float32)
            nsam.set_image(pil); cuda_sync()
            hi, _, _ = nsam.predict(pts, np.array([2, 3])); cuda_sync()
    cuda_sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        dets = detect_on_image(owl, pil, text, enc, args.threshold)
        sel = select_detection(dets, args.target_label)
        if sel is not None:
            box = clip_box(sel["box"], W, H)
            pts = np.array([[box[0], box[1]], [box[2], box[3]]], dtype=np.float32)
            nsam.set_image(pil); cuda_sync()
            hi, _, _ = nsam.predict(pts, np.array([2, 3])); cuda_sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return percentiles(times), times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, choices=["baseline", "optimized"])
    ap.add_argument("--micro", default="joint", choices=["set_image", "joint"])
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--owl-engine", default=DEFAULT_OWL_ENGINE)
    ap.add_argument("--nanosam-enc", default=DEFAULT_NANOSAM_ENC)
    ap.add_argument("--nanosam-dec", default=DEFAULT_NANOSAM_DEC)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompts", default="an owl,a glove,a finger")
    ap.add_argument("--target-label", default="an owl")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--image", default=OWL_IMG)
    ap.add_argument("--dogs-img", default=os.path.join(ROOT, "third_party/nanoowl/assets/owl_glove.jpg"))
    ap.add_argument("--dogs-box", type=float, nargs=4, default=None)
    ap.add_argument("--no-correctness", action="store_true")
    ap.add_argument("--no-stream", action="store_true", help="optimized: disable explicit stream (buffer-reuse only variant)")
    ap.add_argument("--out", default="results/runtime_optimization/stage7a/bench.json")
    args = ap.parse_args()

    cuda_sync()
    t0 = time.perf_counter()
    owl = OwlPredictor(args.model, image_encoder_engine=args.owl_engine); cuda_sync()
    owl_load = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    nsam = make_predictor(args.runtime, args); cuda_sync()
    nsam_load = (time.perf_counter() - t0) * 1000.0
    print("load owl=%.1fms nsam[%s]=%.1fms" % (owl_load, args.runtime, nsam_load), flush=True)

    # correctness gates
    corr = {}
    if not args.no_correctness:
        corr["owl"] = correctness_owl(owl, nsam, args)
        corr["dogs"] = correctness_dogs(owl, nsam, args)
        print("CORRECTNESS owl=%s dogs=%s" % (corr["owl"].get("status"), corr["dogs"].get("status")), flush=True)

    pil = PIL.Image.open(args.image).convert("RGB")

    if args.micro == "set_image":
        stats, times = micro_set_image(nsam, pil, args.warmup, args.iters)
    else:
        stats, times = micro_joint(owl, nsam, pil, args, args.warmup, args.iters)
    print("RESULT[%s][%s] mean=%.3f p50=%.3f p95=%.3f p99=%.3f max=%.3f" %
          (args.runtime, args.micro, stats["mean"], stats["p50"], stats["p95"], stats["p99"], stats["max"]), flush=True)

    out = {
        "runtime": args.runtime + ("_nostream" if (args.runtime=="optimized" and args.no_stream) else ""), "micro": args.micro,
        "warmup": args.warmup, "iters": args.iters,
        "model_load_ms": {"owl": round(owl_load, 3), "nsam": round(nsam_load, 3)},
        "correctness": corr,
        "latency_ms": stats,
        "per_iter_ms": [round(t, 3) for t in times],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
