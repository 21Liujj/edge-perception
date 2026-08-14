# -*- coding: utf-8 -*-
"""Stage 6C - Nsight Systems profiling workload for EdgePerception joint pipeline.

Controlled inference-only workload for nsys capture:
  - models load ONCE (reuse)
  - warmup 20 iters, profile N iters (default 15)
  - fixed owl_glove_small.jpg, prompts "an owl,a glove,a finger", target "an owl"
  - mask slot 0
  - NO image save, NO video write, NO heavy JSON I/O
  - NVTX ranges around pipeline stages (our own profiling script only; no upstream edits)

Reuse Day4/Stage6A validated inference logic verbatim in conventions.
This script is for profiling the production inference path; it ADDS NVTX ranges
and cuda_sync only for measurement. Per Stage 6C requirement, measurement sync must
be distinguishable from production sync.
"""
import argparse
import os
import sys
import json
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
sys.path.insert(0, ROOT)

from nanoowl.owl_predictor import OwlPredictor  # noqa: E402
from nanosam.utils.predictor import Predictor as NanosamPredictor  # noqa: E402

# torch.cuda.nvtx available in torch; used to mark ranges. Safe, no new packages.
try:
    from torch.cuda import nvtx as _nvtx
    def nvtx_range(name):
        return _nvtx.range(name)
    def nvtx_push(name):
        _nvtx.range_push(name)
    def nvtx_pop():
        _nvtx.range_pop()
    HAS_NVTX = True
except Exception:
    HAS_NVTX = False
    import contextlib
    def nvtx_range(name): return contextlib.nullcontext()
    def nvtx_push(name): pass
    def nvtx_pop(): pass

DEFAULT_OWL_ENGINE = os.path.join(ROOT, "third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
DEFAULT_NANOSAM_ENC = os.path.join(ROOT, "third_party/nanosam/data/resnet18_image_encoder.engine")
DEFAULT_NANOSAM_DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
DEFAULT_MODEL = "google/owlvit-base-patch32"
DEFAULT_IMAGE = os.path.join(ROOT, "third_party/nanoowl/assets/owl_glove_small.jpg")
SELECTED_MASK_INDEX = 0


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    cx1 = min(max(float(x1), 0.0), float(w)); cy1 = min(max(float(y1), 0.0), float(h))
    cx2 = min(max(float(x2), 0.0), float(w)); cy2 = min(max(float(y2), 0.0), float(h))
    cx1, cx2 = min(cx1, cx2), max(cx1, cx2); cy1, cy2 = min(cy1, cy2), max(cy1, cy2)
    return [cx1, cy1, cx2, cy2]


def detect_on_image(owl, image_pil_rgb, text, text_encodings, threshold):
    output = owl.predict(image=image_pil_rgb, text=text, text_encodings=text_encodings,
                         threshold=threshold, pad_square=False)
    labels = output.labels.detach().cpu().tolist()
    scores = output.scores.detach().cpu().tolist()
    boxes = output.boxes.detach().cpu().tolist()
    dets = []
    for li, sc, bx in zip(labels, scores, boxes):
        li = int(li)
        dets.append({"label": text[li] if 0 <= li < len(text) else str(li),
                     "score": float(sc), "box": [float(v) for v in bx]})
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets


def select_detection(dets, target_label):
    if not dets: return None
    if target_label is not None:
        m = [d for d in dets if d["label"] == target_label]
        return m[0] if m else None
    return dets[0]


def run_pipeline_once(owl, nsam, frame_bgr, in_w, in_h, text, text_encodings, threshold, target_label):
    """One joint inference pass with NVTX ranges. Returns (rec, status, ok).
    cuda_sync points here are MEASUREMENT sync (to demarcate stages for nsys timeline).
    Production code path sync is what upstream/TRT does internally.
    """
    nvtx_push("pipeline_total")
    # ---- preprocess ----
    nvtx_push("preprocess")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = PIL.Image.fromarray(rgb).convert("RGB")
    nvtx_pop()

    # ---- nanoowl ----
    nvtx_push("nanoowl")
    dets = detect_on_image(owl, image_pil, text, text_encodings, threshold)
    nvtx_pop()

    # ---- nanoowl_postprocess (selection / clip) ----
    nvtx_push("nanoowl_postprocess")
    selected = select_detection(dets, target_label)
    nvtx_pop()

    rec = {"num_detections": len(dets)}
    if selected is None:
        nvtx_pop()  # pipeline_total
        return rec, "NO_DETECTION", True

    nvtx_push("bridge")
    clipped_box = clip_box(selected["box"], in_w, in_h)
    pts = np.array([[clipped_box[0], clipped_box[1]], [clipped_box[2], clipped_box[3]]], dtype=np.float32)
    labs = np.array([2, 3])
    nvtx_pop()

    # ---- nanosam_set_image ----
    nvtx_push("nanosam_set_image")
    nsam.set_image(image_pil)
    cuda_sync()  # MEASUREMENT sync: wall-clock boundary for set_image
    nvtx_pop()

    # ---- nanosam_decode ----
    nvtx_push("nanosam_decode")
    hi, iou_pred, low = nsam.predict(pts, labs)
    cuda_sync()  # MEASUREMENT sync: wall-clock boundary for decode
    nvtx_pop()

    hi_f = hi.detach().float()
    has_nan = bool(torch.isnan(hi_f).any().item())
    has_inf = bool(torch.isinf(hi_f).any().item())
    mask_bool = (hi[0, SELECTED_MASK_INDEX] > 0).detach().cpu().numpy()
    if mask_bool.shape[:2] != (in_h, in_w):
        m_u8 = (mask_bool.astype(np.uint8) * 255)
        m_u8 = cv2.resize(m_u8, (in_w, in_h), interpolation=cv2.INTER_NEAREST)
        mask_bool = (m_u8 > 127)
    rec.update({"label": selected["label"], "score": round(selected["score"], 6),
                "box": clipped_box, "selected_mask_index": SELECTED_MASK_INDEX,
                "mask_area_ratio": round(float(mask_bool.sum()) / float(mask_bool.size), 6),
                "mask_nonzero": int(mask_bool.sum()),
                "has_nan": has_nan, "has_inf": has_inf})
    status = "INFERENCE_NAN_INF" if (has_nan or has_inf) else "DETECTED"

    nvtx_pop()  # pipeline_total
    return rec, status, not (has_nan or has_inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--prompts", default="an owl,a glove,a finger")
    ap.add_argument("--target-label", default="an owl")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--owl-engine", default=DEFAULT_OWL_ENGINE)
    ap.add_argument("--nanosam-enc", default=DEFAULT_NANOSAM_ENC)
    ap.add_argument("--nanosam-dec", default=DEFAULT_NANOSAM_DEC)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--runtime", choices=["baseline", "optimized"], default="baseline")
    ap.add_argument("--out", default="results/nsight_profile/profile_meta.json")
    args = ap.parse_args()

    frame_bgr = cv2.imread(args.image)
    if frame_bgr is None:
        print("ERROR cannot read image", flush=True); return 2
    in_h, in_w = frame_bgr.shape[:2]

    cuda_sync()
    t0 = time.perf_counter()
    owl = OwlPredictor(args.model, image_encoder_engine=args.owl_engine); cuda_sync()
    owl_load = (time.perf_counter() - t0) * 1000.0
    text = [t.strip() for t in args.prompts.split(",") if t.strip()]
    text_encodings = owl.encode_text(text); cuda_sync()
    t0 = time.perf_counter()
    if args.runtime == "optimized":
        from edge_perception.runtime.nanosam_optimized import OptimizedPredictor
        nsam = OptimizedPredictor(args.nanosam_enc, args.nanosam_dec); cuda_sync()
    else:
        nsam = NanosamPredictor(args.nanosam_enc, args.nanosam_dec); cuda_sync()
    nsam_load = (time.perf_counter() - t0) * 1000.0
    print("model_load nanoowl=%.1fms nanosam=%.1fms nvtx=%s img=%dx%d" %
          (owl_load, nsam_load, HAS_NVTX, in_w, in_h), flush=True)

    # warmup
    print("warmup %d..." % args.warmup, flush=True)
    for i in range(args.warmup):
        _, st, _ = run_pipeline_once(owl, nsam, frame_bgr, in_w, in_h, text, text_encodings, args.threshold, args.target_label)
        cuda_sync()
    cuda_sync()
    print("warmup done", flush=True)

    # profile iters
    times = []
    statuses = []
    detected = 0; naninf = 0
    print("profile %d iters (NVTX ranges on)..." % args.iters, flush=True)
    for i in range(args.iters):
        t0 = time.perf_counter()
        rec, st, ok = run_pipeline_once(owl, nsam, frame_bgr, in_w, in_h, text, text_encodings, args.threshold, args.target_label)
        # final MEASUREMENT sync closes the pipeline_total wall-clock
        cuda_sync()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        times.append(ms); statuses.append(st)
        if st == "DETECTED": detected += 1
        if st == "INFERENCE_NAN_INF": naninf += 1
    print("profile done mean=%.2fms" % (sum(times)/len(times)), flush=True)

    a = np.array(times)
    meta = {
        "image": args.image, "in_h": in_h, "in_w": in_w, "runtime": args.runtime,
        "prompts": args.prompts, "target_label": args.target_label, "threshold": args.threshold,
        "warmup": args.warmup, "iters": args.iters,
        "has_nvtx": HAS_NVTX,
        "nvtx_ranges": ["pipeline_total","preprocess","nanoowl","nanoowl_postprocess","bridge","nanosam_set_image","nanosam_decode"],
        "measurement_sync_note": "cuda_sync in this script demarcates NVTX stage boundaries for nsys timeline; production sync is what TRT/upstream does internally.",
        "model_load_ms": {"nanoowl": round(owl_load,3), "nanosam": round(nsam_load,3), "total": round(owl_load+nsam_load,3)},
        "profile_iters": {
            "mean_ms": round(float(a.mean()),3), "min_ms": round(float(a.min()),3), "max_ms": round(float(a.max()),3),
            "p50_ms": round(float(np.percentile(a,50)),3), "p95_ms": round(float(np.percentile(a,95)),3),
            "p99_ms": round(float(np.percentile(a,99)),3),
        },
        "correctness": {"detected_iters": detected, "no_detection_iters": statuses.count("NO_DETECTION"),
                        "nan_inf_iters": naninf,
                        "owl_detection_present": detected == args.iters,
                        "mask_nonzero_last": rec.get("mask_nonzero"),
                        "has_nan": rec.get("has_nan"), "has_inf": rec.get("has_inf")},
        "per_iter_ms": [round(t,3) for t in times],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(meta, open(args.out, "w"), indent=2)
    print("RESULT mean=%.3fms p50=%.3f p95=%.3f detected=%d/%d naninf=%d mask_nonzero=%s" % (
        meta["profile_iters"]["mean_ms"], meta["profile_iters"]["p50_ms"], meta["profile_iters"]["p95_ms"],
        detected, args.iters, naninf, meta["correctness"]["mask_nonzero_last"]), flush=True)
    print("CORRECTNESS_OWL=%s MASK_NONZERO=%s NAN=%s INF=%s" % (
        meta["correctness"]["owl_detection_present"], meta["correctness"]["mask_nonzero_last"] > 0,
        meta["correctness"]["has_nan"], meta["correctness"]["has_inf"]), flush=True)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
