# -*- coding: utf-8 -*-
"""Stage E4 - NanoOWL -> NanoSAM in-process joint perception pipeline.

Open-vocabulary detection (NanoOWL) drives promptable segmentation (NanoSAM):

    Image + Text Prompt -> NanoOWL -> Bounding Box(es) -> NanoSAM Box Prompt -> Mask(s)

This is a CORRECTNESS/demo pipeline (single process, in-process reuse of the
already-validated inference logic from scripts/run_nanoowl.py (NanoOWL) and
the official nanosam Predictor (NanoSAM)). It does NOT subprocess the two
CLIs and does NOT copy third_party model-core implementations.

Coordinate convention (verified from both projects source, NOT from memory):
  * NanoOWL  : OwlPredictor.predict returns output.boxes in xyxy, in ORIGINAL
               image pixel space (when pad_square is False). See owl_predictor:
               _owl_box_roi_to_box_global maps normalized roi boxes onto the
               input image (x0,y0,wh) in original pixels.
  * NanoSAM  : Predictor.predict(points, point_labels) takes points in ORIGINAL
               image XY pixel space; preprocess_points scales them by
               orig_image_encoder_size / max(W,H). A box prompt is encoded as
               points=[[x1,y1],[x2,y2]] with labels=[2,3]
               (2=box top-left, 3=box bottom-right).
  => Both ends use ORIGINAL-image [x,y] / xyxy. Convention is consistent: the
     NanoOWL box [x1,y1,x2,y2] maps directly to NanoSAM point_coords
     [[x1,y1],[x2,y2]] with NO axis swap.

Mask selection rule (same as scripts/run_nanosam_infer.py, official):
  MobileSAM decoder emits 4 candidate masks (low_res_masks[1,4,256,256],
  iou_predictions[1,4]). Official examples/basic_usage.py selects a FIXED slot:
      mask = (mask[0, 0] > 0)   ->  selected_mask_index = 0 (NOT IoU argmax).

Top-k policy:
  --top-k controls how many NanoOWL detections (sorted desc by score, above
  --score-threshold, optionally filtered by --target-label) are pushed through
  NanoSAM. Default is 1 (single best detection -> single mask), matching the
  common "find the X" robotics use case. --top-k -1 means "all detections
  above threshold". Each selected detection gets its own NanoSAM box prompt
  and its own mask; NanoSAM's image embedding (set_image) is computed once
  per input image and reused across all top-k detections.

Production backend:
  NanoSAM default encoder is the FP16 50-epoch self-distilled production
  engine (quality-gate PASS, mIoU 0.6992 on full COCO). The upstream
  third_party default (untrained resnet18_image_encoder.engine) and the INT8
  research backend (quality-gate FAIL) are NOT used as defaults; pass
  --nanosam-image-encoder-engine explicitly to override.
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import PIL.Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)  # so we can import sibling script logic in-process

# Reuse the validated NanoOWL logic (no subprocess, no reimplementation).
from nanoowl.owl_predictor import OwlPredictor  # noqa: E402
from nanoowl.owl_drawing import draw_owl_output  # noqa: E402
from nanosam.utils.predictor import Predictor as NanosamPredictor  # noqa: E402

DEFAULT_OWL_IMAGE = os.path.join(ROOT, "third_party/nanoowl/assets/owl_glove_small.jpg")
DEFAULT_OWL_ENGINE = os.path.join(ROOT, "third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
DEFAULT_NANOSAM_ENC = os.path.join(ROOT, "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine")
DEFAULT_NANOSAM_DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
DEFAULT_MODEL = "google/owlvit-base-patch32"


def parse_args():
    ap = argparse.ArgumentParser(description="NanoOWL -> NanoSAM joint perception pipeline")
    ap.add_argument("--image", default=DEFAULT_OWL_IMAGE)
    ap.add_argument("--text", default="an owl,a glove,a finger",
                    help="comma-separated open-vocabulary text prompt(s)")
    ap.add_argument("--score-threshold", type=float, default=0.10)
    ap.add_argument("--top-k", type=int, default=1,
                    help="max NanoOWL detections (desc by score) to push through NanoSAM; "
                         "-1 = all detections above --score-threshold")
    ap.add_argument("--target-label", default=None,
                    help="if set, restrict detections to this label before applying top-k")
    ap.add_argument("--nanoowl-engine", default=DEFAULT_OWL_ENGINE)
    ap.add_argument("--nanosam-image-encoder-engine", default=DEFAULT_NANOSAM_ENC)
    ap.add_argument("--nanosam-mask-decoder-engine", default=DEFAULT_NANOSAM_DEC)
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "results/joint_pipeline/default"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    return ap.parse_args()


def detect(owl, image_path, prompts_text, threshold, pad_square=False):
    """Run NanoOWL detection. Returns (detections list[dict], image PIL, output obj, text)."""
    text = [t.strip() for t in prompts_text.split(",") if t.strip()]
    image = PIL.Image.open(image_path).convert("RGB")
    text_encodings = owl.encode_text(text)
    output = owl.predict(
        image=image, text=text, text_encodings=text_encodings,
        threshold=threshold, pad_square=pad_square,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    labels = output.labels.detach().cpu().tolist()
    scores = output.scores.detach().cpu().tolist()
    boxes = [[float(v) for v in b] for b in output.boxes.detach().cpu().tolist()]
    dets = []
    for li, sc, bx in zip(labels, scores, boxes):
        li = int(li)
        dets.append({
            "label_index": li,
            "label": text[li] if 0 <= li < len(text) else str(li),
            "score": float(sc),
            "box": bx,  # xyxy, original image px
        })
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets, image, output, text


def select_top_k(dets, target_label, top_k):
    """Selection rule (recorded in JSON).

    1. Optionally restrict to detections whose label == target_label.
    2. Take the top_k highest-score remaining detections (top_k=-1 -> all).
    Returns (chosen list[dict], rule_text).
    """
    pool = dets
    if target_label is not None:
        pool = [d for d in dets if d["label"] == target_label]
        if not pool:
            return [], (f"target-label '{target_label}' had 0 matching detections "
                        f"(of {len(dets)} total) -> NO_DETECTION")
    if not pool:
        return [], "no detections above score-threshold"
    k = len(pool) if top_k is None or top_k < 0 else min(top_k, len(pool))
    chosen = pool[:k]
    rule = (f"top-{top_k if top_k >= 0 else 'ALL'} of "
            f"{'target-label ' + repr(target_label) if target_label else 'all labels'} "
            f"by descending score ({len(chosen)} of {len(pool)} eligible selected)")
    return chosen, rule


def clip_box(box, w, h):
    """Clip xyxy box to image bounds [0,w]x[0,h]. Returns clipped box + out-of-bounds flag."""
    x1, y1, x2, y2 = box
    orig = [float(x1), float(y1), float(x2), float(y2)]
    cx1 = min(max(x1, 0.0), float(w))
    cy1 = min(max(y1, 0.0), float(h))
    cx2 = min(max(x2, 0.0), float(w))
    cy2 = min(max(y2, 0.0), float(h))
    # ensure valid (non-degenerate) order
    cx1, cx2 = min(cx1, cx2), max(cx1, cx2)
    cy1, cy2 = min(cy1, cy2), max(cy1, cy2)
    oob = bool(orig[0] < 0 or orig[1] < 0 or orig[2] > w or orig[3] > h)
    return [round(cx1, 4), round(cy1, 4), round(cx2, 4), round(cy2, 4)], oob, orig


def bridge_box_to_points(clipped_box):
    point_coords = [[clipped_box[0], clipped_box[1]], [clipped_box[2], clipped_box[3]]]
    point_labels = [2, 3]
    return point_coords, point_labels


def run_nanosam_on_box(nsam, clipped_box):
    point_coords, point_labels = bridge_box_to_points(clipped_box)
    pts = np.array(point_coords, dtype=np.float32)
    labs = np.array(point_labels)
    hi, iou, low = nsam.predict(pts, labs)
    pred_ious = iou.detach().cpu().numpy().ravel().tolist()
    sel_idx = 0  # official basic_usage.py: fixed slot 0, NOT IoU argmax
    sel_iou = float(pred_ious[sel_idx])
    mask_bool = (hi[0, sel_idx] > 0).detach().cpu().numpy()
    hi_np = hi.detach().float().cpu().numpy()
    nanosam_info = {
        "mask_selection_rule": "fixed slot 0 (basic_usage.py: mask[0,0]>0), not IoU argmax",
        "selected_mask_index": sel_idx,
        "predicted_ious": [round(float(v), 6) for v in pred_ious],
        "selected_iou": round(sel_iou, 6),
        "mask_shape": list(mask_bool.shape),
        "nonzero_pixels": int(mask_bool.sum()),
        "mask_area_ratio": round(float(mask_bool.sum()) / mask_bool.size, 6) if mask_bool.size else 0.0,
        "has_nan": bool(np.isnan(hi_np).any()),
        "has_inf": bool(np.isinf(hi_np).any()),
    }
    bridge_info = {
        "convention": "both ends original-image xyxy/[x,y]; no axis swap",
        "clipped_box": clipped_box,
        "nanosam_point_coords": point_coords,
        "nanosam_point_labels": point_labels,
    }
    return mask_bool, nanosam_info, bridge_info


def save_mask_png(mask_bool, out_path):
    PIL.Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(out_path)


def overlay_viz(image, entries, out_path):
    """entries: list of (box, label, score, mask_bool)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    colors = plt.cm.get_cmap("tab10").colors
    for i, (box, label, score, mask_bool) in enumerate(entries):
        color = colors[i % len(colors)]
        masked = np.ma.masked_where(~mask_bool, mask_bool.astype(float))
        ax.imshow(masked, alpha=0.5, cmap=matplotlib.colors.ListedColormap([color]))
        x1, y1, x2, y2 = box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor=color, linewidth=2))
        ax.text(x1, max(y1 - 4, 4), f"{label} {score:.3f}",
                color="white", fontsize=10,
                bbox=dict(boxstyle="round", fc="black", alpha=0.6))
    ax.set_axis_off()
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timing = {}

    # ---- load NanoOWL predictor ----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    owl = OwlPredictor(args.model, image_encoder_engine=args.nanoowl_engine)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timing["nanoowl_load_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    # ---- NanoOWL detection ----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    dets, image, owl_output, text = detect(owl, args.image, args.text, args.score_threshold)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timing["nanoowl_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    W, H = image.size  # PIL: (width, height)
    print(f"NanoOWL: {len(dets)} detections; image W={W} H={H}", flush=True)

    # save input + nanoowl visualization (diagnostic, not part of the 3-file contract)
    image.save(os.path.join(args.output_dir, "input.jpg"))
    _vis = np.array(image).copy()
    _ = draw_owl_output(_vis, owl_output, text=text, draw_text=True)
    PIL.Image.fromarray(_vis).save(os.path.join(args.output_dir, "nanoowl_visualization.jpg"))

    # ---- detection selection (top-k) ----
    chosen, rule = select_top_k(dets, args.target_label, args.top_k)

    if not chosen:
        # NO_DETECTION path: do NOT call NanoSAM. Structured, clean exit.
        result = {
            "input": args.image,
            "text": text,
            "score_threshold": args.score_threshold,
            "top_k": args.top_k,
            "target_label": args.target_label,
            "image_width": W,
            "image_height": H,
            "nanoowl": {
                "num_detections": len(dets),
                "detections": dets,
                "selected_detections": [],
                "selection_rule": rule,
            },
            "results": [],
            "status": "NO_DETECTION",
        }
        with open(os.path.join(args.output_dir, "detections.json"), "w") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(args.output_dir, "timing.json"), "w") as f:
            timing["pipeline_total_ms"] = round(timing.get("nanoowl_ms", 0.0), 3)
            json.dump(timing, f, indent=2)
        print("STATUS=NO_DETECTION (execution PASS, no NanoSAM call)", flush=True)
        print("ALL_DONE", flush=True)
        return 0

    # ---- load NanoSAM predictor + set_image once (reused across top-k) ----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    nsam = NanosamPredictor(args.nanosam_image_encoder_engine,
                            args.nanosam_mask_decoder_engine)
    nsam.set_image(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timing["nanosam_set_image_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    per_det_results = []
    viz_entries = []
    decode_ms_total = 0.0
    for i, det in enumerate(chosen):
        clipped_box, oob, orig = clip_box(det["box"], W, H)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        mask_bool, nanosam_info, bridge_info = run_nanosam_on_box(nsam, clipped_box)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t0) * 1000.0
        decode_ms_total += decode_ms

        bridge_info["image_width"] = W
        bridge_info["image_height"] = H
        bridge_info["original_box"] = [round(v, 4) for v in orig]
        bridge_info["out_of_bounds"] = oob

        mask_name = "mask.png" if i == 0 else f"mask_k{i + 1}.png"
        save_mask_png(mask_bool, os.path.join(args.output_dir, mask_name))
        viz_entries.append((clipped_box, det["label"], det["score"], mask_bool))

        per_det_results.append({
            "rank": i + 1,
            "detection": det,
            "bridge": bridge_info,
            "nanosam": nanosam_info,
            "mask_file": mask_name,
            "nanosam_decode_ms": round(decode_ms, 3),
        })
        print(f"[{i + 1}/{len(chosen)}] label={det['label']} score={det['score']:.4f} "
              f"box={clipped_box} sel_iou={nanosam_info['selected_iou']:.4f} "
              f"nonzero={nanosam_info['nonzero_pixels']}", flush=True)

    timing["nanosam_decode_ms_total"] = round(decode_ms_total, 3)
    timing["pipeline_total_ms"] = round(
        timing["nanoowl_ms"] + timing["nanosam_set_image_ms"] + decode_ms_total, 3)

    overlay_viz(image, viz_entries, os.path.join(args.output_dir, "overlay.jpg"))

    result = {
        "input": args.image,
        "text": text,
        "score_threshold": args.score_threshold,
        "top_k": args.top_k,
        "target_label": args.target_label,
        "image_width": W,
        "image_height": H,
        "nanoowl": {
            "num_detections": len(dets),
            "detections": dets,
            "selected_detections": [d for d in chosen],
            "selection_rule": rule,
        },
        "results": per_det_results,
        "status": "OK",
    }
    with open(os.path.join(args.output_dir, "detections.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.output_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)

    any_nan = any(r["nanosam"]["has_nan"] for r in per_det_results)
    any_inf = any(r["nanosam"]["has_inf"] for r in per_det_results)
    print(f"STATUS=OK selected={len(chosen)} nan/inf={any_nan}/{any_inf}", flush=True)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
