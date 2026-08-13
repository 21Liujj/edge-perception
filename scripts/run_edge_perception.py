# -*- coding: utf-8 -*-
"""Stage 4A - NanoOWL -> NanoSAM in-process joint perception pipeline.

Open-vocabulary detection (NanoOWL) drives promptable segmentation (NanoSAM):

    Image + Text Prompt -> NanoOWL -> Bounding Box -> NanoSAM Box Prompt -> Mask

This is a CORRECTNESS pipeline, not a formal benchmark. It runs in ONE process
and reuses the already-validated inference logic from scripts/run_nanoowl.py
(NanoOWL) and the official nanosam Predictor (NanoSAM). It does NOT subprocess
the two CLIs and does NOT copy third_party model-core implementations.

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
from run_nanoowl import load_predictor as load_owl  # noqa: E402
from nanoowl.owl_predictor import OwlPredictor  # noqa: E402
from nanoowl.owl_drawing import draw_owl_output  # noqa: E402
from nanosam.utils.predictor import Predictor as NanosamPredictor  # noqa: E402

DEFAULT_OWL_IMAGE = os.path.join(ROOT, "third_party/nanoowl/assets/owl_glove_small.jpg")
DEFAULT_OWL_ENGINE = os.path.join(ROOT, "third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
DEFAULT_NANOSAM_ENC = os.path.join(ROOT, "third_party/nanosam/data/resnet18_image_encoder.engine")
DEFAULT_NANOSAM_DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
DEFAULT_MODEL = "google/owlvit-base-patch32"


def parse_args():
    ap = argparse.ArgumentParser(description="NanoOWL -> NanoSAM joint perception pipeline")
    ap.add_argument("--image", default=DEFAULT_OWL_IMAGE)
    ap.add_argument("--prompts", default="an owl,a glove,a finger",
                    help="comma-separated text prompts")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--target-label", default=None,
                    help="if set, pick the highest-score detection whose label matches")
    ap.add_argument("--nanoowl-engine", default=DEFAULT_OWL_ENGINE)
    ap.add_argument("--nanosam-image-encoder-engine", default=DEFAULT_NANOSAM_ENC)
    ap.add_argument("--nanosam-mask-decoder-engine", default=DEFAULT_NANOSAM_DEC)
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "results/joint_pipeline/default"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    return ap.parse_args()


def detect(owl, image_path, prompts_text, threshold, pad_square=False):
    """Run NanoOWL detection. Returns (detections list[dict], image PIL, output obj)."""
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


def select_detection(dets, target_label):
    """Selection rule (recorded in JSON).

    If target_label is given: among detections whose label == target_label, pick
    the highest-score one (dets already sorted desc by score).
    Else: pick the global highest-score detection (dets[0]).
    Returns (chosen dict | None, rule_text).
    """
    if not dets:
        return None, "no detections"
    if target_label is not None:
        matches = [d for d in dets if d["label"] == target_label]
        if matches:
            return matches[0], f"highest-score detection matching target-label '{target_label}'"
        return None, (f"target-label '{target_label}' had 0 matching detections "
                      f"(of {len(dets)} total) -> NO_DETECTION")
    return dets[0], "global highest-score detection (no target-label)"


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


def joint_viz(image, box, label, score, mask_bool_np, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.imshow(mask_bool_np, alpha=0.5)
    x1, y1, x2, y2 = box
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor="lime", linewidth=2))
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
    dets, image, owl_output, text = detect(owl, args.image, args.prompts, args.threshold)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timing["nanoowl_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    W, H = image.size  # PIL: (width, height)
    print(f"NanoOWL: {len(dets)} detections; image W={W} H={H}", flush=True)

    # save input + nanoowl visualization
    image.save(os.path.join(args.output_dir, "input.jpg"))
    _vis = np.array(image).copy()
    _ = draw_owl_output(_vis, owl_output, text=text, draw_text=True)
    from PIL import Image as _Image
    _Image.fromarray(_vis).save(os.path.join(args.output_dir, "nanoowl_visualization.jpg"))

    # ---- detection selection ----
    selected, rule = select_detection(dets, args.target_label)

    if selected is None:
        # NO_DETECTION path: do NOT call NanoSAM. Structured, clean exit.
        result = {
            "input": args.image,
            "prompts": text,
            "threshold": args.threshold,
            "target_label": args.target_label,
            "image_width": W,
            "image_height": H,
            "nanoowl": {
                "num_detections": len(dets),
                "detections": dets,
                "selected_detection": None,
                "selection_rule": rule,
            },
            "bridge": None,
            "nanosam": None,
            "status": "NO_DETECTION",
        }
        with open(os.path.join(args.output_dir, "pipeline_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(args.output_dir, "timing.json"), "w") as f:
            timing["pipeline_total_ms"] = round(timing.get("nanoowl_ms", 0.0), 3)
            json.dump(timing, f, indent=2)
        print("STATUS=NO_DETECTION (execution PASS, no NanoSAM call)", flush=True)
        print("ALL_DONE", flush=True)
        return 0

    # ---- bridge: NanoOWL box -> NanoSAM box prompt ----
    orig_box = selected["box"]  # xyxy original px
    clipped_box, oob, orig = clip_box(orig_box, W, H)
    point_coords = [[clipped_box[0], clipped_box[1]], [clipped_box[2], clipped_box[3]]]
    point_labels = [2, 3]
    bridge = {
        "convention": "both ends original-image xyxy/[x,y]; no axis swap",
        "image_width": W,
        "image_height": H,
        "original_box": [round(v, 4) for v in orig],
        "out_of_bounds": oob,
        "clipped_box": clipped_box,
        "nanosam_point_coords": point_coords,
        "nanosam_point_labels": point_labels,
    }
    print(f"Bridge: box {clipped_box} oob={oob} -> points {point_coords} labels {point_labels}",
          flush=True)

    # ---- load NanoSAM predictor + set_image + decode ----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    nsam = NanosamPredictor(args.nanosam_image_encoder_engine,
                            args.nanosam_mask_decoder_engine)
    nsam.set_image(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timing["nanosam_set_image_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    pts = np.array([[point_coords[0][0], point_coords[0][1]],
                    [point_coords[1][0], point_coords[1][1]]])
    labs = np.array(point_labels)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    hi, iou, low = nsam.predict(pts, labs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timing["nanosam_decode_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    pred_ious = iou.detach().cpu().numpy().ravel().tolist()
    sel_idx = 0  # official basic_usage.py: fixed slot 0, NOT IoU argmax
    sel_iou = float(pred_ious[sel_idx])
    mask_bool = (hi[0, sel_idx] > 0).detach().cpu().numpy()
    nz = int(mask_bool.sum())
    total = int(mask_bool.size)
    has_nan = bool(np.isnan(hi.detach().float().cpu().numpy()).any())
    has_inf = bool(np.isinf(hi.detach().float().cpu().numpy()).any())

    timing["pipeline_total_ms"] = round(
        timing["nanoowl_ms"] + timing["nanosam_set_image_ms"] + timing["nanosam_decode_ms"], 3)

    np.save(os.path.join(args.output_dir, "nanosam_mask.npy"), mask_bool)
    joint_viz(image, clipped_box, selected["label"], selected["score"],
              mask_bool, os.path.join(args.output_dir, "joint_visualization.jpg"))

    result = {
        "input": args.image,
        "prompts": text,
        "threshold": args.threshold,
        "target_label": args.target_label,
        "image_width": W,
        "image_height": H,
        "nanoowl": {
            "num_detections": len(dets),
            "detections": dets,
            "selected_detection": selected,
            "selection_rule": rule,
        },
        "bridge": bridge,
        "nanosam": {
            "mask_selection_rule": "fixed slot 0 (basic_usage.py: mask[0,0]>0), not IoU argmax",
            "selected_mask_index": sel_idx,
            "predicted_ious": [round(float(v), 6) for v in pred_ious],
            "selected_iou": round(sel_iou, 6),
            "mask_shape": list(mask_bool.shape),
            "nonzero_pixels": nz,
            "mask_area_ratio": round(nz / total, 6) if total else 0.0,
            "has_nan": has_nan,
            "has_inf": has_inf,
        },
        "status": "OK",
    }
    with open(os.path.join(args.output_dir, "pipeline_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.output_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)

    print(f"STATUS=OK sel_label={selected['label']} score={selected['score']:.4f} "
          f"box={clipped_box} sel_iou={sel_iou:.4f} nonzero={nz} "
          f"nan/inf={has_nan}/{has_inf}", flush=True)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
