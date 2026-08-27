#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage E4 - Joint NanoOWL->NanoSAM pipeline quality eval on 50 fixed COCO cases.

Protocol
--------
Sample unit: COCO instance (one GT annotation), NOT image. 50 instances are
selected deterministically (seed=2027, same seed as the existing NanoOWL COCO
manifests) covering a 3x3 grid of {size_bucket} x {scene_type}:
  size_bucket : small (area<32^2) / medium (32^2<=area<=96^2) / large (area>96^2)
  scene_type  : single (1 valid instance in image) / multi (2-6) / crowded (>6)
(bucket/scene rules match scripts/eval_nanosam_coco.py's _size_bucket and the
same iscrowd==0 + valid-box + decodable-mask instance validity gate.)

Per instance we run TWO NanoSAM box prompts on the SAME NanoSAM image
embedding (set_image called once per unique image_id, reused across all its
instances):
  (a) the NanoOWL-predicted box, obtained by running NanoOWL open-vocabulary
      detection with text=<COCO category name>, target_label=<category name>,
      taking the single highest-score detection (top_k=1) -- this is exactly
      the joint pipeline's own selection rule (scripts/run_edge_perception.py)
  (b) the GT box (xywh->xyxy), i.e. the "perfect detector" reference used by
      scripts/eval_nanosam_coco.py's own NanoSAM-only quality number.

Metrics per instance:
  box_iou          : IoU(NanoOWL box, GT box) in xyxy, image pixel space.
                      None if NanoOWL produced no matching detection.
  mask_iou_owlbox   : IoU(NanoSAM mask from NanoOWL box, GT segmentation mask)
  mask_iou_gtbox    : IoU(NanoSAM mask from GT box, GT segmentation mask)
                      (this is the NanoSAM-only quality ceiling for this
                      instance -- same protocol as eval_nanosam_coco.py)
  degradation       : mask_iou_gtbox - mask_iou_owlbox (box-imprecision cost)

Error decomposition bucket (one of, in this priority order):
  A_detection_failure       : NanoOWL found no detection matching the GT
                              category above --score-threshold.
  C_segmentation_failure    : NanoSAM given the GT box (perfect localization)
                              still scores mask_iou_gtbox < 0.50 -- failure is
                              NanoSAM's, independent of detection.
  B_localization_propagation: detection succeeded, GT-box segmentation was
                              acceptable (>=0.50), but box_iou < 0.50 OR
                              degradation > 0.15 -- NanoOWL's box imprecision
                              measurably hurt the mask.
  OK                        : detection + segmentation both succeeded well.
(Thresholds 0.50 / 0.15 are the same IoU@0.5 convention used throughout this
project's other quality gates; documented here, not tuned post-hoc.)

This script does NOT rebuild engines, retrain, or modify third_party code.
Production NanoSAM backend (FP16 50-epoch self-distilled) is required
explicitly via CLI args, same convention as eval_nanosam_coco.py.
"""
import argparse
import csv
import json
import os
import random
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

from nanoowl.owl_predictor import OwlPredictor  # noqa: E402
from nanosam.utils.predictor import Predictor as NanosamPredictor  # noqa: E402

SEED = 2027
TARGET_TOTAL = 50
SIZE_BUCKETS = ["small", "medium", "large"]
SCENE_TYPES = ["single", "multi", "crowded"]

DEFAULT_OWL_ENGINE = os.path.join(ROOT, "third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
DEFAULT_NANOSAM_ENC = os.path.join(ROOT, "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine")
DEFAULT_NANOSAM_DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
DEFAULT_OWL_MODEL = "google/owlvit-base-patch32"


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

def _size_bucket(area):
    if area < 32 * 32:
        return "small"
    if area <= 96 * 96:
        return "medium"
    return "large"


def _scene_type(n_valid_in_image):
    if n_valid_in_image <= 1:
        return "single"
    if n_valid_in_image <= 6:
        return "multi"
    return "crowded"


def build_candidates(coco, coco_root):
    """Return list of valid-instance dicts tagged with size_bucket/scene_type."""
    valid_by_image = {}
    all_valid = []
    for img_id in coco.getImgIds():
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        insts = []
        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            try:
                gt_mask = coco.annToMask(ann)
            except Exception:
                continue
            if gt_mask.sum() == 0:
                continue
            insts.append(ann)
        valid_by_image[img_id] = insts

    for img_id, insts in valid_by_image.items():
        scene = _scene_type(len(insts))
        if not insts:
            continue
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(coco_root, img_info["file_name"])
        for ann in insts:
            x, y, w, h = ann["bbox"]
            area = float(ann["area"])
            cat = coco.loadCats(ann["category_id"])[0]
            all_valid.append({
                "image_id": img_id,
                "annotation_id": ann["id"],
                "category_id": ann["category_id"],
                "category_name": cat["name"],
                "image_path": image_path,
                "bbox_xywh": [float(x), float(y), float(w), float(h)],
                "bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)],
                "area": area,
                "size_bucket": _size_bucket(area),
                "scene_type": scene,
                "num_valid_in_image": len(insts),
            })
    return all_valid


def select_50(all_valid, per_cell_target):
    """Deterministically sample ~per_cell_target instances per (size,scene) cell."""
    rng = random.Random(SEED)
    cells = {}
    for inst in all_valid:
        key = (inst["size_bucket"], inst["scene_type"])
        cells.setdefault(key, []).append(inst)
    for key in cells:
        cells[key].sort(key=lambda r: (r["image_id"], r["annotation_id"]))
        rng.shuffle(cells[key])

    selected = []
    shortfall_report = {}
    for sb in SIZE_BUCKETS:
        for st in SCENE_TYPES:
            key = (sb, st)
            pool = cells.get(key, [])
            take = pool[:per_cell_target]
            selected.extend(take)
            if len(take) < per_cell_target:
                shortfall_report[f"{sb}/{st}"] = {"available": len(pool), "target": per_cell_target}

    # top up to TARGET_TOTAL from remaining pool (any cell) if short, deterministic order
    if len(selected) < TARGET_TOTAL:
        chosen_ids = {(s["image_id"], s["annotation_id"]) for s in selected}
        leftover = [r for r in all_valid if (r["image_id"], r["annotation_id"]) not in chosen_ids]
        leftover.sort(key=lambda r: (r["image_id"], r["annotation_id"]))
        rng.shuffle(leftover)
        selected.extend(leftover[: TARGET_TOTAL - len(selected)])
    selected = selected[:TARGET_TOTAL]
    return selected, shortfall_report


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mask_iou(pred, gt):
    inter = int(np.count_nonzero(pred & gt))
    union = int(np.count_nonzero(pred | gt))
    return inter / union if union > 0 else 0.0


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    cx1 = min(max(x1, 0.0), float(w))
    cy1 = min(max(y1, 0.0), float(h))
    cx2 = min(max(x2, 0.0), float(w))
    cy2 = min(max(y2, 0.0), float(h))
    cx1, cx2 = min(cx1, cx2), max(cx1, cx2)
    cy1, cy2 = min(cy1, cy2), max(cy1, cy2)
    return [cx1, cy1, cx2, cy2]


def nanosam_mask_from_box(nsam, box_xyxy):
    pts = np.array([[box_xyxy[0], box_xyxy[1]], [box_xyxy[2], box_xyxy[3]]], dtype=np.float32)
    labs = np.array([2, 3])
    hi, iou, low = nsam.predict(pts, labs)
    mask = (hi[0, 0] > 0).detach().cpu().numpy()  # fixed slot 0, not IoU argmax
    return mask


def classify_error(status, box_iou_val, mask_iou_gtbox, mask_iou_owlbox):
    if status == "DETECTION_FAILURE":
        return "A_detection_failure"
    if mask_iou_gtbox < 0.50:
        return "C_segmentation_failure"
    degradation = mask_iou_gtbox - mask_iou_owlbox
    if (box_iou_val is not None and box_iou_val < 0.50) or degradation > 0.15:
        return "B_localization_propagation"
    return "OK"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description="Stage E4 joint pipeline quality eval (50 fixed COCO cases)")
    ap.add_argument("--coco-root", default=os.path.join(ROOT, "data/coco/val2017"))
    ap.add_argument("--annotations", default=os.path.join(ROOT, "data/coco/annotations/instances_val2017.json"))
    ap.add_argument("--per-cell-target", type=int, default=6,
                     help="instances sampled per (size,scene) cell before top-up to 50")
    ap.add_argument("--score-threshold", type=float, default=0.10)
    ap.add_argument("--nanoowl-engine", default=DEFAULT_OWL_ENGINE)
    ap.add_argument("--nanosam-image-encoder-engine", default=DEFAULT_NANOSAM_ENC)
    ap.add_argument("--nanosam-mask-decoder-engine", default=DEFAULT_NANOSAM_DEC)
    ap.add_argument("--model", default=DEFAULT_OWL_MODEL)
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "results/stage_e4_joint_pipeline"))
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.annotations):
        print(f"DATASET_MISSING: {args.annotations}")
        sys.exit(2)
    try:
        from pycocotools.coco import COCO
    except Exception:
        print("MISSING_DEPENDENCY: pycocotools")
        sys.exit(3)

    coco = COCO(args.annotations)
    print("Building candidate instance pool...", flush=True)
    all_valid = build_candidates(coco, args.coco_root)
    print(f"Candidate pool: {len(all_valid)} valid instances", flush=True)

    selected, shortfall = select_50(all_valid, args.per_cell_target)
    print(f"Selected {len(selected)} instances (target {TARGET_TOTAL})", flush=True)
    if shortfall:
        print(f"Cell shortfalls (topped up from other cells): {shortfall}", flush=True)

    manifest = {
        "seed": SEED,
        "target_total": TARGET_TOTAL,
        "per_cell_target": args.per_cell_target,
        "cell_shortfalls": shortfall,
        "instances": [
            {k: v for k, v in inst.items() if k != "image_path"}
            for inst in selected
        ],
    }
    with open(os.path.join(args.output_dir, "eval_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- load models once ----
    print("Loading NanoOWL...", flush=True)
    owl = OwlPredictor(args.model, image_encoder_engine=args.nanoowl_engine)
    print("Loading NanoSAM (production FP16)...", flush=True)
    nsam = NanosamPredictor(args.nanosam_image_encoder_engine, args.nanosam_mask_decoder_engine)

    # group by image_id to reuse NanoSAM embedding
    by_image = {}
    order = []
    for inst in selected:
        iid = inst["image_id"]
        if iid not in by_image:
            by_image[iid] = []
            order.append(iid)
        by_image[iid].append(inst)

    rows = []
    demo_dir = os.path.join(args.output_dir, "demo")
    os.makedirs(demo_dir, exist_ok=True)
    n_demo_saved = 0
    t_start = time.perf_counter()
    done = 0
    total = len(selected)

    for iid in order:
        insts = by_image[iid]
        img_path = insts[0]["image_path"]
        try:
            image = PIL.Image.open(img_path).convert("RGB")
        except Exception as e:
            for inst in insts:
                rows.append(_failed_row(inst, f"IMAGE_LOAD_FAILED: {e}"))
                done += 1
            continue
        W, H = image.size

        try:
            nsam.set_image(image)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            for inst in insts:
                rows.append(_failed_row(inst, f"NANOSAM_SET_IMAGE_FAILED: {e}"))
                done += 1
            continue

        for inst in insts:
            done += 1
            cat_name = inst["category_name"]
            gt_ann = coco.loadAnns([inst["annotation_id"]])[0]
            try:
                gt_mask = coco.annToMask(gt_ann) > 0
            except Exception as e:
                rows.append(_failed_row(inst, f"GT_MASK_FAILED: {e}"))
                continue

            # (1) NanoOWL detection with text = GT category name
            try:
                text = [cat_name]
                text_encodings = owl.encode_text(text)
                owl_out = owl.predict(image=image, text=text, text_encodings=text_encodings,
                                       threshold=args.score_threshold, pad_square=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                scores = owl_out.scores.detach().cpu().tolist()
                boxes = [[float(v) for v in b] for b in owl_out.boxes.detach().cpu().tolist()]
            except Exception as e:
                rows.append(_failed_row(inst, f"NANOOWL_PREDICT_FAILED: {e}"))
                continue

            if not scores:
                status = "DETECTION_FAILURE"
                owl_box = None
                owl_score = None
                box_iou_val = None
                mask_iou_owlbox = 0.0
            else:
                best_i = max(range(len(scores)), key=lambda i: scores[i])
                owl_box_raw = boxes[best_i]
                owl_score = float(scores[best_i])
                owl_box = clip_box(owl_box_raw, W, H)
                box_iou_val = box_iou_xyxy(owl_box, inst["bbox_xyxy"])
                try:
                    pred_owlbox = nanosam_mask_from_box(nsam, owl_box)
                    if pred_owlbox.shape != gt_mask.shape:
                        pred_owlbox = _resize_mask(pred_owlbox, gt_mask.shape)
                    mask_iou_owlbox = mask_iou(pred_owlbox, gt_mask)
                    status = "OK"
                except Exception as e:
                    rows.append(_failed_row(inst, f"NANOSAM_PREDICT_OWLBOX_FAILED: {e}"))
                    continue

            # (2) NanoSAM with GT box (quality ceiling / reference)
            try:
                gt_box_clipped = clip_box(inst["bbox_xyxy"], W, H)
                pred_gtbox = nanosam_mask_from_box(nsam, gt_box_clipped)
                if pred_gtbox.shape != gt_mask.shape:
                    pred_gtbox = _resize_mask(pred_gtbox, gt_mask.shape)
                mask_iou_gtbox = mask_iou(pred_gtbox, gt_mask)
            except Exception as e:
                rows.append(_failed_row(inst, f"NANOSAM_PREDICT_GTBOX_FAILED: {e}"))
                continue

            degradation = mask_iou_gtbox - mask_iou_owlbox
            error_bucket = classify_error(status, box_iou_val, mask_iou_gtbox, mask_iou_owlbox)

            row = {
                "image_id": inst["image_id"],
                "annotation_id": inst["annotation_id"],
                "category": cat_name,
                "size_bucket": inst["size_bucket"],
                "scene_type": inst["scene_type"],
                "gt_bbox_xyxy": inst["bbox_xyxy"],
                "owl_score": owl_score,
                "owl_bbox_xyxy": owl_box,
                "status": status,
                "box_iou": box_iou_val,
                "mask_iou_owlbox": mask_iou_owlbox,
                "mask_iou_gtbox": mask_iou_gtbox,
                "degradation": degradation,
                "error_bucket": error_bucket,
            }
            rows.append(row)

            if n_demo_saved < 8 and status == "OK":
                try:
                    _save_demo(demo_dir, inst, image, owl_box, gt_mask, pred_owlbox, error_bucket)
                    n_demo_saved += 1
                except Exception:
                    pass

            if done % 10 == 0 or done == total:
                elapsed = time.perf_counter() - t_start
                print(f"[{done}/{total}] elapsed={elapsed:.1f}s last: cat={cat_name} "
                      f"status={status} box_iou={box_iou_val} mask_iou_owlbox={mask_iou_owlbox:.3f} "
                      f"mask_iou_gtbox={mask_iou_gtbox:.3f} bucket={error_bucket}", flush=True)

    # ---- write csv ----
    csv_path = os.path.join(args.output_dir, "error_decomposition.csv")
    fieldnames = ["image_id", "annotation_id", "category", "size_bucket", "scene_type",
                  "gt_bbox_xyxy", "owl_score", "owl_bbox_xyxy", "status", "box_iou",
                  "mask_iou_owlbox", "mask_iou_gtbox", "degradation", "error_bucket"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- aggregate summary ----
    summary = summarize(rows)
    with open(os.path.join(args.output_dir, "joint_quality.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print("ALL_DONE", flush=True)
    return 0


def _resize_mask(mask_bool, target_shape):
    pm = PIL.Image.fromarray(mask_bool.astype(np.uint8) * 255)
    pm = pm.resize((target_shape[1], target_shape[0]), PIL.Image.NEAREST)
    return np.asarray(pm) > 127


def _failed_row(inst, error_msg):
    return {
        "image_id": inst["image_id"], "annotation_id": inst["annotation_id"],
        "category": inst["category_name"], "size_bucket": inst["size_bucket"],
        "scene_type": inst["scene_type"], "gt_bbox_xyxy": inst["bbox_xyxy"],
        "owl_score": None, "owl_bbox_xyxy": None, "status": "PIPELINE_FAILED",
        "box_iou": None, "mask_iou_owlbox": None, "mask_iou_gtbox": None,
        "degradation": None, "error_bucket": "PIPELINE_FAILED", "error": error_msg,
    }


def _save_demo(demo_dir, inst, image, owl_box, gt_mask, pred_owlbox, error_bucket):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].imshow(np.ma.masked_where(~gt_mask, gt_mask), alpha=0.5, cmap="Greens")
    x1, y1, x2, y2 = inst["bbox_xyxy"]
    axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
    axes[0].set_title(f"GT: {inst['category_name']}")
    axes[0].set_axis_off()

    axes[1].imshow(image)
    axes[1].imshow(np.ma.masked_where(~pred_owlbox, pred_owlbox), alpha=0.5, cmap="Reds")
    if owl_box is not None:
        ox1, oy1, ox2, oy2 = owl_box
        axes[1].add_patch(plt.Rectangle((ox1, oy1), ox2 - ox1, oy2 - oy1, fill=False, edgecolor="red", linewidth=2))
    axes[1].set_title(f"Joint pred ({error_bucket})")
    axes[1].set_axis_off()
    out_path = os.path.join(demo_dir, f"img{inst['image_id']}_ann{inst['annotation_id']}.jpg")
    plt.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close()


def summarize(rows):
    valid = [r for r in rows if r["error_bucket"] != "PIPELINE_FAILED"]
    n = len(rows)
    bucket_counts = {}
    for r in rows:
        bucket_counts[r["error_bucket"]] = bucket_counts.get(r["error_bucket"], 0) + 1

    ok_rows = [r for r in valid if r["status"] == "OK"]
    box_ious = [r["box_iou"] for r in ok_rows if r["box_iou"] is not None]
    mask_owl = [r["mask_iou_owlbox"] for r in valid if r["mask_iou_owlbox"] is not None]
    mask_gt = [r["mask_iou_gtbox"] for r in valid if r["mask_iou_gtbox"] is not None]
    degr = [r["degradation"] for r in valid if r["degradation"] is not None]

    def stats(a):
        if not a:
            return {"n": 0, "mean": None, "median": None}
        arr = np.asarray(a, dtype=np.float64)
        return {"n": int(arr.size), "mean": float(np.mean(arr)), "median": float(np.median(arr))}

    by_size = {}
    by_scene = {}
    for sb in SIZE_BUCKETS:
        sub = [r for r in valid if r["size_bucket"] == sb and r["mask_iou_owlbox"] is not None]
        by_size[sb] = stats([r["mask_iou_owlbox"] for r in sub])
    for st in SCENE_TYPES:
        sub = [r for r in valid if r["scene_type"] == st and r["mask_iou_owlbox"] is not None]
        by_scene[st] = stats([r["mask_iou_owlbox"] for r in sub])

    return {
        "total_cases": n,
        "error_bucket_counts": bucket_counts,
        "detection_success_rate": round((n - bucket_counts.get("A_detection_failure", 0) - bucket_counts.get("PIPELINE_FAILED", 0)) / n, 4) if n else None,
        "box_iou_owl_vs_gt": stats(box_ious),
        "mask_iou_owlbox_end_to_end": stats(mask_owl),
        "mask_iou_gtbox_ceiling": stats(mask_gt),
        "degradation_from_box_imprecision": stats(degr),
        "mask_iou_owlbox_by_size_bucket": by_size,
        "mask_iou_owlbox_by_scene_type": by_scene,
    }


if __name__ == "__main__":
    raise SystemExit(main())
