#!/usr/bin/env python3
"""Stage F - Prompt Ensemble Prototype + Evaluation.

Lightweight prompt-ensemble implementation (project-side, no third_party mod).
For each category, encode multiple prompt templates, compute their mean in the
normalized 512-d embedding space, re-normalize to unit sphere, use as the
ensemble text_embed. Evaluate on the same E4 50-instance manifest; compare
single-prompt (base) vs ensemble detection_success / box_iou / mask_iou.
"""
import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import PIL.Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from nanoowl.owl_predictor import OwlPredictor
from nanosam.utils.predictor import Predictor as NanosamPredictor

MANIFEST_PATH = os.path.join(ROOT, "results/stage_e4_joint_pipeline/eval_manifest.json")
ANNOTATIONS = os.path.join(ROOT, "data/coco/annotations/instances_val2017.json")
COCO_ROOT = os.path.join(ROOT, "data/coco/val2017")
OWL_ENGINE = os.path.join(ROOT, "third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
NANOSAM_ENC = os.path.join(ROOT, "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine")
NANOSAM_DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
OWL_MODEL = "google/owlvit-base-patch32"
SCORE_THRESHOLD = 0.10
OUT_DIR = os.path.join(ROOT, "results/stage_f_vlm_algorithm")

# Frozen before running. Each category gets 3 templates (not including synonyms
# which showed poor localization in prompt_sensitivity exp).
ENSEMBLE_TEMPLATES = {
    "default": [
        "{cat}",
        "a {cat}",
        "photo of a {cat}",
    ]
}


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
    mask = (hi[0, 0] > 0).detach().cpu().numpy()
    return mask


def resize_mask(mask_bool, target_shape):
    pm = PIL.Image.fromarray(mask_bool.astype(np.uint8) * 255)
    pm = pm.resize((target_shape[1], target_shape[0]), PIL.Image.NEAREST)
    return np.asarray(pm) > 127


def build_ensemble_text_embedding(owl, cat_name):
    """Encode 3 templates, mean+renormalize in the 512-d space."""
    templates = [t.format(cat=cat_name) for t in ENSEMBLE_TEMPLATES["default"]]
    with torch.no_grad():
        text_output = owl.encode_text(templates)
    embeds = text_output.text_embeds  # [3, 512]
    # Mean in the original space (before the decode() normalization)
    mean_embed = embeds.mean(dim=0, keepdim=True)  # [1, 512]
    # Re-normalize to unit sphere (same as decode() will do)
    norm = torch.linalg.norm(mean_embed, dim=-1, keepdim=True) + 1e-6
    ensemble_embed = mean_embed / norm
    # Wrap back in OwlEncodeTextOutput for API compatibility
    from nanoowl.owl_predictor import OwlEncodeTextOutput
    return OwlEncodeTextOutput(text_embeds=ensemble_embed), templates


def main():
    from pycocotools.coco import COCO

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    instances = manifest["instances"]
    print(f"Loaded {len(instances)} instances from E4 manifest", flush=True)

    coco = COCO(ANNOTATIONS)
    owl = OwlPredictor(OWL_MODEL, image_encoder_engine=OWL_ENGINE)
    nsam = NanosamPredictor(NANOSAM_ENC, NANOSAM_DEC)

    by_image = {}
    order = []
    for inst in instances:
        iid = inst["image_id"]
        if iid not in by_image:
            by_image[iid] = []
            order.append(iid)
        by_image[iid].append(inst)

    rows = []
    done = 0
    total = len(instances)

    for iid in order:
        insts = by_image[iid]
        img_info = coco.loadImgs(iid)[0]
        img_path = os.path.join(COCO_ROOT, img_info["file_name"])
        image = PIL.Image.open(img_path).convert("RGB")
        W, H = image.size
        nsam.set_image(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for inst in insts:
            done += 1
            cat_name = inst["category_name"]
            gt_ann = coco.loadAnns([inst["annotation_id"]])[0]
            gt_mask = coco.annToMask(gt_ann) > 0
            gt_box = inst["bbox_xyxy"]

            # Single prompt (base)
            single_text_encoding = owl.encode_text([cat_name])
            owl_out_single = owl.predict(image=image, text=[cat_name], text_encodings=single_text_encoding,
                                          threshold=SCORE_THRESHOLD, pad_square=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            single_scores = owl_out_single.scores.detach().cpu().tolist()
            single_boxes = [[float(v) for v in b] for b in owl_out_single.boxes.detach().cpu().tolist()]

            if not single_scores:
                single_result = {"success": False, "score": None, "box_iou": None, "mask_iou": None}
            else:
                best_i = max(range(len(single_scores)), key=lambda i: single_scores[i])
                s_score = float(single_scores[best_i])
                s_box = clip_box(single_boxes[best_i], W, H)
                s_box_iou = box_iou_xyxy(s_box, gt_box)
                s_mask = nanosam_mask_from_box(nsam, s_box)
                if s_mask.shape != gt_mask.shape:
                    s_mask = resize_mask(s_mask, gt_mask.shape)
                s_mask_iou = mask_iou(s_mask, gt_mask)
                single_result = {
                    "success": True, "score": round(s_score, 6),
                    "box_iou": round(s_box_iou, 6), "mask_iou": round(s_mask_iou, 6)
                }

            # Ensemble prompt
            ensemble_encoding, templates = build_ensemble_text_embedding(owl, cat_name)
            owl_out_ensemble = owl.predict(image=image, text=[cat_name], text_encodings=ensemble_encoding,
                                            threshold=SCORE_THRESHOLD, pad_square=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ens_scores = owl_out_ensemble.scores.detach().cpu().tolist()
            ens_boxes = [[float(v) for v in b] for b in owl_out_ensemble.boxes.detach().cpu().tolist()]

            if not ens_scores:
                ensemble_result = {"success": False, "score": None, "box_iou": None, "mask_iou": None}
            else:
                best_i = max(range(len(ens_scores)), key=lambda i: ens_scores[i])
                e_score = float(ens_scores[best_i])
                e_box = clip_box(ens_boxes[best_i], W, H)
                e_box_iou = box_iou_xyxy(e_box, gt_box)
                e_mask = nanosam_mask_from_box(nsam, e_box)
                if e_mask.shape != gt_mask.shape:
                    e_mask = resize_mask(e_mask, gt_mask.shape)
                e_mask_iou = mask_iou(e_mask, gt_mask)
                ensemble_result = {
                    "success": True, "score": round(e_score, 6),
                    "box_iou": round(e_box_iou, 6), "mask_iou": round(e_mask_iou, 6)
                }

            rows.append({
                "image_id": iid, "annotation_id": inst["annotation_id"], "category": cat_name,
                "single_success": single_result["success"], "single_score": single_result["score"],
                "single_box_iou": single_result["box_iou"], "single_mask_iou": single_result["mask_iou"],
                "ensemble_success": ensemble_result["success"], "ensemble_score": ensemble_result["score"],
                "ensemble_box_iou": ensemble_result["box_iou"], "ensemble_mask_iou": ensemble_result["mask_iou"],
            })

            if done % 10 == 0 or done == total:
                print(f"[{done}/{total}]", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "prompt_ensemble.csv")
    fieldnames = ["image_id", "annotation_id", "category",
                  "single_success", "single_score", "single_box_iou", "single_mask_iou",
                  "ensemble_success", "ensemble_score", "ensemble_box_iou", "ensemble_mask_iou"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}", flush=True)

    # Aggregate
    single_succ = [r for r in rows if r["single_success"]]
    ens_succ = [r for r in rows if r["ensemble_success"]]

    def stats(a):
        if not a:
            return None
        arr = np.array(a, dtype=np.float64)
        return {"mean": float(np.mean(arr)), "median": float(np.median(arr))}

    summary = {
        "protocol": "E4 50-instance manifest, ensemble = mean+renormalize of 3 templates (cat, a cat, photo of a cat)",
        "single_prompt": {
            "detection_success_rate": len(single_succ) / len(rows),
            "box_iou": stats([r["single_box_iou"] for r in single_succ]),
            "mask_iou": stats([r["single_mask_iou"] for r in single_succ]),
        },
        "ensemble_prompt": {
            "detection_success_rate": len(ens_succ) / len(rows),
            "box_iou": stats([r["ensemble_box_iou"] for r in ens_succ]),
            "mask_iou": stats([r["ensemble_mask_iou"] for r in ens_succ]),
        },
        "delta": {
            "detection_success_rate": (len(ens_succ) - len(single_succ)) / len(rows),
            "box_iou_mean_delta": (stats([r["ensemble_box_iou"] for r in ens_succ])["mean"] - stats([r["single_box_iou"] for r in single_succ])["mean"]) if (ens_succ and single_succ) else None,
            "mask_iou_mean_delta": (stats([r["ensemble_mask_iou"] for r in ens_succ])["mean"] - stats([r["single_mask_iou"] for r in single_succ])["mean"]) if (ens_succ and single_succ) else None,
        },
        "decision": "PROMPT_ENSEMBLE_HELPFUL" if (len(ens_succ) > len(single_succ) or (ens_succ and single_succ and stats([r["ensemble_mask_iou"] for r in ens_succ])["mean"] > stats([r["single_mask_iou"] for r in single_succ])["mean"] + 0.01)) else "PROMPT_ENSEMBLE_NO_CLEAR_GAIN",
    }

    with open(os.path.join(OUT_DIR, "prompt_ensemble_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
