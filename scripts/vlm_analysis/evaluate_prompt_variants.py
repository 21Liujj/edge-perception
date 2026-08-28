#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage F - Prompt Sensitivity Controlled Experiment.

Reuses the E4 fixed 50-instance manifest (results/stage_e4_joint_pipeline/
eval_manifest.json, seed=2027) and the E4 pipeline contract (NanoOWL
open-vocab detect with text=<prompt>, threshold=0.10, box->NanoSAM FP16
production backend, fixed-slot-0 mask). The ONLY thing varied across runs is
the text prompt string; images, categories, engine, and threshold are fixed.

Prompt variants per instance (frozen BEFORE running, not tuned after):
  base      : "<category_name>"                    (same as E4's own text)
  template_a: "a <category_name>"
  template_photo: "photo of a <category_name>"
  synonym   : only for categories where a synonym is genuinely equivalent
              (person->human, car->automobile, chair->seat). Skipped for all
              other categories rather than force a non-equivalent synonym.

For each (instance, variant) we record: score (best detection, or None),
bbox (xyxy, or None), box_iou vs GT, mask_iou vs GT (NanoSAM on the OWL box),
detection_success (bool). This isolates whether prompt wording moves the
CLASSIFICATION SCORE, the LOCALIZATION (bbox/box_iou), or both.
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

import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from nanoowl.owl_predictor import OwlPredictor  # noqa: E402
from nanosam.utils.predictor import Predictor as NanosamPredictor  # noqa: E402

MANIFEST_PATH = os.path.join(ROOT, "results/stage_e4_joint_pipeline/eval_manifest.json")
ANNOTATIONS = os.path.join(ROOT, "data/coco/annotations/instances_val2017.json")
COCO_ROOT = os.path.join(ROOT, "data/coco/val2017")
OWL_ENGINE = os.path.join(ROOT, "third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
NANOSAM_ENC = os.path.join(ROOT, "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine")
NANOSAM_DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
OWL_MODEL = "google/owlvit-base-patch32"
SCORE_THRESHOLD = 0.10  # same as E4

# Frozen before running. Genuinely-equivalent synonyms only.
SYNONYM_MAP = {
    "person": "human",
    "car": "automobile",
    "chair": "seat",
}

OUT_DIR = os.path.join(ROOT, "results/stage_f_vlm_algorithm")


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
    mask = (hi[0, 0] > 0).detach().cpu().numpy()  # fixed slot 0, matches E4 convention
    return mask


def resize_mask(mask_bool, target_shape):
    pm = PIL.Image.fromarray(mask_bool.astype(np.uint8) * 255)
    pm = pm.resize((target_shape[1], target_shape[0]), PIL.Image.NEAREST)
    return np.asarray(pm) > 127


def build_variants(cat_name):
    variants = [
        ("base", cat_name),
        ("template_a", f"a {cat_name}"),
        ("template_photo", f"photo of a {cat_name}"),
    ]
    if cat_name in SYNONYM_MAP:
        variants.append(("synonym", SYNONYM_MAP[cat_name]))
    return variants


def main():
    from pycocotools.coco import COCO

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    instances = manifest["instances"]
    print(f"Loaded {len(instances)} fixed instances from E4 manifest (seed={manifest['seed']})", flush=True)

    coco = COCO(ANNOTATIONS)

    print("Loading NanoOWL...", flush=True)
    owl = OwlPredictor(OWL_MODEL, image_encoder_engine=OWL_ENGINE)
    print("Loading NanoSAM (production FP16)...", flush=True)
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
    total = sum(len(build_variants(i["category_name"])) for i in instances)
    done = 0

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
            cat_name = inst["category_name"]
            gt_ann = coco.loadAnns([inst["annotation_id"]])[0]
            gt_mask = coco.annToMask(gt_ann) > 0
            gt_box = inst["bbox_xyxy"]

            for variant_type, prompt_text in build_variants(cat_name):
                done += 1
                text_encodings = owl.encode_text([prompt_text])
                owl_out = owl.predict(image=image, text=[prompt_text], text_encodings=text_encodings,
                                       threshold=SCORE_THRESHOLD, pad_square=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                scores = owl_out.scores.detach().cpu().tolist()
                boxes = [[float(v) for v in b] for b in owl_out.boxes.detach().cpu().tolist()]

                if not scores:
                    row = {
                        "image_id": iid, "annotation_id": inst["annotation_id"], "category": cat_name,
                        "variant_type": variant_type, "prompt_text": prompt_text,
                        "detection_success": False, "score": None, "bbox_xyxy": None,
                        "box_iou": None, "mask_iou": None,
                    }
                else:
                    best_i = max(range(len(scores)), key=lambda i: scores[i])
                    score = float(scores[best_i])
                    box = clip_box(boxes[best_i], W, H)
                    b_iou = box_iou_xyxy(box, gt_box)
                    pred_mask = nanosam_mask_from_box(nsam, box)
                    if pred_mask.shape != gt_mask.shape:
                        pred_mask = resize_mask(pred_mask, gt_mask.shape)
                    m_iou = mask_iou(pred_mask, gt_mask)
                    row = {
                        "image_id": iid, "annotation_id": inst["annotation_id"], "category": cat_name,
                        "variant_type": variant_type, "prompt_text": prompt_text,
                        "detection_success": True, "score": round(score, 6),
                        "bbox_xyxy": box, "box_iou": round(b_iou, 6), "mask_iou": round(m_iou, 6),
                    }
                rows.append(row)

            if done % 40 == 0 or done == total:
                print(f"[{done}/{total}]", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "prompt_sensitivity.csv")
    fieldnames = ["image_id", "annotation_id", "category", "variant_type", "prompt_text",
                  "detection_success", "score", "bbox_xyxy", "box_iou", "mask_iou"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path} ({len(rows)} rows)", flush=True)

    # aggregate per variant_type
    by_variant = {}
    for vt in ["base", "template_a", "template_photo", "synonym"]:
        sub = [r for r in rows if r["variant_type"] == vt]
        if not sub:
            continue
        succ = [r for r in sub if r["detection_success"]]
        by_variant[vt] = {
            "n": len(sub),
            "detection_success_rate": round(len(succ) / len(sub), 4),
            "mean_score": round(float(np.mean([r["score"] for r in succ])), 6) if succ else None,
            "mean_box_iou": round(float(np.mean([r["box_iou"] for r in succ])), 6) if succ else None,
            "mean_mask_iou": round(float(np.mean([r["mask_iou"] for r in succ])), 6) if succ else None,
        }

    base_ids = {(r["image_id"], r["annotation_id"]) for r in rows if r["variant_type"] == "base" and r["detection_success"]}
    flip_analysis = {}
    for vt in ["template_a", "template_photo", "synonym"]:
        sub = [r for r in rows if r["variant_type"] == vt]
        gained = sum(1 for r in sub if r["detection_success"] and (r["image_id"], r["annotation_id"]) not in base_ids)
        lost = sum(1 for r in sub if not r["detection_success"] and (r["image_id"], r["annotation_id"]) in base_ids)
        flip_analysis[vt] = {"gained_detection_vs_base": gained, "lost_detection_vs_base": lost}

    summary = {
        "protocol": "E4 fixed 50-instance manifest (seed=2027), same NanoOWL engine, same NanoSAM FP16 production backend, same threshold=0.10; only text prompt varied.",
        "by_variant": by_variant,
        "detection_flip_vs_base": flip_analysis,
        "interpretation": (
            "Compare mean_score across variants (classification-confidence axis) vs mean_box_iou "
            "across variants (localization axis, restricted to successful detections) to determine "
            "whether prompt wording mainly shifts confidence, localization, or both."
        ),
    }
    with open(os.path.join(OUT_DIR, "prompt_sensitivity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
