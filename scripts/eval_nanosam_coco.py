#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
EdgePerception — NanoSAM COCO quality evaluator (Stage 5A framework).

Evaluates NanoSAM mask quality on COCO val2017 using GT bounding-box prompts.

Protocol (see docs/evaluation_protocol.md):
  - sample unit: COCO instance (per annotation), not image
  - GT box (xywh -> xyxy) -> NanoSAM box prompt points=[[x1,y1],[x2,y2]], labels=[2,3]
  - GT mask: coco.annToMask(ann) > 0  (polygon; crowd excluded)
  - predicted mask: slot 0  (hi_res_mask[0,0] > 0)  -- BASELINE LOCKED, NOT IoU argmax
  - instance IoU = |pred ∩ gt| / |pred ∪ gt|  (union==0 -> 0)

Stage 5A status: framework only. Large-scale inference NOT executed in this stage.
  - dataset missing  -> prints DATASET_MISSING, exit 2
  - pycocotools miss -> prints MISSING_DEPENDENCY: pycocotools, exit 3
  - does NOT fabricate results.

This script does NOT rebuild engines, download data, or modify upstream.
"""

import argparse
import json
import os
import sys
import numpy as np


# --------------------------------------------------------------------------- #
# dependency / dataset gating
# --------------------------------------------------------------------------- #

def _check_pycocotools():
    try:
        from pycocotools.coco import COCO  # noqa: F401
        return True
    except Exception:
        return False


def _check_dataset(coco_root, ann_path):
    if not os.path.isfile(ann_path):
        return False, f"annotations not found: {ann_path}"
    if not os.path.isdir(coco_root):
        return False, f"val2017 images dir not found: {coco_root}"
    return True, ""


# --------------------------------------------------------------------------- #
# subset loading
# --------------------------------------------------------------------------- #

def load_manifest(manifest_path):
    if not os.path.isfile(manifest_path):
        print(f"MANIFEST_MISSING: {manifest_path}", file=sys.stderr)
        sys.exit(4)
    with open(manifest_path, "r") as f:
        return json.load(f)


def build_full_instance_list(coco, coco_root):
    """FULL subset: every legal instance in val2017 (iscrowd==0, valid box, decodable seg)."""
    instances = []
    for img_id in coco.getImgIds():
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        for ann in anns:
            inst = _ann_to_instance(ann, coco, coco_root)
            if inst["status"] == "OK":
                instances.append(inst)
    return instances


def _ann_to_instance(ann, coco, coco_root):
    """Map a COCO annotation to an eval instance dict with an explicit validity status.

    Returns dict with 'status' in {OK, CROWD, INVALID_BOX, INVALID_GT_MASK}.
    This is the single place where the protocol's invalid-sample rules are enforced
    (see evaluation_protocol.md §1.6). Do not change ad-hoc during evaluation.
    """
    ann_id = ann["id"]
    image_id = ann["image_id"]
    iscrowd = ann.get("iscrowd", 0)
    if iscrowd == 1:
        return {"status": "CROWD", "annotation_id": ann_id, "image_id": image_id}

    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0:
        return {"status": "INVALID_BOX", "annotation_id": ann_id, "image_id": image_id}

    cat_id = ann["category_id"]
    cat = coco.loadCats(cat_id)[0]
    area = float(ann["area"])
    size_bucket = _size_bucket(area)

    img_info = coco.loadImgs(image_id)[0]
    image_path = os.path.join(coco_root, img_info["file_name"])

    # GT mask decodability check (annToMask requires pycocotools).
    try:
        gt_mask = coco.annToMask(ann)
    except Exception:
        return {"status": "INVALID_GT_MASK", "annotation_id": ann_id, "image_id": image_id}
    if gt_mask.sum() == 0:
        return {"status": "INVALID_GT_MASK", "annotation_id": ann_id, "image_id": image_id}

    return {
        "status": "OK",
        "image_id": image_id,
        "annotation_id": ann_id,
        "category_id": cat_id,
        "category_name": cat["name"],
        "image_path": image_path,
        "bbox_xywh": [float(x), float(y), float(w), float(h)],
        "bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)],
        "area": area,
        "size_bucket": size_bucket,
        "iscrowd": 0,
    }


def _size_bucket(area):
    if area < 32 * 32:
        return "small"
    if area <= 96 * 96:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# IoU
# --------------------------------------------------------------------------- #

def mask_iou(pred, gt):
    """pred, gt: bool numpy arrays of identical HxW. union==0 -> 0.0."""
    inter = int(np.count_nonzero(pred & gt))
    union = int(np.count_nonzero(pred | gt))
    if union == 0:
        return 0.0
    return inter / union


# --------------------------------------------------------------------------- #
# metrics aggregation
# --------------------------------------------------------------------------- #

def aggregate(ious):
    """ious: list of float instance IoUs (valid instances only)."""
    if not ious:
        return {
            "valid": 0, "mean_iou": None, "median_iou": None,
            "p10": None, "p50": None, "p90": None,
            "iou_ge_050_ratio": None, "iou_ge_075_ratio": None,
        }
    a = np.asarray(ious, dtype=np.float64)
    return {
        "valid": int(a.size),
        "mean_iou": float(np.mean(a)),
        "median_iou": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "iou_ge_050_ratio": float(np.mean(a >= 0.50)),
        "iou_ge_075_ratio": float(np.mean(a >= 0.75)),
    }


# --------------------------------------------------------------------------- #
# evaluation core (model-loaded path)
# --------------------------------------------------------------------------- #

def run_evaluation(predictor, instances, output_dir):
    """Run NanoSAM evaluation over instances.

    - model loaded once (predictor passed in already constructed).
    - same-image instances reuse the image embedding: set_image only when image_id changes.
    - per instance: GT box prompt, slot 0 predicted mask, IoU vs GT mask.
    - per-instance inference failure is caught -> INFERENCE_FAILED, IoU=None, does not abort.
    """
    from PIL import Image
    import torch  # noqa: F401  (for cuda.synchronize if needed)

    # GT mask needs coco object; re-derive lazily via annToMask using a fresh COCO is avoided —
    # instead GT mask is recomputed from the annotation through the coco handle passed via
    # a closure set by the caller (see evaluate()).
    raise NotImplementedError(
        "Model-loaded path is intentionally a framework stub in Stage 5A. "
        "Large-scale inference is not executed in this stage."
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="NanoSAM COCO quality evaluator (Stage 5A framework)")
    parser.add_argument("--coco-root", default="data/coco/val2017")
    parser.add_argument("--annotations", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--subset", choices=["smoke", "quick", "standard", "full"], default="standard")
    parser.add_argument("--manifest", default=None, help="path to subset manifest JSON (ignored for full)")
    parser.add_argument("--image-encoder-engine", required=False)
    parser.add_argument("--mask-decoder-engine", required=False)
    parser.add_argument("--output-dir", default="results/nanosam_coco_eval")
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()

    # gate 1: dataset
    ok, msg = _check_dataset(args.coco_root, args.annotations)
    if not ok:
        print(f"DATASET_MISSING: {msg}")
        print("Acquire official COCO 2017 val (see docs/evaluation_protocol.md §0.2).")
        sys.exit(2)

    # gate 2: pycocotools
    if not _check_pycocotools():
        print("MISSING_DEPENDENCY: pycocotools")
        print("Safe install suggestion (NOT auto-run): pip install pycocotools")
        sys.exit(3)

    # engine args required to actually run
    if not args.image_encoder_engine or not args.mask_decoder_engine:
        print("ENGINE_ARGS_REQUIRED: --image-encoder-engine and --mask-decoder-engine are required to run.")
        print("Stage 5A only defines the framework; no large-scale inference is executed here.")
        sys.exit(5)

    # lazy import (only after gating passed)
    from pycocotools.coco import COCO
    coco = COCO(args.annotations)

    # build instance list per subset
    if args.subset == "full":
        instances = build_full_instance_list(coco, args.coco_root)
    else:
        if not args.manifest:
            print(f"MANIFEST_REQUIRED for subset={args.subset} (use --manifest)")
            sys.exit(4)
        man = load_manifest(args.manifest)
        instances = man.get("instances", [])
        if not instances:
            print("MANIFEST_EMPTY: no instances in manifest")
            sys.exit(4)

    if args.max_instances is not None:
        instances = instances[: args.max_instances]

    print(f"subset={args.subset} instances={len(instances)}")
    print("STAGE_5A_FRAMEWORK_READY: protocol & CLI validated; large-scale inference deferred.")
    # Real per-instance loop (run_evaluation) is intentionally not executed in Stage 5A.
    sys.exit(0)


if __name__ == "__main__":
    main()
