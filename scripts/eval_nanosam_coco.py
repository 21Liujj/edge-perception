#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
EdgePerception - NanoSAM COCO quality evaluator.

Evaluates NanoSAM mask quality on COCO val2017 using GT bounding-box prompts.

Protocol (see docs/evaluation_protocol.md):
  - sample unit: COCO instance (per annotation), not image
  - GT box (xywh -> xyxy) -> NanoSAM box prompt points=[[x1,y1],[x2,y2]], labels=[2,3]
  - GT mask: coco.annToMask(ann) > 0  (polygon; crowd excluded)
  - predicted mask: slot 0  (hi_res_mask[0,0] > 0)  -- BASELINE LOCKED, NOT IoU argmax
  - instance IoU = |pred & gt| / |pred | gt|  (union==0 -> 0)

Gating:
  - dataset missing  -> DATASET_MISSING, exit 2
  - pycocotools miss -> MISSING_DEPENDENCY: pycocotools, exit 3
  - manifest missing -> MANIFEST_MISSING, exit 4
  - engine args missing -> ENGINE_ARGS_REQUIRED, exit 5
  - does NOT fabricate results.

This script does NOT rebuild engines, download data, or modify upstream.
"""

import argparse
import csv
import json
import os
import sys
import time
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

SELECTED_MASK_INDEX = 0  # BASELINE LOCKED (slot 0), NOT IoU argmax.


def run_evaluation(predictor, coco, instances, output_dir, subset_name):
    """Run NanoSAM evaluation over instances.

    - model loaded once (predictor passed in already constructed).
    - same-image instances reuse the image embedding: set_image only when image_id changes.
    - per instance: GT box prompt, slot 0 predicted mask, IoU vs GT mask.
    - per-instance inference failure is caught -> INFERENCE_FAILED, IoU=None, does not abort.
    """
    from PIL import Image
    import torch

    os.makedirs(output_dir, exist_ok=True)
    per_instance_path = os.path.join(output_dir, "per_instance.jsonl")
    failed_path = os.path.join(output_dir, "failed_instances.json")
    runlog_path = os.path.join(output_dir, "run.log")

    runlog = open(runlog_path, "w")
    def log(msg):
        print(msg, flush=True)
        runlog.write(msg + "\n")
        runlog.flush()

    valid_ious = []
    per_records = []
    failed = []
    skipped = {"CROWD": 0, "INVALID_BOX": 0, "INVALID_GT_MASK": 0}

    # group instances by image_id to maximize embedding reuse while preserving
    # the subset order within an image.
    by_image = {}
    order = []
    for inst in instances:
        # full subset may carry only annotation_id/image_id; (re)materialize fields here.
        rec = _materialize(inst, coco)
        if rec["status"] != "OK":
            skipped[rec["status"]] = skipped.get(rec["status"], 0) + 1
            continue
        iid = rec["image_id"]
        if iid not in by_image:
            by_image[iid] = []
            order.append(iid)
        by_image[iid].append(rec)

    unique_images = len(order)
    embedding_set_calls = 0
    embedding_reuse = 0
    t_start = time.perf_counter()

    cur_image_id = None
    total = sum(len(v) for v in by_image.values())
    done = 0
    for iid in order:
        recs = by_image[iid]
        # load image once per unique image
        img_path = recs[0]["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            for rec in recs:
                failed.append({"image_id": iid, "annotation_id": rec["annotation_id"],
                               "status": "IMAGE_LOAD_FAILED", "error": str(e)})
                done += 1
            continue

        # set image embedding once for this image
        try:
            predictor.set_image(image)
            torch.cuda.synchronize()
            embedding_set_calls += 1
            cur_image_id = iid
        except Exception as e:
            for rec in recs:
                failed.append({"image_id": iid, "annotation_id": rec["annotation_id"],
                               "status": "INFERENCE_FAILED", "stage": "set_image", "error": str(e)})
                done += 1
            continue

        for rec in recs:
            done += 1
            # GT mask (cached on coco side via annToMask; cheap to recompute)
            try:
                gt_mask = (coco.annToMask(rec["_ann"]) > 0)
            except Exception as e:
                failed.append({"image_id": iid, "annotation_id": rec["annotation_id"],
                               "status": "GT_MASK_FAILED", "error": str(e)})
                continue

            x1, y1, x2, y2 = rec["bbox_xyxy"]
            points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            point_labels = np.array([2, 3])

            try:
                torch.cuda.synchronize()
                hi_res_mask, mask_iou_preds, low_res_mask = predictor.predict(
                    points=points, point_labels=point_labels
                )
                torch.cuda.synchronize()
            except Exception as e:
                failed.append({"image_id": iid, "annotation_id": rec["annotation_id"],
                               "status": "INFERENCE_FAILED", "stage": "predict", "error": str(e)})
                continue

            # NaN/Inf guard on the mask tensor
            t = hi_res_mask.detach().float()
            if bool(torch.isnan(t).any().item()) or bool(torch.isinf(t).any().item()):
                failed.append({"image_id": iid, "annotation_id": rec["annotation_id"],
                               "status": "INFERENCE_FAILED", "stage": "nan_inf_mask"})
                continue

            # slot 0 (BASELINE LOCKED)
            pred_mask = (hi_res_mask[0, SELECTED_MASK_INDEX] > 0).detach().cpu().numpy()

            # ensure shape match with GT (predictor upscales to image HxW)
            if pred_mask.shape != gt_mask.shape:
                # resize pred to gt shape if mismatch (defensive; should match)
                from PIL import Image as PImage
                pm = PImage.fromarray(pred_mask.astype(np.uint8) * 255)
                pm = pm.resize((gt_mask.shape[1], gt_mask.shape[0]), PImage.NEAREST)
                pred_mask = (np.asarray(pm) > 127)

            iou_val = mask_iou(pred_mask, gt_mask)
            if not np.isfinite(iou_val):
                failed.append({"image_id": iid, "annotation_id": rec["annotation_id"],
                               "status": "INFERENCE_FAILED", "stage": "iou_nan"})
                continue

            valid_ious.append(iou_val)
            rec_out = {
                "image_id": iid,
                "annotation_id": rec["annotation_id"],
                "category_id": rec["category_id"],
                "category_name": rec["category_name"],
                "bbox_xywh": rec["bbox_xywh"],
                "area": rec["area"],
                "size_bucket": rec["size_bucket"],
                "selected_mask_index": SELECTED_MASK_INDEX,
                "predicted_iou_slot0": float(mask_iou_preds[0, SELECTED_MASK_INDEX].detach().cpu().item())
                                        if mask_iou_preds is not None else None,
                "iou": iou_val,
            }
            per_records.append(rec_out)
            if done % 50 == 0 or done == total:
                log(f"  progress {done}/{total}  valid={len(valid_ious)} failed={len(failed)}")

    # embedding reuse accounting: every instance beyond the first per image reused the embedding
    total_instances_evaluated = len(per_records) + len([f for f in failed if f["status"] == "INFERENCE_FAILED"])
    instances_seen = sum(len(v) for v in by_image.values())
    embedding_reuse = max(0, instances_seen - embedding_set_calls)

    wall_time_s = time.perf_counter() - t_start
    eval_per_sec = (len(per_records) / wall_time_s) if wall_time_s > 0 else None

    # write per-instance jsonl
    with open(per_instance_path, "w") as f:
        for r in per_records:
            f.write(json.dumps(r) + "\n")
    with open(failed_path, "w") as f:
        json.dump(failed, f, indent=2)

    agg = aggregate(valid_ious)
    # size-bucket mIoU
    bucket_ious = {"small": [], "medium": [], "large": []}
    for r in per_records:
        bucket_ious[r["size_bucket"]].append(r["iou"])
    bucket_miou = {}
    for b, vals in bucket_ious.items():
        bucket_miou[b] = float(np.mean(vals)) if vals else None
        bucket_miou[b + "_count"] = len(vals)

    summary = {
        "subset": subset_name,
        "mask_selection": "slot 0 (fixed, not IoU argmax)",
        "total_instances_in_subset": total,
        "skipped": skipped,
        "valid": agg["valid"],
        "failed": len(failed),
        "mean_iou": agg["mean_iou"],
        "median_iou": agg["median_iou"],
        "p10": agg["p10"],
        "p50": agg["p50"],
        "p90": agg["p90"],
        "iou_ge_050_ratio": agg["iou_ge_050_ratio"],
        "iou_ge_075_ratio": agg["iou_ge_075_ratio"],
        "mIoU_small": bucket_miou.get("small"),
        "mIoU_medium": bucket_miou.get("medium"),
        "mIoU_large": bucket_miou.get("large"),
        "small_count": bucket_miou.get("small_count", 0),
        "medium_count": bucket_miou.get("medium_count", 0),
        "large_count": bucket_miou.get("large_count", 0),
        "unique_images": unique_images,
        "embedding_set_calls": embedding_set_calls,
        "embedding_reuse_count": embedding_reuse,
        "wall_time_s": wall_time_s,
        "instances_per_sec_eval": eval_per_sec,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"=== {subset_name} DONE === valid={summary['valid']} failed={summary['failed']} "
        f"mIoU={summary['mean_iou']} wall={wall_time_s:.1f}s")
    runlog.close()
    return summary


def _materialize(inst, coco):
    """Given a manifest instance dict (or already-materialized), ensure bbox_xyxy /
    image_path / category_name / size_bucket / _ann are present.

    Manifest instances already carry most fields; we attach the live COCO annotation
    object so annToMask can be called. For FULL (annotation_id only) we load from coco.
    """
    # manifest instances carry bbox_xywh + image_path (frozen schema), not bbox_xyxy.
    # Detect a manifest instance by image_path + annotation_id (without _ann); derive
    # bbox_xyxy from bbox_xywh and keep the manifest image_path. Fixes path loss
    # that caused IMAGE_LOAD_FAILED. No protocol change.
    if "_ann" not in inst and "image_path" in inst and "annotation_id" in inst:
        ann = coco.loadAnns(inst["annotation_id"])
        inst = dict(inst)
        inst["_ann"] = ann[0] if ann else None
        if inst.get("_ann") is None:
            return {"status": "INVALID_GT_MASK", "annotation_id": inst.get("annotation_id"),
                    "image_id": inst.get("image_id")}
        if "bbox_xyxy" not in inst and "bbox_xywh" in inst:
            x, y, w, h = inst["bbox_xywh"]
            inst["bbox_xyxy"] = [float(x), float(y), float(x + w), float(y + h)]
        if "size_bucket" not in inst and "area" in inst:
            inst["size_bucket"] = _size_bucket(float(inst["area"]))
        inst["status"] = "OK"
        inst["image_path"] = os.path.abspath(inst["image_path"]) if not os.path.isabs(inst["image_path"]) else inst["image_path"]
        return inst
    if "_ann" in inst:
        return inst
    # minimal (annotation_id only): build from coco
    ann = coco.loadAnns(inst["annotation_id"])
    if not ann:
        return {"status": "INVALID_GT_MASK", "annotation_id": inst.get("annotation_id")}
    ann = ann[0]
    rec = _ann_to_instance(ann, coco, coco.dataset.get("images_root", ""))
    rec["_ann"] = ann
    return rec


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="NanoSAM COCO quality evaluator")
    parser.add_argument("--coco-root", default="data/coco/val2017")
    parser.add_argument("--annotations", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--subset", choices=["smoke", "quick", "standard", "full"], default="standard")
    parser.add_argument("--manifest", default=None, help="path to subset manifest JSON (ignored for full)")
    parser.add_argument("--image-encoder-engine", required=False)
    parser.add_argument("--mask-decoder-engine", required=False)
    parser.add_argument("--output-dir", default="results/nanosam_coco_eval")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--runtime", choices=["baseline", "optimized", "cuda"], default="baseline", help="Stage7A: baseline (upstream) or optimized wrapper")
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

    if args.runtime not in ("baseline", "optimized", "cuda"):
        print("RUNTIME_INVALID")
        sys.exit(7)
    if args.max_instances is not None:
        instances = instances[: args.max_instances]

    print(f"subset={args.subset} instances={len(instances)}")

    # build predictor once (model loaded once)
    import warnings
    warnings.filterwarnings("ignore")
    print(f"loading engines: enc={args.image_encoder_engine} dec={args.mask_decoder_engine} runtime={args.runtime}", flush=True)
    t0 = time.perf_counter()
    if args.runtime == "optimized":
        import sys as _sys
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from edge_perception.runtime.nanosam_optimized import OptimizedPredictor
        predictor = OptimizedPredictor(args.image_encoder_engine, args.mask_decoder_engine)
    elif args.runtime == "cuda":
        import sys as _sys
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from edge_perception.runtime.cuda_preprocess import CudaPreprocessPredictor
        predictor = CudaPreprocessPredictor(args.image_encoder_engine, args.mask_decoder_engine)
    else:
        from nanosam.utils.predictor import Predictor
        predictor = Predictor(args.image_encoder_engine, args.mask_decoder_engine)
    print(f"model_load_ms={(time.perf_counter() - t0) * 1000.0:.3f}", flush=True)

    summary = run_evaluation(predictor, coco, instances, args.output_dir, args.subset)

    # exit non-zero if every instance failed (so caller can stop the chain)
    if summary["valid"] == 0 and summary["failed"] > 0:
        print("ALL_FAILED: no valid instances; aborting chain.")
        sys.exit(6)
    sys.exit(0)


if __name__ == "__main__":
    main()
