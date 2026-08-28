#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage G - 评估 training-free bbox localization refinement methods。

对比三种方法：
  Baseline: top-1 highest-score patch (E4 current)
  Method A: Score-Weighted Box Fusion (softmax weight, temperature=0.1)
  Method B: Best Spatial Consensus Candidate (score * mean_pairwise_iou^2)

仅在 evaluation_split.json (33 instances, 冻结) 上运行最终评估。
所有超参（top_k=5, temperature=0.1, gamma=2.0）已在 frozen_methods.json 中冻结。
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import PIL.Image
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = "/home/nvidia/jetson-deploy-lab"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party/nanoowl"))

from nanoowl.owl_predictor import OwlPredictor
from nanosam.utils.predictor import Predictor as NanosamPredictor

DEFAULT_OWL_ENGINE = os.path.join(ROOT, "third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
DEFAULT_NANOSAM_ENC = os.path.join(ROOT, "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine")
DEFAULT_NANOSAM_DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
DEFAULT_MODEL = "google/owlvit-base-patch32"

TOP_K = 5
TEMPERATURE = 0.1
GAMMA = 2.0


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


def extract_top_k_candidates(owl, image_pil, text_query, threshold, top_k):
    """提取 top-K 候选 patches，返回 list of dict(rank, patch_id, score, bbox_xyxy)。"""
    text_encodings = owl.encode_text([text_query])
    image_tensor = owl.image_preprocessor.preprocess_pil_image(image_pil)
    W, H = image_pil.size
    rois = torch.tensor([[0, 0, W, H]], dtype=image_tensor.dtype, device=image_tensor.device)
    image_output = owl.encode_rois(image_tensor, rois, pad_square=False)
    
    image_class_embeds = image_output.image_class_embeds
    image_class_embeds_norm = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
    query_embeds = text_encodings.text_embeds
    query_embeds_norm = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)
    logits = torch.einsum("bpd,qd->bpq", image_class_embeds_norm, query_embeds_norm)
    logits = (logits + image_output.logit_shift) * image_output.logit_scale
    scores_sigmoid = torch.sigmoid(logits)
    
    patch_scores = scores_sigmoid[0, :, 0].detach().cpu().numpy()
    pred_boxes = image_output.pred_boxes[0].detach().cpu().numpy()
    
    sorted_indices = np.argsort(patch_scores)[::-1]
    candidates = []
    for rank, patch_id in enumerate(sorted_indices[:top_k]):
        score = float(patch_scores[patch_id])
        if score < threshold:
            break
        bbox = pred_boxes[patch_id].tolist()
        candidates.append({"rank": rank, "patch_id": int(patch_id), "score": score, "bbox_xyxy": bbox})
    return candidates


def method_baseline(candidates):
    """Baseline: top-1 highest-score candidate。"""
    if not candidates:
        return None
    return candidates[0]["bbox_xyxy"]


def method_a_score_weighted_fusion(candidates, temperature=TEMPERATURE):
    """Method A: softmax-weighted box fusion。"""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]["bbox_xyxy"]
    
    scores = np.array([c["score"] for c in candidates])
    weights = np.exp(scores / temperature)
    weights /= weights.sum()
    
    boxes = np.array([c["bbox_xyxy"] for c in candidates])
    fused = (weights[:, None] * boxes).sum(axis=0)
    return fused.tolist()


def method_b_spatial_consensus(candidates, gamma=GAMMA):
    """Method B: 选择 consensus_score 最高的候选 (score * mean_pairwise_iou^gamma)。"""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]["bbox_xyxy"]
    
    N = len(candidates)
    boxes = [c["bbox_xyxy"] for c in candidates]
    consensus_scores = []
    for i in range(N):
        ious = [box_iou_xyxy(boxes[i], boxes[j]) for j in range(N) if j != i]
        mean_iou = np.mean(ious) if ious else 0.0
        consensus = candidates[i]["score"] * (mean_iou ** gamma)
        consensus_scores.append(consensus)
    
    best_idx = np.argmax(consensus_scores)
    return candidates[best_idx]["bbox_xyxy"]


def nanosam_mask_from_box(nsam, box_xyxy):
    """NanoSAM box-prompt segmentation (slot 0 fixed)。"""
    pts = np.array([[box_xyxy[0], box_xyxy[1]], [box_xyxy[2], box_xyxy[3]]], dtype=np.float32)
    labs = np.array([2, 3])
    hi, iou, low = nsam.predict(pts, labs)
    mask = (hi[0, 0] > 0).detach().cpu().numpy()
    return mask


def main():
    ap = argparse.ArgumentParser(description="Stage G - evaluate bbox refinement methods")
    ap.add_argument("--eval-split", default=os.path.join(ROOT, "results/stage_g_bbox_refinement/evaluation_split.json"))
    ap.add_argument("--eval-manifest", default=os.path.join(ROOT, "results/stage_e4_joint_pipeline/eval_manifest.json"))
    ap.add_argument("--coco-annotations", default=os.path.join(ROOT, "data/coco/annotations/instances_val2017.json"))
    ap.add_argument("--coco-root", default=os.path.join(ROOT, "data/coco/val2017"))
    ap.add_argument("--score-threshold", type=float, default=0.10)
    ap.add_argument("--nanoowl-engine", default=DEFAULT_OWL_ENGINE)
    ap.add_argument("--nanosam-image-encoder-engine", default=DEFAULT_NANOSAM_ENC)
    ap.add_argument("--nanosam-mask-decoder-engine", default=DEFAULT_NANOSAM_DEC)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "results/stage_g_bbox_refinement"))
    args = ap.parse_args()
    
    with open(args.eval_split) as f:
        eval_split = json.load(f)
    eval_ids = {(r["image_id"], r["annotation_id"]) for r in eval_split["instance_ids"]}
    print(f"Eval split: {len(eval_ids)} instances", flush=True)
    
    with open(args.eval_manifest) as f:
        manifest = json.load(f)
    instances = [inst for inst in manifest["instances"] if (inst["image_id"], inst["annotation_id"]) in eval_ids]
    print(f"Matched eval instances: {len(instances)}", flush=True)
    
    try:
        from pycocotools.coco import COCO
        coco = COCO(args.coco_annotations)
    except Exception:
        print("COCO API not available")
        sys.exit(1)
    
    print("Loading NanoOWL...", flush=True)
    owl = OwlPredictor(args.model, image_encoder_engine=args.nanoowl_engine)
    print("Loading NanoSAM (FP16 production)...", flush=True)
    nsam = NanosamPredictor(args.nanosam_image_encoder_engine, args.nanosam_mask_decoder_engine)
    
    # group by image_id for NanoSAM embedding reuse
    by_image = {}
    order = []
    for inst in instances:
        iid = inst["image_id"]
        if iid not in by_image:
            by_image[iid] = []
            order.append(iid)
        by_image[iid].append(inst)
    
    results = []
    latencies_baseline = []
    latencies_method_a = []
    latencies_method_b = []
    
    for img_idx, iid in enumerate(order, 1):
        insts = by_image[iid]
        img_path = os.path.join(args.coco_root, f"{iid:012d}.jpg")
        if not os.path.isfile(img_path):
            print(f"[{img_idx}/{len(order)}] SKIP img_id={iid} (file not found)", flush=True)
            continue
        
        try:
            image = PIL.Image.open(img_path).convert("RGB")
        except Exception:
            print(f"[{img_idx}/{len(order)}] SKIP img_id={iid} (load failed)", flush=True)
            continue
        
        W, H = image.size
        try:
            nsam.set_image(image)
            torch.cuda.synchronize()
        except Exception:
            print(f"[{img_idx}/{len(order)}] SKIP img_id={iid} (NanoSAM set_image failed)", flush=True)
            continue
        
        for inst in insts:
            ann_id = inst["annotation_id"]
            cat_name = inst["category_name"]
            size_bucket = inst["size_bucket"]
            scene_type = inst["scene_type"]
            gt_bbox = inst["bbox_xyxy"]
            
            try:
                gt_ann = coco.loadAnns([ann_id])[0]
                gt_mask = coco.annToMask(gt_ann) > 0
            except Exception:
                print(f"  SKIP ann_id={ann_id} (GT mask failed)", flush=True)
                continue
            
            # GT-box NanoSAM mask (upper bound reference)
            try:
                gt_box_clipped = clip_box(gt_bbox, W, H)
                pred_gtbox = nanosam_mask_from_box(nsam, gt_box_clipped)
                if pred_gtbox.shape != gt_mask.shape:
                    from scipy.ndimage import zoom
                    scale = (gt_mask.shape[0] / pred_gtbox.shape[0], gt_mask.shape[1] / pred_gtbox.shape[1])
                    pred_gtbox = zoom(pred_gtbox.astype(float), scale, order=0) > 0.5
                mask_iou_gtbox = mask_iou(pred_gtbox, gt_mask)
            except Exception:
                print(f"  SKIP ann_id={ann_id} (GT-box NanoSAM failed)", flush=True)
                continue
            
            # extract candidates
            try:
                candidates = extract_top_k_candidates(owl, image, cat_name, args.score_threshold, TOP_K)
                torch.cuda.synchronize()
            except Exception:
                print(f"  SKIP ann_id={ann_id} (candidate extraction failed)", flush=True)
                continue
            
            if not candidates:
                # DETECTION_FAILURE for all methods
                result = {
                    "image_id": iid,
                    "annotation_id": ann_id,
                    "category": cat_name,
                    "size_bucket": size_bucket,
                    "scene_type": scene_type,
                    "gt_bbox_xyxy": gt_bbox,
                    "num_candidates": 0,
                    "baseline_box": None,
                    "baseline_box_iou": None,
                    "baseline_mask_iou": 0.0,
                    "method_a_box": None,
                    "method_a_box_iou": None,
                    "method_a_mask_iou": 0.0,
                    "method_b_box": None,
                    "method_b_box_iou": None,
                    "method_b_mask_iou": 0.0,
                    "mask_iou_gtbox": mask_iou_gtbox,
                    "baseline_latency_ms": 0.0,
                    "method_a_latency_ms": 0.0,
                    "method_b_latency_ms": 0.0,
                }
                results.append(result)
                continue
            
            # Baseline
            t0 = time.perf_counter()
            baseline_box = method_baseline(candidates)
            baseline_box_clipped = clip_box(baseline_box, W, H)
            t_baseline = (time.perf_counter() - t0) * 1000
            latencies_baseline.append(t_baseline)
            
            try:
                pred_baseline = nanosam_mask_from_box(nsam, baseline_box_clipped)
                if pred_baseline.shape != gt_mask.shape:
                    from scipy.ndimage import zoom
                    scale = (gt_mask.shape[0] / pred_baseline.shape[0], gt_mask.shape[1] / pred_baseline.shape[1])
                    pred_baseline = zoom(pred_baseline.astype(float), scale, order=0) > 0.5
                baseline_mask_iou = mask_iou(pred_baseline, gt_mask)
            except Exception:
                baseline_mask_iou = 0.0
            baseline_box_iou = box_iou_xyxy(baseline_box_clipped, gt_bbox)
            
            # Method A
            t0 = time.perf_counter()
            method_a_box = method_a_score_weighted_fusion(candidates, TEMPERATURE)
            method_a_box_clipped = clip_box(method_a_box, W, H)
            t_method_a = (time.perf_counter() - t0) * 1000
            latencies_method_a.append(t_method_a)
            
            try:
                pred_method_a = nanosam_mask_from_box(nsam, method_a_box_clipped)
                if pred_method_a.shape != gt_mask.shape:
                    from scipy.ndimage import zoom
                    scale = (gt_mask.shape[0] / pred_method_a.shape[0], gt_mask.shape[1] / pred_method_a.shape[1])
                    pred_method_a = zoom(pred_method_a.astype(float), scale, order=0) > 0.5
                method_a_mask_iou = mask_iou(pred_method_a, gt_mask)
            except Exception:
                method_a_mask_iou = 0.0
            method_a_box_iou = box_iou_xyxy(method_a_box_clipped, gt_bbox)
            
            # Method B
            t0 = time.perf_counter()
            method_b_box = method_b_spatial_consensus(candidates, GAMMA)
            method_b_box_clipped = clip_box(method_b_box, W, H)
            t_method_b = (time.perf_counter() - t0) * 1000
            latencies_method_b.append(t_method_b)
            
            try:
                pred_method_b = nanosam_mask_from_box(nsam, method_b_box_clipped)
                if pred_method_b.shape != gt_mask.shape:
                    from scipy.ndimage import zoom
                    scale = (gt_mask.shape[0] / pred_method_b.shape[0], gt_mask.shape[1] / pred_method_b.shape[1])
                    pred_method_b = zoom(pred_method_b.astype(float), scale, order=0) > 0.5
                method_b_mask_iou = mask_iou(pred_method_b, gt_mask)
            except Exception:
                method_b_mask_iou = 0.0
            method_b_box_iou = box_iou_xyxy(method_b_box_clipped, gt_bbox)
            
            result = {
                "image_id": iid,
                "annotation_id": ann_id,
                "category": cat_name,
                "size_bucket": size_bucket,
                "scene_type": scene_type,
                "gt_bbox_xyxy": gt_bbox,
                "num_candidates": len(candidates),
                "baseline_box": baseline_box_clipped,
                "baseline_box_iou": baseline_box_iou,
                "baseline_mask_iou": baseline_mask_iou,
                "method_a_box": method_a_box_clipped,
                "method_a_box_iou": method_a_box_iou,
                "method_a_mask_iou": method_a_mask_iou,
                "method_b_box": method_b_box_clipped,
                "method_b_box_iou": method_b_box_iou,
                "method_b_mask_iou": method_b_mask_iou,
                "mask_iou_gtbox": mask_iou_gtbox,
                "baseline_latency_ms": t_baseline,
                "method_a_latency_ms": t_method_a,
                "method_b_latency_ms": t_method_b,
            }
            results.append(result)
        
        if img_idx % 5 == 0 or img_idx == len(order):
            print(f"[{img_idx}/{len(order)}] Processed {len(results)} instances", flush=True)
    
    # Write CSV
    fieldnames = [
        "image_id", "annotation_id", "category", "size_bucket", "scene_type",
        "gt_bbox_xyxy", "num_candidates",
        "baseline_box", "baseline_box_iou", "baseline_mask_iou",
        "method_a_box", "method_a_box_iou", "method_a_mask_iou",
        "method_b_box", "method_b_box_iou", "method_b_mask_iou",
        "mask_iou_gtbox",
        "baseline_latency_ms", "method_a_latency_ms", "method_b_latency_ms",
    ]
    
    for method_name, method_key in [("baseline", "baseline"), ("method_a", "method_a"), ("method_b", "method_b")]:
        output_csv = os.path.join(args.output_dir, f"{method_key}_results.csv")
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote: {output_csv}")
    
    # Latency summary
    latency_summary = {
        "baseline": {"mean": float(np.mean(latencies_baseline)), "p50": float(np.percentile(latencies_baseline, 50)), "p95": float(np.percentile(latencies_baseline, 95)), "p99": float(np.percentile(latencies_baseline, 99))} if latencies_baseline else {},
        "method_a": {"mean": float(np.mean(latencies_method_a)), "p50": float(np.percentile(latencies_method_a, 50)), "p95": float(np.percentile(latencies_method_a, 95)), "p99": float(np.percentile(latencies_method_a, 99))} if latencies_method_a else {},
        "method_b": {"mean": float(np.mean(latencies_method_b)), "p50": float(np.percentile(latencies_method_b, 50)), "p95": float(np.percentile(latencies_method_b, 95)), "p99": float(np.percentile(latencies_method_b, 99))} if latencies_method_b else {},
    }
    with open(os.path.join(args.output_dir, "latency.json"), "w") as f:
        json.dump(latency_summary, f, indent=2)
    print(f"Wrote latency.json")
    
    print(f"\nEvaluation complete. Total instances: {len(results)}")


if __name__ == "__main__":
    main()
