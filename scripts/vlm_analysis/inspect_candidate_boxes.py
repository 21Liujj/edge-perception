#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage G - 提取并分析 NanoOWL candidate-level predictions (top-K patches)。"""
import argparse
import csv
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import PIL.Image
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 固定 ROOT 为 jetson-deploy-lab 根目录
ROOT = "/home/nvidia/jetson-deploy-lab"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party/nanoowl"))

from nanoowl.owl_predictor import OwlPredictor

DEFAULT_OWL_ENGINE = os.path.join(ROOT, "third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
DEFAULT_MODEL = "google/owlvit-base-patch32"


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


def box_center_distance_normalized(a, b, image_diag):
    """两个 bbox 中心点距离，归一化到图像对角线长度。"""
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    dist = np.sqrt((acx - bcx)**2 + (acy - bcy)**2)
    return float(dist / image_diag) if image_diag > 0 else 0.0


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def extract_top_k_candidates(owl, image_pil, text_query, threshold, top_k):
    """提取 top-K 候选 patches。"""
    text_encodings = owl.encode_text([text_query])
    image_tensor = owl.image_preprocessor.preprocess_pil_image(image_pil)
    W, H = image_pil.size
    
    rois = torch.tensor([[0, 0, W, H]], dtype=image_tensor.dtype, device=image_tensor.device)
    image_output = owl.encode_rois(image_tensor, rois, pad_square=False)
    
    # language matching (复制 decode 逻辑，不做 threshold mask)
    image_class_embeds = image_output.image_class_embeds  # [1, 576, 512]
    image_class_embeds_norm = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
    query_embeds = text_encodings.text_embeds  # [1, 512]
    query_embeds_norm = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)
    
    logits = torch.einsum("bpd,qd->bpq", image_class_embeds_norm, query_embeds_norm)  # [1, 576, 1]
    logits = (logits + image_output.logit_shift) * image_output.logit_scale
    scores_sigmoid = torch.sigmoid(logits)  # [1, 576, 1]
    
    patch_scores = scores_sigmoid[0, :, 0].detach().cpu().numpy()  # [576]
    pred_boxes = image_output.pred_boxes[0].detach().cpu().numpy()  # [576, 4]
    
    sorted_indices = np.argsort(patch_scores)[::-1]
    candidates = []
    for rank, patch_id in enumerate(sorted_indices[:top_k]):
        score = float(patch_scores[patch_id])
        if score < threshold:
            break
        bbox = pred_boxes[patch_id].tolist()
        candidates.append({
            "rank": rank,
            "patch_id": int(patch_id),
            "language_score": score,
            "bbox_xyxy": bbox,
        })
    
    return candidates


def analyze_candidates_spatial(candidates, image_diag):
    """候选框空间关系分析。"""
    if len(candidates) < 2:
        return {"pairwise_iou": [], "center_distances": [], "size_ratios": []}
    
    boxes = [c["bbox_xyxy"] for c in candidates]
    N = len(boxes)
    pairs_iou = []
    pairs_dist = []
    pairs_size_ratio = []
    
    for i in range(N):
        for j in range(i + 1, N):
            iou = box_iou_xyxy(boxes[i], boxes[j])
            dist = box_center_distance_normalized(boxes[i], boxes[j], image_diag)
            area_i = box_area(boxes[i])
            area_j = box_area(boxes[j])
            size_ratio = min(area_i, area_j) / max(area_i, area_j) if max(area_i, area_j) > 0 else 0.0
            
            pairs_iou.append(iou)
            pairs_dist.append(dist)
            pairs_size_ratio.append(size_ratio)
    
    return {
        "pairwise_iou": pairs_iou,
        "center_distances": pairs_dist,
        "size_ratios": pairs_size_ratio,
    }


def main():
    ap = argparse.ArgumentParser(description="Stage G - inspect NanoOWL candidate boxes")
    ap.add_argument("--eval-manifest", default=os.path.join(ROOT, "results/stage_e4_joint_pipeline/eval_manifest.json"))
    ap.add_argument("--error-decomposition", default=os.path.join(ROOT, "results/stage_e4_joint_pipeline/error_decomposition.csv"))
    ap.add_argument("--coco-root", default=os.path.join(ROOT, "data/coco/val2017"))
    ap.add_argument("--score-threshold", type=float, default=0.10)
    ap.add_argument("--top-k", type=int, default=5, help="extract top-K candidates")
    ap.add_argument("--nanoowl-engine", default=DEFAULT_OWL_ENGINE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output-csv", default=os.path.join(ROOT, "results/stage_g_bbox_refinement/candidate_analysis.csv"))
    args = ap.parse_args()
    
    with open(args.eval_manifest) as f:
        manifest = json.load(f)
    instances = manifest["instances"]
    
    e4_rows = {}
    with open(args.error_decomposition, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ann_id = int(row["annotation_id"])
            e4_rows[ann_id] = row
    
    print(f"Loading NanoOWL...", flush=True)
    owl = OwlPredictor(args.model, image_encoder_engine=args.nanoowl_engine)
    
    results = []
    for i, inst in enumerate(instances, 1):
        img_id = inst["image_id"]
        ann_id = inst["annotation_id"]
        cat_name = inst["category_name"]
        size_bucket = inst["size_bucket"]
        scene_type = inst["scene_type"]
        gt_bbox = inst["bbox_xyxy"]
        
        e4_row = e4_rows.get(ann_id, {})
        error_bucket = e4_row.get("error_bucket", "UNKNOWN")
        owl_bbox_e4 = e4_row.get("owl_bbox_xyxy", "")
        box_iou_e4 = e4_row.get("box_iou", "")
        
        img_path = os.path.join(args.coco_root, f"{img_id:012d}.jpg")
        if not os.path.isfile(img_path):
            print(f"[{i}/{len(instances)}] SKIP img_id={img_id} (file not found)", flush=True)
            continue
        
        try:
            image = PIL.Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[{i}/{len(instances)}] SKIP img_id={img_id} (load failed)", flush=True)
            continue
        
        W, H = image.size
        image_diag = np.sqrt(W**2 + H**2)
        
        try:
            candidates = extract_top_k_candidates(owl, image, cat_name, args.score_threshold, args.top_k)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[{i}/{len(instances)}] SKIP ann_id={ann_id} (extract failed: {e})", flush=True)
            continue
        
        spatial = analyze_candidates_spatial(candidates, image_diag)
        
        top1_bbox = candidates[0]["bbox_xyxy"] if candidates else None
        top1_score = candidates[0]["language_score"] if candidates else None
        
        results.append({
            "image_id": img_id,
            "annotation_id": ann_id,
            "category": cat_name,
            "size_bucket": size_bucket,
            "scene_type": scene_type,
            "error_bucket": error_bucket,
            "gt_bbox_xyxy": gt_bbox,
            "e4_owl_bbox_xyxy": owl_bbox_e4,
            "e4_box_iou": box_iou_e4,
            "num_candidates": len(candidates),
            "top1_bbox_xyxy": top1_bbox,
            "top1_score": top1_score,
            "pairwise_iou_mean": float(np.mean(spatial["pairwise_iou"])) if spatial["pairwise_iou"] else None,
            "pairwise_iou_max": float(np.max(spatial["pairwise_iou"])) if spatial["pairwise_iou"] else None,
            "center_distance_mean": float(np.mean(spatial["center_distances"])) if spatial["center_distances"] else None,
            "center_distance_min": float(np.min(spatial["center_distances"])) if spatial["center_distances"] else None,
            "size_ratio_mean": float(np.mean(spatial["size_ratios"])) if spatial["size_ratios"] else None,
            "candidates_json": json.dumps(candidates, ensure_ascii=False),
        })
        
        if i % 10 == 0 or i == len(instances):
            print(f"[{i}/{len(instances)}] ann_id={ann_id}, candidates={len(candidates)}", flush=True)
    
    fieldnames = [
        "image_id", "annotation_id", "category", "size_bucket", "scene_type", "error_bucket",
        "gt_bbox_xyxy", "e4_owl_bbox_xyxy", "e4_box_iou",
        "num_candidates", "top1_bbox_xyxy", "top1_score",
        "pairwise_iou_mean", "pairwise_iou_max",
        "center_distance_mean", "center_distance_min",
        "size_ratio_mean",
        "candidates_json",
    ]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nCandidate analysis complete: {args.output_csv}")
    print(f"Total: {len(results)}")


if __name__ == "__main__":
    main()
