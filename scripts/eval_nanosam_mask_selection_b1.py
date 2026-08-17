#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/nanosam"))
from nanosam.utils.predictor import Predictor

EQUAL_TOLERANCE = 1e-4
POLICIES = {
    "fixed_slot_0": "fixed_iou",
    "predicted_iou_argmax": "predicted_argmax_iou",
    "oracle_best_of_4": "oracle_iou",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = int(np.count_nonzero(pred & gt))
    union = int(np.count_nonzero(pred | gt))
    return intersection / union if union else 0.0


def aggregate(values: list[float], failed: int = 0) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mIoU": float(array.mean()) if array.size else None,
        "median_iou": float(np.median(array)) if array.size else None,
        "p10": float(np.percentile(array, 10)) if array.size else None,
        "p50": float(np.percentile(array, 50)) if array.size else None,
        "p90": float(np.percentile(array, 90)) if array.size else None,
        "iou_ge_0_50": float(np.mean(array >= 0.50)) if array.size else None,
        "iou_ge_0_75": float(np.mean(array >= 0.75)) if array.size else None,
        "valid": int(array.size),
        "failed": int(failed),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def policy_metrics(records: list[dict], field: str, failed: int) -> dict:
    values = [float(record[field]) for record in records]
    result = aggregate(values, failed)
    for bucket in ("small", "medium", "large"):
        bucket_values = [float(record[field]) for record in records if record["size_bucket"] == bucket]
        result[f"{bucket}_mIoU"] = float(np.mean(bucket_values)) if bucket_values else None
        result[f"{bucket}_count"] = len(bucket_values)
    return result


def update_extreme(pool: list[tuple[float, int]], key: float, annotation_id: int, largest: bool) -> None:
    pool.append((key, annotation_id))
    pool.sort(key=lambda item: (item[0], item[1]), reverse=largest)
    del pool[10:]


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int = 120) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    color_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    color_image[mask] = (*color, alpha)
    return Image.alpha_composite(base, Image.fromarray(color_image, mode="RGBA")).convert("RGB")


def label_panel(panel: Image.Image, title: str) -> Image.Image:
    bar_height = 52
    canvas = Image.new("RGB", (panel.width, panel.height + bar_height), "white")
    canvas.paste(panel, (0, bar_height))
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((6, 5), title, fill="black", font=ImageFont.load_default(), spacing=2)
    return canvas


def visualize_case(group: str, rank: int, record: dict, snapshot: dict, output_dir: Path) -> str:
    image = snapshot["image"]
    gt = snapshot["gt"]
    masks = snapshot["masks"]
    indices = [0, int(record["predicted_argmax_index"]), int(record["oracle_index"])]
    panels = [label_panel(image.copy(), f"original\n{record['category_name']} ann={record['annotation_id']}")]
    panels.append(label_panel(overlay(image, gt, (0, 220, 0)), "GT mask"))
    labels = ["fixed", "predicted argmax", "oracle"]
    colors = [(220, 40, 40), (40, 100, 230), (240, 180, 20)]
    for label, color, index in zip(labels, colors, indices):
        score = record[f"predicted_iou_{index}"]
        actual = record[f"gt_iou_mask_{index}"]
        panels.append(label_panel(overlay(image, masks[index], color), f"{label}: slot {index}\npred={score:.4f} actual={actual:.4f}"))
    max_width = 300
    resized = []
    for panel in panels:
        if panel.width > max_width:
            height = int(panel.height * max_width / panel.width)
            panel = panel.resize((max_width, height), Image.Resampling.BILINEAR)
        resized.append(panel)
    width = sum(panel.width for panel in resized)
    height = max(panel.height for panel in resized)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in resized:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{group}_{rank:02d}_ann_{record['annotation_id']}.jpg"
    canvas.save(path, quality=90)
    return str(path.relative_to(ROOT))


def benchmark_selection(hi_res_mask: torch.Tensor, predicted_iou: torch.Tensor) -> dict:
    warmup = 100
    iterations = 1000

    def fixed() -> torch.Tensor:
        return hi_res_mask[0, 0] > 0

    def argmax() -> torch.Tensor:
        index = torch.argmax(predicted_iou[0])
        return hi_res_mask[0, index] > 0

    def measure(function) -> np.ndarray:
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        samples = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            started = time.perf_counter_ns()
            function()
            torch.cuda.synchronize()
            samples[index] = (time.perf_counter_ns() - started) / 1000.0
        return samples

    fixed_times = measure(fixed)
    argmax_times = measure(argmax)

    def summarize(values: np.ndarray) -> dict:
        return {
            "mean_us": float(values.mean()),
            "p50_us": float(np.percentile(values, 50)),
            "p95_us": float(np.percentile(values, 95)),
            "p99_us": float(np.percentile(values, 99)),
        }

    fixed_summary = summarize(fixed_times)
    argmax_summary = summarize(argmax_times)
    delta = argmax_times - fixed_times
    return {
        "scope": "GPU mask selection and threshold extraction only; same decoder outputs and image embedding",
        "warmup": warmup,
        "iterations": iterations,
        "synchronization": "torch.cuda.synchronize after each measured operation",
        "fixed_slot_0": fixed_summary,
        "predicted_iou_argmax": argmax_summary,
        "paired_delta_argmax_minus_fixed": summarize(delta),
        "conclusion": "NEGLIGIBLE_WITHIN_MEASUREMENT_RESOLUTION" if abs(float(np.median(delta))) < 10.0 else "MEASURABLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/eval_manifests/standard_manifest.json")
    parser.add_argument("--annotations", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--image-encoder-engine", default="third_party/nanosam/data/resnet18_image_encoder.engine")
    parser.add_argument("--mask-decoder-engine", default="third_party/nanosam/data/mobile_sam_mask_decoder.engine")
    parser.add_argument("--output-dir", default="results/algorithm_b1")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    instances = manifest["instances"]
    if len(instances) != 500 or len({item["image_id"] for item in instances}) != 455:
        raise RuntimeError("STANDARD manifest contract changed")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != "19a00a836c2f1d1f236ea34175d12c2afc670b9cb5136b30e6b39bf7ca78f015":
        raise RuntimeError(f"STANDARD manifest SHA256 changed: {manifest_sha}")

    coco = COCO(str(ROOT / args.annotations))
    predictor = Predictor(str(ROOT / args.image_encoder_engine), str(ROOT / args.mask_decoder_engine))
    by_image: dict[int, list[dict]] = defaultdict(list)
    image_order = []
    for instance in instances:
        image_id = int(instance["image_id"])
        if image_id not in by_image:
            image_order.append(image_id)
        by_image[image_id].append(instance)

    records: list[dict] = []
    failures: list[dict] = []
    pools = {"fixed_worst": [], "argmax_improved": [], "argmax_degraded": [], "oracle_headroom": []}
    snapshots: dict[int, dict] = {}
    last_hi_res = None
    last_scores = None
    started = time.perf_counter()

    for image_number, image_id in enumerate(image_order, 1):
        group = by_image[image_id]
        image_path = ROOT / group[0]["image_path"]
        image = Image.open(image_path).convert("RGB")
        try:
            predictor.set_image(image)
            torch.cuda.synchronize()
        except Exception as error:
            failures.extend({"annotation_id": item["annotation_id"], "image_id": image_id, "stage": "set_image", "error": repr(error)} for item in group)
            continue

        for item in group:
            annotation_id = int(item["annotation_id"])
            try:
                annotation = coco.loadAnns([annotation_id])[0]
                gt = coco.annToMask(annotation) > 0
                x, y, width, height = [float(value) for value in item["bbox_xywh"]]
                points = np.array([[x, y], [x + width, y + height]], dtype=np.float32)
                labels = np.array([2, 3], dtype=np.int64)
                hi_res, predicted_iou, _ = predictor.predict(points=points, point_labels=labels)
                torch.cuda.synchronize()
                if tuple(hi_res.shape[:2]) != (1, 4) or tuple(predicted_iou.shape) != (1, 4):
                    raise RuntimeError(f"decoder contract mismatch masks={tuple(hi_res.shape)} scores={tuple(predicted_iou.shape)}")
                if not bool(torch.isfinite(hi_res).all()) or not bool(torch.isfinite(predicted_iou).all()):
                    raise RuntimeError("non-finite decoder output")
                masks = (hi_res[0] > 0).detach().cpu().numpy().astype(bool)
                scores = predicted_iou[0].detach().float().cpu().numpy()
                actual = np.asarray([mask_iou(mask, gt) for mask in masks], dtype=np.float64)
                predicted_index = int(np.argmax(scores))
                oracle_index = int(np.argmax(actual))
                record = {
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(item["category_id"]),
                    "category_name": item["category_name"],
                    "coco_area": float(item["area"]),
                    "size_bucket": item["size_bucket"],
                    **{f"predicted_iou_{index}": float(scores[index]) for index in range(4)},
                    **{f"gt_iou_mask_{index}": float(actual[index]) for index in range(4)},
                    "fixed_selected_index": 0,
                    "fixed_iou": float(actual[0]),
                    "predicted_argmax_index": predicted_index,
                    "predicted_argmax_iou": float(actual[predicted_index]),
                    "oracle_index": oracle_index,
                    "oracle_iou": float(actual[oracle_index]),
                    "delta_iou": float(actual[predicted_index] - actual[0]),
                    "oracle_headroom": float(actual[oracle_index] - actual[0]),
                }
                records.append(record)
                update_extreme(pools["fixed_worst"], record["fixed_iou"], annotation_id, False)
                update_extreme(pools["argmax_improved"], record["delta_iou"], annotation_id, True)
                update_extreme(pools["argmax_degraded"], record["delta_iou"], annotation_id, False)
                update_extreme(pools["oracle_headroom"], record["oracle_headroom"], annotation_id, True)
                active = {candidate_id for pool in pools.values() for _, candidate_id in pool}
                if annotation_id in active:
                    snapshots[annotation_id] = {"image": image.copy(), "gt": gt.copy(), "masks": masks.copy()}
                for stale in set(snapshots) - active:
                    del snapshots[stale]
                last_hi_res, last_scores = hi_res, predicted_iou
            except Exception as error:
                failures.append({"annotation_id": annotation_id, "image_id": image_id, "stage": "instance", "error": repr(error)})
        if image_number % 25 == 0 or image_number == len(image_order):
            print(f"progress images={image_number}/{len(image_order)} valid={len(records)} failed={len(failures)}", flush=True)

    if last_hi_res is None or last_scores is None:
        raise RuntimeError("No successful decoder output available")
    runtime_overhead = benchmark_selection(last_hi_res, last_scores)

    fieldnames = [
        "annotation_id", "image_id", "category_id", "category_name", "coco_area", "size_bucket",
        *[f"predicted_iou_{index}" for index in range(4)],
        *[f"gt_iou_mask_{index}" for index in range(4)],
        "fixed_selected_index", "fixed_iou", "predicted_argmax_index", "predicted_argmax_iou",
        "oracle_index", "oracle_iou", "delta_iou", "oracle_headroom",
    ]
    with (output_dir / "per_instance_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    policy_summary = {
        "manifest": {"path": args.manifest, "sha256": manifest_sha, "instances": len(instances), "unique_images": len(image_order)},
        "prompt": "COCO GT bounding box, points=[top-left,bottom-right], labels=[2,3]",
        "mask_threshold": "decoder logits > 0",
        "candidate_count": 4,
        "policies": {name: policy_metrics(records, field, len(failures)) for name, field in POLICIES.items()},
    }
    (output_dir / "policy_summary.json").write_text(json.dumps(policy_summary, indent=2) + "\n")

    deltas = np.asarray([record["delta_iou"] for record in records], dtype=np.float64)
    headroom = np.asarray([record["oracle_headroom"] for record in records], dtype=np.float64)
    predicted_distribution = Counter(record["predicted_argmax_index"] for record in records)
    oracle_distribution = Counter(record["oracle_index"] for record in records)
    paired = {
        "equal_tolerance": EQUAL_TOLERANCE,
        "predicted_argmax_minus_fixed": {
            "mean_delta": float(deltas.mean()), "median_delta": float(np.median(deltas)),
            "mean_absolute_delta": float(np.abs(deltas).mean()),
            "improved_count": int(np.sum(deltas >= EQUAL_TOLERANCE)),
            "degraded_count": int(np.sum(deltas <= -EQUAL_TOLERANCE)),
            "approximately_equal_count": int(np.sum(np.abs(deltas) < EQUAL_TOLERANCE)),
        },
        "oracle_minus_fixed": {
            "mean_headroom": float(headroom.mean()), "median_headroom": float(np.median(headroom)),
            "mean_absolute_headroom": float(np.abs(headroom).mean()),
            "improved_count": int(np.sum(headroom >= EQUAL_TOLERANCE)),
            "approximately_equal_count": int(np.sum(headroom < EQUAL_TOLERANCE)),
        },
        "predicted_argmax_slot_distribution": {str(index): {"count": predicted_distribution[index], "ratio": predicted_distribution[index] / len(records)} for index in range(4)},
        "oracle_slot_distribution": {str(index): {"count": oracle_distribution[index], "ratio": oracle_distribution[index] / len(records)} for index in range(4)},
    }
    (output_dir / "paired_comparison.json").write_text(json.dumps(paired, indent=2) + "\n")

    predicted_all = np.asarray([record[f"predicted_iou_{index}"] for record in records for index in range(4)], dtype=np.float64)
    actual_all = np.asarray([record[f"gt_iou_mask_{index}"] for record in records for index in range(4)], dtype=np.float64)
    pearson = float(np.corrcoef(predicted_all, actual_all)[0, 1])
    spearman = float(np.corrcoef(average_ranks(predicted_all), average_ranks(actual_all))[0, 1])
    calibration = {
        "candidate_masks": int(predicted_all.size),
        "pearson_correlation": pearson,
        "spearman_rank_correlation": spearman,
        "mean_absolute_error": float(np.mean(np.abs(predicted_all - actual_all))),
        "root_mean_squared_error": float(np.sqrt(np.mean((predicted_all - actual_all) ** 2))),
        "argmax_top1_selection_accuracy": float(np.mean([record["predicted_argmax_index"] == record["oracle_index"] for record in records])),
        "argmax_top1_correct_count": int(sum(record["predicted_argmax_index"] == record["oracle_index"] for record in records)),
        "instances": len(records),
        "note": "Predicted IoU is treated as a quality regression output, not a calibrated probability; classification ECE is not applied.",
    }
    (output_dir / "iou_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")

    size_analysis = {}
    for bucket in ("small", "medium", "large"):
        subset = [record for record in records if record["size_bucket"] == bucket]
        fixed = float(np.mean([record["fixed_iou"] for record in subset]))
        predicted = float(np.mean([record["predicted_argmax_iou"] for record in subset]))
        oracle = float(np.mean([record["oracle_iou"] for record in subset]))
        size_analysis[bucket] = {"count": len(subset), "fixed_mIoU": fixed, "predicted_argmax_mIoU": predicted, "oracle_mIoU": oracle, "predicted_delta": predicted - fixed, "oracle_headroom": oracle - fixed}
    (output_dir / "size_bucket_analysis.json").write_text(json.dumps(size_analysis, indent=2) + "\n")

    category_groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for record in records:
        category_groups[(record["category_id"], record["category_name"])].append(record)
    categories = []
    for (category_id, category_name), group in category_groups.items():
        if len(group) < 5:
            continue
        categories.append({
            "category_id": category_id, "category_name": category_name, "count": len(group),
            "fixed_mIoU": float(np.mean([record["fixed_iou"] for record in group])),
            "predicted_argmax_mIoU": float(np.mean([record["predicted_argmax_iou"] for record in group])),
            "oracle_mIoU": float(np.mean([record["oracle_iou"] for record in group])),
        })
    category_analysis = {
        "minimum_count": 5,
        "eligible_category_count": len(categories),
        "categories": sorted(categories, key=lambda item: item["fixed_mIoU"], reverse=True),
        "best_5_by_fixed_mIoU": sorted(categories, key=lambda item: item["fixed_mIoU"], reverse=True)[:5],
        "worst_5_by_fixed_mIoU": sorted(categories, key=lambda item: item["fixed_mIoU"])[:5],
        "caution": "Category metrics are descriptive for this fixed 500-instance subset and are not population-level conclusions.",
    }
    (output_dir / "category_analysis.json").write_text(json.dumps(category_analysis, indent=2) + "\n")

    record_by_id = {record["annotation_id"]: record for record in records}
    failure_groups = {}
    visualizations = []
    for group_name, pool in pools.items():
        group_records = [record_by_id[annotation_id] for _, annotation_id in pool]
        failure_groups[group_name] = group_records
        for rank, record in enumerate(group_records[:5], 1):
            snapshot = snapshots.get(record["annotation_id"])
            if snapshot is not None:
                visualizations.append(visualize_case(group_name, rank, record, snapshot, output_dir / "visualizations"))
    failure_cases = {"groups": failure_groups, "visualizations": visualizations, "visualization_limit_per_group": 5}
    (output_dir / "failure_cases.json").write_text(json.dumps(failure_cases, indent=2) + "\n")
    (output_dir / "runtime_overhead.json").write_text(json.dumps(runtime_overhead, indent=2) + "\n")
    (output_dir / "failed_instances.json").write_text(json.dumps(failures, indent=2) + "\n")

    run_metadata = {
        "wall_time_seconds": time.perf_counter() - started,
        "valid": len(records), "failed": len(failures), "unique_images": len(image_order),
        "image_encoder_engine": {"path": args.image_encoder_engine, "sha256": sha256_file(ROOT / args.image_encoder_engine)},
        "mask_decoder_engine": {"path": args.mask_decoder_engine, "sha256": sha256_file(ROOT / args.mask_decoder_engine)},
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(json.dumps({"policy_summary": policy_summary, "paired": paired, "calibration": calibration, "size": size_analysis, "runtime": runtime_overhead, "run": run_metadata}, indent=2))


if __name__ == "__main__":
    main()
