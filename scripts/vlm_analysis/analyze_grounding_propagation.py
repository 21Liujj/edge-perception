#!/usr/bin/env python3
"""Stage F - Grounding Error Propagation + Correlation Analysis.

Uses E4's existing per-case data (results/stage_e4_joint_pipeline/
error_decomposition.csv) which already has: gt box, owl box, box_iou,
mask_iou_owlbox, mask_iou_gtbox for 50 fixed COCO instances. No new inference
is run; this is pure quantitative re-analysis of frozen E4 evidence.

Computes:
  - Δmask (degradation) = mask_iou_owlbox - mask_iou_gtbox (per E4 CSV values;
    note E4's own 'degradation' column already = mask_iou_gtbox - mask_iou_owlbox,
    so Δmask here is defined as the NEGATIVE of that, i.e. box-imprecision cost
    is positive when localization hurts mask quality -- reported explicitly to
    avoid sign confusion).
  - Pearson and Spearman correlation: box_iou vs mask_iou_owlbox, box_iou vs
    degradation.
  - Bucketed analysis by box_iou range: [0,0.25) [0.25,0.5) [0.5,0.75) [0.75,1.0]
"""
import csv
import json
import os
from scipy import stats as scipy_stats

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
E4_CSV = os.path.join(ROOT, "results/stage_e4_joint_pipeline/error_decomposition.csv")
OUT_DIR = os.path.join(ROOT, "results/stage_f_vlm_algorithm")


def load_e4_rows():
    rows = []
    with open(E4_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(v):
    if v is None or v == "" or v == "None":
        return None
    return float(v)


def main():
    rows = load_e4_rows()
    print(f"Loaded {len(rows)} rows from E4 error_decomposition.csv", flush=True)

    # Filter to rows with valid box_iou (i.e. detection succeeded)
    valid_rows = []
    for r in rows:
        box_iou = to_float(r.get("box_iou"))
        mask_iou_owlbox = to_float(r.get("mask_iou_owlbox"))
        mask_iou_gtbox = to_float(r.get("mask_iou_gtbox"))
        if box_iou is None or mask_iou_owlbox is None or mask_iou_gtbox is None:
            continue
        degradation_cost = mask_iou_gtbox - mask_iou_owlbox  # positive = box hurt mask (E4 convention)
        valid_rows.append({
            "image_id": r["image_id"],
            "annotation_id": r["annotation_id"],
            "category": r["category"],
            "box_iou": box_iou,
            "mask_iou_owlbox": mask_iou_owlbox,
            "mask_iou_gtbox": mask_iou_gtbox,
            "degradation_cost": degradation_cost,
            "error_bucket": r["error_bucket"],
        })

    n_excluded = len(rows) - len(valid_rows)
    print(f"Valid rows (detection succeeded, box_iou present): {len(valid_rows)}; excluded (detection failure): {n_excluded}", flush=True)

    # Write grounding_propagation.csv
    csv_path = os.path.join(OUT_DIR, "grounding_propagation.csv")
    fieldnames = ["image_id", "annotation_id", "category", "box_iou",
                  "mask_iou_owlbox", "mask_iou_gtbox", "degradation_cost", "error_bucket"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in valid_rows:
            w.writerow(r)
    print(f"Wrote {csv_path}", flush=True)

    box_ious = [r["box_iou"] for r in valid_rows]
    mask_ious = [r["mask_iou_owlbox"] for r in valid_rows]
    degradations = [r["degradation_cost"] for r in valid_rows]

    # Pearson and Spearman: box_iou vs mask_iou_owlbox
    pearson_iou = scipy_stats.pearsonr(box_ious, mask_ious)
    spearman_iou = scipy_stats.spearmanr(box_ious, mask_ious)

    # Pearson and Spearman: box_iou vs degradation
    pearson_degr = scipy_stats.pearsonr(box_ious, degradations)
    spearman_degr = scipy_stats.spearmanr(box_ious, degradations)

    # Bucket analysis
    buckets = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001)]  # 1.0001 to include exactly 1.0
    bucket_labels = ["[0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1.0]"]
    bucket_stats = {}
    for (lo, hi), label in zip(buckets, bucket_labels):
        sub = [r for r in valid_rows if lo <= r["box_iou"] < hi]
        if not sub:
            bucket_stats[label] = {"count": 0, "mean_mask_iou": None, "mean_degradation": None}
            continue
        mean_mask = sum(r["mask_iou_owlbox"] for r in sub) / len(sub)
        mean_degr = sum(r["degradation_cost"] for r in sub) / len(sub)
        bucket_stats[label] = {
            "count": len(sub),
            "mean_mask_iou": round(mean_mask, 4),
            "mean_degradation": round(mean_degr, 4),
        }

    correlation_result = {
        "n_valid_cases": len(valid_rows),
        "n_excluded_detection_failure": n_excluded,
        "box_iou_vs_mask_iou": {
            "pearson_r": round(float(pearson_iou[0]), 4),
            "pearson_p_value": round(float(pearson_iou[1]), 6),
            "spearman_rho": round(float(spearman_iou[0]), 4),
            "spearman_p_value": round(float(spearman_iou[1]), 6),
        },
        "box_iou_vs_degradation": {
            "pearson_r": round(float(pearson_degr[0]), 4),
            "pearson_p_value": round(float(pearson_degr[1]), 6),
            "spearman_rho": round(float(spearman_degr[0]), 4),
            "spearman_p_value": round(float(spearman_degr[1]), 6),
        },
        "bucketed_by_box_iou": bucket_stats,
        "caveat": "Correlation does NOT imply causation. n=42 (50 minus 8 detection failures) is a small-sample controlled study, not a statistically powered claim about the general NanoOWL/NanoSAM population. Sign convention: degradation_cost = mask_iou_gtbox - mask_iou_owlbox; POSITIVE means the imperfect NanoOWL box hurt mask quality relative to the GT-box ceiling. A negative pearson_r for box_iou_vs_degradation is EXPECTED and consistent with 'higher box_iou -> lower degradation cost'.",
    }

    json_path = os.path.join(OUT_DIR, "grounding_correlation.json")
    with open(json_path, "w") as f:
        json.dump(correlation_result, f, indent=2)
    print(f"Wrote {json_path}", flush=True)
    print(json.dumps(correlation_result, indent=2), flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
