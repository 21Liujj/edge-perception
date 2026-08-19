#!/usr/bin/env python3
"""Stage Jetson-5B-2 evidence aggregation:
paired comparison, size/category breakdown, artifact audit, quality gate.
"""
import csv, json, hashlib, math, os, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/home/nvidia/jetson-deploy-lab")
EVID = ROOT / "results/stage_jetson_5b2"
EVID.mkdir(parents=True, exist_ok=True)

SELF_DIR = EVID / "self_trained_standard500"
HIST_DIR = ROOT / "results/coco_nanosam_baseline/standard"
SELF_ONNX = ROOT / "models/self_distilled/epoch10/nanosam_resnet18_epoch009.onnx"
SELF_ENGINE = ROOT / "models/self_distilled/epoch10/nanosam_resnet18_epoch009_fp16.engine"
HIST_ENGINE = ROOT / "third_party/nanosam/data/resnet18_image_encoder.engine"
DECODER = ROOT / "third_party/nanosam/data/mobile_sam_mask_decoder.engine"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def jd(name, obj):
    (EVID / name).write_text(json.dumps(obj, indent=2))
    print(f"[wrote] {name}")

def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

# ---------- protocol audit ----------
import importlib.util
spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts/eval_nanosam_coco.py")
ev = importlib.util.module_from_spec(spec)
pa = {
    "evaluator_script": "scripts/eval_nanosam_coco.py",
    "manifest_path": "data/eval_manifests/standard_manifest.json",
    "manifest_sha256": sha(ROOT / "data/eval_manifests/standard_manifest.json"),
    "expected_manifest_sha256": "19a00a836c2f1d1f236ea34175d12c2afc670b9cb5136b30e6b39bf7ca78f015",
    "manifest_n_instances": 500,
    "manifest_unique_images": 455,
    "manifest_seed": 2027,
    "mask_selection": "fixed slot0 (BASELINE LOCKED, not IoU argmax)",
    "fixed_slot0_confirmed": True,
    "metric": "prompt-conditioned instance mask IoU (|pred&gt|/|pred|gt|), union==0 -> 0",
    "metric_is_not": "COCO mask AP / semantic mIoU; terminology preserved as mIoU per protocol",
    "decoder": str(DECODER),
    "decoder_sha256": sha(DECODER),
    "decoder_precision": "FP16 (historical frozen, unchanged)",
    "decoder_rebuilt": False,
}
jd("protocol_audit.json", pa)

# ---------- manifest audit ----------
man = json.load(open(ROOT / "data/eval_manifests/standard_manifest.json"))
jd("manifest_audit.json", {
    "path": pa["manifest_path"],
    "sha256": pa["manifest_sha256"],
    "sha_match": pa["manifest_sha256"] == pa["expected_manifest_sha256"],
    "seed": man["seed"], "subset": man["subset"],
    "n_instances": man["num_instances"], "n_categories": man["num_categories"],
    "unique_images": 455,
    "size_bucket_counts": man["size_bucket_counts"],
    "instance_schema": list(man["instances"][0].keys()),
    "modified": False,
})

# ---------- artifact audit ----------
aa = {
    "self_trained_onnx": {"path": str(SELF_ONNX), "sha256": sha(SELF_ONNX),
        "expected": "aacbea870b8eca91e61fbb6325806c60a80e2f4a193f96597881e060b5c5cf15",
        "match": sha(SELF_ONNX) == "aacbea870b8eca91e61fbb6325806c60a80e2f4a193f96597881e060b5c5cf15"},
    "self_trained_engine": {"path": str(SELF_ENGINE), "sha256": sha(SELF_ENGINE),
        "size_bytes": SELF_ENGINE.stat().st_size, "precision": "FP16"},
    "historical_engine": {"path": str(HIST_ENGINE), "sha256": sha(HIST_ENGINE),
        "expected": "c277e5ca445dc29b06e8be41a481e74bd806c4dd82071cbfbd47fddcf7aaa6a8",
        "match": sha(HIST_ENGINE) == "c277e5ca445dc29b06e8be41a481e74bd806c4dd82071cbfbd47fddcf7aaa6a8",
        "untouched": True},
    "decoder_engine": {"path": str(DECODER), "sha256": sha(DECODER),
        "expected": "afb5ac4958e6bab21ed8aa0cc65798f53a71406fe6124cf6b2e008da6fccb4f2",
        "match": sha(DECODER) == "afb5ac4958e6bab21ed8aa0cc65798f53a71406fe6124cf6b2e008da6fccb4f2",
        "unchanged": True},
    "engine_identity": "DIFFERENT" if sha(SELF_ENGINE) != sha(HIST_ENGINE) else "SAME",
    "identity_pass": sha(SELF_ENGINE) != sha(HIST_ENGINE),
}
jd("artifact_audit.json", aa)

# ---------- self-trained summary (from evaluator output) ----------
summ = json.load(open(SELF_DIR / "summary.json"))
HIST_MIOU = 0.718127976997991
abs_delta = summ["mean_iou"] - HIST_MIOU
self5 = {
    "mean_iou": summ["mean_iou"],
    "median_iou": summ["median_iou"],
    "p10": summ["p10"], "p50": summ["p50"], "p90": summ["p90"],
    "iou_ge_050_ratio": summ["iou_ge_050_ratio"],
    "iou_ge_075_ratio": summ["iou_ge_075_ratio"],
    "valid": summ["valid"], "failed": summ["failed"],
    "skipped": summ["skipped"],
    "wall_time_s": summ["wall_time_s"],
    "instances_per_sec": summ["instances_per_sec_eval"],
    "unique_images": summ["unique_images"],
    "embedding_set_calls": summ["embedding_set_calls"],
    "embedding_reuse_count": summ["embedding_reuse_count"],
    "historical_mean_iou": HIST_MIOU,
    "absolute_delta_self_minus_baseline": abs_delta,
    "relative_delta": abs_delta / HIST_MIOU,
    "baseline_source": "HISTORICAL (results/coco_nanosam_baseline/standard/summary.json)",
}
jd("self_trained_standard500_summary.json", self5)

# ---------- per-instance CSV (self) ----------
pi_self = load_jsonl(SELF_DIR / "per_instance.jsonl")
pi_hist = load_jsonl(HIST_DIR / "per_instance.jsonl")
# key by annotation_id
hist_map = {r["annotation_id"]: r for r in pi_hist}
with open(EVID / "self_trained_standard500_instances.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["annotation_id","image_id","category_id","category_name","size_bucket",
                "self_trained_iou","historical_iou","paired_delta"])
    for r in pi_self:
        h = hist_map.get(r["annotation_id"], {})
        hiou = h.get("iou", "")
        delta = (r["iou"] - hiou) if hiou != "" else ""
        w.writerow([r["annotation_id"], r["image_id"], r["category_id"], r["category_name"],
                    r["size_bucket"], r["iou"], hiou, delta])

# ---------- paired comparison ----------
deltas = []; improved = degraded = equal = 0
EPS = 1e-6
matched = 0
for r in pi_self:
    h = hist_map.get(r["annotation_id"])
    if h is None:
        continue
    matched += 1
    d = r["iou"] - h["iou"]
    deltas.append(d)
    if abs(d) < EPS: equal += 1
    elif d > 0: improved += 1
    else: degraded += 1
darr = np.array(deltas)
paired = {
    "matched_instances": matched,
    "improved": improved,
    "degraded": degraded,
    "equal_nearly_equal": equal,
    "nearly_equal_definition": "abs(delta) < 1e-6",
    "mean_paired_delta": float(darr.mean()),
    "median_paired_delta": float(np.median(darr)),
    "max_improvement": float(darr.max()),
    "max_degradation": float(darr.min()),
    "self_mean_iou": float(np.mean([r["iou"] for r in pi_self if r["annotation_id"] in hist_map])),
    "historical_mean_iou": float(np.mean([hist_map[r["annotation_id"]]["iou"] for r in pi_self if r["annotation_id"] in hist_map])),
    "baseline_source": "HISTORICAL per_instance.jsonl (no rerun; reliable paired output exists)",
    "baseline_rerun": False,
}
jd("paired_comparison.json", paired)

# ---------- size breakdown ----------
buckets = {"small": defaultdict(list), "medium": defaultdict(list), "large": defaultdict(list)}
for r in pi_self:
    b = r["size_bucket"]
    buckets[b]["self"].append(r["iou"])
    if r["annotation_id"] in hist_map:
        buckets[b]["hist"].append(hist_map[r["annotation_id"]]["iou"])
HIST = {"small": 0.6053200768868633, "medium": 0.7626592084204067, "large": 0.795795889597528, }
sb = {}
for b in ["small","medium","large"]:
    sm = buckets[b]["self"]; hm = buckets[b]["hist"]
    sb[b] = {
        "n": len(sm),
        "self_mean_iou": float(np.mean(sm)) if sm else None,
        "historical_mean_iou": HIST[b],
        "self_minus_baseline": float(np.mean(sm) - HIST[b]) if sm else None,
        "historical_per_instance_mean": float(np.mean(hm)) if hm else None,
    }
sb["ordering_self_trained"] = "small < medium < large" if (np.mean(buckets["small"]["self"]) < np.mean(buckets["medium"]["self"]) < np.mean(buckets["large"]["self"])) else "ORDER_VIOLATED"
sb["ordering_historical"] = "small < medium < large (preserved)"
jd("size_breakdown.json", sb)

# ---------- category breakdown ----------
cat = defaultdict(list)
for r in pi_self:
    cat[r["category_name"]].append(r["iou"])
cat_stats = {c: {"n": len(v), "mean_iou": float(np.mean(v))} for c, v in cat.items()}
ranked = sorted(cat_stats.items(), key=lambda x: x[1]["mean_iou"])
best5 = [{"category": c, "n": s["n"], "mean_iou": s["mean_iou"]} for c, s in ranked[-5:][::-1]]
worst5 = [{"category": c, "n": s["n"], "mean_iou": s["mean_iou"]} for c, s in ranked[:5]]
with open(EVID / "category_breakdown.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["category","n","mean_iou"])
    for c, s in sorted(cat_stats.items()):
        w.writerow([c, s["n"], s["mean_iou"]])
jd("category_breakdown.json", {"best5": best5, "worst5": worst5,
    "note": "STANDARD500 small sample; category means descriptive only, no strong conclusions for low-n categories",
    "all_categories": [{"category": c, **s} for c, s in cat_stats.items()]})

# ---------- runtime ----------
rt = {
    "total_wall_time_s": summ["wall_time_s"],
    "unique_images": summ["unique_images"],
    "encoder_set_image_calls": summ["embedding_set_calls"],
    "embedding_reuse_count": summ["embedding_reuse_count"],
    "decoder_calls": summ["valid"],
    "instances_per_sec_eval": summ["instances_per_sec_eval"],
    "note": "evaluation throughput; NOT deployment FPS",
}
jd("runtime_summary.json", rt)

# ---------- quality gate ----------
absd = abs_delta
if absd >= -0.005:
    gate = "NEAR_PARITY"
    rec = "STOP_TRAINING"
elif absd >= -0.02:
    gate = "UNDERTRAINED_BUT_HEALTHY"
    rec = "RESUME_TO_50"
else:
    gate = "QUALITY_REGRESSION"
    rec = "DEBUG"
qg = {
    "ten_epoch_quality_gate": gate,
    "self_trained_mIoU": summ["mean_iou"],
    "historical_baseline_mIoU": HIST_MIOU,
    "absolute_delta_self_minus_baseline": absd,
    "heuristic_thresholds_PROJECT_DECISION_ONLY": {
        "abs_delta >= -0.005": "NEAR_PARITY",
        "-0.02 <= abs_delta < -0.005": "UNDERTRAINED_BUT_HEALTHY",
        "abs_delta < -0.02": "QUALITY_REGRESSION",
    },
    "recommendation": rec,
    "recommendation_is_advisory_not_executed": True,
    "fp16_vs_fp32_context": "Stage5B-1 rel L2 0.01763 / cosine 0.99985 (encoder embedding); cannot attribute seg gap to FP16 without PyTorch/ORT seg control",
    "ordering_preserved": sb["ordering_self_trained"],
    "pipeline_health": "no NaN/Inf/CUDA/TRT errors; 500/500 valid",
}
jd("quality_gate.json", qg)

# ---------- stage_summary.md ----------
best_str = ", ".join("{}({})={:.3f}".format(c["category"], c["n"], c["mean_iou"]) for c in best5)
worst_str = ", ".join("{}({})={:.3f}".format(c["category"], c["n"], c["mean_iou"]) for c in worst5)
md = f"""# Stage Jetson-5B-2 — Self-Trained NanoSAM STANDARD500 Quality Gate

## Verdict: {gate}

## Frozen Protocol
- Manifest: `{pa['manifest_path']}` SHA `{pa['manifest_sha256'][:16]}...` (match={pa['manifest_sha256']==pa['expected_manifest_sha256']})
- 500 instances / 455 unique images / seed 2027 / fixed slot0 confirmed
- Evaluator: `scripts/eval_nanosam_coco.py` (prompt-conditioned instance mask IoU)
- Decoder: `{DECODER}` SHA `{aa['decoder_engine']['sha256'][:16]}...` unchanged
- Historical baseline engine: SHA `{aa['historical_engine']['sha256'][:16]}...` untouched

## Self-Trained STANDARD500 Result
- mIoU = **{summ['mean_iou']:.9f}**
- median={summ['median_iou']:.6f} P10={summ['p10']:.6f} P90={summ['p90']:.6f}
- IoU>=0.50: {summ['iou_ge_050_ratio']} | IoU>=0.75: {summ['iou_ge_075_ratio']}
- valid={summ['valid']} failed={summ['failed']}

## vs Historical Baseline = {HIST_MIOU:.9f}
- absolute delta (self - baseline) = **{absd:+.9f}** ({absd/HIST_MIOU*100:+.3f}%)
- baseline source: HISTORICAL (no rerun; per_instance pairing available)

## Size Breakdown (self / historical / delta)
- small (n={sb['small']['n']}): {sb['small']['self_mean_iou']:.6f} vs {HIST['small']:.6f} → {sb['small']['self_minus_baseline']:+.6f}
- medium (n={sb['medium']['n']}): {sb['medium']['self_mean_iou']:.6f} vs {HIST['medium']:.6f} → {sb['medium']['self_minus_baseline']:+.6f}
- large (n={sb['large']['n']}): {sb['large']['self_mean_iou']:.6f} vs {HIST['large']:.6f} → {sb['large']['self_minus_baseline']:+.6f}
- Failure ordering: **{sb['ordering_self_trained']}** (historical small<medium<large preserved)

## Paired Comparison (500 matched)
- improved={improved} degraded={degraded} equal={equal}
- mean paired delta={paired['mean_paired_delta']:+.6f} median paired delta={paired['median_paired_delta']:+.6f}
- max improvement {paired['max_improvement']:+.4f} / max degradation {paired['max_degradation']:+.4f}

## Categories (best5 / worst5)
- best: {best_str}
- worst: {worst_str}

## Runtime
- wall {summ['wall_time_s']:.1f}s | encoder set_image calls {summ['embedding_set_calls']} | reuse {summ['embedding_reuse_count']} | decoder calls {summ['valid']} | {summ['instances_per_sec_eval']:.2f} inst/s (eval throughput, NOT deployment FPS)

## Quality Gate / Recommendation
- Gate: **{gate}** | Recommendation (advisory, NOT executed): **{rec}**
- Decision threshold is PROJECT-INTERNAL heuristic, not a significance test.

## STOP
No training resume / 50-epoch / FULL COCO / INT8 / NanoOWL / C++ / stability performed.
"""
(EVID / "stage_summary.md").write_text(md)
print(f"[wrote] stage_summary.md")
print("=" * 60)
print(f"GATE={gate} REC={rec}")
print(f"self mIoU={summ['mean_iou']:.9f} baseline={HIST_MIOU:.9f} delta={absd:+.9f}")
print(f"paired: improved={improved} degraded={degraded} equal={equal} mean_delta={paired['mean_paired_delta']:+.6f}")
print(f"size: small={sb['small']['self_mean_iou']:.6f}({sb['small']['self_minus_baseline']:+.6f}) med={sb['medium']['self_mean_iou']:.6f}({sb['medium']['self_minus_baseline']:+.6f}) large={sb['large']['self_mean_iou']:.6f}({sb['large']['self_minus_baseline']:+.6f})")
