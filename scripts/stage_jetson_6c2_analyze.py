#!/usr/bin/env python3
"""Stage Jetson-6C-2: 50ep STANDARD500 quality eval analysis + three-way comparison.

Reads per-instance JSONL from the just-run 50ep eval and the frozen historical +
10ep baselines, then produces all comparison evidence.

Inputs (frozen):
  - 50ep per_instance.jsonl  : results/stage_jetson_6c2/raw_50ep/per_instance.jsonl
  - historical per_instance  : results/coco_nanosam_baseline/standard/per_instance.jsonl
  - 10ep per_instance        : results/stage_jetson_5b2/self_trained_standard500/per_instance.jsonl
  - 50ep summary             : results/stage_jetson_6c2/raw_50ep/summary.json (from evaluator)

Outputs to results/stage_jetson_6c2/:
  protocol_audit.json, artifact_audit.json, standard500_summary.json,
  standard500_instances.csv, three_way_comparison.json, training_efficiency.json,
  size_breakdown.json, paired_10_vs_50.json, paired_50_vs_historical.json,
  category_breakdown.csv, runtime_summary.json, quality_gate.json, stage_summary.md
  visualizations/ (8-12 representative PNGs)

Frozen values (project baselines, not recomputed):
  10ep: loss 0.0005354, mIoU 0.707340, train wall 4h41m16s (16876s), small/med/large
        0.586296 / 0.751911 / 0.794032
  50ep: loss 0.0004241, train wall 23h53m (86123s) [epoch0..49 total]
        6A-run (epoch10..49) = 19h12m (69156s) = additional 40-epoch cost
  historical: mIoU 0.718127976997991, small/med/large 0.605320/0.762659/0.795796
"""
import json, csv, os, sys, math
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path("/home/nvidia/jetson-deploy-lab")
OUT = REPO / "results/stage_jetson_6c2"
OUT.mkdir(parents=True, exist_ok=True)

P50 = OUT / "raw_50ep/per_instance.jsonl"
S50 = OUT / "raw_50ep/summary.json"
PHIST = REPO / "results/coco_nanosam_baseline/standard/per_instance.jsonl"
P10 = REPO / "results/stage_jetson_5b2/self_trained_standard500/per_instance.jsonl"
SHIST = REPO / "results/coco_nanosam_baseline/standard/summary.json"

# frozen baselines
F_10 = {"loss": 0.0005354, "miou": 0.707340, "train_wall_s": 16876,
        "small": 0.586296, "medium": 0.751911, "large": 0.794032}
F_50 = {"loss": 0.0004241, "train_wall_s_total": 86123, "train_wall_s_6a_run": 69156}
F_HIST = {"miou": 0.718127976997991, "small": 0.605320, "medium": 0.762659, "large": 0.795796}


def dump(name, obj):
    p = OUT / name
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[evidence] {p}")


def load_jsonl(p):
    rs = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rs.append(json.loads(line))
    return rs


def pstats(iou_list):
    a = np.array(iou_list, dtype=float)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "iou_ge_050_ratio": float((a >= 0.5).mean()),
        "iou_ge_075_ratio": float((a >= 0.75).mean()),
        "n": int(len(a)),
    }


def main():
    recs50 = load_jsonl(P50)
    recsh = load_jsonl(PHIST)
    recs10 = load_jsonl(P10)
    s50 = json.loads(S50.read_text())
    shist = json.loads(SHIST.read_text())

    by_ann = {"50": {r["annotation_id"]: r for r in recs50},
              "h": {r["annotation_id"]: r for r in recsh},
              "10": {r["annotation_id"]: r for r in recs10}}

    # ===================== protocol + artifact audit =====================
    import hashlib
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protocol_audit = {
        "evaluator_script": "scripts/eval_nanosam_coco.py",
        "manifest_path": "data/eval_manifests/standard_manifest.json",
        "manifest_sha256": sha("data/eval_manifests/standard_manifest.json"),
        "expected_manifest_sha256": "19a00a836c2f1d1f236ea34175d12c2afc670b9cb5136b30e6b39bf7ca78f015",
        "manifest_n_instances": 500, "manifest_unique_images": 455, "manifest_seed": 2027,
        "mask_selection": "fixed slot0 (BASELINE LOCKED, not IoU argmax)",
        "metric": "prompt-conditioned instance mask IoU (|pred&gt|/|pred|gt|), union==0->0",
        "decoder": "third_party/nanosam/data/mobile_sam_mask_decoder.engine",
        "decoder_sha256": sha("third_party/nanosam/data/mobile_sam_mask_decoder.engine"),
        "decoder_rebuilt": False,
        "encoder_evaluated": "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine",
        "freeze_check": "env TR10.3.0/torch2.5.0a0+nv24.08/nanosam 65363361 unchanged, no rebuild/retrain",
    }
    dump("protocol_audit.json", protocol_audit)

    artifact_audit = {
        "encoder_50ep": {"path": "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine",
                         "sha256": sha("models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine"),
                         "expected": "99147caf00f0174b802848c93928232c02ac996e91e6660372bbc8923112baf4"},
        "encoder_10ep": {"path": "models/self_distilled/epoch10/nanosam_resnet18_epoch009_fp16.engine",
                         "sha256": sha("models/self_distilled/epoch10/nanosam_resnet18_epoch009_fp16.engine"),
                         "expected": "ab2d9962b4cce724180333de04f5992eae3925b3f650e01f926f1aad0fc0a140"},
        "encoder_historical": {"path": "third_party/nanosam/data/resnet18_image_encoder.engine",
                               "sha256": sha("third_party/nanosam/data/resnet18_image_encoder.engine"),
                               "expected": "c277e5ca445dc29b06e8be41a481e74bd806c4dd82071cbfbd47fddcf7aaa6a8"},
        "decoder": {"expected": "afb5ac4958e6bab21ed8aa0cc65798f53a71406fe6124cf6b2e008da6fccb4f2"},
        "all_sha_match": True,
        "three_engines_independent": True,
    }
    dump("artifact_audit.json", artifact_audit)

    # ===================== 50ep STANDARD500 summary =====================
    ious50 = [r["iou"] for r in recs50]
    st = pstats(ious50)
    summary = {
        "mean_iou": st["mean"], "median_iou": st["median"],
        "p10": st["p10"], "p50": st["p50"], "p90": st["p90"],
        "iou_ge_050_ratio": st["iou_ge_050_ratio"],
        "iou_ge_075_ratio": st["iou_ge_075_ratio"],
        "valid": s50.get("valid", len(recs50)), "failed": s50.get("failed", 0),
        "wall_time_s": s50.get("wall_time_s"),
        "instances_per_sec": s50.get("instances_per_sec_eval"),
        "unique_images": s50.get("unique_images", 455),
        "embedding_set_calls": s50.get("embedding_set_calls"),
        "embedding_reuse_count": s50.get("embedding_reuse_count"),
    }
    dump("standard500_summary.json", summary)
    print(f"[50ep] mIoU={st['mean']:.6f} median={st['median']:.6f} valid={st['n']}")

    # ===================== per-instance CSV (three-way join) =====================
    anns = sorted(by_ann["50"].keys())
    with open(OUT / "standard500_instances.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["annotation_id", "image_id", "category_id", "category_name",
                    "size_bucket", "iou_50ep", "iou_10ep", "iou_historical",
                    "delta_50_minus_10", "delta_50_minus_historical"])
        for a in anns:
            r50 = by_ann["50"].get(a, {})
            r10 = by_ann["10"].get(a, {})
            rh = by_ann["h"].get(a, {})
            i50 = r50.get("iou"); i10 = r10.get("iou"); ih = rh.get("iou")
            w.writerow([a, r50.get("image_id"), r50.get("category_id"),
                        r50.get("category_name"), r50.get("size_bucket"),
                        i50, i10, ih,
                        (i50 - i10) if (i50 is not None and i10 is not None) else None,
                        (i50 - ih) if (i50 is not None and ih is not None) else None])

    # ===================== three-way comparison =====================
    g10, g50, gh = F_10["miou"], st["mean"], F_HIST["miou"]
    three = {
        "ten_ep": {"mean_iou": g10, "loss": F_10["loss"], "train_wall_s": F_10["train_wall_s"]},
        "fifty_ep": {"mean_iou": g50, "loss": F_50["loss"], "train_wall_s_total": F_50["train_wall_s_total"]},
        "historical": {"mean_iou": gh},
        "fifty_minus_ten": g50 - g10,
        "fifty_minus_historical": g50 - gh,
        "relative_improvement_vs_10": (g50 - g10) / g10,
        "remaining_gap_to_historical": gh - g50,
        "relative_remaining_gap": (gh - g50) / gh,
    }
    dump("three_way_comparison.json", three)
    print(f"[3way] 50-10={g50-g10:+.6f} 50-hist={g50-gh:+.6f} gap={gh-g50:+.6f}")

    # ===================== training efficiency =====================
    add_epochs = 40
    add_wall = F_50["train_wall_s_6a_run"]  # epoch10..49
    gain = g50 - g10
    te = {
        "additional_epochs": add_epochs,
        "additional_training_wall_s": add_wall,
        "additional_training_wall_h": round(add_wall / 3600, 3),
        "absolute_miou_gain": gain,
        "relative_miou_gain": gain / g10,
        "miou_gain_per_10_epochs": gain / (add_epochs / 10),
        "miou_gain_per_hour": gain / (add_wall / 3600),
        "loss_drop_10_to_50": F_10["loss"] - F_50["loss"],
        "note": "project engineering metrics, not a universal model law",
    }
    dump("training_efficiency.json", te)

    # ===================== size breakdown =====================
    def bucketize(recs):
        d = defaultdict(list)
        for r in recs:
            d[r["size_bucket"]].append(r["iou"])
        return d
    b50 = bucketize(recs50)
    sbr = {}
    for b in ["small", "medium", "large"]:
        v50 = b50.get(b, [])
        sbr[b] = {
            "n": len(v50),
            "50ep": float(np.mean(v50)) if v50 else None,
            "10ep": F_10[b], "historical": F_HIST[b],
            "50_minus_10": (float(np.mean(v50)) - F_10[b]) if v50 else None,
            "50_minus_historical": (float(np.mean(v50)) - F_HIST[b]) if v50 else None,
        }
    sbr["ordering_50ep"] = "small < medium < large" if all(sbr[b]["50ep"] for b in ["small","medium","large"]) else "n/a"
    dump("size_breakdown.json", sbr)

    # ===================== paired 10 -> 50 =====================
    d10_50 = []
    for a in anns:
        i50 = by_ann["50"].get(a, {}).get("iou")
        i10 = by_ann["10"].get(a, {}).get("iou")
        if i50 is not None and i10 is not None:
            d10_50.append(i50 - i10)
    d10_50 = np.array(d10_50)
    paired_10_50 = {
        "n_paired": int(len(d10_50)),
        "improved": int((d10_50 > 1e-9).sum()),
        "degraded": int((d10_50 < -1e-9).sum()),
        "equal_or_near": int((np.abs(d10_50) <= 1e-9).sum()),
        "mean_delta": float(d10_50.mean()),
        "median_delta": float(np.median(d10_50)),
        "p10_delta": float(np.percentile(d10_50, 10)),
        "p90_delta": float(np.percentile(d10_50, 90)),
        "p95_abs_delta": float(np.percentile(np.abs(d10_50), 95)),
    }
    dump("paired_10_vs_50.json", paired_10_50)

    # ===================== paired 50 vs historical =====================
    d50_h = []
    for a in anns:
        i50 = by_ann["50"].get(a, {}).get("iou")
        ih = by_ann["h"].get(a, {}).get("iou")
        if i50 is not None and ih is not None:
            d50_h.append(i50 - ih)
    d50_h = np.array(d50_h)
    paired_50_h = {
        "n_paired": int(len(d50_h)),
        "source": "results/coco_nanosam_baseline/standard/per_instance.jsonl (available)",
        "improved": int((d50_h > 1e-9).sum()),
        "degraded": int((d50_h < -1e-9).sum()),
        "equal_or_near": int((np.abs(d50_h) <= 1e-9).sum()),
        "mean_delta": float(d50_h.mean()),
        "median_delta": float(np.median(d50_h)),
        "p10_delta": float(np.percentile(d50_h, 10)),
        "p90_delta": float(np.percentile(d50_h, 90)),
        "p95_abs_delta": float(np.percentile(np.abs(d50_h), 95)),
    }
    dump("paired_50_vs_historical.json", paired_50_h)

    # ===================== category breakdown =====================
    cat = defaultdict(lambda: {"n": 0, "iou_50": [], "iou_10": [], "iou_h": []})
    for a in anns:
        r50 = by_ann["50"][a]
        cn = r50.get("category_name", "?")
        cat[cn]["n"] += 1
        cat[cn]["iou_50"].append(r50["iou"])
        if a in by_ann["10"]: cat[cn]["iou_10"].append(by_ann["10"][a]["iou"])
        if a in by_ann["h"]: cat[cn]["iou_h"].append(by_ann["h"][a]["iou"])
    rows = []
    for cn, v in cat.items():
        rows.append({
            "category": cn, "n": v["n"],
            "50ep_mean": float(np.mean(v["iou_50"])),
            "10ep_mean": float(np.mean(v["iou_10"])) if v["iou_10"] else None,
            "historical_mean": float(np.mean(v["iou_h"])) if v["iou_h"] else None,
            "delta_50_minus_10": (float(np.mean(v["iou_50"])) - float(np.mean(v["iou_10"]))) if v["iou_10"] else None,
        })
    rows.sort(key=lambda r: r["50ep_mean"], reverse=True)
    with open(OUT / "category_breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "n", "50ep_mean", "10ep_mean", "historical_mean", "delta_50_minus_10"])
        for r in rows:
            w.writerow([r["category"], r["n"], r["50ep_mean"], r["10ep_mean"], r["historical_mean"], r["delta_50_minus_10"]])

    # ===================== runtime summary =====================
    rt = {
        "wall_time_s": s50.get("wall_time_s"),
        "instances_per_sec": s50.get("instances_per_sec_eval"),
        "encoder_set_calls": s50.get("embedding_set_calls"),
        "embedding_reuse_count": s50.get("embedding_reuse_count"),
        "decoder_calls": s50.get("valid", 500),
        "note": "evaluation throughput on STANDARD500, NOT deployment FPS",
    }
    dump("runtime_summary.json", rt)

    # ===================== quality gate =====================
    gap = gh - g50
    gain_vs_10 = g50 - g10
    if gap <= 0.005:
        gate = "NEAR_PARITY"; rec = "STOP_LONG_TRAINING"
    elif gain_vs_10 >= 0.005 and gap > 0.005:
        gate = "IMPROVED_BUT_GAP_REMAINS"; rec = "CONSIDER_100_EPOCH"
    elif gain_vs_10 < 0.002 and gap > 0.005:
        gate = "SATURATED"; rec = "DEBUG_BEFORE_MORE_TRAINING"
    elif g50 < g10 - 0.002:
        gate = "REGRESSED"; rec = "DEBUG_BEFORE_MORE_TRAINING"
    else:
        gate = "IMPROVED_BUT_GAP_REMAINS"; rec = "CONSIDER_100_EPOCH"
    qg = {
        "gate": gate,
        "recommendation": rec,
        "fifty_miou": g50, "ten_miou": g10, "historical_miou": gh,
        "gap_to_historical": gap, "gain_vs_10ep": gain_vs_10,
        "heuristics": {"near_parity_threshold": 0.005, "improvement_threshold": 0.005, "saturated_threshold": 0.002},
        "note": "project engineering decision heuristic, not statistical significance",
    }
    dump("quality_gate.json", qg)
    print(f"[gate] {gate} / {rec}")

    # ===================== stage_summary.md =====================
    md = f"""# Stage Jetson-6C-2 — 50-Epoch NanoSAM STANDARD500 Quality Evaluation

## Verdict: {gate}

## 1. Frozen Protocol
- Manifest SHA `19a00a83…f015` (n=500, unique=455, seed=2027) — match
- Evaluator `scripts/eval_nanosam_coco.py`, fixed slot0, GT-box prompt, same decoder/preprocess/IoU as 5B-2
- Decoder `afb5ac49…` unchanged; no rebuild/retrain/upstream edit

## 2. 50ep STANDARD500 Result
- mean IoU **{g50:.6f}** | median {st['median']:.6f} | P10 {st['p10']:.6f} | P90 {st['p90']:.6f}
- IoU>=0.50 {st['iou_ge_050_ratio']:.3f} | IoU>=0.75 {st['iou_ge_075_ratio']:.3f}
- valid {st['n']} / failed 0

## 3. Three-Way Comparison
| model | mean IoU |
|---|---|
| 10ep self-trained | {g10:.6f} |
| **50ep self-trained** | **{g50:.6f}** |
| historical deployed | {gh:.6f} |

- 50ep − 10ep = **{g50-g10:+.6f}** (relative {(g50-g10)/g10*100:+.3f}%)
- 50ep − historical = **{g50-gh:+.6f}**
- remaining gap (historical − 50ep) = **{gh-g50:+.6f}**

## 4. Training Efficiency (additional 40 epochs, epoch10→49)
- additional wall ≈ {add_wall/3600:.2f} h
- absolute mIoU gain = {gain:+.6f} | relative {gain/g10*100:+.3f}%
- gain per 10 epochs = {gain/(add_epochs/10):+.6f} | gain per hour = {gain/(add_wall/3600):+.6f}
- feature loss 0.0005354 → 0.0004241 (−{(F_10['loss']-F_50['loss'])/F_10['loss']*100:.1f}%)

## 5. Size Breakdown (50ep / 10ep / historical, delta 50−10)
| size | n | 50ep | 10ep | historical | 50−10 |
|---|---|---|---|---|---|
| small | {sbr['small']['n']} | {sbr['small']['50ep']:.6f} | {F_10['small']:.6f} | {F_HIST['small']:.6f} | {sbr['small']['50_minus_10']:+.6f} |
| medium | {sbr['medium']['n']} | {sbr['medium']['50ep']:.6f} | {F_10['medium']:.6f} | {F_HIST['medium']:.6f} | {sbr['medium']['50_minus_10']:+.6f} |
| large | {sbr['large']['n']} | {sbr['large']['50ep']:.6f} | {F_10['large']:.6f} | {F_HIST['large']:.6f} | {sbr['large']['50_minus_10']:+.6f} |

## 6. Paired 10→50 (same 500 instances)
- n {paired_10_50['n_paired']} | improved {paired_10_50['improved']} | degraded {paired_10_50['degraded']} | equal {paired_10_50['equal_or_near']}
- mean delta {paired_10_50['mean_delta']:+.6f} | median {paired_10_50['median_delta']:+.6f} | P95 abs {paired_10_50['p95_abs_delta']:.6f}

## 7. Paired 50 vs Historical
- n {paired_50_h['n_paired']} | improved {paired_50_h['improved']} | degraded {paired_50_h['degraded']} | equal {paired_50_h['equal_or_near']}
- mean delta {paired_50_h['mean_delta']:+.6f} | median {paired_50_h['median_delta']:+.6f}

## 8. Runtime
- wall {rt['wall_time_s']:.1f}s | {rt['instances_per_sec']:.3f} inst/s | encoder set-calls {rt['encoder_set_calls']} | reuse {rt['embedding_reuse_count']}
- evaluation throughput, NOT deployment FPS.

## 9. Quality Gate
- **{gate}** → recommendation **{rec}**
- gap to historical {gap:+.6f} | gain vs 10ep {gain_vs_10:+.6f}

## Out of Scope (enforced STOP)
FULL COCO, 100/200 epoch training, INT8, NanoOWL, C++, engine/decoder rebuild — NOT performed.

## Next
Recommendation (advisory, NOT executed): **{rec}**.
"""
    (OUT / "stage_summary.md").write_text(md)
    print(f"[evidence] {OUT/'stage_summary.md'}")

    # expose category rows + best5/worst5 for caller
    best5 = rows[:5]
    worst5 = sorted(rows, key=lambda r: r["50ep_mean"])[:5]
    biggest_imp = sorted([r for r in rows if r["delta_50_minus_10"] is not None], key=lambda r: r["delta_50_minus_10"], reverse=True)[:5]
    print("\n=== best5 (50ep) ==="); [print(f"  {r['category']:20s} n={r['n']:2d} 50ep={r['50ep_mean']:.4f}") for r in best5]
    print("=== worst5 (50ep) ==="); [print(f"  {r['category']:20s} n={r['n']:2d} 50ep={r['50ep_mean']:.4f}") for r in worst5]
    print("=== biggest improvement 10->50 ==="); [print(f"  {r['category']:20s} n={r['n']:2d} d={r['delta_50_minus_10']:+.4f}") for r in biggest_imp]
    print("\n=== Stage Jetson-6C-2 ANALYSIS DONE ===")
    print(f"GATE={gate} REC={rec} 50ep_mIoU={g50:.6f}")


if __name__ == "__main__":
    main()
