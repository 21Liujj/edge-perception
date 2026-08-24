#!/usr/bin/env python3
"""Stage Jetson-6D: 50ep NanoSAM FULL COCO formal validation.

Frozen protocol: import Stage B3's evaluator (scripts/eval_nanosam_full_b3.py),
swap ONLY the encoder to the 50ep FP16 engine, run the FULL 36335-instance
evaluation. Bypasses B3's main() standard-gate (which would fail-exit because
50ep mIoU != STD_MIOU by design) by calling evaluate() + analyze() directly.

Evidence shape IDENTICAL to historical results/algorithm_b3_nanosam_full/
(full_metrics.json, size_bucket_analysis.json, category_analysis.json, ...),
plus Stage-6D additions: paired_comparison.json, historical_comparison.json,
quality_gate.json, distillation_final_decision.json, stage_summary.md.

Out of scope (STOP): 100/200ep, INT8, NanoOWL, C++, rebuild, training.
"""
import os, sys, json, csv, time, hashlib
from pathlib import Path
from collections import defaultdict
import importlib.util

import numpy as np
import torch

ROOT = Path("/home/nvidia/jetson-deploy-lab")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/nanosam"))

# ---- import B3 frozen evaluator ----
spec = importlib.util.spec_from_file_location("b3", ROOT / "scripts/eval_nanosam_full_b3.py")
b3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b3)

OUT = ROOT / "results/stage_jetson_6d"
OUT.mkdir(parents=True, exist_ok=True)

ANN = ROOT / "data/coco/annotations/instances_val2017.json"
IMG = ROOT / "data/coco/val2017"
FULL_MAN = ROOT / "data/eval_manifests/nanosam_coco_full_instances.json"
ENC = ROOT / "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine"
DEC = ROOT / "third_party/nanosam/data/mobile_sam_mask_decoder.engine"

# frozen historical FULL results (produced by the same B3 code, historical encoder)
HIST_DIR = ROOT / "results/algorithm_b3_nanosam_full"
HIST_CSV = HIST_DIR / "per_instance_primary.csv"
HIST_FULL = HIST_DIR / "full_metrics.json"
HIST_SIZE = HIST_DIR / "size_bucket_analysis.json"

# frozen reference values (from project memory / B3)
HIST_MEAN = 0.7032011354593914  # historical FULL mIoU (the gate anchor)
STD_MIOU = 0.718127976997991    # historical STANDARD500 mIoU (for standard_vs_full proxy)

ENC_EXP = "99147caf00f0174b802848c93928232c02ac996e91e6660372bbc8923112baf4"
DEC_EXP = "afb5ac4958e6bab21ed8aa0cc65798f53a71406fe6124cf6b2e008da6fccb4f2"
MAN_EXP = "ad8843c78b3ea4c94c368bfbeb7ae26138c761a8adcabb277a4a63600c00a764"


def dump(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"[evidence] {p}")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    t_start = time.perf_counter()

    # ===================== audit =====================
    enc_sha = sha(ENC); dec_sha = sha(DEC); man_sha = sha(FULL_MAN)
    man = json.loads(FULL_MAN.read_text())
    insts = man["instances"]
    unique_imgs = len({x["image_id"] for x in insts})

    dump("manifest_audit.json", {
        "path": "data/eval_manifests/nanosam_coco_full_instances.json",
        "sha256": man_sha, "expected": MAN_EXP, "match": man_sha == MAN_EXP,
        "legal_instances": len(insts), "unique_images": unique_imgs,
        "regenerated": False})
    dump("artifact_audit.json", {
        "encoder_50ep": {"path": str(ENC), "sha256": enc_sha,
                         "expected": ENC_EXP, "match": enc_sha == ENC_EXP},
        "decoder": {"path": str(DEC), "sha256": dec_sha,
                    "expected": DEC_EXP, "match": dec_sha == DEC_EXP},
        "rebuilt": False})
    dump("protocol_audit.json", {
        "evaluator": "scripts/eval_nanosam_full_b3.py (frozen, imported)",
        "functions_used": ["evaluate", "analyze", "aggregate", "materialize", "bucket"],
        "bypassed": "main() standard-gate (STD_MIOU check — would exit 10 with 50ep encoder)",
        "encoder_swapped_only": True, "decoder_unchanged": True, "manifest_unchanged": True,
        "mask_selection": "fixed slot0", "prompt": "clipped GT box corners, points labels [2,3]",
        "size_split": "COCO area small<32^2 medium<96^2 large>=96^2",
        "crowd_excluded": True, "embedding_reuse": True,
        "env": {"trt": "10.3.0", "torch": "2.5.0a0+nv24.08",
                "nanosam": "653633614b2eb93b06ba3be9adb2aeffb117bd72"}})

    if not (enc_sha == ENC_EXP and dec_sha == DEC_EXP and man_sha == MAN_EXP):
        print("STOP: artifact/manifest SHA drift — refusing to evaluate")
        sys.exit(2)
    print("[audit] all SHA match, proceeding", flush=True)

    # ===================== FULL evaluate (the long step) =====================
    csv_path = OUT / "per_instance_primary.csv"
    print(f"[FULL] start: {len(insts)} instances / {unique_imgs} images", flush=True)
    # B3 evaluate() takes Path objects (materialize uses image_root/im["file_name"])
    rows, runtime, fails = b3.evaluate(
        FULL_MAN, ANN, IMG, ENC, DEC, csv_path, OUT, "FULL_50ep")
    print(f"[FULL] done: valid={runtime['valid_count']} failed={runtime['failed_count']}", flush=True)

    # ===================== B3 frozen analyze (identical evidence shape) =====================
    # standard proxy = historical STANDARD500 aggregate, for the standard_vs_full descriptive table.
    hist_full = json.loads(HIST_FULL.read_text())
    hist_size = json.loads(HIST_SIZE.read_text())
    standard_proxy = {
        "mIoU": STD_MIOU,
        "median_iou": 0.7516989022477784,
        "iou_ge_0_50_ratio": 0.858,
        "iou_ge_0_75_ratio": 0.530,
        "small_mIoU": hist_size["small"]["mIoU"],
        "medium_mIoU": hist_size["medium"]["mIoU"],
        "large_mIoU": hist_size["large"]["mIoU"],
    }
    # B3 analyze writes: full_metrics.json, size_bucket_analysis.json, mask_selection_full.json,
    # predicted_iou_analysis.json, category_analysis.json, category_size_analysis.json,
    # iou_distribution.json(+png), failure_cases.json, standard_vs_full.json,
    # AND runtime_summary.json + failed_instances.json were already written by evaluate().
    primary, size, selection, pred_iou, cats, fc = b3.analyze(rows, fails, OUT, standard_proxy)
    print(f"[analyze] mIoU={primary['mean_iou']:.6f} wrote B3 evidence set", flush=True)

    g50 = primary["mean_iou"]
    gh = HIST_MEAN

    # ===================== 50ep summary (Stage-6D layer) =====================
    dump("full_summary.json", {
        "mean_iou": primary["mean_iou"], "median_iou": primary["median_iou"],
        "p10": primary["p10"], "p90": primary["p90"],
        "iou_ge_0_50_ratio": primary["iou_ge_0_50_ratio"],
        "iou_ge_0_75_ratio": primary["iou_ge_0_75_ratio"],
        "valid": runtime["valid_count"], "failed": runtime["failed_count"],
        "count": primary["count"]})

    # ===================== historical FULL comparison =====================
    dump("historical_comparison.json", {
        "fifty": {"mean": primary["mean_iou"], "median": primary["median_iou"],
                  "p10": primary["p10"], "p90": primary["p90"],
                  "ge050": primary["iou_ge_0_50_ratio"], "ge075": primary["iou_ge_0_75_ratio"]},
        "historical": {"mean": hist_full["mean_iou"], "median": hist_full["median_iou"],
                       "p10": hist_full["p10"], "p90": hist_full["p90"],
                       "ge050": hist_full["iou_ge_0_50_ratio"], "ge075": hist_full["iou_ge_0_75_ratio"]},
        "absolute_delta": g50 - gh, "relative_delta": (g50 - gh) / gh})

    # ===================== size breakdown (50ep vs historical, with STD500 trend) =====================
    sbr = {}
    for b in ("small", "medium", "large"):
        m50 = size[b]["mIoU"]
        mh = hist_size[b]["mIoU"]
        sbr[b] = {"n": size[b]["count"], "fifty": m50, "historical": mh, "delta": m50 - mh}
    small_par = abs(sbr["small"]["delta"]) <= 0.005
    med_par = abs(sbr["medium"]["delta"]) <= 0.005
    large_down = sbr["large"]["delta"] < -0.002
    sbr["std500_trend_on_full"] = ("CONFIRMED" if (small_par and med_par and large_down)
                                   else ("PARTIAL" if (small_par or med_par or large_down)
                                         else "NOT_CONFIRMED"))
    sbr["std500_trend_detail"] = {"small_near_par": small_par,
                                  "medium_near_par": med_par, "large_down": large_down}
    dump("size_breakdown.json", sbr)

    # ===================== paired 50ep vs historical (per-instance) =====================
    paired = {"available": False}
    try:
        hist_iou = {}
        with open(HIST_CSV) as f:
            for row in csv.DictReader(f):
                if row.get("status") == "VALID" and row.get("slot0_iou"):
                    hist_iou[int(row["annotation_id"])] = float(row["slot0_iou"])
        deltas = []
        for r in rows:
            if r["status"] == "VALID" and r["annotation_id"] in hist_iou:
                deltas.append(r["slot0_iou"] - hist_iou[r["annotation_id"]])
        d = np.asarray(deltas, dtype=np.float64)
        paired = {
            "available": True,
            "n_paired": int(d.size),
            "source": "results/algorithm_b3_nanosam_full/per_instance_primary.csv",
            "improved": int((d > 1e-9).sum()),
            "degraded": int((d < -1e-9).sum()),
            "equal_or_near": int((np.abs(d) <= 1e-9).sum()),
            "mean_delta": float(d.mean()),
            "median_delta": float(np.median(d)),
            "p10_delta": float(np.percentile(d, 10)),
            "p90_delta": float(np.percentile(d, 90)),
            "p95_abs_delta": float(np.percentile(np.abs(d), 95)),
        }
    except Exception as e:
        paired = {"available": False, "error": repr(e)}
    dump("paired_comparison.json", paired)

    # ===================== quality gate + distillation final decision =====================
    gap = abs(g50 - gh)
    if gap <= 0.005:
        gate = "NEAR_PARITY"
    elif gap <= 0.015:
        gate = "ACCEPTABLE_GAP"
    else:
        gate = "QUALITY_REGRESSION"
    complete = gate in ("NEAR_PARITY", "ACCEPTABLE_GAP")
    dump("quality_gate.json", {
        "gate": gate, "fifty_miou": g50, "historical_miou": gh,
        "absolute_delta": g50 - gh, "abs_gap": gap,
        "heuristics": {"near_parity": 0.005, "acceptable": 0.015}})

    rec = "STOP_LONG_TRAINING" if complete else "INVESTIGATE"
    dump("distillation_final_decision.json", {
        "distillation_module": "COMPLETE" if complete else "NEEDS_MORE_INVESTIGATION",
        "recommendation": rec,
        "evidence_chain": {
            "training_loss": {"epoch0": 0.001122, "epoch9": 0.0005354, "epoch49": 0.0004241},
            "standard500": {"10ep": 0.707340, "50ep": 0.715650, "historical": 0.718128},
            "full_coco": {"50ep": g50, "historical": gh}},
        "next_module": "INT8 PTQ (awaiting authorization)"})

    # ===================== stage_summary.md =====================
    md = f"""# Stage Jetson-6D — 50-Epoch NanoSAM FULL COCO Formal Validation

## Verdict: {gate} — Distillation Module: {'COMPLETE' if complete else 'NEEDS_MORE_INVESTIGATION'}

## 1. Frozen FULL Protocol
- Manifest `nanosam_coco_full_instances.json` SHA `{man_sha[:12]}…` — match (legal {len(insts)}, images {unique_imgs})
- Evaluator = B3 frozen `evaluate()` + `analyze()`, fixed slot0, clipped GT-box prompt [2,3], same decoder/preprocess/IoU
- Encoder swapped ONLY to 50ep FP16; decoder `afb5ac49…` unchanged; no rebuild/retrain/upstream edit
- B3 `main()` standard-gate bypassed (would exit 10: 50ep != STD_MIOU by design); `analyze()` called directly

## 2. 50ep FULL Result
- mean IoU **{g50:.6f}** | median {primary['median_iou']:.6f} | P10 {primary['p10']:.6f} | P90 {primary['p90']:.6f}
- IoU>=0.50 {primary['iou_ge_0_50_ratio']:.6f} | IoU>=0.75 {primary['iou_ge_0_75_ratio']:.6f}
- valid {runtime['valid_count']} / failed {runtime['failed_count']}

## 3. Historical FULL Comparison
| metric | 50ep | historical | delta |
|---|---|---|---|
| mean IoU | {g50:.6f} | {gh:.6f} | {g50-gh:+.6f} |
| median | {primary['median_iou']:.6f} | {hist_full['median_iou']:.6f} | {primary['median_iou']-hist_full['median_iou']:+.6f} |
| P10 | {primary['p10']:.6f} | {hist_full['p10']:.6f} | {primary['p10']-hist_full['p10']:+.6f} |
| P90 | {primary['p90']:.6f} | {hist_full['p90']:.6f} | {primary['p90']-hist_full['p90']:+.6f} |
| >=0.5 | {primary['iou_ge_0_50_ratio']:.6f} | {hist_full['iou_ge_0_50_ratio']:.6f} | {primary['iou_ge_0_50_ratio']-hist_full['iou_ge_0_50_ratio']:+.6f} |
| >=0.75 | {primary['iou_ge_0_75_ratio']:.6f} | {hist_full['iou_ge_0_75_ratio']:.6f} | {primary['iou_ge_0_75_ratio']-hist_full['iou_ge_0_75_ratio']:+.6f} |

- absolute delta {g50-gh:+.6f} | relative {(g50-gh)/gh*100:+.4f}%

## 4. Size Breakdown (50ep / historical / delta)
| size | n | 50ep | historical | delta |
|---|---|---|---|---|
| small | {sbr['small']['n']} | {sbr['small']['fifty']:.6f} | {sbr['small']['historical']:.6f} | {sbr['small']['delta']:+.6f} |
| medium | {sbr['medium']['n']} | {sbr['medium']['fifty']:.6f} | {sbr['medium']['historical']:.6f} | {sbr['medium']['delta']:+.6f} |
| large | {sbr['large']['n']} | {sbr['large']['fifty']:.6f} | {sbr['large']['historical']:.6f} | {sbr['large']['delta']:+.6f} |

- STANDARD500 size trend on FULL: **{sbr['std500_trend_on_full']}**
  (small_near_par={small_par}, medium_near_par={med_par}, large_down={large_down})

## 5. Paired 50ep vs Historical (per-instance)
"""
    if paired.get("available"):
        md += (f"- n {paired['n_paired']} | improved {paired['improved']} | "
               f"degraded {paired['degraded']} | equal {paired['equal_or_near']}\n")
        md += (f"- mean delta {paired['mean_delta']:+.6f} | median {paired['median_delta']:+.6f} "
               f"| P95 abs {paired['p95_abs_delta']:.6f}\n")
    else:
        md += f"- NOT AVAILABLE ({paired.get('error','?')})\n"
    md += f"""
## 6. Runtime
- wall {runtime['wall_time_seconds']:.1f}s | {runtime['instances_per_second']:.3f} inst/s | {runtime['images_per_second']:.3f} img/s
- encoder calls {runtime['encoder_calls']} | decoder calls {runtime['decoder_calls']} | reuse {runtime['embedding_reuse_instances']}
- peak RSS {runtime['peak_rss_bytes']/1e9:.2f}GB | CUDA alloc {runtime['torch_cuda_peak_allocated_bytes']/1e9:.2f}GB | reserved {runtime['torch_cuda_peak_reserved_bytes']/1e9:.2f}GB
- evaluation throughput, NOT deployment FPS

## 7. Quality Gate
- **{gate}** (abs gap {gap:.6f}) → Distillation Module **{'COMPLETE' if complete else 'NEEDS_MORE_INVESTIGATION'}**
- recommendation (advisory, NOT executed): **{rec}**

## Out of Scope (STOP)
100/200ep, INT8, NanoOWL, C++, rebuild — NOT performed.
"""
    (OUT / "stage_summary.md").write_text(md)
    print(f"[evidence] {OUT/'stage_summary.md'}")

    print(f"\n=== Stage Jetson-6D DONE ===")
    print(f"GATE={gate} 50ep_FULL_mIoU={g50:.6f} historical={gh:.6f} delta={g50-gh:+.6f}")
    print(f"distillation_module={'COMPLETE' if complete else 'NEEDS_MORE_INVESTIGATION'} rec={rec}")
    print(f"wall_total={time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
