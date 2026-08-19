#!/usr/bin/env python3
"""Stage Jetson-5B-3 evidence aggregation: ORT backend audit, FP32 summary,
FP32-vs-FP16 paired comparison, size precision delta, quality decomposition,
FP16 quality impact gate, and stage_summary.md."""
import json, hashlib, os
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/home/nvidia/jetson-deploy-lab")
EVID = ROOT / "results" / "stage_jetson_5b3"
ONNX = ROOT / "models/self_distilled/epoch10/nanosam_resnet18_epoch009.onnx"
ENGINE = ROOT / "models/self_distilled/epoch10/nanosam_resnet18_epoch009_fp16.engine"
DECODER = ROOT / "third_party/nanosam/data/mobile_sam_mask_decoder.engine"
ORT_DIR = EVID / "ort_standard500"
FP16_DIR = ROOT / "results/stage_jetson_5b2/self_trained_standard500"

HIST = {"small": 0.6053200768868633, "medium": 0.7626592084204067, "large": 0.795795889597528}
HIST_MIOU = 0.718127976997991
FP16_MIOU = 0.7073402744674683

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def jd(name, obj):
    (EVID / name).write_text(json.dumps(obj, indent=2)); print(f"[wrote] {name}")

def load_jsonl(p): return [json.loads(l) for l in open(p) if l.strip()]

# ---- protocol audit ----
pa = {
  "stage": "Jetson-5B-3",
  "purpose": "isolate FP16 (TRT) vs FP32 (ORT CPU) downstream mask-quality impact; same model, same protocol, encoder-only backend swap",
  "manifest": "data/eval_manifests/standard_manifest.json",
  "manifest_sha256": sha(ROOT/"data/eval_manifests/standard_manifest.json"),
  "manifest_n_instances": 500, "manifest_unique_images": 455, "manifest_seed": 2027,
  "decoder": str(DECODER), "decoder_sha256": sha(DECODER),
  "decoder_precision": "FP16 frozen TRT, unchanged",
  "mask_selection": "fixed slot0",
  "metric": "prompt-conditioned instance mask IoU (terminology mIoU per protocol)",
  "encoder_backend_swap": "image encoder TRT FP16 -> ORT CPU FP32; nothing else changed",
  "upstream_modified": False,
  "engine_rebuilt": False,
}
jd("protocol_audit.json", pa)

# ---- artifact audit ----
aa = {
  "self_trained_onnx": {"path": str(ONNX), "sha256": sha(ONNX),
    "expected": "aacbea870b8eca91e61fbb6325806c60a80e2f4a193f96597881e060b5c5cf15",
    "match": sha(ONNX)=="aacbea870b8eca91e61fbb6325806c60a80e2f4a193f96597881e060b5c5cf15"},
  "self_trained_trt_engine": {"path": str(ENGINE), "sha256": sha(ENGINE),
    "expected": "ab2d9962b4cce724180333de04f5992eae3925b3f650e01f926f1aad0fc0a140",
    "match": sha(ENGINE)=="ab2d9962b4cce724180333de04f5992eae3925b3f650e01f926f1aad0fc0a140",
    "rebuilt_this_stage": False},
  "decoder_engine": {"path": str(DECODER), "sha256": sha(DECODER),
    "expected":"afb5ac4958e6bab21ed8aa0cc65798f53a71406fe6124cf6b2e008da6fccb4f2",
    "match": sha(DECODER)=="afb5ac4958e6bab21ed8aa0cc65798f53a71406fe6124cf6b2e008da6fccb4f2"},
  "engine_sha_process_note": "Stage5B-1 captured 0d63103c...; Stage5B-2 found disk SHA ab2d9962... due to concurrent trtexec builds writing same --saveEngine. Frozen artifact for this stage = ab2d9962... Build system should use unique temp output -> validation -> atomic rename. No rebuild performed here.",
}
jd("artifact_audit.json", aa)

# ---- ORT backend audit ----
import onnxruntime as ort
sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
oi, oo = sess.get_inputs()[0], sess.get_outputs()[0]
ob = {
  "backend": "ONNXRuntime CPUExecutionProvider",
  "onnxruntime_version": ort.__version__,
  "provider_options": {"intra_op_num_threads": 8, "inter_op_num_threads": 1},
  "input_name": oi.name, "input_shape": list(oi.shape), "input_dtype": str(oi.type),
  "output_name": oo.name, "output_shape": list(oo.shape), "output_dtype": str(oo.type),
  "precision": "FP32 CPU",
  "embedding_transfer_to_decoder": "ORT FP32 numpy -> torch float32 .cuda() (decoder is TRT)",
}
jd("ort_backend.json", ob)

# ---- ORT FP32 summary ----
s = json.load(open(ORT_DIR/"summary.json"))
s2 = {
  "backend": "ORT FP32 CPU encoder + frozen TRT FP16 decoder",
  "mean_iou": s["mean_iou"], "median_iou": s["median_iou"],
  "p10": s["p10"], "p50": s["p50"], "p90": s["p90"],
  "iou_ge_050_ratio": s["iou_ge_050_ratio"], "iou_ge_075_ratio": s["iou_ge_075_ratio"],
  "valid": s["valid"], "failed": s["failed"],
  "mIoU_small": s["mIoU_small"], "mIoU_medium": s["mIoU_medium"], "mIoU_large": s["mIoU_large"],
  "small_count": s["small_count"], "medium_count": s["medium_count"], "large_count": s["large_count"],
  "unique_images": s["unique_images"],
  "embedding_set_calls": s["embedding_set_calls"], "embedding_reuse_count": s["embedding_reuse_count"],
  "wall_time_s": s["wall_time_s"], "instances_per_sec": s["instances_per_sec_eval"],
}
jd("ort_standard500_summary.json", s2)

# ---- per-instance CSV (ORT FP32) ----
pi_ort = load_jsonl(ORT_DIR/"per_instance.jsonl")
import csv
with open(EVID/"ort_standard500_instances.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["annotation_id","image_id","category_id","category_name","size_bucket","ort_fp32_iou"])
    for r in pi_ort:
        w.writerow([r["annotation_id"],r["image_id"],r["category_id"],r["category_name"],r["size_bucket"],r["iou"]])

# ---- paired FP32 vs FP16 (FP16 - FP32) ----
pi_fp16 = {r["annotation_id"]: r for r in load_jsonl(FP16_DIR/"per_instance.jsonl")}
deltas=[]; improved=degraded=equal=0; EPS=1e-6; rows=[]
for r in pi_ort:
    f=pi_fp16.get(r["annotation_id"])
    if f is None: continue
    d = f["iou"] - r["iou"]  # FP16 - FP32
    deltas.append(d); rows.append((r["annotation_id"], f["iou"], r["iou"], d))
    if abs(d)<EPS: equal+=1
    elif d>0: improved+=1
    else: degraded+=1
darr=np.array(deltas)
absarr=np.abs(darr)
pp={
  "definition": "FP16_iou - FP32_iou (TRT FP16 minus ORT FP32, same instance)",
  "matched": len(deltas),
  "improved_FP16_higher": improved, "degraded_FP16_lower": degraded, "equal": equal,
  "mean_delta_FP16_minus_FP32": float(darr.mean()),
  "median_delta": float(np.median(darr)),
  "max_abs_per_instance_delta": float(absarr.max()),
  "P95_abs_delta": float(np.percentile(absarr,95)),
  "FP16_mean_iou": float(np.mean([a for _,a,_,_ in rows])),
  "FP32_mean_iou": float(np.mean([b for _,_,b,_ in rows])),
}
jd("fp32_vs_fp16_paired.json", pp)

# ---- size precision delta (FP16 - FP32 per bucket) ----
buk = defaultdict(lambda: {"fp16":[], "fp32":[]})
for r in pi_ort:
    b=r["size_bucket"]; buk[b]["fp32"].append(r["iou"])
    f=pi_fp16.get(r["annotation_id"]);
    if f: buk[b]["fp16"].append(f["iou"])
sz={}
for bkey in ["small","medium","large"]:
    fp16m=float(np.mean(buk[bkey]["fp16"])); fp32m=float(np.mean(buk[bkey]["fp32"]))
    sz[bkey]={"n":len(buk[bkey]["fp32"]),
      "fp16_mean_iou":fp16m, "fp32_mean_iou":fp32m,
      "delta_fp16_minus_fp32":fp16m-fp32m}
jd("size_precision_delta.json", sz)

# ---- quality decomposition ----
fp32_m=s2["mean_iou"]; fp16_m=FP16_MIOU
total_gap = HIST_MIOU - fp16_m                 # baseline - FP16 (total observed regression)
precision_gap = fp32_m - fp16_m                # FP32 - FP16 (loss attributable to FP16)
training_gap = HIST_MIOU - fp32_m              # baseline - FP32 (loss NOT explained by FP16)
# additive ENGINEERING approximation (nonlinear decoder -> not strictly additive)
additive_residual = (HIST_MIOU - fp16_m) - (training_gap + precision_gap)
qd={
  "historical_baseline_mIoU": HIST_MIOU,
  "self_trained_ORT_FP32_mIoU": fp32_m,
  "self_trained_TRT_FP16_mIoU": fp16_m,
  "total_gap_baseline_minus_FP16": total_gap,
  "FP16_precision_gap_FP32_minus_FP16": precision_gap,
  "training_model_gap_baseline_minus_FP32": training_gap,
  "additive_engineering_approx_note": "nonlinear decoder => gaps are NOT strictly mathematically additive; additive residual shown as sanity check only",
  "additive_residual": additive_residual,
  "fraction_of_total_gap_attributed_to_FP16": precision_gap/total_gap if total_gap!=0 else None,
}
jd("quality_decomposition.json", qd)

# ---- FP16 quality impact gate ----
ad = precision_gap  # FP32 - FP16 (positive => FP16 worse)
ab = abs(ad)
if ab <= 0.002: impact="NEGLIGIBLE"
elif ab <= 0.005: impact="MINOR"
else: impact="MATERIAL"
gate={
  "FP16_QUALITY_IMPACT": impact,
  "definition": "abs(FP32_mIoU - FP16_mIoU) = "+f"{ab:.6f}",
  "heuristic_PROJECT_DECISION_ONLY": {"<=0.002":"NEGLIGIBLE","0.002-0.005":"MINOR",">0.005":"MATERIAL"},
  "self_trained_FP32_mIoU": fp32_m,
  "self_trained_FP16_mIoU": fp16_m,
  "historical_baseline_mIoU": HIST_MIOU,
}
jd("quality_gate.json", gate)

# ---- main conclusion + recommendation ----
if fp32_m < HIST_MIOU - 0.002 and impact in ("NEGLIGIBLE","MINOR"):
    concl="TRAINING_GAP_DOMINANT"
elif impact=="MATERIAL":
    concl="FP16_GAP_MATERIAL"
else:
    concl="MIXED"
rec = "INVESTIGATE_FP16_FIRST" if impact=="MATERIAL" else ("RESUME_TO_50" if concl=="TRAINING_GAP_DOMINANT" else "DEBUG")
rec={"main_conclusion":concl,
     "recommendation":rec,
     "recommendation_advisory_not_executed":True,
     "rationale": f"FP16 impact {impact} (|delta|={ab:.6f}). FP32 mIoU {fp32_m:.6f} vs baseline {HIST_MIOU:.6f}: training/model gap {training_gap:+.6f} dominates total gap {total_gap:+.6f}."}
jd("conclusion.json", rec)

# ---- stage_summary.md ----
md=f"""# Stage Jetson-5B-3 — FP32 vs FP16 Downstream Quality Control

## Purpose
Separate the 10-epoch **training gap** from the **FP16 deployment-precision gap**.
Same model, frozen STANDARD500 protocol; only the image encoder backend swapped
(TRT FP16 -> ORT CPU FP32). No upstream edit, no rebuild, no training.

## Artifacts (frozen, read-only)
- ONNX `{aa['self_trained_onnx']['sha256'][:16]}...` (match={aa['self_trained_onnx']['match']})
- TRT engine `{aa['self_trained_trt_engine']['sha256'][:16]}...` (match={aa['self_trained_trt_engine']['match']})
- decoder `{aa['decoder_engine']['sha256'][:16]}...` (match={aa['decoder_engine']['match']})
- Engine SHA process note: 5B-1 `0d63103c` -> 5B-2 `ab2d9962` (concurrent trtexec builds; future build system should use temp-output->validate->atomic-rename). Frozen here = `ab2d9962`.

## ORT FP32 STANDARD500
- mIoU = **{s2['mean_iou']:.9f}** | median {s2['median_iou']:.6f} P10 {s2['p10']:.6f} P90 {s2['p90']:.6f}
- >=0.50 {s2['iou_ge_050_ratio']} | >=0.75 {s2['iou_ge_075_ratio']} | valid {s2['valid']} failed {s2['failed']}
- wall {s2['wall_time_s']:.1f}s ({s2['instances_per_sec']:.2f} inst/s; CPU ORT, quality control not perf bench)

## FP16 vs FP32
- FP16 (TRT) mIoU = **{fp16_m:.9f}** | FP32 (ORT) mIoU = **{fp32_m:.9f}**
- FP16_DOWNSTREAM_DELTA (FP16 - FP32) = **{fp16_m-fp32_m:+.9f}**
- Paired (n={len(deltas)}): FP16 higher {improved} / FP16 lower {degraded} / equal {equal}
  mean delta {pp['mean_delta_FP16_minus_FP32']:+.6f} | median {pp['median_delta']:+.6f}
  | max abs {pp['max_abs_per_instance_delta']:.6f} | P95 abs {pp['P95_abs_delta']:.6f}

## Size Sensitivity (FP32 / FP16 / delta FP16-FP32)
- small n={sz['small']['n']}: FP32 {sz['small']['fp32_mean_iou']:.6f} | FP16 {sz['small']['fp16_mean_iou']:.6f} | Δ {sz['small']['delta_fp16_minus_fp32']:+.6f}
- medium n={sz['medium']['n']}: FP32 {sz['medium']['fp32_mean_iou']:.6f} | FP16 {sz['medium']['fp16_mean_iou']:.6f} | Δ {sz['medium']['delta_fp16_minus_fp32']:+.6f}
- large n={sz['large']['n']}: FP32 {sz['large']['fp32_mean_iou']:.6f} | FP16 {sz['large']['fp16_mean_iou']:.6f} | Δ {sz['large']['delta_fp16_minus_fp32']:+.6f}

## Quality Decomposition (engineering approximation, nonlinear decoder)
- total gap (baseline - FP16) = {qd['total_gap_baseline_minus_FP16']:+.6f}
- FP16 precision gap (FP32 - FP16) = {qd['FP16_precision_gap_FP32_minus_FP16']:+.6f}
- training/model gap (baseline - FP32) = {qd['training_model_gap_baseline_minus_FP32']:+.6f}
- fraction of total gap attributable to FP16 = {qd['fraction_of_total_gap_attributed_to_FP16']*100:.1f}% (approx)

## Gate
- FP16_QUALITY_IMPACT = **{impact}** (|delta|={ab:.6f})
- Main conclusion = **{concl}** | recommendation (advisory) = **{rec['recommendation']}**

## Key takeaway
Encoder embedding numerical parity (5B-1 cosine 0.99985) does NOT by itself prove
downstream mask-quality parity; this stage measures that mask quality directly.
Result: FP16 cost on downstream mIoU is {impact.lower()} ({ad:+.6f}); the self-trained
regression vs baseline is dominated by the training/model gap, not FP16.
"""
(EVID/"stage_summary.md").write_text(md)
print("[wrote] stage_summary.md")
print("="*60)
print(f"FP32 mIoU={fp32_m:.9f} FP16 mIoU={fp16_m:.9f} FP16delta={ad:+.9f}")
print(f"IMPACT={impact} CONCLUSION={concl} REC={rec['recommendation']}")
print(f"size delta small={sz['small']['delta_fp16_minus_fp32']:+.6f} med={sz['medium']['delta_fp16_minus_fp32']:+.6f} large={sz['large']['delta_fp16_minus_fp32']:+.6f}")
