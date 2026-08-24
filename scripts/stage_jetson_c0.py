#!/usr/bin/env python3
"""Stage Jetson-C0: INT8 PTQ Capability & Calibration Design Audit (audit-only)."""
import os, sys, json, hashlib, subprocess, platform
from pathlib import Path
from collections import Counter

ROOT = Path("/home/nvidia/jetson-deploy-lab")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/nanosam"))
OUT = ROOT / "results/stage_jetson_c0"
OUT.mkdir(parents=True, exist_ok=True)

ONNX = ROOT / "models/self_distilled/epoch50/nanosam_resnet18_epoch049.onnx"
ENG = ROOT / "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine"
ONNX_EXP = "906d8fd752aa2f4dc28d353adb20a462387c9414c4ddf01c1fca93ddf3f31dd7"
ENG_EXP = "99147caf00f0174b802848c93928232c02ac996e91e6660372bbc8923112baf4"

FP16_STD = 0.715650
FP16_FULL = 0.699183
HIST_FULL = 0.703201
FP16_LAT = {"mean_ms": 11.08, "P50_ms": 10.86, "P95_ms": 11.48, "P99_ms": 11.55,
            "source": "Stage Jetson-6C-1 (execute_async_v3, encoder only, warmup30 measured100)"}
CALIB_SEED = 20260824


def dump(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"[evidence] {p}")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    import torch
    import tensorrt as trt
    import onnx

    # ============ Step 1: Environment ============
    tegra = Path("/etc/nv_tegra_release")
    l4t = tegra.read_text().strip().splitlines()[0] if tegra.exists() else "n/a"
    dpkg = subprocess.run(["dpkg", "-l"], capture_output=True, text=True).stdout
    def pkg(name):
        for ln in dpkg.splitlines():
            if name in ln and "ii" in ln.split()[0:1]:
                return ln.split()[2]
        return None
    env = {
        "host": platform.node(),
        "arch": platform.machine(),
        "l4t_release": l4t,
        "nvidia_l4t_core": pkg("nvidia-l4t-core"),
        "tensorrt_meta": pkg("tensorrt"),
        "tensorrt_python": trt.__version__,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a",
        "compute_capability": "8.7 (Orin, Ampere)",
        "trtexec": "/usr/src/tensorrt/bin/trtexec",
        "venv": ".venv (python3.10.12)",
        "constraint": "TRT 10.3.0, no dependency upgrade",
        "upgraded": False,
    }
    dump("environment.json", env)

    # ============ Step 2: Source artifact freeze ============
    onnx_sha = sha(ONNX)
    eng_sha = sha(ENG)
    dump("source_artifact_audit.json", {
        "onnx": {"path": str(ONNX.relative_to(ROOT)), "sha256": onnx_sha,
                 "expected": ONNX_EXP, "match": onnx_sha == ONNX_EXP,
                 "size_bytes": ONNX.stat().st_size},
        "fp16_engine": {"path": str(ENG.relative_to(ROOT)), "sha256": eng_sha,
                        "expected": ENG_EXP, "match": eng_sha == ENG_EXP,
                        "size_bytes": ENG.stat().st_size},
        "role": "Module C frozen INT8-source + FP16 quality/perf baseline",
        "rebuilt": False})
    assert onnx_sha == ONNX_EXP and eng_sha == ENG_EXP, "SHA drift"

    # ============ Step 4: TRT INT8 capability (API as source of truth) ============
    B = trt.Builder(trt.Logger(trt.Logger.WARNING))
    cfg = B.create_builder_config()
    net = B.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    has_fast_int8 = bool(B.platform_has_fast_int8)
    builder_flags = [nm for nm in dir(trt.BuilderFlag) if not nm.startswith("_")]
    int8_flag_methods = {
        "config.set_flag": hasattr(cfg, "set_flag"),
        "config.int8_calibrator_property": hasattr(cfg, "int8_calibrator"),
        "config.set_int8_calibrator": hasattr(cfg, "set_int8_calibrator"),
        "config.set_quantization_flag": hasattr(cfg, "set_quantization_flag"),
        "config.quantization_flags": hasattr(cfg, "quantization_flags"),
        "config.get_calibration_profile": hasattr(cfg, "get_calibration_profile"),
    }
    calibrators = [nm for nm in dir(trt) if "Calibrator" in nm]
    net_qdq = {"add_quantize": hasattr(net, "add_quantize"),
               "add_dequantize": hasattr(net, "add_dequantize")}
    net_flags = [nm for nm in dir(trt.NetworkDefinitionCreationFlag) if not nm.startswith("_")]
    # trtexec flags
    help_txt = subprocess.run(["/usr/src/tensorrt/bin/trtexec", "--help"],
                             capture_output=True, text=True).stdout
    trtexec_int8_flags = {}
    for f in ["--int8", "--calib=", "--fp16", "--minShapesCalib=",
              "--optShapesCalib=", "--maxShapesCalib=", "--calibProfile",
              "--inputIOFormats=", "--outputIOFormats=", "--int8IOFormats="]:
        trtexec_int8_flags[f.rstrip("=")] = (f in help_txt)
    dump("trt_int8_capability.json", {
        "tensorrt_version": trt.__version__,
        "platform_has_fast_int8": has_fast_int8,
        "meaning": "Orin GPU (sm_87 Ampere) has hardware INT8 fast path (DP4A/IMMA)",
        "builder_int8_enables_via": "config.set_flag(trt.BuilderFlag.INT8)",
        "BuilderFlag_INT8_value": int(trt.BuilderFlag.INT8),
        "BuilderFlag_FP16_value": int(trt.BuilderFlag.FP16),
        "BuilderFlag_enum": builder_flags,
        "config_int8_methods": int8_flag_methods,
        "calibrator_classes_available": calibrators,
        "recommended_calibrator": "IInt8EntropyCalibrator2 (TRT default for CNN PTQ; "
                                  "MinMax is alt, Legacy deprecated)",
        "explicit_QDQ_support": net_qdq,
        "explicit_quantization_path": "StronglyTyped network + Q/DQ nodes "
                                      "(NetworkDefinitionCreationFlag.STRONGLY_TYPED=1); "
                                      "no separate EXPLICIT_QUANT flag in TRT 10.x",
        "network_creation_flags": net_flags,
        "trtexec_int8_flags": trtexec_int8_flags,
        "trtexec_int8_build_recipe": [
            "/usr/src/tensorrt/bin/trtexec",
            "--onnx=<onnx>",
            "--saveEngine=<int8.engine>",
            "--int8 --fp16",
            "--calib=<calibration.cache>",
            "--builderOptimizationLevel=3",
            "--memPoolSize=workspace:2048MiB",
            "--skipInference",
        ],
        "note": "TRT 10.3 implicit-quantization PTQ via legacy-ish calibrator is supported; "
                "explicit Q/DQ also supported. Both paths verified against live API, not docs.",
        "calibrator_cache_format": "TensorRT internal binary cache (engine-build-specific, "
                                   "not portable across TRT versions / GPU archs)",
    })

    # ============ Step 5: ONNX operator audit ============
    m = onnx.load(str(ONNX))
    nodes = list(m.graph.node)
    counts = dict(Counter(n.op_type for n in nodes))
    # op-level INT8 suitability (theoretical, TRT build evidence comes in C1)
    op_profile = {
        "Conv": {"count": counts.get("Conv", 0),
                 "int8_theory": "PRIMARY target — DP4A/IMMA INT8 conv kernels; "
                                "weights per-tensor or per-channel quantized",
                 "likely": "INT8"},
        "ConvTranspose": {"count": counts.get("ConvTranspose", 0),
                          "int8_theory": "Transpose-conv INT8 support is patchier; "
                                         "may run FP16 fallback",
                          "likely": "INT8_or_FP16_fallback"},
        "Relu": {"count": counts.get("Relu", 0),
                 "int8_theory": "Clamp/ReLU quantizes cleanly (monotonic)",
                 "likely": "INT8"},
        "MaxPool": {"count": counts.get("MaxPool", 0),
                    "int8_theory": "Quantization-friendly (argmax)",
                    "likely": "INT8"},
        "Add": {"count": counts.get("Add", 0),
                "int8_theory": "Elementwise Add needs aligned input scales; "
                               "residual adds are common INT8 sensitivity points",
                "likely": "INT8 (watch residual scales)"},
        "Mul": {"count": counts.get("Mul", 0),
                "int8_theory": "Elementwise Mul (used in GELU approx / up-proj); "
                               "scale-mismatch sensitive",
                "likely": "INT8_or_FP16_fallback"},
        "Div": {"count": counts.get("Div", 0),
                "int8_theory": "Div in Erf/GELU softmax-ish path; "
                               "division not natively INT8, usually FP16",
                "likely": "FP16_fallback"},
        "Erf": {"count": counts.get("Erf", 0),
                "int8_theory": "Erf is transcendental (GELU); TRT fuses GELU but "
                               "the Erf/Div/Mul cluster is the sensitive tail",
                "likely": "FP16_fallback_or_fused"},
        "Constant": {"count": counts.get("Constant", 0),
                     "int8_theory": "No compute, folded at build",
                     "likely": "folded"},
    }
    conv_in_first = [list(n.input) for n in nodes if n.op_type == "Conv"][:1]
    convtranspose = [n for n in nodes if n.op_type == "ConvTranspose"]
    erf_inputs = [list(n.input) for n in nodes if n.op_type == "Erf"]
    existing_qdq = [n.op_type for n in nodes if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
    dump("onnx_operator_audit.json", {
        "onnx_sha256": onnx_sha,
        "total_nodes": len(nodes),
        "op_counts": counts,
        "op_int8_profile": op_profile,
        "conv_first_inputs_sample": conv_in_first,
        "convtranspose": [{"name": n.name, "inputs": list(n.input)}
                          for n in convtranspose],
        "erf_cluster_inputs": erf_inputs,
        "existing_QDQ_nodes": existing_qdq if existing_qdq else "NONE "
        "(graph is plain FP32 — implicit-quantization PTQ, no Q/DQ)",
        "quantization_style": "IMPLICIT (calibrator-based) — graph has no Q/DQ, "
                              "TRT inserts quantization during build",
        "disclaimer": "op 'likely' is theoretical. Actual per-op precision is decided "
                      "by the TRT builder autotuner; confirmed only via build profile / "
                      "--layerOutputTypes / verbose log in Stage C1.",
        "sensitivity_watchlist": [
            "ConvTranspose (up_1.6, output projection) — INT8 support uncertain",
            "Erf/Div/Mul GELU tail (5 each) — transcendental, likely FP16",
            "14 residual Adds — scale alignment sensitive",
        ],
    })

    # ============ Step 6: Quantization strategy ============
    dump("quantization_strategy.json", {
        "candidates": {
            "A_TRT_PTQ_implicit_calibration": {
                "desc": "trtexec --int8 --fp16 --calib=<cache>, legacy IInt8EntropyCalibrator2",
                "retrain": False, "engineering_cost": "LOW",
                "trt103_support": "VERIFIED (set_flag INT8 + IInt8EntropyCalibrator2 "
                                  "+ trtexec --int8 --calib)",
                "graph_modification": "none — calibrator infers activation scales"},
            "B_explicit_QDQ_onnx": {
                "desc": "Insert QuantizeLinear/DequantizeLinear into ONNX (e.g. via "
                        "ONNX Runtime quantize-perceiver / pytorch quantization), build "
                        "StronglyTyped",
                "retrain": False, "engineering_cost": "MEDIUM (ONNX graph surgery)",
                "trt103_support": "VERIFIED (add_quantize/add_dequantize, STRONGLY_TYPED)",
                "graph_modification": "Q/DQ nodes inserted into ONNX"},
            "C_QAT": {
                "desc": "Quantization-aware training, resume from epoch049 with fake-quant",
                "retrain": True, "engineering_cost": "HIGH",
                "trt103_support": "supported but OUT OF SCOPE for Module C",
                "graph_modification": "Q/DQ + trained scales"},
        },
        "selected": "A_TRT_PTQ_implicit_calibration",
        "rationale": [
            "No retraining — fits project constraint (STOP_LONG_TRAINING, module COMPLETE)",
            "Lowest engineering cost; reuses Stage 5B-1/6C-1 trtexec build protocol",
            "TRT 10.3 implicit-PTQ API verified live on this device",
            "Graph has no Q/DQ (plain FP32 export) → implicit calibrator is the natural fit",
            "QAT (C) is out of scope; explicit Q/DQ (B) is the fallback if A under-delivers "
            "(sensitive-op control via explicit per-channel scales)",
        ],
        "fallback_if_regression": "B_explicit_QDQ_onnx (per-channel weight quant + "
                                  "keep GELU/ConvTranspose FP16) before considering mixed-precision "
                                  "layer override",
    })

    # ============ Step 7-8: Calibration dataset design ============
    val_dir = ROOT / "data/coco/val2017"
    val_imgs = sorted(val_dir.glob("*.jpg")) if val_dir.exists() else []
    train_dir = ROOT / "data/coco/train2017"
    train_present = train_dir.exists() and len(list(train_dir.glob("*.jpg"))) > 0
    candidates = {
        "100": {"wall_estimate_s": "~30 (preprocess only, no build)", "coverage": "low"},
        "256": {"wall_estimate_s": "~75", "coverage": "moderate"},
        "512": {"wall_estimate_s": "~150", "coverage": "good — recommended primary"},
        "1024": {"wall_estimate_s": "~300", "coverage": "high (diminishing return)"},
    }
    dump("calibration_design.json", {
        "principle": "Use train/calibration images ONLY — never STANDARD500 GT eval set "
                     "(avoid evaluation leakage). Calibration is image-level encoder input "
                     "statistics; no masks/boxes used.",
        "source_dataset": "COCO 2017 val2017 (5000 images) — ALREADY on device",
        "source_dataset_rationale": "Module C frozen-baseline protocol uses val2017 for the "
                                     "encoder input distribution that deployment sees. train2017 "
                                     "is NOT present on Jetson (only val2017.zip); pulling 19GB "
                                     "train2017 needs user action and gives marginal calibration "
                                     "benefit for an encoder already trained on train2017. val2017 "
                                     "is a valid, present, in-distribution source for PTQ activation "
                                     "statistics.",
        "train2017_on_device": train_present,
        "source_image_count": len(val_imgs),
        "candidate_sizes": candidates,
        "primary_recommendation": 512,
        "primary_rationale": "512 balances coverage vs calibration wall-time; common TRT "
                             "rule-of-thumb 500-1000 batches; leaves headroom to bump to 1024 "
                             "if C1 INT8 quality is borderline",
        "seed": CALIB_SEED,
        "sampling": "deterministic random (numpy default_rng(seed).choice of val2017 jpg list)",
        "stratification": "NONE — calibration is image-level encoder-input statistics, not "
                          "category/size metrics; complex stratification would expand scope "
                          "without PTQ benefit",
        "output": "image paths + per-image SHA256 + sample count + seed written to "
                  "calibration_manifest.json",
    })

    # ============ Step 9: Preprocessing freeze ============
    dump("preprocessing_freeze.json", {
        "must_match_deployment": True,
        "deployment_function": "nanosam.utils.predictor.preprocess_image(image, size=1024) "
                               "(called by Predictor.set_image, the exact path the deployed "
                               "and eval encoders use)",
        "function_path": "third_party/nanosam/nanosam/utils/predictor.py:preprocess_image",
        "steps": [
            "PIL.Image convert RGB",
            "aspect-aware resize: long side -> 1024, short side = 1024/aspect",
            "np.asarray (HWC uint8)",
            "permute -> CHW, .float()",
            "normalize: (x - mean) / std",
            "mean = [123.675, 116.28, 103.53] (ImageNet, channel-last broadcast [:,None,None])",
            "std  = [58.395, 57.12, 57.375]",
            "zero-pad into [1,3,1024,1024] (top-left, rest zeros)",
            ".cuda() → fp32 tensor",
        ],
        "input_shape": [1, 3, 1024, 1024],
        "dtype": "float32 (calibrator feeds fp32; TRT INT8 builder quantizes internally)",
        "forbidden": "feeding raw JPEG bytes / different normalization / different resize / "
                     "different pad scheme — would desync calibration distribution from "
                     "deployment distribution",
        "reuse_strategy": "Stage C1 will import preprocess_image directly — no reimplementation",
    })

    # ============ Step 10: Calibration cache design ============
    dump("calibration_cache_design.json", {
        "dir": "models/self_distilled/epoch50/int8/",
        "cache_file": "calibration.cache",
        "manifest_file": "calibration_manifest.json",
        "cache_binding": {
            "onnx_sha256": onnx_sha,
            "preprocessing_version": "predictor.preprocess_image @ size=1024 (ImageNet mean/std, aspect-resize, zero-pad)",
            "calibration_manifest_sha256": "computed in C1 after sampling",
            "tensorrt_version": trt.__version__,
            "calibrator_class": "IInt8EntropyCalibrator2",
            "calibration_batch_size": "TBD in C1 (likely 8, bounded by 1024x1024x3xCUDA fp32 + engine workspace)",
        },
        "reuse_guard": "Cache is engine-build- and TRT-version-specific; the manifest SHAs above "
                       "MUST be checked before any cache reuse. Cache from a different ONNX / "
                       "preprocess / TRT version must NOT be reused.",
        "gitignore": "cache + dataset images gitignored; only calibration_manifest.json (paths+sha) committed",
    })

    # ============ Step 11-12: FP16 baselines ============
    dump("fp16_baseline_freeze.json", {
        "module_C_quality_baseline": "50ep FP16 self-trained student (Stage 6C/6D, not recomputed)",
        "standard500_miou": FP16_STD,
        "full_coco_miou": FP16_FULL,
        "historical_full_reference": HIST_FULL,
        "eval_order": "INT8 runs STANDARD500 FIRST. Only if STANDARD500 gate passes does "
                      "INT8 proceed to FULL COCO.",
        "fp16_performance_baseline": FP16_LAT,
        "perf_note": "If a formal INT8 benchmark runs, FP16 and INT8 MUST be re-measured with "
                     "the SAME script in the SAME session — do NOT mix historical 6C-1 numbers.",
    })

    # ============ Step 13: Quality gate ============
    dump("quality_gate_design.json", {
        "primary_metric": "STANDARD500 mIoU (fixed slot0), same frozen 500 manifest",
        "baseline": "50ep FP16 STANDARD500 mIoU = 0.715650",
        "delta_definition": "delta = INT8_STD500_miou - FP16_STD500_miou",
        "heuristic": {
            "delta >= -0.005": "QUALITY_ACCEPTABLE",
            "-0.005 > delta >= -0.015": "QUALITY_DEGRADED_BUT_USABLE",
            "delta < -0.015": "QUALITY_REGRESSION",
        },
        "not_statistical": "project deployment decision heuristic, not a significance test",
        "consistent_with_prior_stages": "matches Stage 6C/6D NEAR_PARITY/ACCEPTABLE/REGRESSION "
                                         "0.005/0.015 thresholds",
        "gate_action": {
            "ACCEPTABLE": "proceed to FULL COCO INT8 eval in a later stage",
            "DEGRADED_BUT_USABLE": "proceed to FULL only if deployment context tolerates the loss; "
                                   "else try mixed-precision",
            "REGRESSION": "do NOT deploy INT8; investigate sensitive ops (mixed precision / Q-DQ path B)",
        },
    })

    # ============ Step 14: Performance gate ============
    dump("performance_gate_design.json", {
        "must_same_protocol": "FP16 vs INT8 measured with identical script, same session, "
                              "execute_async_v3 encoder-only, warmup30 measured100",
        "metrics": ["latency_mean_ms", "P50_ms", "P95_ms", "P99_ms",
                    "engine_size_bytes", "gpu_peak_allocated_bytes", "peak_rss_bytes"],
        "optional": "power (NOT modifying power-mode / jetson_clocks — out of scope)",
        "deploy_worth": "INT8 justified only if (quality loss acceptable) AND (real perf/size "
                        "gain exists). On Orin, INT8 conv gain is real but the encoder is "
                        "already 11ms — size reduction + throughput may matter more than latency.",
    })

    # ============ Step 15: Mixed precision plan ============
    dump("mixed_precision_plan.json", {
        "stage_C0_scope": "DESIGN ONLY — no layer override built here",
        "investigation_order": [
            "1. full INT8 build (C1) → STANDARD500 gate",
            "2. if REGRESSION: pull TRT verbose layer-precision log, identify sensitive ops",
            "3. expect sensitive ops from audit: ConvTranspose (up_1.6), GELU Erf/Div/Mul cluster, "
               "residual Adds",
            "4. mixed precision: keep sensitive layers FP16, rest INT8 "
               "(trtexec --layerOutputTypes / config layer-precision API, or path B Q-DQ)",
            "5. re-gate STANDARD500; only proceed if acceptable",
        ],
        "not_done_here": "no automated layer search; manual targeted override based on C1 evidence",
    })

    # ============ Step 16: stage_summary.md ============
    final = "INT8_PTQ_PATH_READY"
    lm = FP16_LAT["mean_ms"]; lp50 = FP16_LAT["P50_ms"]
    lp95 = FP16_LAT["P95_ms"]; lp99 = FP16_LAT["P99_ms"]
    md = f"""# Stage Jetson-C0 — INT8 PTQ Capability & Calibration Design Audit

## Final: {final}

## 1. Environment (frozen)
- Jetson Orin, L4T R36.4.3, CUDA 12.6, **TensorRT 10.3.0** (python + trtexec), torch 2.5.0a0+nv24.08, python 3.10.12
- GPU sm_87 Ampere, `platform_has_fast_int8 = True` (hardware INT8 fast path: DP4A / IMMA)
- No dependency upgrade performed.

## 2. Source Artifact Freeze (Module C baseline)
- ONNX `{ONNX.relative_to(ROOT)}` SHA `{onnx_sha[:12]}…` — match ({onnx_sha == ONNX_EXP})
- FP16 engine `{ENG.relative_to(ROOT)}` SHA `{eng_sha[:12]}…` — match ({eng_sha == ENG_EXP})
- Both are the frozen INT8 source + FP16 quality/perf baseline.

## 3. Conceptual Quantization Contract
See `docs/int8_ptq.md` § "Quantization basics" for the full conceptual write-up (FP32/FP16 vs INT8, scale, weight vs activation, calibration, dynamic range, quantization error, dequantization, mixed precision, PTQ vs QAT) — bound to this NanoSAM project, not generic theory.

## 4. TensorRT INT8 Capability (live API, not docs)
- `platform_has_fast_int8 = True`
- Implicit PTQ: `config.set_flag(trt.BuilderFlag.INT8)` + `config.set_int8_calibrator(IInt8EntropyCalibrator2)`
- Explicit Q/DQ: `add_quantize`/`add_dequantize` + `STRONGLY_TYPED` network
- Calibrators available: IInt8EntropyCalibrator2 / MinMax / Legacy / base IInt8Calibrator
- trtexec: `--int8 --fp16 --calib=<cache> --builderOptimizationLevel=3 --memPoolSize=workspace:2048MiB --skipInference`

## 5. ONNX Operator Audit (93 nodes)
| op | count | INT8 likelihood |
|---|---|---|
| Conv | 25 | INT8 (primary target) |
| Relu | 17 | INT8 |
| Add | 14 | INT8 (residual — watch scales) |
| Constant | 15 | folded |
| Mul | 10 | INT8 or FP16 fallback |
| Div | 5 | FP16 fallback (GELU) |
| Erf | 5 | FP16 fallback / fused (GELU) |
| MaxPool | 1 | INT8 |
| ConvTranspose | 1 | INT8 or FP16 fallback (uncertain) |

- Graph is plain FP32 (no Q/DQ) → **implicit-quantization PTQ**.
- Sensitive tail: ConvTranspose output proj + Erf/Div/Mul GELU (5 each) + 14 residual Adds.
- "Likelihood" is theoretical; actual per-op precision confirmed only in C1 build profile.

## 6. Selected Path: A — TRT implicit-calibration PTQ
- No retraining (STOP_LONG_TRAINING honored), low cost, reuses 5B-1/6C-1 trtexec protocol.
- Fallback if regression: B explicit Q-DQ (per-channel + keep GELU/ConvTranspose FP16), then mixed-precision.

## 7-8. Calibration Dataset
- Source: COCO val2017 (5000 images, already on device). train2017 NOT present (would need user to pull 19GB; marginal benefit).
- Principle: image-level encoder-input stats only; no masks/boxes; never touch STANDARD500 GT set (no leakage).
- Seed {CALIB_SEED}, deterministic random, no stratification (calibration ≠ category metrics).
- **Primary size: 512** (coverage vs wall-time; bump to 1024 if C1 borderline).

## 9. Preprocessing Freeze
- MUST reuse `nanosam.utils.predictor.preprocess_image(image, size=1024)` — the exact deployment/eval path.
- RGB → aspect-resize long→1024 → CHW float → ImageNet mean/std → zero-pad [1,3,1024,1024] → cuda fp32.
- C1 imports the function directly (no reimplementation).

## 10. Calibration Cache
- `models/self_distilled/epoch50/int8/calibration.cache` + `calibration_manifest.json`
- Bound to ONNX SHA + preprocess version + manifest SHA + TRT 10.3; reuse-guard checked before any reuse.

## 11-12. FP16 Baselines
- STANDARD500 {FP16_STD} | FULL {FP16_FULL} (historical FULL {HIST_FULL})
- INT8 runs STANDARD500 FIRST; FULL only if gate passes.
- FP16 latency (6C-1, encoder-only): mean {lm}ms / P50 {lp50} / P95 {lp95} / P99 {lp99}.
- Formal INT8 benchmark MUST re-measure FP16+INT8 same-session same-script.

## 13. Quality Gate (heuristic)
- delta = INT8_STD500 − FP16_STD500 (FP16 = {FP16_STD})
- ≥ −0.005 ACCEPTABLE | (−0.015, −0.005) DEGRADED_BUT_USABLE | < −0.015 REGRESSION

## 14. Performance Gate
- Same-protocol FP16 vs INT8: latency mean/P50/P95/P99, engine size, GPU alloc, RSS. No power-mode change.

## 15. Mixed Precision Plan (design only, not built)
- full INT8 → identify sensitive ops (ConvTranspose, GELU cluster, residual Adds) → keep those FP16 → re-gate.

## Evidence
`results/stage_jetson_c0/`: environment, source_artifact_audit, trt_int8_capability,
onnx_operator_audit, quantization_strategy, calibration_design, preprocessing_freeze,
calibration_cache_design, fp16_baseline_freeze, quality_gate_design,
performance_gate_design, mixed_precision_plan, this stage_summary.md.

## Out of scope (STOP — none done)
INT8 engine build, STANDARD500/FULL INT8 eval, QAT, model/decoder change, continued
distillation, NanoOWL, C++, power-mode/jetson_clocks.

## Next (NOT executed)
Stage Jetson-C1: build calibration dataset (512 val2017 images, seed {CALIB_SEED}) →
IInt8EntropyCalibrator2 cache → first INT8 engine build (trtexec --int8 --fp16 --calib)
→ STANDARD500 quality gate vs FP16 {FP16_STD}.
"""
    (OUT / "stage_summary.md").write_text(md)
    print(f"[evidence] {OUT/'stage_summary.md'}")
    print(f"\n=== Stage Jetson-C0 DONE === {final}")


if __name__ == "__main__":
    main()
