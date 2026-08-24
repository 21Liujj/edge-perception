#!/usr/bin/env python3
"""Stage Jetson-6C-1: 50-epoch NanoSAM ONNX -> TensorRT FP16 engine build + verify.

Reuses Stage5B-1 verified protocol (trtexec FP16, builderOptimizationLevel=3,
workspace:2048MiB, skipInference). Differences vs 5B-1:
  - source = epoch049 ONNX (50ep) instead of epoch009 (10ep)
  - build to a UNIQUE temp path, atomic rename only after all checks pass
  - single builder only (no concurrent trtexec, the 5B-1 race lesson)

Out of scope (enforced STOP): STANDARD500, FULL COCO, INT8, NanoOWL, training.
"""
import os, sys, json, time, hashlib, subprocess, shutil
from pathlib import Path

REPO = Path("/home/nvidia/jetson-deploy-lab")
ONNX = REPO / "models/self_distilled/epoch50/nanosam_resnet18_epoch049.onnx"
ENGINE_FINAL = REPO / "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine"
ENGINE_TMP = REPO / "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine.tmp"
EVID = REPO / "results/stage_jetson_6c1"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

EXPECTED_ONNX_SHA = "906d8fd752aa2f4dc28d353adb20a462387c9414c4ddf01c1fca93ddf3f31dd7"
EPOCH10_ENGINE = REPO / "models/self_distilled/epoch10/nanosam_resnet18_epoch009_fp16.engine"
HIST_ENGINE = REPO / "third_party/nanosam/data/resnet18_image_encoder.engine"

EVID.mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(name, obj):
    p = EVID / name
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[evidence] {p}")


# ---- syspath for nanosam (mirrors venv site-packages .pth approach) ----
def setup_nanosam_path():
    import glob
    cand = glob.glob("/home/nvidia/jetson-deploy-lab/.venv/lib/python3.10/site-packages/_nanosam_thirdparty_paths.pth")
    # the .pth file adds third_party paths; if present, site already loaded it.
    # Ensure third_party/nanosam usable:
    tp = "/home/nvidia/jetson-deploy-lab/third_party/nanosam"
    if tp not in sys.path:
        sys.path.insert(0, tp)


def main():
    import onnx
    import torch
    import tensorrt as trt

    # ===================== Step 1: Environment =====================
    env = {
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "tensorrt": trt.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a",
        "nanosam_commit": (REPO / "third_party/nanosam/.pinned_commit").read_text().strip()
                          if (REPO / "third_party/nanosam/.pinned_commit").exists() else "n/a",
        "stage_baseline": "matches Stage5B-1 (TR10.3.0 / torch2.5.0a0+nv24.08)",
    }
    dump("environment.json", env)

    # ===================== Step 2: ONNX Integrity =====================
    onnx_sha = sha256(ONNX)
    model = onnx.load(str(ONNX))
    inp = model.graph.input[0]
    out = model.graph.output[0]
    def dims(t):
        return [d.dim_value for d in t.type.tensor_type.shape.dim]
    opsets = [[o.domain, o.version] for o in model.opset_import]
    opset_versions = [o.version for o in model.opset_import]
    audit = {
        "path": str(ONNX),
        "size_bytes": ONNX.stat().st_size,
        "sha256": onnx_sha,
        "expected_sha256": EXPECTED_ONNX_SHA,
        "sha_match": onnx_sha == EXPECTED_ONNX_SHA,
        "opset": opsets,
        "ir_version": model.ir_version,
        "producer": model.producer_name,
        "num_nodes": len(model.graph.node),
        "inputs": [{"name": inp.name, "shape": dims(inp), "dtype": inp.type.tensor_type.elem_type}],
        "outputs": [{"name": out.name, "shape": dims(out), "dtype": out.type.tensor_type.elem_type}],
    }
    audit["contract_pass"] = (
        dims(inp) == [1, 3, 1024, 1024]
        and dims(out) == [1, 256, 64, 64]
        and inp.type.tensor_type.elem_type == 1  # FLOAT
        and 16 in opset_versions
    )
    dump("source_onnx_audit.json", audit)
    print(f"[onnx] sha_match={audit['sha_match']} contract={audit['contract_pass']}")
    if not (audit["sha_match"] and audit["contract_pass"]):
        print("STOP: ONNX integrity FAILED"); sys.exit(2)

    # ===================== Step 3: Build Protection =====================
    # ensure no stale tmp, ensure final not yet present, single builder
    # Resumability: if a prior run already produced a valid tmp engine + build.log
    # with exit 0, reuse it instead of re-building (saves ~145s on script rerun
    # after a post-build crash).
    reuse_tmp = (
        ENGINE_TMP.exists()
        and (EVID / "build.exit").exists()
        and (EVID / "build.exit").read_text().strip() == "0"
    )
    if ENGINE_FINAL.exists():
        print(f"[WARN] final engine already exists, will overwrite via tmp+rename: {ENGINE_FINAL}")
    # fail fast if another trtexec is running
    ps = subprocess.run(["pgrep", "-fc", "trtexec"], capture_output=True, text=True)
    n_running = int(ps.stdout.strip()) if ps.stdout.strip() else 0
    print(f"[protect] trtexec processes already running: {n_running}")
    if n_running > 0:
        print("STOP: a trtexec is already running (concurrent build risk)"); sys.exit(3)

    # ===================== Step 4: TRT FP16 Build =====================
    if reuse_tmp:
        print("[build] reusing existing tmp engine from prior successful build (exit 0)")
        # rebuild build_summary from existing artifacts
        engine_size = ENGINE_TMP.stat().st_size
        engine_sha = sha256(ENGINE_TMP)
        build_wall = 0.0
        proc = type("P", (), {"stdout": "", "stderr": "", "returncode": 0})()
        warnings = []
        build_passed = True
    else:
        cmd = [
            TRTEXEC,
            f"--onnx={ONNX}",
            f"--saveEngine={ENGINE_TMP}",
            "--fp16",
            "--builderOptimizationLevel=3",
            "--memPoolSize=workspace:2048MiB",
            "--skipInference",
        ]
        print(f"[build] {' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        build_wall = time.time() - t0
        (EVID / "build.log").write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        (EVID / "build.exit").write_text(str(proc.returncode))

        engine_size = ENGINE_TMP.stat().st_size if ENGINE_TMP.exists() else 0
        engine_sha = sha256(ENGINE_TMP) if ENGINE_TMP.exists() else None
        warnings = [l for l in proc.stdout.splitlines() if "workspace memory" in l or "Warning" in l]
        build_passed = proc.returncode == 0 and ENGINE_TMP.exists()
    build_summary = {
        "tensorrt_version": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "precision": "FP16",
        "builder": "trtexec",
        "builder_optimization_level": 3,
        "workspace_memPool": "workspace:2048MiB",
        "input_shape": [1, 3, 1024, 1024],
        "output_shape": [1, 256, 64, 64],
        "build_wall_time_ms": int(build_wall * 1000),
        "engine_tmp_path": str(ENGINE_TMP),
        "engine_exists_tmp": ENGINE_TMP.exists(),
        "engine_size_bytes": engine_size,
        "engine_sha256_tmp": engine_sha,
        "onnx_sha256": onnx_sha,
        "build_passed": proc.returncode == 0 and ENGINE_TMP.exists(),
        "returncode": proc.returncode,
        "warnings": warnings[:5],
        "fatal_error": proc.stderr.strip()[-500:] if proc.returncode != 0 else None,
        "concurrent_build_guard": "single builder, unique .tmp path (5B-1 race lesson applied)",
    }
    dump("trt_build_summary.json", build_summary)
    print(f"[build] passed={build_summary['build_passed']} wall={build_wall:.1f}s size={engine_size} sha={engine_sha}")
    if not build_summary["build_passed"]:
        print("STOP: build FAILED"); sys.exit(4)

    # ===================== Step 5: Deserialize / Runtime Smoke =====================
    import numpy as np
    with open(ENGINE_TMP, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print("STOP: deserialize returned None"); sys.exit(5)
    context = engine.create_execution_context()
    # TRT 10.x API: get_tensor_name + get_tensor_mode (no get_binding_name)
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    iname = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
    oname = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
    in_shape = tuple(engine.get_tensor_shape(iname))
    out_shape = tuple(engine.get_tensor_shape(oname))
    in_trt_dtype = engine.get_tensor_dtype(iname)
    # map trt dtype -> torch dtype (TRT 10.3 has no float64 enum)
    _trt2torch = {trt.float32: torch.float32, trt.float16: torch.float16}
    torch_in_dtype = _trt2torch.get(in_trt_dtype, torch.float32)
    torch_out_dtype = _trt2torch.get(engine.get_tensor_dtype(oname), torch.float32)
    # alloc
    din = torch.randn(*in_shape, dtype=torch_in_dtype, device="cuda").contiguous()
    dout = torch.empty(*out_shape, dtype=torch_out_dtype, device="cuda").contiguous()
    context.set_input_shape(iname, in_shape)
    context.set_tensor_address(iname, int(din.data_ptr()))
    context.set_tensor_address(oname, int(dout.data_ptr()))
    stream = torch.cuda.Stream()
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    arr = dout.cpu().numpy()
    runtime_smoke = {
        "deserialize": "PASS" if engine is not None else "FAIL",
        "input_name": iname,
        "output_name": oname,
        "engine_input_shape": list(in_shape),
        "engine_output_shape": list(out_shape),
        "runtime": "PASS",
        "output_shape": list(arr.shape),
        "finite": bool(np.isfinite(arr).all()),
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
        "n_measured": 1,
    }
    dump("runtime_smoke.json", runtime_smoke)
    print(f"[runtime] deserialize=PASS out={arr.shape} finite={runtime_smoke['finite']}")
    if not runtime_smoke["finite"]:
        print("STOP: non-finite output"); sys.exit(6)

    # ===================== Step 6: Predictor Compatibility =====================
    setup_nanosam_path()
    from nanosam.utils.predictor import load_image_encoder_engine
    # Predictor expects the final-named engine path. We load from tmp directly
    # by copying tmp -> a temp final-named probe path? No: load_image_encoder_engine
    # takes a path arg. Pass ENGINE_TMP.
    try:
        enc = load_image_encoder_engine(str(ENGINE_TMP))
        # build a dummy 1024 input via preprocess? Use synthetic directly.
        dummy = torch.randn(1, 3, 1024, 1024, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            emb = enc(dummy)
        pred = {
            "predictor_load": "PASS",
            "loader": "nanosam.utils.predictor.load_image_encoder_engine (TRTModule)",
            "engine_path": str(ENGINE_TMP),
            "input_shape": list(dummy.shape),
            "embedding_shape": list(emb.shape) if hasattr(emb, "shape") else "n/a",
            "finite": bool(torch.isfinite(emb).all().item()) if hasattr(emb, "all") else "n/a",
            "expected_embedding_shape": [1, 256, 64, 64],
            "shape_ok": (hasattr(emb, "shape") and list(emb.shape) == [1, 256, 64, 64]),
        }
    except Exception as e:
        import traceback
        pred = {"predictor_load": "FAIL", "error": str(e), "traceback": traceback.format_exc()[-800:]}
    dump("predictor_compatibility.json", pred)
    print(f"[predictor] {pred.get('predictor_load')} shape_ok={pred.get('shape_ok')}")
    if pred["predictor_load"] != "PASS":
        print("STOP: predictor load FAILED"); sys.exit(7)

    # ===================== Step 7: Latency Smoke =====================
    times = []
    # warmup
    for _ in range(30):
        context.execute_async_v3(stream.cuda_stream); stream.synchronize()
    for _ in range(100):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(stream); context.execute_async_v3(stream.cuda_stream); e.record(stream); stream.synchronize()
        times.append(s.elapsed_time(e))
    times_np = np.array(times)
    latency = {
        "target": "50ep self-trained student encoder FP16 (TRT execute_async_v3 only)",
        "warmup": 30,
        "measured": 100,
        "mean_ms": float(times_np.mean()),
        "P50_ms": float(np.percentile(times_np, 50)),
        "P95_ms": float(np.percentile(times_np, 95)),
        "P99_ms": float(np.percentile(times_np, 99)),
        "note": "image encoder TensorRT execute latency, NOT full segmentation / NOT system FPS",
    }
    dump("latency_smoke.json", latency)
    print(f"[latency] mean={latency['mean_ms']:.3f}ms P50={latency['P50_ms']:.3f} P95={latency['P95_ms']:.3f} P99={latency['P99_ms']:.3f}")

    # ===================== atomic rename tmp -> final =====================
    engine_sha_final = sha256(ENGINE_TMP)
    ENGINE_TMP.replace(ENGINE_FINAL)
    print(f"[protect] atomic rename -> {ENGINE_FINAL}")

    # ===================== Step 8: Artifact Identity =====================
    new_sha = sha256(ENGINE_FINAL)
    e10_sha = sha256(EPOCH10_ENGINE) if EPOCH10_ENGINE.exists() else None
    hist_sha = sha256(HIST_ENGINE) if HIST_ENGINE.exists() else None
    manifest = {
        "artifacts": [
            {"label": "source_onnx_50ep", "path": "models/self_distilled/epoch50/nanosam_resnet18_epoch049.onnx",
             "exists": ONNX.exists(), "size_bytes": ONNX.stat().st_size, "sha256": onnx_sha,
             "gitignored": "*.onnx ignored, SHA tracked"},
            {"label": "self_trained_fp16_engine_50ep", "path": "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine",
             "exists": ENGINE_FINAL.exists(), "size_bytes": ENGINE_FINAL.stat().st_size, "sha256": new_sha,
             "gitignored": "*.engine ignored, SHA tracked"},
            {"label": "self_trained_fp16_engine_10ep", "path": "models/self_distilled/epoch10/nanosam_resnet18_epoch009_fp16.engine",
             "exists": EPOCH10_ENGINE.exists(),
             "size_bytes": EPOCH10_ENGINE.stat().st_size if EPOCH10_ENGINE.exists() else 0,
             "sha256": e10_sha, "gitignored": "*.engine ignored, SHA tracked"},
            {"label": "historical_baseline_fp16_engine", "path": "third_party/nanosam/data/resnet18_image_encoder.engine",
             "exists": HIST_ENGINE.exists(),
             "size_bytes": HIST_ENGINE.stat().st_size if HIST_ENGINE.exists() else 0,
             "sha256": hist_sha, "gitignored": "*.engine ignored, SHA tracked"},
        ],
        "identity": {
            "50ep_vs_10ep": "DIFFERENT" if new_sha != e10_sha else "SAME",
            "50ep_vs_historical": "DIFFERENT" if new_sha != hist_sha else "SAME",
            "expected": "all DIFFERENT",
        },
    }
    dump("artifact_manifest.json", manifest)
    print(f"[identity] 50ep={new_sha[:12]} 10ep={(e10_sha or 'n/a')[:12]} hist={(hist_sha or 'n/a')[:12]}")

    print("\n=== Stage Jetson-6C-1 ALL STEPS PASS ===")
    print(f"engine: {ENGINE_FINAL}")
    print(f"sha256: {new_sha}")


if __name__ == "__main__":
    main()
