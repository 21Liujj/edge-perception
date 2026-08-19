#!/usr/bin/env python3
"""Stage Jetson-5B-1: Self-trained NanoSAM ONNX -> TensorRT FP16 build + runtime smoke.

Single-script driver. Expects the FP16 engine already built by trtexec
(see stage_jetson_5b1_build.sh) OR builds nothing if engine missing.
Produces all evidence JSON under results/stage_jetson_5b1/.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import torch

ROOT = Path("/home/nvidia/jetson-deploy-lab")
OUT = ROOT / "results" / "stage_jetson_5b1"
OUT.mkdir(parents=True, exist_ok=True)

ONNX_PATH = ROOT / "models/self_distilled/epoch10/nanosam_resnet18_epoch009.onnx"
ENGINE_PATH = ROOT / "models/self_distilled/epoch10/nanosam_resnet18_epoch009_fp16.engine"
BASELINE_ENGINE = ROOT / "third_party/nanosam/data/resnet18_image_encoder.engine"
BASELINE_ENGINE_FP32 = ROOT / "third_party/nanosam/data/resnet18_image_encoder_fp32.engine"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
SEED = 20260819
INPUT_SHAPE = (1, 3, 1024, 1024)


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(name, obj):
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[wrote] {OUT/name}")


def det_input():
    rng = np.random.default_rng(SEED)
    # deterministic fp32 input in [0,255] range (like a real image)
    return rng.random(INPUT_SHAPE, dtype=np.float32) * 255.0


# ---------------------------------------------------------------- runtime
class TRTRunner:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine is not None, "deserialize failed"
        self.ctx = self.engine.create_execution_context()
        # tensor names
        self.iname = self.engine.get_tensor_name(0) if self.engine.num_io_tensors >= 1 else "image"
        self.oname = self.engine.get_tensor_name(1)
        # try to find image/image_embeddings explicitly
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.iname = n
            elif self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                self.oname = n
        self.d_in = torch.zeros(INPUT_SHAPE, dtype=torch.float32, device="cuda")
        # allocate output buffer using reported shape
        oshape = tuple(self.engine.get_tensor_shape(self.oname))
        self.oshape = oshape
        self.d_out = torch.zeros(oshape, dtype=torch.float32, device="cuda")

    def run(self, x_np):
        self.d_in.copy_(torch.from_numpy(x_np))
        self.ctx.set_input_shape(self.iname, INPUT_SHAPE)
        self.ctx.set_tensor_address(self.iname, int(self.d_in.data_ptr()))
        self.ctx.set_tensor_address(self.oname, int(self.d_out.data_ptr()))
        stream = torch.cuda.current_stream().cuda_stream
        ok = self.ctx.execute_async_v3(stream)
        assert ok, "execute_async_v3 failed"
        torch.cuda.synchronize()
        return self.d_out.detach().cpu().numpy().copy()


def main():
    # ------------------------------------------------------------------ 7-8
    print("=== deserialize + runtime smoke ===")
    runner = TRTRunner(ENGINE_PATH)
    ein = runner.engine.get_tensor_shape(runner.iname)
    eout = runner.oshape
    smoke = {
        "deserialize": "PASS",
        "input_name": runner.iname,
        "output_name": runner.oname,
        "engine_input_shape": [int(d) for d in ein] if list(ein) else list(INPUT_SHAPE),
        "engine_output_shape": [int(d) for d in eout],
    }

    x = det_input()
    # warmup 10, measured 30
    for _ in range(10):
        runner.run(x)
    outs = []
    for _ in range(30):
        outs.append(runner.run(x))
    y = outs[-1]
    finite = bool(np.isfinite(y).all())
    smoke.update({
        "runtime": "PASS",
        "output_shape": list(y.shape),
        "finite": finite,
        "has_nan": bool(np.isnan(y).any()),
        "has_inf": bool(np.isinf(y).any()),
        "n_measured": 30,
    })
    save("runtime_smoke.json", smoke)
    assert smoke["engine_output_shape"] == [1, 256, 64, 64], smoke
    assert finite, "output not finite"

    # ------------------------------------------------------------------ 9
    print("=== numerical sanity vs ORT ===")
    sanity = {}
    try:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        sess = ort.InferenceSession(str(ONNX_PATH), sess_options=so,
                                    providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, {"image": x})[0]
        abs_err = np.abs(y.astype(np.float64) - y_ort.astype(np.float64))
        rel_l2 = float(np.linalg.norm(y - y_ort) / max(np.linalg.norm(y_ort), 1e-12))
        cos = float(np.dot(y.ravel(), y_ort.ravel()) /
                    max(np.linalg.norm(y) * np.linalg.norm(y_ort), 1e-12))
        sanity = {
            "reference_backend": "ONNXRuntime CPU FP32",
            "reference_available": True,
            "max_abs": float(abs_err.max()),
            "mean_abs": float(abs_err.mean()),
            "rmse": float(np.sqrt((abs_err ** 2).mean())),
            "relative_l2": rel_l2,
            "cosine": cos,
            "trt_shape": list(y.shape),
            "ort_shape": list(y_ort.shape),
        }
    except Exception as e:
        sanity = {"reference_backend": "NOT_AVAILABLE",
                  "reference_available": False, "error": str(e)}
    save("numerical_sanity.json", sanity)
    # save reference arrays for the record (small)
    np.save(OUT / "trt_out_smoke.npy", y)
    if "reference_available" in sanity and sanity["reference_available"]:
        np.save(OUT / "ort_out_smoke.npy", y_ort)

    # ------------------------------------------------------------------ 10
    print("=== predictor compatibility ===")
    pred = {}
    try:
        sys.path.insert(0, str(ROOT / "third_party/nanosam"))
        from nanosam.utils.predictor import load_image_encoder_engine
        enc = load_image_encoder_engine(str(ENGINE_PATH))
        # use predictor's own preprocess on a real-ish deterministic image
        from nanosam.utils.predictor import preprocess_image
        pil_in = PIL.Image.fromarray(
            (np.random.default_rng(SEED).random((720, 1280, 3)) * 255).astype(np.uint8))
        import PIL
        tin = preprocess_image(pil_in, size=1024)
        with torch.no_grad():
            emb = enc(tin)
        pred = {
            "predictor_load": "PASS",
            "loader": "nanosam.utils.predictor.load_image_encoder_engine (TRTModule)",
            "embedding_shape": list(emb.shape),
            "finite": bool(torch.isfinite(emb).all().item()),
            "source": "deterministic synthetic image (1280x720) via preprocess_image",
        }
        assert list(emb.shape) == [1, 256, 64, 64], pred
    except Exception as e:
        import traceback
        pred = {"predictor_load": "FAIL", "error": str(e),
                "trace": traceback.format_exc()}
    save("predictor_compatibility.json", pred)
    import PIL  # noqa

    # ------------------------------------------------------------------ 11
    print("=== latency smoke (self-trained) ===")
    runner2 = TRTRunner(ENGINE_PATH)
    for _ in range(30):
        runner2.run(x)
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        runner2.run(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    lat = {
        "target": "self-trained student encoder FP16",
        "warmup": 30,
        "measured": 100,
        "mean_ms": float(arr.mean()),
        "P50_ms": float(np.percentile(arr, 50)),
        "P95_ms": float(np.percentile(arr, 95)),
        "P99_ms": float(np.percentile(arr, 99)),
    }
    save("latency_smoke.json", lat)

    # ------------------------------------------------------------------ 12
    print("=== baseline same-protocol latency ===")
    base = {}
    try:
        rb = TRTRunner(BASELINE_ENGINE)
        for _ in range(30):
            rb.run(x)
        bt = []
        for _ in range(100):
            t0 = time.perf_counter()
            rb.run(x)
            torch.cuda.synchronize()
            bt.append((time.perf_counter() - t0) * 1000.0)
        barr = np.array(bt)
        base = {
            "target": "historical NanoSAM student encoder FP16 (Day3 baseline)",
            "engine_sha256": sha256(BASELINE_ENGINE),
            "warmup": 30, "measured": 100,
            "mean_ms": float(barr.mean()),
            "P50_ms": float(np.percentile(barr, 50)),
            "P95_ms": float(np.percentile(barr, 95)),
            "P99_ms": float(np.percentile(barr, 99)),
        }
    except Exception as e:
        base = {"target": "historical baseline", "status": "NOT_RUN", "error": str(e)}
    save("latency_baseline_same_protocol.json", base)

    print("=== DONE ===")
    print(json.dumps({"runtime": smoke, "sanity": sanity, "latency": lat, "baseline": base},
                     indent=2)[:4000])


if __name__ == "__main__":
    main()
