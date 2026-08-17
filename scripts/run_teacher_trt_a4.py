from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/toolchain_trt_a4"
A3 = ROOT / "results/toolchain_parity_a3"
NANOSAM = ROOT / "third_party/nanosam"
ENGINE_PATH = OUT / "artifacts/mobile_sam_teacher_fp32.engine"
ONNX_PATH = ROOT / "results/toolchain_export_a2/artifacts/mobile_sam_image_encoder_exported.onnx"
INPUT_PATH = A3 / "input_reference.npy"
INPUT_RAW_SHA = "fda4c3d2ad945c3441a7bd79de1711655d5324d2921831549003a769892aba93"
CHECKPOINT_SHA = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
WARMUP = 20
ITERATIONS = 100

sys.path.insert(0, str(NANOSAM))
from nanosam.mobile_sam import sam_model_registry

warnings.filterwarnings("ignore", message="Overwriting tiny_vit")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(array: np.ndarray) -> dict:
    values = array.astype(np.float64)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "has_nan": bool(np.isnan(array).any()),
        "has_inf": bool(np.isinf(array).any()),
        "sha256_raw": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def reference(backend: str, device: str, array: np.ndarray, elapsed: float) -> dict:
    return {
        "backend": backend,
        "device": device,
        "model": "MobileSAM vit_t TinyViT image_encoder",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "input_sha256_raw_float32": INPUT_RAW_SHA,
        "elapsed_seconds_single_run": elapsed,
        "output": stats(array),
    }


def compare(reference_array: np.ndarray, candidate_array: np.ndarray, reference_name: str, candidate_name: str) -> dict:
    ref = reference_array.astype(np.float64).ravel()
    candidate = candidate_array.astype(np.float64).ravel()
    difference = candidate - ref
    absolute = np.abs(difference)
    percentiles = np.percentile(absolute, [50, 90, 95, 99, 99.9])
    ref_norm = np.linalg.norm(ref)
    candidate_norm = np.linalg.norm(candidate)
    return {
        "reference": reference_name,
        "candidate": candidate_name,
        "shape_match": reference_array.shape == candidate_array.shape,
        "dtype_match": reference_array.dtype == candidate_array.dtype,
        "element_count": int(ref.size),
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "absolute_error_percentiles": {
            "p50": float(percentiles[0]),
            "p90": float(percentiles[1]),
            "p95": float(percentiles[2]),
            "p99": float(percentiles[3]),
            "p99_9": float(percentiles[4]),
        },
        "relative_l2_error": float(np.linalg.norm(difference) / ref_norm),
        "cosine_similarity": float(np.dot(ref, candidate) / (ref_norm * candidate_norm)),
        "pearson_correlation": float(np.corrcoef(ref, candidate)[0, 1]),
        "threshold_proportions": {
            ">1e-1": float(np.mean(absolute > 1e-1)),
            ">5e-2": float(np.mean(absolute > 5e-2)),
            ">1e-2": float(np.mean(absolute > 1e-2)),
            ">1e-3": float(np.mean(absolute > 1e-3)),
        },
    }


def run_pytorch_cuda(input_array: np.ndarray) -> tuple[np.ndarray, float]:
    state = torch.load(NANOSAM / "assets/mobile_sam.pt", map_location="cpu", weights_only=False)
    model = sam_model_registry["vit_t"](checkpoint=None)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    encoder = model.eval().image_encoder.eval().cuda()
    image = torch.from_numpy(input_array.copy()).cuda()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        output = encoder(image)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    array = output.cpu().numpy()
    del output, image, encoder, model, state
    gc.collect()
    torch.cuda.empty_cache()
    return array, elapsed


def run_ort_cpu(input_array: np.ndarray) -> tuple[np.ndarray, float]:
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    output = session.run(["image_embeddings"], {"image": input_array})[0]
    elapsed = time.perf_counter() - started
    del session
    gc.collect()
    return output, elapsed


def tensor_contract(engine: trt.ICudaEngine) -> list[dict]:
    tensors = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        item = {
            "index": index,
            "name": name,
            "mode": str(mode),
            "dtype": str(engine.get_tensor_dtype(name)),
            "engine_shape": list(engine.get_tensor_shape(name)),
        }
        try:
            profile = engine.get_tensor_profile_shape(name, 0)
            item["profile_0"] = {
                "min": list(profile[0]),
                "opt": list(profile[1]),
                "max": list(profile[2]),
            }
        except Exception:
            item["profile_0"] = None
        tensors.append(item)
    return tensors


def run_trt(input_array: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    engine_bytes = ENGINE_PATH.read_bytes()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("TensorRT engine deserialization returned None")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT execution context creation returned None")

    input_name = next(engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT)
    output_name = next(engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT)
    if not context.set_input_shape(input_name, tuple(input_array.shape)):
        raise RuntimeError("TensorRT rejected input shape")
    output_shape = tuple(context.get_tensor_shape(output_name))
    if output_shape != (1, 256, 64, 64):
        raise RuntimeError(f"Unexpected TensorRT output shape: {output_shape}")

    stream = torch.cuda.Stream()
    input_tensor = torch.from_numpy(input_array.copy()).cuda(non_blocking=False).contiguous()
    output_tensor = torch.empty(output_shape, dtype=torch.float32, device="cuda")
    if not context.set_tensor_address(input_name, input_tensor.data_ptr()):
        raise RuntimeError("Could not bind TensorRT input tensor")
    if not context.set_tensor_address(output_name, output_tensor.data_ptr()):
        raise RuntimeError("Could not bind TensorRT output tensor")

    def execute() -> None:
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")

    with torch.cuda.stream(stream):
        for _ in range(WARMUP):
            execute()
    stream.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERATIONS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERATIONS)]
    with torch.cuda.stream(stream):
        for start, end in zip(starts, ends):
            start.record(stream)
            execute()
            end.record(stream)
    stream.synchronize()
    timings = np.array([start.elapsed_time(end) for start, end in zip(starts, ends)], dtype=np.float64)
    output = output_tensor.cpu().numpy()

    benchmark = {
        "backend": "TensorRT FP32",
        "warmup_iterations": WARMUP,
        "measured_iterations": ITERATIONS,
        "unit": "milliseconds",
        "mean": float(timings.mean()),
        "p50": float(np.percentile(timings, 50)),
        "p95": float(np.percentile(timings, 95)),
        "p99": float(np.percentile(timings, 99)),
        "minimum": float(timings.min()),
        "maximum": float(timings.max()),
        "scope": "Runtime correctness baseline; no performance tuning.",
    }
    contract = {
        "engine": {
            "path": str(ENGINE_PATH.relative_to(ROOT)),
            "size_bytes": ENGINE_PATH.stat().st_size,
            "sha256": sha256_file(ENGINE_PATH),
        },
        "tensorrt_version": trt.__version__,
        "deserialize": "PASS",
        "execution_context": "PASS",
        "num_io_tensors": engine.num_io_tensors,
        "num_optimization_profiles": engine.num_optimization_profiles,
        "tensors": tensor_contract(engine),
        "runtime_input_shape": list(input_array.shape),
        "runtime_output_shape": list(output.shape),
        "tf32": "disabled at build with --noTF32",
        "precision": "FP32; --fp16 and INT8 not enabled",
    }
    return output, benchmark, contract


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    input_array = np.load(INPUT_PATH)
    if input_array.shape != (1, 3, 1024, 1024) or input_array.dtype != np.float32:
        raise RuntimeError(f"Input contract mismatch: {input_array.shape}, {input_array.dtype}")
    if hashlib.sha256(input_array.tobytes(order="C")).hexdigest() != INPUT_RAW_SHA:
        raise RuntimeError("Input raw SHA256 mismatch")

    pytorch_cuda, pytorch_seconds = run_pytorch_cuda(input_array)
    ort_cpu, ort_seconds = run_ort_cpu(input_array)
    trt_output, benchmark, contract = run_trt(input_array)

    np.save(OUT / "pytorch_cuda_output.npy", pytorch_cuda)
    np.save(OUT / "ort_cpu_output.npy", ort_cpu)
    np.save(OUT / "trt_output.npy", trt_output)

    (OUT / "pytorch_cuda_reference.json").write_text(json.dumps(reference("PyTorch", "CUDA FP32", pytorch_cuda, pytorch_seconds), indent=2) + "\n")
    (OUT / "ort_cpu_reference.json").write_text(json.dumps(reference("ONNX Runtime 1.20.0", "CPUExecutionProvider FP32", ort_cpu, ort_seconds), indent=2) + "\n")
    (OUT / "trt_reference.json").write_text(json.dumps(reference(f"TensorRT {trt.__version__}", "CUDA FP32, TF32 disabled", trt_output, benchmark["mean"] / 1000.0), indent=2) + "\n")
    (OUT / "runtime_benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    (OUT / "trt_engine_contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    matrix = {
        "input": {
            "path": str(INPUT_PATH.relative_to(ROOT)),
            "shape": list(input_array.shape),
            "dtype": str(input_array.dtype),
            "sha256_raw_float32": INPUT_RAW_SHA,
        },
        "comparisons": {
            "pytorch_cuda_vs_ort_cpu": compare(pytorch_cuda, ort_cpu, "PyTorch CUDA FP32", "ONNX Runtime CPU FP32"),
            "pytorch_cuda_vs_trt_fp32": compare(pytorch_cuda, trt_output, "PyTorch CUDA FP32", "TensorRT FP32, TF32 disabled"),
            "ort_cpu_vs_trt_fp32": compare(ort_cpu, trt_output, "ONNX Runtime CPU FP32", "TensorRT FP32, TF32 disabled"),
        },
    }
    (OUT / "parity_matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"contract": contract, "benchmark": benchmark, "parity": matrix["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
