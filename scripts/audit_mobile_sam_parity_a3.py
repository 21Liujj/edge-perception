from pathlib import Path
import gc, hashlib, json, sys, time, warnings
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path.cwd()
OUT = ROOT / "results/toolchain_parity_a3"
NANOSAM = ROOT / "third_party/nanosam"
sys.path.insert(0, str(NANOSAM))
from nanosam.mobile_sam import sam_model_registry

warnings.filterwarnings("ignore", message="Overwriting tiny_vit")
INPUT_SHA = "fda4c3d2ad945c3441a7bd79de1711655d5324d2921831549003a769892aba93"
CHECKPOINT_SHA = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
input_array = np.load(OUT / "input_reference.npy")
state = torch.load(NANOSAM / "assets/mobile_sam.pt", map_location="cpu", weights_only=False)


def stats(array):
    values = array.astype(np.float64)
    return {
        "shape": list(array.shape), "dtype": str(array.dtype),
        "min": float(array.min()), "max": float(array.max()),
        "mean": float(values.mean()), "std": float(values.std()),
        "has_nan": bool(np.isnan(array).any()), "has_inf": bool(np.isinf(array).any()),
        "sha256_raw": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def run_pytorch(device):
    model = sam_model_registry["vit_t"](checkpoint=None)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    encoder = model.eval().image_encoder.eval().to(device)
    image = torch.from_numpy(input_array.copy()).to(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = encoder(image)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    array = output.cpu().numpy()
    del output, image, encoder, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return array, elapsed


def reference(backend, device, array, elapsed):
    return {
        "backend": backend, "device": device,
        "model": "MobileSAM vit_t TinyViT image_encoder",
        "model_eval": True, "no_grad": True,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "input_sha256_raw_float32": INPUT_SHA,
        "elapsed_seconds": elapsed, "output": stats(array),
    }


def compare(reference_array, candidate_array, reference_name, candidate_name):
    ref = reference_array.astype(np.float64).ravel()
    candidate = candidate_array.astype(np.float64).ravel()
    difference = candidate - ref
    absolute = np.abs(difference)
    percentiles = np.percentile(absolute, [50, 90, 95, 99, 99.9])
    ref_norm = np.linalg.norm(ref)
    candidate_norm = np.linalg.norm(candidate)
    return {
        "reference": reference_name, "candidate": candidate_name,
        "shape_match": reference_array.shape == candidate_array.shape,
        "dtype_match": reference_array.dtype == candidate_array.dtype,
        "element_count": int(ref.size),
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "absolute_error_percentiles": {
            "p50": float(percentiles[0]), "p90": float(percentiles[1]),
            "p95": float(percentiles[2]), "p99": float(percentiles[3]),
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
        "max_relative_error_audit_only": float(
            np.max(absolute / np.maximum(np.abs(ref), 1e-12))
        ),
        "max_relative_error_note": "Audit only; unstable where reference approaches zero.",
    }


cpu, cpu_seconds = run_pytorch("cpu")
np.save(OUT / "pytorch_cpu_output.npy", cpu)
cuda, cuda_seconds = run_pytorch("cuda")
np.save(OUT / "pytorch_cuda_output.npy", cuda)
started = time.perf_counter()
session = ort.InferenceSession(
    str(ROOT / "results/toolchain_export_a2/artifacts/mobile_sam_image_encoder_exported.onnx"),
    providers=["CPUExecutionProvider"],
)
ort_cpu = session.run(["image_embeddings"], {"image": input_array})[0]
ort_seconds = time.perf_counter() - started
np.save(OUT / "ort_cpu_output.npy", ort_cpu)

references = [
    ("pytorch_cpu_reference.json", reference("PyTorch", "cpu", cpu, cpu_seconds)),
    ("pytorch_cuda_reference.json", reference("PyTorch", "cuda", cuda, cuda_seconds)),
    ("ort_cpu_reference.json", reference("ONNX Runtime 1.20.0", "CPUExecutionProvider", ort_cpu, ort_seconds)),
]
comparisons = [
    ("pytorch_cpu_vs_cuda.json", compare(cpu, cuda, "PyTorch CPU FP32", "PyTorch CUDA FP32")),
    ("pytorch_cpu_vs_ort_cpu.json", compare(cpu, ort_cpu, "PyTorch CPU FP32", "ONNX Runtime CPU FP32")),
    ("pytorch_cuda_vs_ort_cpu.json", compare(cuda, ort_cpu, "PyTorch CUDA FP32", "ONNX Runtime CPU FP32")),
]
for name, data in references + comparisons:
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")

values = cpu.astype(np.float64)
absolute_values = np.abs(values)
percentiles = np.percentile(absolute_values, [50, 90, 95, 99])
distribution = {
    "reference": "PyTorch CPU FP32", "shape": list(cpu.shape), "dtype": str(cpu.dtype),
    "min": float(values.min()), "max": float(values.max()),
    "mean": float(values.mean()), "std": float(values.std()),
    "absolute_output_percentiles": {
        "p50": float(percentiles[0]), "p90": float(percentiles[1]),
        "p95": float(percentiles[2]), "p99": float(percentiles[3]),
        "max": float(absolute_values.max()),
    },
    "mean_abs_error_context": {
        "cpu_vs_cuda_mean_abs_over_output_std": comparisons[0][1]["mean_absolute_error"] / values.std(),
        "cpu_vs_ort_mean_abs_over_output_std": comparisons[1][1]["mean_absolute_error"] / values.std(),
    },
}
(OUT / "output_distribution.json").write_text(json.dumps(distribution, indent=2) + "\n")
print(json.dumps({"timing": [cpu_seconds, cuda_seconds, ort_seconds], "A": comparisons[0][1], "B": comparisons[1][1], "C": comparisons[2][1], "scale": distribution}, indent=2))
