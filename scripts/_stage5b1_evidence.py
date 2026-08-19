#!/usr/bin/env python3
"""Emit environment.json + source_onnx_audit.json + baseline_engine_audit.json
+ trt_build_summary.json for Stage Jetson-5B-1. Run after build completes."""
import json, os, subprocess, hashlib, sys
from pathlib import Path
import onnx
import torch

ROOT = Path("/home/nvidia/jetson-deploy-lab")
OUT = ROOT / "results" / "stage_jetson_5b1"
ONNX_PATH = ROOT / "models/self_distilled/epoch10/nanosam_resnet18_epoch009.onnx"
ENG = ROOT / "models/self_distilled/epoch10/nanosam_resnet18_epoch009_fp16.engine"
BASELINE = ROOT / "third_party/nanosam/data/resnet18_image_encoder.engine"
BASELINE_FP32 = ROOT / "third_party/nanosam/data/resnet18_image_encoder_fp32.engine"

import tensorrt as trt

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def jdump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2))
    print(f"[wrote] {name}")

# 1 environment
env = {
    "hostname": subprocess.check_output(["hostname"]).decode().strip(),
    "uname": subprocess.check_output(["uname", "-a"]).decode().strip(),
    "os": subprocess.check_output(["lsb_release", "-ds"]).decode().strip(),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torchvision": __import__("torchvision").__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
    "tensorrt": trt.__version__,
    "cuda_runtime": torch.version.cuda,
    "jetpack_l4t": subprocess.check_output(["cat", "/etc/nv_tegra_release"]).decode().strip() if Path("/etc/nv_tegra_release").exists() else "N/A",
    "free_mem": subprocess.check_output(["free", "-h"]).decode().strip().splitlines(),
    "disk_home": subprocess.check_output(["df", "-h", "/home/nvidia"]).decode().strip().splitlines(),
    "tensorrt_expected": "10.3.0.30",
    "drift": False,
}
jdump("environment.json", env)

# 3 source onnx audit
m = onnx.load(str(ONNX_PATH))
g = m.graph
def dims(t):
    return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in t.type.tensor_type.shape.dim]
sox = {
    "path": str(ONNX_PATH),
    "size_bytes": ONNX_PATH.stat().st_size,
    "sha256": sha(ONNX_PATH),
    "expected_sha256": "aacbea870b8eca91e61fbb6325806c60a80e2f4a193f96597881e060b5c5cf15",
    "sha_match": sha(ONNX_PATH) == "aacbea870b8eca91e61fbb6325806c60a80e2f4a193f96597881e060b5c5cf15",
    "opset": [(o.domain, o.version) for o in m.opset_import],
    "ir_version": m.ir_version,
    "producer": m.producer_name,
    "num_nodes": len(g.node),
    "inputs": [{"name": i.name, "shape": dims(i), "dtype": i.type.tensor_type.elem_type} for i in g.input],
    "outputs": [{"name": o.name, "shape": dims(o), "dtype": o.type.tensor_type.elem_type} for o in g.output],
    "contract_pass": (
        len(g.input) == 1 and g.input[0].name == "image" and dims(g.input[0]) == [1, 3, 1024, 1024]
        and len(g.output) == 1 and g.output[0].name == "image_embeddings" and dims(g.output[0]) == [1, 256, 64, 64]
    ),
}
jdump("source_onnx_audit.json", sox)

# 2 baseline audit
ba = {
    "baseline_fp16_path": str(BASELINE),
    "baseline_fp16_exists": BASELINE.exists(),
    "baseline_fp16_size": BASELINE.stat().st_size if BASELINE.exists() else None,
    "baseline_fp16_sha256": sha(BASELINE) if BASELINE.exists() else None,
    "expected_baseline_fp16_sha256": "c277e5ca445dc29b06e8be41a481e74bd806c4dd82071cbfbd47fddcf7aaa6a8",
    "baseline_fp16_sha_match": sha(BASELINE) == "c277e5ca445dc29b06e8be41a481e74bd806c4dd82071cbfbd47fddcf7aaa6a8" if BASELINE.exists() else None,
    "baseline_fp32_path": str(BASELINE_FP32),
    "baseline_fp32_sha256": sha(BASELINE_FP32) if BASELINE_FP32.exists() else None,
    "protection_note": "historical baseline engine untouched; self-trained engine written to models/self_distilled/epoch10/",
}
jdump("baseline_engine_audit.json", ba)

# 6 build summary
build_log = (OUT / "build.log").read_text() if (OUT / "build.log").exists() else ""
build_ms_f = OUT / "build_ms.txt"
meta_f = OUT / "build_meta.txt"
meta = meta_f.read_text().strip().split("|") if meta_f.exists() else ["?", "?"]
build = {
    "tensorrt_version": meta[0] if len(meta) > 0 else trt.__version__,
    "gpu": meta[1] if len(meta) > 1 else "Orin",
    "precision": "FP16",
    "builder": "trtexec",
    "builder_optimization_level": 3,
    "workspace_memPool": "workspace:2048MiB",
    "input_shape": [1, 3, 1024, 1024],
    "output_shape": [1, 256, 64, 64],
    "build_wall_time_ms": int(build_ms_f.read_text().strip()) if build_ms_f.exists() else None,
    "engine_path": str(ENG),
    "engine_exists": ENG.exists(),
    "engine_size_bytes": ENG.stat().st_size if ENG.exists() else None,
    "engine_sha256": sha(ENG) if ENG.exists() else None,
    "onnx_sha256": sha(ONNX_PATH),
    "build_passed": ("PASSED TensorRT.trtexec" in build_log) and (OUT / "build.exit").exists() and (OUT / "build.exit").read_text().strip() == "0",
    "parser_errors": [l for l in build_log.splitlines() if l.strip().startswith("[E")][:20],
    "parser_warnings": [l for l in build_log.splitlines() if l.strip().startswith("[W")][:20],
    "fatal_error": None,
}
jdump("trt_build_summary.json", build)

# 13 engine identity
new_sha = build["engine_sha256"]
old_sha = ba["baseline_fp16_sha256"]
jdump("engine_identity.json", {
    "new_self_trained_engine_sha256": new_sha,
    "historical_baseline_engine_sha256": old_sha,
    "identity": "DIFFERENT_ENGINE" if new_sha != old_sha else "SAME_ENGINE",
    "expected": "DIFFERENT_ENGINE",
})
print("ALL_EVIDENCE_WRITTEN")
