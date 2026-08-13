#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
RAW_OUTPUT="${RESULTS_DIR}/inventory.txt"
JSON_OUTPUT="${RESULTS_DIR}/env.json"
if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
mkdir -p "${RESULTS_DIR}"

run_optional() {
    local title="$1"
    shift
    printf '\n=== %s ===\n' "${title}"
    "$@" 2>&1 || true
}

collect_inventory() {
    printf 'collected_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'model='; tr -d '\000' < /proc/device-tree/model; printf '\n'

    run_optional L4T cat /etc/nv_tegra_release
    run_optional OS grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
    run_optional KERNEL uname -a
    run_optional CUDA_TOOLKIT /usr/local/cuda/bin/nvcc --version
    run_optional CUDA_LINK readlink -f /usr/local/cuda
    run_optional TENSORRT_PACKAGES dpkg-query -W -f='${Package}\t${Version}\n' tensorrt libnvinfer10 python3-libnvinfer
    run_optional CUDNN_PACKAGES dpkg-query -W -f='${Package}\t${Version}\n' libcudnn9-cuda-12 libcudnn9-dev-cuda-12
    run_optional PYTHON "${PYTHON_BIN}" --version

    printf '\n=== PYTHON_PACKAGES ===\n'
    "${PYTHON_BIN}" - <<'PY' || true
import importlib
for name in ("tensorrt", "torch", "onnx", "onnxruntime", "cv2", "numpy"):
    try:
        module = importlib.import_module(name)
        print(f"{name}\t{getattr(module, '__version__', 'unknown')}")
        if name == "torch":
            print(f"torch.cuda_available\t{module.cuda.is_available()}")
            print(f"torch.cuda_version\t{module.version.cuda}")
            print(f"torch.cudnn_version\t{module.backends.cudnn.version()}")
        if name == "onnxruntime":
            print(f"onnxruntime.providers\t{module.get_available_providers()}")
    except Exception as exc:
        print(f"{name}\tMISSING\t{type(exc).__name__}: {exc}")
PY

    run_optional POWER_MODE nvpmodel -q
    run_optional MEMORY free -b
    run_optional STORAGE df -B1 /
    run_optional BLOCK_DEVICES lsblk -b -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
    run_optional PROFILERS bash -c 'command -v tegrastats; command -v nsys; command -v /usr/local/cuda/bin/ncu'
    run_optional DOCKER bash -c 'command -v docker; docker --version; systemctl is-active docker'
    run_optional CAMERA_DEVICES bash -c 'ls -l /dev/video* /dev/media* 2>/dev/null'
    run_optional CAMERA_PLUGIN gst-inspect-1.0 nvarguscamerasrc
    run_optional TOOLCHAIN bash -c 'gcc --version | head -1; g++ --version | head -1; cmake --version | head -1; git --version'
    run_optional TEGRASTATS timeout 5s tegrastats --interval 1000
}

collect_inventory > "${RAW_OUTPUT}"

"${PYTHON_BIN}" - "${JSON_OUTPUT}" <<'PY'
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])


def command(args):
    try:
        return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def package_version(name):
    return command(["dpkg-query", "-W", "-f=${Version}", name])


def module_version(name):
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None

model_path = Path("/proc/device-tree/model")
model = model_path.read_bytes().replace(b"\x00", b"").decode() if model_path.exists() else None
l4t = command(["dpkg-query", "-W", "-f=${Version}", "nvidia-l4t-core"])
cuda_root = os.path.realpath("/usr/local/cuda") if Path("/usr/local/cuda").exists() else None
nvcc_version = command(["/usr/local/cuda/bin/nvcc", "--version"])

try:
    import torch
    torch_info = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    }
except Exception as exc:
    torch_info = {"version": None, "error": f"{type(exc).__name__}: {exc}"}

try:
    import onnxruntime as ort
    ort_info = {"version": ort.__version__, "providers": ort.get_available_providers()}
except Exception as exc:
    ort_info = {"version": None, "error": f"{type(exc).__name__}: {exc}"}

memory_lines = Path("/proc/meminfo").read_text().splitlines()
meminfo = {line.split(":", 1)[0]: int(line.split()[1]) * 1024 for line in memory_lines if line.split()[1].isdigit()}
stat = os.statvfs("/")
video_devices = sorted(str(path) for path in Path("/dev").glob("video*"))
media_devices = sorted(str(path) for path in Path("/dev").glob("media*"))

data = {
    "schema_version": 1,
    "collected_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "device": {
        "hostname": platform.node(),
        "model": model,
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": meminfo.get("MemTotal"),
        "swap_total_bytes": meminfo.get("SwapTotal"),
        "root_storage_total_bytes": stat.f_blocks * stat.f_frsize,
        "root_storage_available_bytes": stat.f_bavail * stat.f_frsize,
    },
    "software": {
        "os": command(["lsb_release", "-ds"]),
        "kernel": platform.release(),
        "l4t_package": l4t,
        "cuda_root": cuda_root,
        "cuda_nvcc_available": nvcc_version is not None,
        "cuda_nvcc_release": next((line.strip() for line in (nvcc_version or "").splitlines() if "release" in line), None),
        "cudnn_package": package_version("libcudnn9-cuda-12"),
        "tensorrt_package": package_version("tensorrt"),
        "python": platform.python_version(),
        "python_modules": {
            "tensorrt": module_version("tensorrt"),
            "onnx": module_version("onnx"),
            "opencv": module_version("cv2"),
            "numpy": module_version("numpy"),
        },
        "pytorch": torch_info,
        "onnxruntime": ort_info,
    },
    "system": {
        "power_mode": command(["nvpmodel", "-q"]),
        "tegrastats_available": command(["bash", "-lc", "command -v tegrastats"]) is not None,
        "nsight_systems_available": command(["bash", "-lc", "command -v nsys"]) is not None,
        "nsight_compute_available": Path("/usr/local/cuda/bin/ncu").exists(),
        "docker_available": command(["bash", "-lc", "command -v docker"]) is not None,
        "docker_active": command(["systemctl", "is-active", "docker"]) == "active",
        "camera": {
            "video_devices": video_devices,
            "media_devices": media_devices,
            "argus_plugin_available": command(["gst-inspect-1.0", "nvarguscamerasrc"]) is not None,
            "argus_daemon_active": command(["systemctl", "is-active", "nvargus-daemon"]) == "active",
        },
    },
    "readiness": {
        "required_missing": ["python-onnx"] if module_version("onnx") is None else [],
        "configuration_issues": [],
        "optional_missing": [
            item for item, missing in {
                "docker": command(["bash", "-lc", "command -v docker"]) is None,
                "camera-sensor": len(video_devices) == 0,
            }.items() if missing
        ],
    },
}
output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
PY

printf 'Inventory written to:\n  %s\n  %s\n' "${RAW_OUTPUT}" "${JSON_OUTPUT}"
