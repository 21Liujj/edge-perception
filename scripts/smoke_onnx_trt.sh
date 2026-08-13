#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PYTHON="${VENV_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${PROJECT_ROOT}/results/smoke}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
mkdir -p "${ARTIFACT_DIR}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    printf 'Missing virtual environment Python: %s\n' "${VENV_PYTHON}" >&2
    exit 1
fi
if [[ ! -x "${TRTEXEC}" ]]; then
    printf 'Missing trtexec: %s\n' "${TRTEXEC}" >&2
    exit 1
fi

"${VENV_PYTHON}" - "${ARTIFACT_DIR}/identity.onnx" <<'PY'
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper, shape_inference

output = Path(sys.argv[1])
input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
node = helper.make_node("Identity", ["input"], ["output"], name="identity")
graph = helper.make_graph([node], "jetson_deploy_lab_smoke", [input_info], [output_info])
model = helper.make_model(
    graph,
    producer_name="jetson-deploy-lab",
    opset_imports=[helper.make_opsetid("", 17)],
)
model.ir_version = 9
model = shape_inference.infer_shapes(model)
checker.check_model(model)
onnx.save(model, output)
print(f"wrote={output}")
print(f"onnx_version={onnx.__version__}")
PY

rm -f "${ARTIFACT_DIR}/identity.engine"
START_NS="$(date +%s%N)"
"${TRTEXEC}" \
    --onnx="${ARTIFACT_DIR}/identity.onnx" \
    --saveEngine="${ARTIFACT_DIR}/identity.engine" \
    --builderOptimizationLevel=3 \
    --profilingVerbosity=detailed \
    --skipInference \
    2>&1 | tee "${ARTIFACT_DIR}/build.log"
END_NS="$(date +%s%N)"
printf '%s\n' "$(( (END_NS - START_NS) / 1000000 ))" > "${ARTIFACT_DIR}/build_time_ms.txt"

"${TRTEXEC}" \
    --loadEngine="${ARTIFACT_DIR}/identity.engine" \
    --warmUp=500 \
    --duration=3 \
    --iterations=100 \
    --separateProfileRun \
    --percentile=50,90,95,99 \
    --exportTimes="${ARTIFACT_DIR}/times.json" \
    --exportProfile="${ARTIFACT_DIR}/profile.json" \
    --exportLayerInfo="${ARTIFACT_DIR}/layers.json" \
    2>&1 | tee "${ARTIFACT_DIR}/run.log"

grep -q '&&&& PASSED TensorRT.trtexec' "${ARTIFACT_DIR}/build.log"
grep -q '&&&& PASSED TensorRT.trtexec' "${ARTIFACT_DIR}/run.log"
test -s "${ARTIFACT_DIR}/identity.engine"
test -e "${ARTIFACT_DIR}/times.json"
test -e "${ARTIFACT_DIR}/profile.json"
test -e "${ARTIFACT_DIR}/layers.json"

sha256sum \
    "${ARTIFACT_DIR}/identity.onnx" \
    "${ARTIFACT_DIR}/identity.engine" \
    > "${ARTIFACT_DIR}/sha256.txt"

printf 'ONNX -> TensorRT smoke test passed. Artifacts: %s\n' "${ARTIFACT_DIR}"
