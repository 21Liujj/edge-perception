#!/usr/bin/env python3
"""Run INT8 Calibration to generate cache."""
import sys, time
from pathlib import Path

ROOT = Path("/home/nvidia/jetson-deploy-lab")
sys.path.insert(0, str(ROOT / "scripts/int8"))

import torch
import tensorrt as trt

from simple_calibrator import SimpleCalibrator

ONNX = ROOT / "models/self_distilled/epoch50/nanosam_resnet18_epoch049.onnx"
MANIFEST = ROOT / "data/calibration/nanosam_int8_train2017_512.json"
CACHE_DIR = ROOT / "models/self_distilled/epoch50/int8"
CACHE_FILE = CACHE_DIR / "calibration_train2017_512.cache"

print("=== NanoSAM INT8 Calibration ===")
print(f"ONNX: {ONNX}")
print(f"Manifest: {MANIFEST}")
print(f"Cache: {CACHE_FILE}")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Create calibrator
calibrator = SimpleCalibrator(
    manifest_json=str(MANIFEST),
    batch_size=1,
    cache_file=str(CACHE_FILE)
)

# Build network
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

print("\\nParsing ONNX...")
with open(ONNX, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f"Error: {parser.get_error(i)}")
        sys.exit(1)

print(f"Network: {network.num_layers} layers")

# Configure for INT8
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)
config.int8_calibrator = calibrator
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

print("\\nRunning calibration (this will take a few minutes)...")
print("Building INT8 engine...")

t0 = time.time()
serialized = builder.build_serialized_network(network, config)
elapsed = time.time() - t0

if serialized is None:
    print("ERROR: Build failed")
    sys.exit(1)

print(f"\\nCalibration complete: {elapsed:.1f}s")
print(f"Engine size: {len(serialized)} bytes")

# Save temporary engine
temp_engine = CACHE_DIR / "calibration_temp.engine"
with open(temp_engine, "wb") as f:
    f.write(serialized)

print(f"Temp engine: {temp_engine}")

# Verify cache
if CACHE_FILE.exists():
    cache_size = CACHE_FILE.stat().st_size
    print(f"\\nCache verified: {CACHE_FILE}")
    print(f"Cache size: {cache_size} bytes")
else:
    print("ERROR: Cache not created!")
    sys.exit(1)

print("\\n=== SUCCESS ===")
