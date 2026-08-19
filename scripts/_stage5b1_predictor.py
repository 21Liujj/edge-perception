#!/usr/bin/env python3
"""Re-run ONLY the NanoSAM predictor compatibility smoke (Stage 5B-1 section 10)."""
import json, sys
from pathlib import Path
import numpy as np
import torch
import PIL.Image

ROOT = Path("/home/nvidia/jetson-deploy-lab")
OUT = ROOT / "results" / "stage_jetson_5b1"
ENGINE = str(ROOT / "models/self_distilled/epoch10/nanosam_resnet18_epoch009_fp16.engine")
SEED = 20260819

def main():
    sys.path.insert(0, str(ROOT / "third_party/nanosam"))
    from nanosam.utils.predictor import load_image_encoder_engine, preprocess_image

    enc = load_image_encoder_engine(ENGINE)
    # deterministic synthetic image
    rng = np.random.default_rng(SEED)
    arr = (rng.random((720, 1280, 3)) * 255).astype(np.uint8)
    pil_in = PIL.Image.fromarray(arr)
    tin = preprocess_image(pil_in, size=1024)  # already on cuda
    with torch.no_grad():
        emb = enc(tin)
    pred = {
        "predictor_load": "PASS",
        "loader": "nanosam.utils.predictor.load_image_encoder_engine (TRTModule)",
        "engine_path": ENGINE,
        "input_source": "deterministic synthetic 1280x720 via preprocess_image(size=1024)",
        "input_shape": list(tin.shape),
        "embedding_shape": list(emb.shape),
        "finite": bool(torch.isfinite(emb).all().item()),
        "expected_embedding_shape": [1, 256, 64, 64],
        "shape_ok": list(emb.shape) == [1, 256, 64, 64],
    }
    (OUT / "predictor_compatibility.json").write_text(json.dumps(pred, indent=2))
    print(json.dumps(pred, indent=2))

if __name__ == "__main__":
    main()
