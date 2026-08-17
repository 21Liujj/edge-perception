#!/usr/bin/env python3
"""Reproduce the Stage A2 MobileSAM teacher reference and ONNX export."""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import torch


def tensor_stats(array):
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "has_nan": bool(np.isnan(array).any()),
        "has_inf": bool(np.isinf(array).any()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output-dir", type=Path, default=Path("results/toolchain_export_a2"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    nanosam = root / "third_party/nanosam"
    sys.path.insert(0, str(nanosam))
    from nanosam.mobile_sam import sam_model_registry

    if not torch.cuda.is_available():
        raise RuntimeError("Stage A2 reference contract requires CUDA")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.output_dir / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    checkpoint = nanosam / "assets/mobile_sam.pt"
    checkpoint_bytes = checkpoint.read_bytes()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = sam_model_registry["vit_t"](checkpoint=None)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint contract mismatch: {incompatible}")

    encoder = model.eval().image_encoder.eval().cuda()
    image = torch.randn(1, 3, 1024, 1024, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        embedding = encoder(image)
    torch.cuda.synchronize()
    image_np = image.cpu().numpy()
    embedding_np = embedding.cpu().numpy()

    checkpoint_info = {
        "path": str(checkpoint.relative_to(root)),
        "size_bytes": len(checkpoint_bytes),
        "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "top_level_type": f"{type(state).__module__}.{type(state).__name__}",
        "state_dict_key_count": len(state),
        "first_keys": list(state)[:12],
        "value_types": sorted({type(value).__name__ for value in state.values()}),
    }
    reference = {
        "random_seed": args.seed,
        "device": "cuda",
        "model_class": f"{type(encoder).__module__}.{type(encoder).__name__}",
        "model_eval": not encoder.training,
        "no_grad": True,
        "checkpoint_load": {
            "strict_contract": "PASS",
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        },
        "input": tensor_stats(image_np) | {
            "sha256_raw_float32": hashlib.sha256(image_np.tobytes(order="C")).hexdigest(),
            "saved_in_repository": False,
        },
        "output": tensor_stats(embedding_np),
        "forward": "PASS",
    }
    (args.output_dir / "checkpoint_info.json").write_text(json.dumps(checkpoint_info, indent=2) + "\n")
    (args.output_dir / "pytorch_reference.json").write_text(json.dumps(reference, indent=2) + "\n")

    onnx_path = artifact_dir / "mobile_sam_image_encoder_exported.onnx"
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (image,),
            str(onnx_path),
            input_names=["image"],
            output_names=["image_embeddings"],
            dynamic_axes={"image": {0: "batch_size"}},
            opset_version=16,
            training=torch.onnx.TrainingMode.EVAL,
        )
    print(onnx_path)


if __name__ == "__main__":
    main()
