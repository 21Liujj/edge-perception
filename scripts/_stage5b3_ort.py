#!/usr/bin/env python3
"""Stage Jetson-5B-3: ORT FP32 image-encoder backend + frozen TRT mask decoder,
run through the EXACT same run_evaluation code path used in Stage 5B-2.

Only the image encoder backend is swapped (TRT FP16 -> ORT CPU FP32). No
upstream file is modified: this project-side predictor reuses upstream
preprocessing / decoder / upscale / selection.
"""
import argparse, importlib.util, os, sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

ROOT = Path("/home/nvidia/jetson-deploy-lab")
sys.path.insert(0, str(ROOT / "third_party/nanosam"))
from nanosam.utils.predictor import (
    preprocess_image, preprocess_points, run_mask_decoder, upscale_mask,
    load_mask_decoder_engine,
)


class ORTFP32Predictor:
    """Same interface as nanosam.utils.predictor.Predictor; image encoder runs
    on ONNX Runtime CPUExecutionProvider (FP32). Mask decoder stays on the
    frozen TRT engine."""

    def __init__(self, encoder_onnx, mask_decoder_engine,
                 image_encoder_size=1024, orig_image_encoder_size=1024):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 8
        so.inter_op_num_threads = 1
        self.enc_sess = ort.InferenceSession(
            str(encoder_onnx), sess_options=so,
            providers=["CPUExecutionProvider"])
        self.iname = self.enc_sess.get_inputs()[0].name
        self.mask_decoder_engine = load_mask_decoder_engine(mask_decoder_engine)
        self.image_encoder_size = image_encoder_size
        self.orig_image_encoder_size = orig_image_encoder_size

    def set_image(self, image):
        self.image = image
        self.image_tensor = preprocess_image(image, self.image_encoder_size)
        x = self.image_tensor.detach().cpu().float().numpy()
        emb = self.enc_sess.run(None, {self.iname: x})[0]
        # decoder is TRT on CUDA; put embedding on device as FP32
        self.features = torch.from_numpy(emb).float().cuda()

    def predict(self, points, point_labels, mask_input=None):
        points = preprocess_points(points, (self.image.height, self.image.width),
                                   self.orig_image_encoder_size)
        mask_iou, low_res_mask = run_mask_decoder(
            self.mask_decoder_engine, self.features, points, point_labels, mask_input)
        hi_res_mask = upscale_mask(low_res_mask, (self.image.height, self.image.width))
        return hi_res_mask, mask_iou, low_res_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", default="data/coco/val2017")
    ap.add_argument("--annotations", default="data/coco/annotations/instances_val2017.json")
    ap.add_argument("--subset", default="standard")
    ap.add_argument("--manifest", default="data/eval_manifests/standard_manifest.json")
    ap.add_argument("--encoder-onnx", required=True)
    ap.add_argument("--mask-decoder-engine", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-instances", type=int, default=None)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts/eval_nanosam_coco.py")
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

    # identical gating as evaluator
    from pycocotools.coco import COCO
    ok, msg = ev._check_dataset(args.coco_root, args.annotations)
    if not ok: print(f"DATASET_MISSING: {msg}"); sys.exit(2)
    if not ev._check_pycocotools(): print("MISSING_DEPENDENCY: pycocotools"); sys.exit(3)
    if not os.path.isfile(args.manifest): print(f"MANIFEST_MISSING: {args.manifest}"); sys.exit(4)

    man = ev.load_manifest(args.manifest)
    instances = man.get("instances", [])
    if not instances: print("MANIFEST_EMPTY"); sys.exit(4)
    if args.max_instances is not None:
        instances = instances[:args.max_instances]

    coco = COCO(args.annotations)
    print(f"subset={args.subset} instances={len(instances)}", flush=True)

    import warnings; warnings.filterwarnings("ignore")
    print(f"loading ORT encoder: {args.encoder_onnx}", flush=True)
    print(f"decoder: {args.mask_decoder_engine}", flush=True)
    predictor = ORTFP32Predictor(args.encoder_onnx, args.mask_decoder_engine)
    print("ORT+decoder ready", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    summary = ev.run_evaluation(predictor, coco, instances, args.output_dir, args.subset)
    import json
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
