#!/usr/bin/env python3
"""Reusable NanoSAM promptable-segmentation CLI.

Reuses the official nanosam Predictor (no upstream modification). Runs a single
prompt (box or point) on one image with the fixed TensorRT engines and writes
legality fields + reusable mask data + visualization + timing.

Mask selection rule (read from source, NOT from memory):
  * The MobileSAM mask decoder emits 4 candidate masks per prompt:
        low_res_masks  shape [1, 4, 256, 256]
        iou_predictions shape [1, 4]            (predicted IoU of each candidate)
    See nanosam/utils/predictor.py: run_mask_decoder() returns iou_predictions,
    low_res_masks both with leading dim 4 candidates; Predictor.predict() upscales
    the full 4-candidate batch to hi_res_mask [1, 4, H, W] via upscale_mask().
  * The official example (examples/basic_usage.py) selects a FIXED slot, not the
    IoU argmax:
        mask = (mask[0, 0] > 0).detach().cpu().numpy()
    => selected mask index = 0 ; selected predicted IoU = iou_predictions[0].
  * This CLI follows that same convention (configurable but default slot 0) and
    records both the selected index and its predicted IoU.
"""
import os
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import PIL.Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from nanosam.utils.predictor import Predictor

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))


def stats(mask_bool_np):
    nz = int(mask_bool_np.sum())
    total = int(mask_bool_np.size)
    return {
        "nonzero_pixels": nz,
        "mask_area_ratio": round(nz / total, 6) if total else 0.0,
        "mask_shape": list(mask_bool_np.shape),
    }


def has_nan_inf(t):
    t = t.detach().float()
    return {"nan": bool(torch.isnan(t).any().item()), "inf": bool(torch.isinf(t).any().item())}


def viz(image, mask_bool_np, points_xy, labels, out_path, box=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.imshow(mask_bool_np, alpha=0.5)
    if box is not None:
        x0, y0, x1, y1 = box
        plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, x0], "g-", linewidth=2)
    for (x, y), l in zip(points_xy, labels):
        if l == 1:
            plt.plot(x, y, "r.", markersize=12)
        elif l == 0:
            plt.plot(x, y, "b.", markersize=12)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()


def build_prompt(args):
    """Return (points np[N,2], labels np[N], bbox_or_None) from CLI args."""
    if args.prompt_type == "box":
        bbox = list(args.box)
        pts = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
        labels = np.array([2, 3])  # 2=box TL, 3=box BR (NanoSAM/SAM convention)
        return pts, labels, bbox
    else:  # point
        pts = np.array([list(args.point)])
        labels = np.array([args.point_label])  # 1=fg, 0=bg
        return pts, labels, None


def main():
    ap = argparse.ArgumentParser(description="NanoSAM reusable segmentation CLI")
    ap.add_argument("--image", default=os.path.join(ROOT, "third_party/nanosam/assets/dogs.jpg"))
    ap.add_argument("--prompt-type", choices=["box", "point"], default="box")
    ap.add_argument("--box", nargs=4, type=int, default=[100, 100, 850, 759],
                    metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--point", nargs=2, type=int, default=[475, 430],
                    metavar=("X", "Y"))
    ap.add_argument("--point-label", type=int, default=1, choices=[0, 1])
    ap.add_argument("--image-encoder-engine",
                    default=os.path.join(ROOT, "third_party/nanosam/data/resnet18_image_encoder.engine"))
    ap.add_argument("--mask-decoder-engine",
                    default=os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine"))
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "results/nanosam_baseline"))
    ap.add_argument("--mask-index", type=int, default=0,
                    help="which of the 4 decoder candidates to select (default 0 = basic_usage.py)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    # ---- model load (timed) ----
    t0 = time.perf_counter()
    predictor = Predictor(args.image_encoder_engine, args.mask_decoder_engine)
    torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"model_load_ms={model_load_ms:.3f}", flush=True)

    image = PIL.Image.open(args.image).convert("RGB")
    print(f"image size (W,H): {image.size}", flush=True)

    # ---- set_image (timed) ----
    t0 = time.perf_counter()
    predictor.set_image(image)
    torch.cuda.synchronize()
    set_image_ms = (time.perf_counter() - t0) * 1000.0
    emb = predictor.features
    print(f"image_embeddings shape: {tuple(emb.shape)}", flush=True)

    # ---- prompt + decode (timed) ----
    pts, labels, bbox = build_prompt(args)
    t0 = time.perf_counter()
    hi, iou, low = predictor.predict(pts, labels)
    torch.cuda.synchronize()
    prompt_decode_ms = (time.perf_counter() - t0) * 1000.0
    full_segmentation_ms = set_image_ms + prompt_decode_ms

    iou_np = iou.detach().cpu().numpy().ravel()
    sel_idx = int(args.mask_index)
    sel_iou = float(iou_np[sel_idx])
    mask_best = (hi[0, sel_idx] > 0).detach().cpu().numpy()

    # prefix output files by prompt type so box/point runs don't overwrite each other
    tag = args.prompt_type
    result = {
        "prompt_type": args.prompt_type,
        "image_path": args.image,
        "image_size_W_H": list(image.size),
        "image_embeddings_shape": list(emb.shape),
        "bbox_xyxy": bbox,
        "points": pts.tolist(),
        "point_labels": labels.tolist(),
        "mask_decoder_output_count": 4,
        "mask_selection_rule": "fixed slot (basic_usage.py: mask[0,0] > 0); not IoU argmax",
        "selected_mask_index": sel_idx,
        "selected_predicted_iou": sel_iou,
        "iou_predictions_all": iou_np.tolist(),
        "low_res_mask_shape": list(low.shape),
        "hi_res_mask_shape": list(hi.shape),
        "has_nan_inf": has_nan_inf(hi),
        "timing_ms": {
            "model_load_ms": round(model_load_ms, 3),
            "set_image_ms": round(set_image_ms, 3),
            "prompt_decode_ms": round(prompt_decode_ms, 3),
            "full_segmentation_ms": round(full_segmentation_ms, 3),
        },
    }
    result.update(stats(mask_best))

    # ---- write outputs ----
    with open(os.path.join(args.output_dir, f"{tag}_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.save(os.path.join(args.output_dir, f"{tag}_mask.npy"), mask_best)
    with open(os.path.join(args.output_dir, f"{tag}_timing.json"), "w") as f:
        json.dump(result["timing_ms"], f, indent=2)
    viz(image, mask_best, pts.tolist(), labels.tolist(),
        os.path.join(args.output_dir, f"{tag}_visualization.jpg"), box=bbox)

    print(f"{tag.upper()} done: index={sel_idx} iou={sel_iou:.4f} "
          f"nonzero={result['nonzero_pixels']} ratio={result['mask_area_ratio']} "
          f"nan/inf={result['has_nan_inf']}", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
