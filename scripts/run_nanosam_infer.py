#!/usr/bin/env python3
# Stage 3B NanoSAM inference: Box prompt + Point prompt on assets/dogs.jpg.
# Reuses official nanosam Predictor (no upstream modification). Records legality fields
# and saves JSON + visualization into results/nanosam_baseline/.
import os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import PIL.Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nanosam.utils.predictor import Predictor

# Resolve project root = parent of the scripts/ dir this file lives in.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DATA = os.path.join(ROOT, "third_party", "nanosam", "data")
IMG = os.path.join(ROOT, "third_party", "nanosam", "assets", "dogs.jpg")
OUT = os.path.join(ROOT, "results", "nanosam_baseline")
ENC = os.path.join(DATA, "resnet18_image_encoder.engine")
DEC = os.path.join(DATA, "mobile_sam_mask_decoder.engine")

os.makedirs(OUT, exist_ok=True)


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
        plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, x0], 'g-', linewidth=2)
    # draw points
    for (x, y), l in zip(points_xy, labels):
        if l == 1:
            plt.plot(x, y, 'r.', markersize=12)
        elif l == 0:
            plt.plot(x, y, 'b.', markersize=12)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()


import torch  # noqa: E402

print("Loading Predictor ...", flush=True)
p = Predictor(ENC, DEC)
print("PREDICTOR_LOADED_OK", flush=True)

image = PIL.Image.open(IMG).convert("RGB")
print("image size (W,H):", image.size, flush=True)
p.set_image(image)
emb = p.features
print("image_embeddings shape:", tuple(emb.shape), flush=True)

results = {}
results["image_path"] = IMG
results["image_size_W_H"] = list(image.size)
results["image_embeddings_shape"] = list(emb.shape)

# ---- Box prompt ----
bbox = [100, 100, 850, 759]
pts = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
labels = np.array([2, 3])
print("BOX predict ...", flush=True)
hi, iou, low = p.predict(pts, labels)
emb_box = {
    "prompt_type": "box",
    "bbox_xyxy": bbox,
    "points": pts.tolist(),
    "point_labels": labels.tolist(),
    "iou_predictions": iou.detach().cpu().numpy().ravel().tolist(),
    "low_res_mask_shape": list(low.shape),
    "hi_res_mask_shape": list(hi.shape),
    "has_nan_inf": has_nan_inf(hi),
}
# official basic_usage.py convention: best mask = index 0
mask_best = (hi[0, 0] > 0).detach().cpu().numpy()
emb_box.update(stats(mask_best))
emb_box["mask_convention"] = "mask[0,0] > 0 ( NanoSAM basic_usage.py best-mask slot 0)"
print("BOX done:", emb_box, flush=True)
viz(image, mask_best, [[bbox[0], bbox[1]], [bbox[2], bbox[3]]], [2, 3],
    os.path.join(OUT, "box_prompt_visualization.jpg"), box=bbox)
results["box_prompt"] = emb_box

with open(os.path.join(OUT, "box_prompt_result.json"), "w") as f:
    json.dump(emb_box, f, indent=2)

# ---- Point prompt (foreground, label=1) ----
# Choose a point clearly inside a dog. dogs.jpg bbox was [100,100,850,759] (left/upper region).
# A foreground point near the centroid of that region.
point = np.array([[475, 430]])
plabel = np.array([1])
print("POINT predict ...", flush=True)
hi2, iou2, low2 = p.predict(point, plabel)
emb_pt = {
    "prompt_type": "point",
    "point_xy": point.tolist(),
    "point_labels": plabel.tolist(),
    "iou_predictions": iou2.detach().cpu().numpy().ravel().tolist(),
    "low_res_mask_shape": list(low2.shape),
    "hi_res_mask_shape": list(hi2.shape),
    "has_nan_inf": has_nan_inf(hi2),
}
mask_best2 = (hi2[0, 0] > 0).detach().cpu().numpy()
emb_pt.update(stats(mask_best2))
emb_pt["mask_convention"] = "mask[0,0] > 0 (best-mask slot 0)"
print("POINT done:", emb_pt, flush=True)
viz(image, mask_best2, [[475, 430]], [1],
    os.path.join(OUT, "point_prompt_visualization.jpg"), box=None)
results["point_prompt"] = emb_pt
with open(os.path.join(OUT, "point_prompt_result.json"), "w") as f:
    json.dump(emb_pt, f, indent=2)

with open(os.path.join(OUT, "inference_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print("ALL_DONE", flush=True)
