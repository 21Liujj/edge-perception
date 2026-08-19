#!/usr/bin/env python3
"""Stage 5B-2: ~10 representative visualizations.
Re-runs NanoSAM predictor (self-trained encoder + frozen decoder, slot0) on
selected instances to overlay predicted vs GT mask. Selection covers:
- worst self-trained degradations
- best self-trained improvements
- small-object cases
Saves small PNGs to results/stage_jetson_5b2/visualizations/.
"""
import json, sys, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/nvidia/jetson-deploy-lab")
EVID = ROOT / "results" / "stage_jetson_5b2"
VIZ = EVID / "visualizations"; VIZ.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "third_party/nanosam"))
sys.path.insert(0, str(ROOT / "scripts"))
from pycocotools.coco import COCO
from nanosam.utils.predictor import Predictor

ENC = str(ROOT / "models/self_distilled/epoch10/nanosam_resnet18_epoch009_fp16.engine")
DEC = str(ROOT / "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
COCO_ANN = str(ROOT / "data/coco/annotations/instances_val2017.json")

coco = COCO(COCO_ANN)

# load paired deltas
pi_self = [json.loads(l) for l in open(EVID / "self_trained_standard500/per_instance.jsonl")]
hist_map = {json.loads(l)["annotation_id"]: json.loads(l) for l in open(ROOT / "results/coco_nanosam_baseline/standard/per_instance.jsonl")}
rows = []
for r in pi_self:
    h = hist_map.get(r["annotation_id"])
    if h:
        rows.append({**r, "hist_iou": h["iou"], "delta": r["iou"] - h["iou"]})

# selection: 4 worst delta, 3 best delta, 3 small-object (any)
rows_by_delta = sorted(rows, key=lambda x: x["delta"])
worst = rows_by_delta[:4]
best = sorted(rows, key=lambda x: x["delta"], reverse=True)[:3]
small = [r for r in rows if r["size_bucket"] == "small"][:3]
sel = worst + best + small
# dedupe
seen = set(); selected = []
for r in sel:
    if r["annotation_id"] in seen: continue
    seen.add(r["annotation_id"]); selected.append(r)

predictor = Predictor(ENC, DEC)

def iou(a, b):
    inter = np.logical_and(a, b).sum(); uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni > 0 else 0.0

idx = 0
manifest_entries = []
for r in selected:
    img_id = r["image_id"]; ann_id = r["annotation_id"]
    img_info = coco.loadImgs(img_id)[0]
    img_path = str(ROOT / "data/coco/val2017" / img_info["file_name"])
    ann = coco.loadAnns(ann_id)[0]
    gt = coco.annToMask(ann) > 0
    H, W = gt.shape
    try:
        img_pil = Image.open(img_path).convert("RGB")
    except Exception as e:
        continue
    # box prompt from GT bbox_xywh
    x, y, w, h = r["bbox_xywh"]
    points = [[x, y], [x + w, y + h]]
    predictor.set_image(img_pil)
    hi_res_mask, mask_iou, low_res_mask = predictor.predict(points=np.array(points), point_labels=np.array([2, 3]))
    pred = hi_res_mask[0, 0].cpu().numpy() > 0
    # resize pred to image size if needed
    if pred.shape != (H, W):
        pred = np.array(Image.fromarray(pred.astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
    gt_r = np.array(Image.fromarray(gt.astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
    i = iou(pred, gt_r)
    # overlay
    ov = np.array(img_pil.resize((W, H)))
    red = (pred.astype(np.uint8) * 255)
    green = (gt_r.astype(np.uint8) * 255)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(ov); ax[0].set_title("image"); ax[0].axis("off")
    ax[1].imshow(pred, cmap="Reds"); ax[1].set_title(f"pred slot0 IoU={i:.3f}"); ax[1].axis("off")
    ax[2].imshow(gt_r, cmap="Greens"); ax[2].set_title("GT mask"); ax[2].axis("off")
    ov2 = ov.copy(); ov2[pred, 0] = 255; ov2[gt_r, 1] = 255
    ax[3].imshow(ov2); ax[3].set_title("overlay (R=pred G=GT)"); ax[3].axis("off")
    tag = r["size_bucket"]
    fig.suptitle(f"ann={ann_id} cat={r['category_name']} {tag} | self={r['iou']:.3f} hist={r['hist_iou']:.3f} d={r['delta']:+.3f}", fontsize=10)
    plt.tight_layout()
    fn = VIZ / f"viz_{idx:02d}_ann{ann_id}_{tag}_{('worse' if r['delta']<-0.001 else 'better' if r['delta']>0.001 else 'equal')}.png"
    plt.savefig(fn, dpi=70, bbox_inches="tight"); plt.close()
    manifest_entries.append({"file": fn.name, "annotation_id": ann_id,
        "category": r["category_name"], "size_bucket": tag,
        "self_iou": r["iou"], "historical_iou": r["hist_iou"], "delta": r["delta"],
        "rerun_iou": i, "selection": "worst" if r in worst else ("best" if r in best else "small")})
    idx += 1
    if idx >= 10: break

json.dump(manifest_entries, open(VIZ / "manifest.json", "w"), indent=2)
print(f"wrote {idx} visualizations to {VIZ}")
for m in manifest_entries:
    print(m["file"], m["delta"], m["size_bucket"])
