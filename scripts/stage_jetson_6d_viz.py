#!/usr/bin/env python3
"""Stage Jetson-6D: 50ep vs historical FULL COCO representative visualization.

Loads 50ep + historical encoders (same frozen decoder), re-runs predict on
representative instances drawn from the paired per-instance comparison, draws
a 4-panel image: original | GT | 50ep pred | historical pred.

Selection: 50ep-wins, historical-wins, near-parity, thin/small-low.
"""
import os, sys, json, csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/home/nvidia/jetson-deploy-lab")
sys.path.insert(0, str(REPO / "third_party/nanosam"))
COCO_ROOT = "data/coco/val2017"
ANN = "data/coco/annotations/instances_val2017.json"
DEC = "third_party/nanosam/data/mobile_sam_mask_decoder.engine"
ENC_50 = "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine"
ENC_H = "third_party/nanosam/data/resnet18_image_encoder.engine"
OUT = REPO / "results/stage_jetson_6d/visualizations"
CSV50 = REPO / "results/stage_jetson_6d/per_instance_primary.csv"
CSVH = REPO / "results/algorithm_b3_nanosam_full/per_instance_primary.csv"
SELECTED_MASK_INDEX = 0


def overlay(image, mask, color, alpha=120):
    arr = np.array(image).copy()
    m = mask.astype(bool)
    for c in range(3):
        ch = arr[:, :, c]
        ch[m] = ((ch[m].astype(int) * (255 - alpha) + color[c] * alpha) // 255).astype(np.uint8)
    return Image.fromarray(arr)


def label_panel(panel, title):
    canvas = Image.new("RGB", (panel.width, panel.height + 34), "white")
    canvas.paste(panel, (0, 34))
    ImageDraw.Draw(canvas).multiline_text((6, 3), title, fill="black",
                                          font=ImageFont.load_default(), spacing=2)
    return canvas


def main():
    import torch
    from pycocotools.coco import COCO
    from nanosam.utils.predictor import Predictor

    coco = COCO(ANN)

    def load_csv(p):
        out = {}
        for r in csv.DictReader(open(p)):
            if r.get("status") == "VALID" and r.get("slot0_iou"):
                out[int(r["annotation_id"])] = {
                    "iou": float(r["slot0_iou"]),
                    "category": r.get("category_name", "?"),
                    "size_bucket": r.get("size_bucket", "?"),
                    "image_id": int(r["image_id"]),
                }
        return out

    r50 = load_csv(CSV50)
    rh = load_csv(CSVH)
    common = sorted(set(r50) & set(rh))
    paired = []
    for aid in common:
        d = r50[aid]["iou"] - rh[aid]["iou"]
        paired.append({"aid": aid, "i50": r50[aid]["iou"], "ih": rh[aid]["iou"],
                       "d": d, "cat": r50[aid]["category"], "sz": r50[aid]["size_bucket"],
                       "image_id": r50[aid]["image_id"]})

    f50_wins = sorted(paired, key=lambda r: -r["d"])[:3]
    hist_wins = sorted(paired, key=lambda r: r["d"])[:3]
    near = sorted([r for r in paired if abs(r["d"]) < 0.005 and r["i50"] > 0.75],
                  key=lambda r: -r["i50"])[:3]
    thin = sorted([r for r in paired if r["cat"] in ("skis", "fork", "knife", "spoon", "scissors", "tie", "snowboard")],
                  key=lambda r: r["i50"])[:3]
    small_low = sorted([r for r in paired if r["sz"] == "small" and r["i50"] < 0.4],
                       key=lambda r: r["i50"])[:2]

    picked = []
    for bucket, lst, tag in [("f50_wins", f50_wins, "fifty_wins"),
                             ("hist_wins", hist_wins, "historical_wins"),
                             ("near_par", near, "near_parity"),
                             ("thin", thin, "thin_structure"),
                             ("small_low", small_low, "small_low_iou")]:
        for r in lst:
            if r["aid"] not in [p["aid"] for p in picked]:
                r["tag"] = tag
                picked.append(r)
    picked = picked[:12]
    if not picked:
        print("no viz candidates"); return
    print(f"[viz] {len(picked)} instances")
    OUT.mkdir(parents=True, exist_ok=True)

    print("[viz] loading 2 predictors (50ep/historical)...")
    p50 = Predictor(ENC_50, DEC)
    ph = Predictor(ENC_H, DEC)

    def predict_one(pred, image, bbox):
        pred.set_image(image)
        x1, y1, x2, y2 = bbox
        pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        labels = np.array([2, 3])
        hi, mi, lo = pred.predict(points=pts, point_labels=labels)
        return (hi[0, SELECTED_MASK_INDEX] > 0).detach().cpu().numpy()

    manifest = []
    by_img = {}
    for idx, r in enumerate(picked):
        a = coco.loadAnns([r["aid"]])[0]
        img_info = coco.loadImgs([a["image_id"]])[0]
        ip = os.path.join(COCO_ROOT, img_info["file_name"])
        x, y, w, h = a["bbox"]
        bbox = [float(x), float(y), float(x + w), float(y + h)]
        image = Image.open(ip).convert("RGB")
        gt = (coco.annToMask(a) > 0)
        try:
            m50 = predict_one(p50, image, bbox)
            mh = predict_one(ph, image, bbox)
        except Exception as e:
            print(f"  ann {r['aid']} predict failed: {e}"); continue

        def iou(a, b):
            u = int(np.count_nonzero(a | b))
            return float(np.count_nonzero(a & b) / u) if u else 0.0
        i50, ih = iou(m50, gt), iou(mh, gt)

        panels = [
            label_panel(image.copy(), f"{r['cat']} ann={r['aid']}\n{r['sz']} [{r['tag']}]"),
            label_panel(overlay(image, gt, (0, 220, 0)), "GT"),
            label_panel(overlay(image, m50, (40, 100, 230)), f"50ep\nIoU={i50:.3f}"),
            label_panel(overlay(image, mh, (240, 180, 20)), f"historical\nIoU={ih:.3f}"),
        ]
        maxw = 300
        rp = [p.resize((maxw, int(p.height * maxw / p.width)), Image.BILINEAR) if p.width > maxw else p for p in panels]
        W = sum(p.width for p in rp); H = max(p.height for p in rp)
        canvas = Image.new("RGB", (W, H), "white")
        x = 0
        for p in rp:
            canvas.paste(p, (x, 0)); x += p.width
        fn = f"viz_{idx:02d}_ann{r['aid']}_{r['sz']}_{r['tag']}.png"
        canvas.save(OUT / fn, quality=90)
        manifest.append({"file": fn, "annotation_id": r["aid"], "category": r["cat"],
                         "size_bucket": r["sz"], "selection": r["tag"],
                         "iou_50ep": i50, "iou_historical": ih, "delta": i50 - ih})
        print(f"  {fn}  50={i50:.3f} hist={ih:.3f} d={i50-ih:+.3f}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[viz] done, {len(manifest)} pngs in {OUT}")


if __name__ == "__main__":
    main()
