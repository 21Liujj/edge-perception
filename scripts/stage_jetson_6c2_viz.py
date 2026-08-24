#!/usr/bin/env python3
"""Stage Jetson-6C-2: failure visualization for the three-way comparison.

Loads three encoders (50ep / 10ep / historical) with the same frozen decoder,
re-runs predict on a handful of representative instances, and draws a panel
per instance: original | GT | 50ep pred | 10ep pred | historical pred.

Selection aims to cover: 10ep-worse-50ep-improved, both-worse-small, thin,
boundary leakage — drawn from the per-instance CSV produced by analyze.py.
"""
import os, sys, json, csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/home/nvidia/jetson-deploy-lab")
sys.path.insert(0, str(REPO / "third_party/nanosam"))
COCO_ROOT = "data/coco/val2017"
ANN = "data/coco/annotations/instances_val2017.json"
MANIFEST = "data/eval_manifests/standard_manifest.json"
DEC = "third_party/nanosam/data/mobile_sam_mask_decoder.engine"
ENC_50 = "models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine"
ENC_10 = "models/self_distilled/epoch10/nanosam_resnet18_epoch009_fp16.engine"
ENC_H = "third_party/nanosam/data/resnet18_image_encoder.engine"
OUT = REPO / "results/stage_jetson_6c2/visualizations"
SELECTED_MASK_INDEX = 0
CSV_PATH = REPO / "results/stage_jetson_6c2/standard500_instances.csv"


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
    # load the manifest instances (frozen, carries image_path + bbox_xywh)
    man = json.loads(Path(MANIFEST).read_text())
    insts = man["instances"] if isinstance(man, dict) and "instances" in man else man

    # build annotation_id -> coco ann + image_path + bbox_xyxy
    by_ann = {}
    for inst in insts:
        aid = inst["annotation_id"]
        a = coco.loadAnns([aid])[0]
        img_info = coco.loadImgs([a["image_id"]])[0]
        ip = os.path.join(COCO_ROOT, img_info["file_name"])
        x, y, w, h = a["bbox"]
        by_ann[aid] = {"ann": a, "image_path": ip,
                       "bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)],
                       "category": coco.loadCats([a["category_id"]])[0]["name"]}

    # read three-way CSV to pick representative anns
    rows = list(csv.DictReader(open(CSV_PATH)))
    def f(v):
        try: return float(v)
        except: return None
    for r in rows:
        r["i50"] = f(r["iou_50ep"]); r["i10"] = f(r["iou_10ep"]); r["ih"] = f(r["iou_historical"])
        r["d50_10"] = (r["i50"] - r["i10"]) if (r["i50"] is not None and r["i10"] is not None) else None
        r["category"] = r.get("category_name") or r.get("category")

    # selection buckets
    valid = [r for r in rows if r["i50"] is not None and r["i10"] is not None and r["ih"] is not None]
    improved = sorted([r for r in valid if (r["d50_10"] or 0) > 0.02], key=lambda r: r["d50_10"], reverse=True)
    both_bad_small = sorted([r for r in valid if r["size_bucket"] == "small" and r["i50"] < 0.45 and r["i10"] < 0.45], key=lambda r: r["i50"])
    near_par = sorted([r for r in valid if abs((r["i50"] or 0) - (r["ih"] or 0)) < 0.01 and r["i50"] > 0.7], key=lambda r: r["i50"], reverse=True)
    thin = sorted([r for r in valid if r["category"] in ("skis", "scissors", "snowboard", "fork", "tie")], key=lambda r: r["i50"])

    picked = []
    for bucket, lst, tag in [("improved", improved, "10worse_50better"),
                             ("bothbad_small", both_bad_small, "both_bad_small"),
                             ("near_par", near_par, "near_parity"),
                             ("thin", thin, "thin_structure")]:
        for r in lst[:2]:
            if r["annotation_id"] not in [p["annotation_id"] for p in picked]:
                r["tag"] = tag; picked.append(r)
    picked = picked[:8]

    if not picked:
        print("no viz candidates"); return

    print(f"[viz] {len(picked)} instances")
    OUT.mkdir(parents=True, exist_ok=True)

    # load three predictors (lazy: only when first needed, to bound GPU mem)
    def mk(enc):
        return Predictor(enc, DEC)
    print("[viz] loading 3 predictors (50ep/10ep/historical)...")
    p50, p10, ph = mk(ENC_50), mk(ENC_10), mk(ENC_H)

    def predict_one(pred, image, bbox):
        pred.set_image(image)
        x1, y1, x2, y2 = bbox
        pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        labels = np.array([2, 3])
        hi, mi, lo = pred.predict(points=pts, point_labels=labels)
        return (hi[0, SELECTED_MASK_INDEX] > 0).detach().cpu().numpy()

    manifest = []
    for idx, r in enumerate(picked):
        aid = int(r["annotation_id"])
        meta = by_ann[aid]
        image = Image.open(meta["image_path"]).convert("RGB")
        gt = (coco.annToMask(meta["ann"]) > 0)
        bbox = meta["bbox_xyxy"]
        try:
            m50 = predict_one(p50, image, bbox)
            m10 = predict_one(p10, image, bbox)
            mh = predict_one(ph, image, bbox)
        except Exception as e:
            print(f"  ann {aid} predict failed: {e}"); continue

        def iou(a, b):
            u = int(np.count_nonzero(a | b))
            if u == 0: return 0.0
            return float(np.count_nonzero(a & b) / u)
        i50, i10, ih = iou(m50, gt), iou(m10, gt), iou(mh, gt)
        cn = meta["category"]; sb = r["size_bucket"]; tag = r["tag"]

        panels = [
            label_panel(image.copy(), f"{cn} ann={aid}\n{sb} [{tag}]"),
            label_panel(overlay(image, gt, (0, 220, 0)), f"GT"),
            label_panel(overlay(image, m50, (40, 100, 230)), f"50ep\nIoU={i50:.3f}"),
            label_panel(overlay(image, m10, (220, 40, 40)), f"10ep\nIoU={i10:.3f}"),
            label_panel(overlay(image, mh, (240, 180, 20)), f"historical\nIoU={ih:.3f}"),
        ]
        # resize each to max width 280
        maxw = 280
        rp = []
        for p in panels:
            if p.width > maxw:
                p = p.resize((maxw, int(p.height * maxw / p.width)), Image.BILINEAR)
            rp.append(p)
        W = sum(p.width for p in rp); H = max(p.height for p in rp)
        canvas = Image.new("RGB", (W, H), "white")
        x = 0
        for p in rp:
            canvas.paste(p, (x, 0)); x += p.width
        fn = f"viz_{idx:02d}_ann{aid}_{sb}_{tag}.png"
        canvas.save(OUT / fn, quality=90)
        manifest.append({"file": fn, "annotation_id": aid, "category": cn,
                         "size_bucket": sb, "selection": tag,
                         "iou_50ep": i50, "iou_10ep": i10, "iou_historical": ih})
        print(f"  {fn}  50={i50:.3f} 10={i10:.3f} hist={ih:.3f}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[viz] done, {len(manifest)} pngs in {OUT}")


if __name__ == "__main__":
    main()
