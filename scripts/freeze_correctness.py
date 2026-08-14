# -*- coding: utf-8 -*-
"""Stage 7B-Freeze — correctness gates for the experimental fused CUDA preprocess.

Produces:
  tensor_numerical_comparison.json   (baseline PIL-BICUBIC tensor vs CUDA-bilinear tensor)
  owl_regression.json                (owl joint: label/score/box/mask; baseline-vs-cuda mask IoU)
  dogs_regression.json               (dogs box [100,100,850,759], slot 0; baseline-vs-cuda mask IoU)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np
import torch
import PIL.Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/runtime_optimization/stage7b_freeze")
os.makedirs(OUT, exist_ok=True)
SIZE = 1024
MEAN = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
STD = torch.tensor([58.395, 57.12, 57.375])[:, None, None]
ENC = os.path.join(ROOT, "third_party/nanosam/data/resnet18_image_encoder.engine")
DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
OWL = os.path.join(ROOT, "third_party/nanoowl/assets/owl_glove_small.jpg")
DOGS = os.path.join(ROOT, "third_party/nanosam/assets/dogs.jpg")


def resized_dims(image, size=SIZE):
    aspect = image.width / image.height
    if aspect >= 1:
        return int(size / aspect), size
    return size, int(size * aspect)


def baseline_tensor(image):
    rh, rw = resized_dims(image)
    arr = np.asarray(image.resize((rw, rh)))  # PIL default == BICUBIC
    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    out = torch.zeros((1, 3, SIZE, SIZE))
    out[0, :, :rh, :rw] = (t - MEAN) / STD
    return out, rh, rw


def pct(a, p):
    a = sorted(a) if isinstance(a, list) else sorted(a.tolist())
    if not a:
        return float("nan")
    k = (len(a) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(a) - 1)
    return a[f] + (a[c] - a[f]) * (k - f)


def mask_iou(a, b):
    a = a > 0.5
    b = b > 0.5
    inter = int((a & b).sum())
    uni = int((a | b).sum())
    return float(inter / uni) if uni > 0 else 0.0


def tensor_compare():
    from edge_perception.runtime.cuda_preprocess import CudaPreprocessor
    prep = CudaPreprocessor(size=SIZE)
    recs = {}
    for name, path in {"owl": OWL, "dogs": DOGS}.items():
        img = PIL.Image.open(path).convert("RGB")
        bt, rh, rw = baseline_tensor(img)
        src_h, src_w = prep.prepare_source(img)
        prep.launch_fused(src_h, src_w, rh, rw)
        prep._stream.synchronize()
        ct = prep.image_tensor_gpu.detach().cpu()
        d = (bt - ct).abs()
        rd = (bt[0, :, :rh, :rw] - ct[0, :, :rh, :rw]).abs().flatten()
        recs[name] = {
            "src_size": list(img.size), "rh": rh, "rw": rw, "size": SIZE,
            "baseline_resample": "PIL_default_BICUBIC", "cuda_resample": "bilinear_half_pixel",
            "shape_match": list(bt.shape) == list(ct.shape),
            "dtype": str(ct.dtype),
            "region_max_abs": float(rd.max()), "region_mean_abs": float(rd.mean()),
            "region_rmse": float(rd.pow(2).mean().sqrt()),
            "region_p50_abs": float(pct(rd, 50)), "region_p95_abs": float(pct(rd, 95)),
            "region_p99_abs": float(pct(rd, 99)),
            "full_max_abs": float(d.max()), "pad_max_abs_cuda": float(ct[0, :, rh:, :].abs().max()),
        }
    return recs


def run_predictor(P, pil, points, labels):
    p = P(ENC, DEC)
    p.set_image(pil)
    hm, miou, low = p.predict(np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int64))
    hm = hm.detach().cpu()
    mask_bool = hm[0, 0].numpy() > 0
    return {
        "mask_bool": mask_bool,
        "pred_iou": float(miou[0, 0]),
        "low_res_shape": list(low.shape),
        "has_nan": bool(np.isnan(hm.numpy()).any()),
        "has_inf": bool(np.isinf(hm.numpy()).any()),
        "mask_nonzero": int(mask_bool.sum()),
    }


def gate_status(b, c):
    return (b["mask_nonzero"] > 0 and c["mask_nonzero"] > 0
            and not b["has_nan"] and not c["has_nan"]
            and not b["has_inf"] and not c["has_inf"])


def owl_regression():
    from nanosam.utils.predictor import Predictor
    from edge_perception.runtime.cuda_preprocess import CudaPreprocessPredictor
    from nanoowl.owl_predictor import OwlPredictor
    import cv2
    owl_engine = os.path.join(ROOT, "third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
    owl = OwlPredictor("google/owlvit-base-patch32", image_encoder_engine=owl_engine)
    bgr = cv2.imread(OWL)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = PIL.Image.fromarray(rgb).convert("RGB")
    H, W = pil.height, pil.width
    text = ["an owl", "a glove", "a finger"]
    enc = owl.encode_text(text)
    torch.cuda.synchronize()
    out = owl.predict(image=pil, text=text, text_encodings=enc, threshold=0.10, pad_square=False)
    torch.cuda.synchronize()
    labels = out.labels.detach().cpu().tolist()
    scores = out.scores.detach().cpu().tolist()
    boxes = out.boxes.detach().cpu().tolist()
    dets = [{"label": text[int(li)] if 0 <= int(li) < len(text) else str(li),
             "score": float(sc), "box": [float(v) for v in bx]}
            for li, sc, bx in zip(labels, scores, boxes)]
    # select owl detection
    owls = [d for d in dets if d["label"] == "an owl"]
    if not owls:
        return {"owl_PASS": False, "status": "NO_OWL_DETECTION"}
    sel = max(owls, key=lambda d: d["score"])
    # clip box
    x1, y1, x2, y2 = sel["box"]
    x1 = min(max(x1, 0.0), W); y1 = min(max(y1, 0.0), H)
    x2 = min(max(x2, 0.0), W); y2 = min(max(y2, 0.0), H)
    box_xyxy = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    pts = np.array([[box_xyxy[0], box_xyxy[1]], [box_xyxy[2], box_xyxy[3]]], dtype=np.float32)
    b = run_predictor(Predictor, pil, pts, [2, 3])
    c = run_predictor(CudaPreprocessPredictor, pil, pts, [2, 3])
    iou = mask_iou(b["mask_bool"], c["mask_bool"])
    out = {"detected_label": sel["label"], "score": round(sel["score"], 6), "box_xyxy": box_xyxy,
           "baseline": {k: b[k] for k in ("pred_iou", "low_res_shape", "has_nan", "has_inf", "mask_nonzero")},
           "cuda": {k: c[k] for k in ("pred_iou", "low_res_shape", "has_nan", "has_inf", "mask_nonzero")}}
    out["mask_iou_baseline_vs_cuda"] = iou
    out["mask_area_ratio"] = float(c["mask_nonzero"] / max(1, b["mask_nonzero"]))
    out["owl_PASS"] = gate_status(b, c)
    return out


def dogs_regression():
    from nanosam.utils.predictor import Predictor
    from edge_perception.runtime.cuda_preprocess import CudaPreprocessPredictor
    import cv2
    bgr = cv2.imread(DOGS)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = PIL.Image.fromarray(rgb)
    box = [100.0, 100.0, 850.0, 759.0]
    pts = np.array([[box[0], box[1]], [box[2], box[3]]], dtype=np.float32)
    b = run_predictor(Predictor, pil, pts, [2, 3])
    c = run_predictor(CudaPreprocessPredictor, pil, pts, [2, 3])
    iou = mask_iou(b["mask_bool"], c["mask_bool"])
    out = {"box": box, "mask_slot": 0,
           "baseline": {k: b[k] for k in ("pred_iou", "low_res_shape", "has_nan", "has_inf", "mask_nonzero")},
           "cuda": {k: c[k] for k in ("pred_iou", "low_res_shape", "has_nan", "has_inf", "mask_nonzero")}}
    out["mask_iou_baseline_vs_cuda"] = iou
    out["mask_area_ratio"] = float(c["mask_nonzero"] / max(1, b["mask_nonzero"]))
    out["dogs_PASS"] = gate_status(b, c)
    return out


def main():
    print("### tensor numerical comparison ###")
    tc = tensor_compare()
    json.dump(tc, open(os.path.join(OUT, "tensor_numerical_comparison.json"), "w"), indent=2)
    print(json.dumps(tc, indent=2))
    print("### owl regression ###")
    owl = owl_regression()
    json.dump(owl, open(os.path.join(OUT, "owl_regression.json"), "w"), indent=2)
    print(json.dumps(owl, indent=2))
    print("### dogs regression ###")
    dogs = dogs_regression()
    json.dump(dogs, open(os.path.join(OUT, "dogs_regression.json"), "w"), indent=2)
    print(json.dumps(dogs, indent=2))


if __name__ == "__main__":
    main()
