# -*- coding: utf-8 -*-
"""Stage 6A - MP4 -> NanoOWL -> NanoSAM -> Overlay -> MP4 video pipeline.

Basic, correct, reusable video inference loop. NOT a real-time optimization.
Input MP4 -> OpenCV VideoCapture -> frames -> NanoOWL -> selected box ->
NanoSAM (slot 0) -> mask overlay -> output MP4.

Models are loaded ONCE and reused across frames (no per-frame engine reload).
Per-frame preprocessing / coordinate conventions are identical to the Day 4
single-image joint pipeline (scripts/run_edge_perception.py):

  * OpenCV frame is BGR; convert to RGB PIL.Image before NanoOWL/NanoSAM (same
    RGB convention as the single-image path which opens with PIL RGB).
  * NanoOWL boxes are xyxy in original image pixels (pad_square=False).
  * NanoSAM box prompt: points=[[x1,y1],[x2,y2]], labels=[2,3], original px.
  * Mask selection: fixed slot 0 (official basic_usage.py), NOT IoU argmax.

NanoSAM image embedding is re-set_image'd each processed frame (the synthetic
video pans, so the embedding changes per frame). Correct path; the video
pipeline does NOT share embeddings across different frames.

NO_DETECTION branch: skip NanoSAM, write the original frame with a marker, do
not crash.

This script does NOT rebuild engines, modify upstream, or define real-time FPS.
Timing here is correctness profiling only.
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import cv2  # noqa: E402
import PIL.Image  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from nanoowl.owl_predictor import OwlPredictor  # noqa: E402
from nanosam.utils.predictor import Predictor as NanosamPredictor  # noqa: E402

DEFAULT_OWL_ENGINE = os.path.join(ROOT, "third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
DEFAULT_NANOSAM_ENC = os.path.join(ROOT, "third_party/nanosam/data/resnet18_image_encoder.engine")
DEFAULT_NANOSAM_DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
DEFAULT_MODEL = "google/owlvit-base-patch32"
SELECTED_MASK_INDEX = 0  # official slot 0, NOT IoU argmax


def parse_args():
    ap = argparse.ArgumentParser(description="MP4 EdgePerception joint pipeline (NanoOWL->NanoSAM->overlay)")
    ap.add_argument("--input-video", required=True)
    ap.add_argument("--output-video", required=True)
    ap.add_argument("--prompts", default="an owl,a glove,a finger")
    ap.add_argument("--target-label", default="an owl")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--nanoowl-engine", default=DEFAULT_OWL_ENGINE)
    ap.add_argument("--nanosam-image-encoder-engine", default=DEFAULT_NANOSAM_ENC)
    ap.add_argument("--nanosam-mask-decoder-engine", default=DEFAULT_NANOSAM_DEC)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--process-every-n", type=int, default=1,
                    help="process 1-in-N frames through the model; skipped frames still written")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    return ap.parse_args()


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    cx1 = min(max(float(x1), 0.0), float(w))
    cy1 = min(max(float(y1), 0.0), float(h))
    cx2 = min(max(float(x2), 0.0), float(w))
    cy2 = min(max(float(y2), 0.0), float(h))
    cx1, cx2 = min(cx1, cx2), max(cx1, cx2)
    cy1, cy2 = min(cy1, cy2), max(cy1, cy2)
    return [cx1, cy1, cx2, cy2]


def detect_on_image(owl, image_pil_rgb, text, text_encodings, threshold):
    """NanoOWL on a PIL RGB image. Returns list of (label_idx, label, score, box_xyxy)."""
    output = owl.predict(image=image_pil_rgb, text=text, text_encodings=text_encodings,
                         threshold=threshold, pad_square=False)
    cuda_sync()
    labels = output.labels.detach().cpu().tolist()
    scores = output.scores.detach().cpu().tolist()
    boxes = output.boxes.detach().cpu().tolist()
    dets = []
    for li, sc, bx in zip(labels, scores, boxes):
        li = int(li)
        dets.append({
            "label_index": li,
            "label": text[li] if 0 <= li < len(text) else str(li),
            "score": float(sc),
            "box": [float(v) for v in bx],
        })
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets


def select_detection(dets, target_label):
    if not dets:
        return None
    if target_label is not None:
        matches = [d for d in dets if d["label"] == target_label]
        if matches:
            return matches[0]
        return None
    return dets[0]


def overlay_mask_box(frame_bgr, mask_bool, box_xyxy, label, score, status):
    out = frame_bgr.copy()
    if mask_bool is not None and mask_bool.any():
        color = np.array([0, 255, 0], dtype=np.uint8)
        if mask_bool.shape[:2] == out.shape[:2]:
            out[mask_bool] = (0.5 * out[mask_bool].astype(np.float32)
                              + 0.5 * color.astype(np.float32)).astype(np.uint8)
    if box_xyxy is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{label} {score:.3f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"status={status}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
    text = [t.strip() for t in args.prompts.split(",") if t.strip()]

    runlog_path = os.path.join(os.path.dirname(os.path.abspath(args.output_video)), "run.log")
    runlog = open(runlog_path, "w")

    def log(msg):
        print(msg, flush=True)
        runlog.write(msg + "\n")
        runlog.flush()

    timing = {"model_load": {}}

    cuda_sync()
    t0 = time.perf_counter()
    owl = OwlPredictor(args.model, image_encoder_engine=args.nanoowl_engine)
    cuda_sync()
    owl_load_ms = (time.perf_counter() - t0) * 1000.0
    timing["model_load"]["nanoowl_load_ms"] = round(owl_load_ms, 3)

    text_encodings = owl.encode_text(text)
    cuda_sync()

    cuda_sync()
    t0 = time.perf_counter()
    nsam = NanosamPredictor(args.nanosam_image_encoder_engine, args.nanosam_mask_decoder_engine)
    cuda_sync()
    nsam_load_ms = (time.perf_counter() - t0) * 1000.0
    timing["model_load"]["nanosam_load_ms"] = round(nsam_load_ms, 3)
    timing["model_load"]["total_model_load_ms"] = round(owl_load_ms + nsam_load_ms, 3)
    log(f"model_load: nanoowl={owl_load_ms:.1f}ms nanosam={nsam_load_ms:.1f}ms")

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        log(f"ERROR: cannot open input video {args.input_video}")
        return 2
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = in_fps if in_fps and in_fps > 0 else 10.0
    writer = cv2.VideoWriter(args.output_video, fourcc, out_fps, (in_w, in_h))
    if not writer.isOpened():
        log("ERROR: cannot open output video writer")
        return 3

    frames_jsonl = os.path.join(os.path.dirname(os.path.abspath(args.output_video)), "frames.jsonl")
    fj = open(frames_jsonl, "w")

    per_frame_timing = []
    detected = 0
    no_detection = 0
    processed = 0
    written = 0
    out_shape = None
    seen_shapes = set()

    frame_index = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame_index >= args.max_frames:
            break

        seen_shapes.add(frame.shape)
        rec = {
            "frame_index": frame_index,
            "input_timestamp_ms": round(frame_index * 1000.0 / out_fps, 3),
        }
        ft = {}

        t0 = time.perf_counter()
        frame_bgr = frame
        decode_ms = (time.perf_counter() - t0) * 1000.0
        ft["decode_ms"] = round(decode_ms, 3)

        do_process = (frame_index % max(1, args.process_every_n)) == 0
        status = "SKIPPED"
        selected = None
        mask_bool = None
        clipped_box = None

        if do_process:
            processed += 1
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_pil = PIL.Image.fromarray(rgb).convert("RGB")
            bridge_ms_decode = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            dets = detect_on_image(owl, image_pil, text, text_encodings, args.threshold)
            nanoowl_ms = (time.perf_counter() - t0) * 1000.0
            ft["nanoowl_ms"] = round(nanoowl_ms, 3)

            t0 = time.perf_counter()
            selected = select_detection(dets, args.target_label)
            if selected is not None:
                clipped_box = clip_box(selected["box"], in_w, in_h)
                point_coords = [[clipped_box[0], clipped_box[1]], [clipped_box[2], clipped_box[3]]]
                point_labels = [2, 3]
            bridge_ms = (time.perf_counter() - t0) * 1000.0 + bridge_ms_decode
            ft["bridge_ms"] = round(bridge_ms, 3)

            if selected is None:
                status = "NO_DETECTION"
                no_detection += 1
                ft["nanosam_set_image_ms"] = 0.0
                ft["nanosam_decode_ms"] = 0.0
            else:
                status = "DETECTED"
                t0 = time.perf_counter()
                nsam.set_image(image_pil)
                cuda_sync()
                nanosam_set_image_ms = (time.perf_counter() - t0) * 1000.0
                ft["nanosam_set_image_ms"] = round(nanosam_set_image_ms, 3)

                pts = np.array([[point_coords[0][0], point_coords[0][1]],
                                [point_coords[1][0], point_coords[1][1]]], dtype=np.float32)
                labs = np.array(point_labels)

                t0 = time.perf_counter()
                hi, iou_pred, low = nsam.predict(pts, labs)
                cuda_sync()
                nanosam_decode_ms = (time.perf_counter() - t0) * 1000.0
                ft["nanosam_decode_ms"] = round(nanosam_decode_ms, 3)

                hi_f = hi.detach().float()
                has_nan = bool(torch.isnan(hi_f).any().item())
                has_inf = bool(torch.isinf(hi_f).any().item())
                if has_nan or has_inf:
                    status = "INFERENCE_NAN_INF"

                mask_bool = (hi[0, SELECTED_MASK_INDEX] > 0).detach().cpu().numpy()
                if mask_bool.shape[:2] != (in_h, in_w):
                    m_u8 = (mask_bool.astype(np.uint8) * 255)
                    m_u8 = cv2.resize(m_u8, (in_w, in_h), interpolation=cv2.INTER_NEAREST)
                    mask_bool = (m_u8 > 127)
                detected += 1
                rec["selected_label"] = selected["label"]
                rec["score"] = round(selected["score"], 6)
                rec["box"] = clipped_box
                rec["mask_area_ratio"] = round(float(mask_bool.sum()) / float(mask_bool.size), 6) if mask_bool.size else 0.0
                rec["selected_mask_index"] = SELECTED_MASK_INDEX
                rec["has_nan"] = has_nan
                rec["has_inf"] = has_inf
        else:
            ft["nanoowl_ms"] = 0.0
            ft["bridge_ms"] = 0.0
            ft["nanosam_set_image_ms"] = 0.0
            ft["nanosam_decode_ms"] = 0.0

        t0 = time.perf_counter()
        out_frame = overlay_mask_box(
            frame_bgr, mask_bool, clipped_box,
            (selected["label"] if selected else ""), (selected["score"] if selected else 0.0),
            status)
        overlay_ms = (time.perf_counter() - t0) * 1000.0
        ft["overlay_ms"] = round(overlay_ms, 3)

        t0 = time.perf_counter()
        writer.write(out_frame)
        written += 1
        if out_shape is None:
            out_shape = out_frame.shape
        encode_write_ms = (time.perf_counter() - t0) * 1000.0
        ft["encode_write_ms"] = round(encode_write_ms, 3)

        ft["frame_total_ms"] = round(sum(v for v in ft.values()), 3)
        rec["status"] = status
        rec["timing"] = ft
        fj.write(json.dumps(rec) + "\n")
        per_frame_timing.append(ft)

        if frame_index % 20 == 0 and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024*1024)
            log(f"  frame {frame_index}: status={status} alloc_MB={alloc:.1f}")

        if (frame_index + 1) % 10 == 0 or (frame_index + 1) == args.max_frames:
            log(f"  progress frame {frame_index+1}/{args.max_frames} processed={processed} "
                f"detected={detected} no_det={no_detection}")

    cap.release()
    writer.release()
    fj.close()

    vcap = cv2.VideoCapture(args.output_video)
    out_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_cap_w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_cap_h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_cap_fps = vcap.get(cv2.CAP_PROP_FPS)
    ok_read = vcap.isOpened() and out_count > 0
    vcap.release()

    def col_avg(key):
        vals = [t[key] for t in per_frame_timing if key in t]
        return round(float(np.mean(vals)), 3) if vals else 0.0

    keys = ["decode_ms", "nanoowl_ms", "bridge_ms", "nanosam_set_image_ms",
            "nanosam_decode_ms", "overlay_ms", "encode_write_ms", "frame_total_ms"]
    averages = {k: col_avg(k) for k in keys}

    summary = {
        "input_video": args.input_video,
        "output_video": args.output_video,
        "input_fps": in_fps,
        "input_frame_count": in_count,
        "input_resolution_WxH": [in_w, in_h],
        "max_frames": args.max_frames,
        "process_every_n": args.process_every_n,
        "processed_frames": processed,
        "detected_frames": detected,
        "no_detection_frames": no_detection,
        "written_frames": written,
        "output_frame_count_via_videocapture": out_count,
        "output_resolution_WxH_via_videocapture": [out_cap_w, out_cap_h],
        "output_fps_via_videocapture": out_cap_fps,
        "output_reread_ok": bool(ok_read),
        "input_frame_shapes_seen": [list(s) for s in sorted(seen_shapes)],
        "output_frame_shape_first": list(out_shape) if out_shape else None,
        "model_load": timing["model_load"],
        "prompts": text,
        "target_label": args.target_label,
        "threshold": args.threshold,
        "mask_selection": "slot 0 (fixed, not IoU argmax)",
        "timing_note": "correctness profiling only; NOT real-time FPS (Stage 6B defines throughput)",
        "averages_ms_over_processed_frames": averages,
    }
    all_totals = [t["frame_total_ms"] for t in per_frame_timing]
    summary["avg_frame_total_ms_over_all_written"] = round(float(np.mean(all_totals)), 3) if all_totals else 0.0

    ts_path = os.path.join(os.path.dirname(os.path.abspath(args.output_video)), "timing_summary.json")
    with open(ts_path, "w") as f:
        json.dump(summary, f, indent=2)
    runlog.close()

    print(json.dumps(summary, indent=2), flush=True)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
