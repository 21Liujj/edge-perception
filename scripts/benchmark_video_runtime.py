# -*- coding: utf-8 -*-
"""Stage 6B - Realtime video runtime / backpressure strategy benchmark.

Compares three runtime strategies under REALTIME frame arrival (producer paces
frame release at the source video FPS, NOT the disk decode rate):

  1. sequential : read -> inference -> next (no queue; queue metrics N/A)
  2. fifo       : producer thread -> bounded FIFO queue -> consumer; BLOCK on
                  full (backpressure), no silent drop; can lag input
  3. latest     : effective queue capacity 1; newer pending frame supersedes /
                  drops older unprocessed frame; sacrifices completeness for low
                  live latency

Reuses the Day4/Stage6A validated inference logic (NanoOWL -> NanoSAM slot 0)
verbatim in conventions. Models load ONCE. Does NOT rebuild engines, modify
upstream, or define this as a formal algorithm benchmark (synthetic video only).

Per source-frame timestamps recorded:
  source_timestamp_ms, scheduled_arrival_ms, arrival_wall_time,
  enqueue_time, dequeue_time, inference_start, inference_end, completion_time
Derived: queue_wait_ms, processing_ms, frame_age_at_inference_start_ms,
         live_e2e_latency_ms = completion - scheduled_arrival.

Output video writing is DISABLED during the formal benchmark (VideoWriter I/O
would pollute runtime timing); this is noted in summary.json.

timing_note: processing throughput here is offline pipeline throughput under
realtime-paced input, NOT a claim of real-time camera FPS.
"""
import argparse
import csv
import json
import os
import queue as _queue
import subprocess
import sys
import threading
import time

import numpy as np
import cv2
import PIL.Image
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from nanoowl.owl_predictor import OwlPredictor  # noqa: E402
from nanosam.utils.predictor import Predictor as NanosamPredictor  # noqa: E402

DEFAULT_OWL_ENGINE = os.path.join(ROOT, "third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
DEFAULT_NANOSAM_ENC = os.path.join(ROOT, "third_party/nanosam/data/resnet18_image_encoder.engine")
DEFAULT_NANOSAM_DEC = os.path.join(ROOT, "third_party/nanosam/data/mobile_sam_mask_decoder.engine")
DEFAULT_MODEL = "google/owlvit-base-patch32"
SELECTED_MASK_INDEX = 0
DEFAULT_INPUT = os.path.join(ROOT, "results/video_pipeline/baseline/input.mp4")
DEFAULT_OUTDIR = os.path.join(ROOT, "results/video_runtime_benchmark")


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
    output = owl.predict(image=image_pil_rgb, text=text, text_encodings=text_encodings,
                         threshold=threshold, pad_square=False)
    cuda_sync()
    labels = output.labels.detach().cpu().tolist()
    scores = output.scores.detach().cpu().tolist()
    boxes = output.boxes.detach().cpu().tolist()
    dets = []
    for li, sc, bx in zip(labels, scores, boxes):
        li = int(li)
        dets.append({"label": text[li] if 0 <= li < len(text) else str(li),
                     "score": float(sc), "box": [float(v) for v in bx]})
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets


def select_detection(dets, target_label):
    if not dets:
        return None
    if target_label is not None:
        m = [d for d in dets if d["label"] == target_label]
        return m[0] if m else None
    return dets[0]


def run_one_frame(owl, nsam, frame_bgr, in_w, in_h, text, text_encodings, threshold, target_label):
    """Single-frame joint inference. Returns (rec_dict, status, inferred_ok)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = PIL.Image.fromarray(rgb).convert("RGB")
    dets = detect_on_image(owl, image_pil, text, text_encodings, threshold)
    selected = select_detection(dets, target_label)
    rec = {"num_detections": len(dets)}
    if selected is None:
        return rec, "NO_DETECTION", True
    clipped = clip_box(selected["box"], in_w, in_h)
    nsam.set_image(image_pil)
    cuda_sync()
    pts = np.array([[clipped[0], clipped[1]], [clipped[2], clipped[3]]], dtype=np.float32)
    labs = np.array([2, 3])
    hi, iou_pred, low = nsam.predict(pts, labs)
    cuda_sync()
    hi_f = hi.detach().float()
    has_nan = bool(torch.isnan(hi_f).any().item())
    has_inf = bool(torch.isinf(hi_f).any().item())
    mask_bool = (hi[0, SELECTED_MASK_INDEX] > 0).detach().cpu().numpy()
    if mask_bool.shape[:2] != (in_h, in_w):
        m_u8 = (mask_bool.astype(np.uint8) * 255)
        m_u8 = cv2.resize(m_u8, (in_w, in_h), interpolation=cv2.INTER_NEAREST)
        mask_bool = (m_u8 > 127)
    rec.update({"label": selected["label"], "score": round(selected["score"], 6),
                "box": clipped, "selected_mask_index": SELECTED_MASK_INDEX,
                "mask_area_ratio": round(float(mask_bool.sum()) / float(mask_bool.size), 6),
                "has_nan": has_nan, "has_inf": has_inf})
    status = "INFERENCE_NAN_INF" if (has_nan or has_inf) else "DETECTED"
    return rec, status, not (has_nan or has_inf)


# ---- timestamped source frame ----
class SrcFrame:
    __slots__ = ("idx", "source_timestamp_ms", "scheduled_arrival_ms", "arrival_wall",
                 "enqueue_time", "dequeue_time", "frame_bgr", "dropped_reason")
    def __init__(self, idx, source_timestamp_ms, scheduled_arrival_ms, arrival_wall, frame_bgr):
        self.idx = idx
        self.source_timestamp_ms = source_timestamp_ms
        self.scheduled_arrival_ms = scheduled_arrival_ms
        self.arrival_wall = arrival_wall
        self.enqueue_time = None
        self.dequeue_time = None
        self.frame_bgr = frame_bgr
        self.dropped_reason = None


def percentiles(vals):
    if not vals:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    a = np.array(vals, dtype=np.float64)
    return {"mean": round(float(a.mean()), 4), "p50": round(float(np.percentile(a, 50)), 4),
            "p95": round(float(np.percentile(a, 95)), 4), "p99": round(float(np.percentile(a, 99)), 4),
            "max": round(float(a.max()), 4)}


# ---- tegrastats sampler (background thread) ----
def parse_tegrastats(raw):
    import re
    ram, gr3d, gtemp, ctemp, vdd = [], [], [], [], []
    for line in raw.splitlines():
        m = re.search(r"RAM\s+(\d+)/(\d+)", line)
        if m:
            ram.append(int(m.group(1)))
        m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if m:
            gr3d.append(int(m.group(1)))
        m = re.search(r"gpu@(-?\d+\.?\d*)C", line)
        if m:
            gtemp.append(float(m.group(1)))
        m = re.search(r"cpu@(-?\d+\.?\d*)C", line)
        if m:
            ctemp.append(float(m.group(1)))
        m = re.search(r"VDD_IN\s+(\d+)mW", line)
        if m:
            vdd.append(int(m.group(1)))
    def stat(v):
        if not v:
            return {"mean": None, "max": None}
        return {"mean": round(float(sum(v) / len(v)), 2), "max": round(float(max(v)), 2)}
    return {"RAM_MB": stat(ram), "GR3D_pct": stat(gr3d), "GPU_temp_C": stat(gtemp),
            "CPU_temp_C": stat(ctemp), "VDD_IN_mW": stat(vdd), "EMC": {"mean": None, "max": None}}


class TegraSampler(threading.Thread):
    def __init__(self, out_log, interval_ms=200):
        super().__init__(daemon=True)
        self.out_log = out_log
        self.interval_ms = interval_ms
        self.proc = None
        self.buf = []
        self._stop = False

    def run(self):
        try:
            self.proc = subprocess.Popen(["/usr/bin/tegrastats", "--interval", str(self.interval_ms)],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            for line in self.proc.stdout:
                self.buf.append(line)
                if self._stop:
                    break
        except Exception as e:
            self.buf.append("TEGRASTATS_ERROR: %s\n" % e)

    def stop_and_save(self):
        self._stop = True
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        raw = "".join(self.buf)
        with open(self.out_log, "w") as f:
            f.write(raw)
        return raw


def load_models(args):
    cuda_sync()
    t0 = time.perf_counter()
    owl = OwlPredictor(args.model, image_encoder_engine=args.owl_engine)
    cuda_sync()
    owl_load = (time.perf_counter() - t0) * 1000.0
    text = [t.strip() for t in args.prompts.split(",") if t.strip()]
    text_encodings = owl.encode_text(text)
    cuda_sync()
    t0 = time.perf_counter()
    nsam = NanosamPredictor(args.nanosam_enc, args.nanosam_dec)
    cuda_sync()
    nsam_load = (time.perf_counter() - t0) * 1000.0
    return owl, nsam, text, text_encodings, {"nanoowl_load_ms": round(owl_load, 3),
                                             "nanosam_load_ms": round(nsam_load, 3),
                                             "total_load_ms": round(owl_load + nsam_load, 3)}


def replay_frames(cap, loops, fps, max_total, max_qsize):
    """Yield SrcFrame objects paced at source FPS, monotonic source timestamp across loops."""
    frame_interval_ms = 1000.0 / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # read all frames once into memory (60 frames small) for clean realtime pacing
    all_frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        all_frames.append(f)
    n_per = len(all_frames)
    produced = 0
    start_wall = time.perf_counter()
    for lp in range(loops):
        for i in range(n_per):
            if produced >= max_total:
                return
            src_idx = lp * n_per + i
            source_timestamp_ms = src_idx * frame_interval_ms
            scheduled_arrival_ms = source_timestamp_ms  # relative to replay start
            # pace to scheduled arrival
            target_wall = start_wall + (src_idx * frame_interval_ms / 1000.0)
            now = time.perf_counter()
            if now < target_wall:
                time.sleep(target_wall - now)
            arrival_wall = time.perf_counter()
            yield SrcFrame(src_idx, source_timestamp_ms, scheduled_arrival_ms, arrival_wall, all_frames[i])
            produced += 1


def run_sequential(args, owl, nsam, text, text_encodings, cap, in_w, in_h, n_total, fps):
    outdir = os.path.join(args.outdir, "sequential")
    os.makedirs(outdir, exist_ok=True)
    runlog = open(os.path.join(outdir, "run.log"), "w")
    def log(m):
        print(m, flush=True); runlog.write(m + "\n"); runlog.flush()
    tegra = TegraSampler(os.path.join(outdir, "tegrastats.log"))
    tegra.start()
    t0_all = time.perf_counter()
    fj = open(os.path.join(outdir, "frames.jsonl"), "w")
    processing_ms = []
    live_lat = []
    frame_age_ms = []
    nan_inf = 0
    detections = 0
    processed = 0
    for sf in replay_frames(cap, args.loops, fps, n_total, 1):
        inf_start = time.perf_counter()
        rec, status, ok = run_one_frame(owl, nsam, sf.frame_bgr, in_w, in_h, text, text_encodings,
                                        args.threshold, args.target_label)
        cuda_sync()
        inf_end = time.perf_counter()
        completion = time.perf_counter()
        if status == "DETECTED":
            detections += 1
        if status == "INFERENCE_NAN_INF":
            nan_inf += 1
        if status != "NO_DETECTION":
            processed += 1
        p_ms = (inf_end - inf_start) * 1000.0
        processing_ms.append(p_ms)
        queue_wait_ms_val = 0.0  # no queue in sequential
        age_val = (inf_start - sf.arrival_wall) * 1000.0
        frame_age_ms.append(age_val)
        live_lat_val = (completion - t0_all) * 1000.0 - sf.scheduled_arrival_ms
        live_lat.append(live_lat_val)
        row = {"frame_index": sf.idx, "source_timestamp_ms": round(sf.source_timestamp_ms, 3),
               "scheduled_arrival_ms": round(sf.scheduled_arrival_ms, 3),
               "arrival_wall_s": round(sf.arrival_wall - t0_all, 4),
               "inference_start_s": round(inf_start - t0_all, 4),
               "inference_end_s": round(inf_end - t0_all, 4),
               "completion_s": round(completion - t0_all, 4),
               "processing_ms": round(p_ms, 3),
               "queue_wait_ms": None,
               "frame_age_at_inference_start_ms": round((inf_start - sf.arrival_wall) * 1000.0, 3),
               "live_e2e_latency_ms": round((completion - t0_all) * 1000.0 - sf.scheduled_arrival_ms, 3),
               "status": status, "inference": rec}
        fj.write(json.dumps(row) + "\n")
        if (sf.idx + 1) % 50 == 0:
            log("  seq progress frame %d status=%s" % (sf.idx + 1, status))
    fj.close()
    wall = time.perf_counter() - t0_all
    raw = tegra.stop_and_save()
    parsed = parse_tegrastats(raw)
    with open(os.path.join(outdir, "tegrastats_parsed.json"), "w") as f:
        json.dump(parsed, f, indent=2)
    summary = {
        "mode": "sequential",
        "source_fps": fps,
        "source_frames": n_total,
        "processed_frames": processed,
        "dropped_frames": 0,
        "drop_rate": 0.0,
        "detection_frames": detections,
        "nan_inf_frames": nan_inf,
        "processing_throughput_fps": round(processed / wall, 4) if wall else None,
        "wall_time_s": round(wall, 3),
        "processing_ms": percentiles(processing_ms),
        "live_e2e_latency_ms": percentiles(live_lat),
        "queue_wait_ms": "N/A (no queue)",
        "frame_age_at_inference_start_ms": percentiles(frame_age_ms),
        "queue_depth": "N/A (no queue)",
        "producer_blocked_ms": "N/A",
        "output_video": "DISABLED (benchmark; VideoWriter I/O would pollute timing)",
        "model_load": args._model_load,
        "tegrastats": parsed,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("sequential DONE processed=%d throughput=%.2ffps" % (processed, summary["processing_throughput_fps"]))
    runlog.close()
    return summary


def run_fifo(args, owl, nsam, text, text_encodings, cap, in_w, in_h, n_total, fps):
    outdir = os.path.join(outdir_path(args, "fifo"))
    os.makedirs(outdir, exist_ok=True)
    runlog = open(os.path.join(outdir, "run.log"), "w")
    def log(m):
        print(m, flush=True); runlog.write(m + "\n"); runlog.flush()
    qsize = args.queue_size
    q = _queue.Queue(maxsize=qsize)
    done_frame = {"idx": -1}
    results = {}
    results_lock = threading.Lock()
    producer_blocked_total = [0.0]
    max_depth = [0]
    stop_producer = threading.Event()

    def producer():
        for sf in replay_frames(cap, args.loops, fps, n_total, qsize):
            if stop_producer.is_set():
                break
            sf.enqueue_time = time.perf_counter()
            t_bl = time.perf_counter()
            try:
                q.put(sf, timeout=60)
            except _queue.Full:
                # backpressure: block until space (bounded wait)
                q.put(sf)
            blocked = time.perf_counter() - t_bl
            producer_blocked_total[0] += blocked
            with results_lock:
                max_depth[0] = max(max_depth[0], q.qsize())
        # sentinel
        q.put(None)

    def consumer():
        while True:
            sf = q.get()
            if sf is None:
                break
            sf.dequeue_time = time.perf_counter()
            inf_start = time.perf_counter()
            rec, status, ok = run_one_frame(owl, nsam, sf.frame_bgr, in_w, in_h, text, text_encodings,
                                            args.threshold, args.target_label)
            cuda_sync()
            inf_end = time.perf_counter()
            completion = time.perf_counter()
            with results_lock:
                results[sf.idx] = {"sf": sf, "rec": rec, "status": status,
                                   "inf_start": inf_start, "inf_end": inf_end, "completion": completion}

    tegra = TegraSampler(os.path.join(outdir, "tegrastats.log"))
    tegra.start()
    t0_all = time.perf_counter()
    pt = threading.Thread(target=producer, daemon=True)
    ct = threading.Thread(target=consumer, daemon=True)
    pt.start(); ct.start()
    ct.join()
    stop_producer.set()
    pt.join(timeout=5)
    wall = time.perf_counter() - t0_all

    fj = open(os.path.join(outdir, "frames.jsonl"), "w")
    processing_ms = []; qwait_ms = []; frame_age_ms = []; live_lat = []
    processed = 0; detections = 0; nan_inf = 0
    depth_samples = []
    for idx in sorted(results.keys()):
        r = results[idx]
        sf = r["sf"]
        p_ms = (r["inf_end"] - r["inf_start"]) * 1000.0
        qw = (sf.dequeue_time - sf.enqueue_time) * 1000.0 if (sf.enqueue_time and sf.dequeue_time) else 0.0
        age = (r["inf_start"] - sf.arrival_wall) * 1000.0
        latency = (r["completion"] - t0_all) * 1000.0 - sf.scheduled_arrival_ms
        processing_ms.append(p_ms); qwait_ms.append(qw); frame_age_ms.append(age); live_lat.append(latency)
        status = r["status"]
        if status == "DETECTED": detections += 1
        if status == "INFERENCE_NAN_INF": nan_inf += 1
        if status != "NO_DETECTION": processed += 1
        row = {"frame_index": sf.idx, "source_timestamp_ms": round(sf.source_timestamp_ms, 3),
               "scheduled_arrival_ms": round(sf.scheduled_arrival_ms, 3),
               "arrival_wall_s": round(sf.arrival_wall - t0_all, 4),
               "enqueue_time_s": round(sf.enqueue_time - t0_all, 4) if sf.enqueue_time else None,
               "dequeue_time_s": round(sf.dequeue_time - t0_all, 4) if sf.dequeue_time else None,
               "inference_start_s": round(r["inf_start"] - t0_all, 4),
               "inference_end_s": round(r["inf_end"] - t0_all, 4),
               "completion_s": round(r["completion"] - t0_all, 4),
               "processing_ms": round(p_ms, 3), "queue_wait_ms": round(qw, 3),
               "frame_age_at_inference_start_ms": round(age, 3),
               "live_e2e_latency_ms": round(latency, 3), "status": status, "inference": r["rec"]}
        fj.write(json.dumps(row) + "\n")
    fj.close()
    raw = tegra.stop_and_save()
    parsed = parse_tegrastats(raw)
    with open(os.path.join(outdir, "tegrastats_parsed.json"), "w") as f:
        json.dump(parsed, f, indent=2)
    summary = {
        "mode": "fifo",
        "source_fps": fps,
        "source_frames": n_total,
        "queue_size": qsize,
        "processed_frames": processed,
        "dropped_frames": 0,
        "drop_rate": 0.0,
        "detection_frames": detections,
        "nan_inf_frames": nan_inf,
        "processing_throughput_fps": round(processed / wall, 4) if wall else None,
        "wall_time_s": round(wall, 3),
        "processing_ms": percentiles(processing_ms),
        "live_e2e_latency_ms": percentiles(live_lat),
        "queue_wait_ms": percentiles(qwait_ms),
        "frame_age_at_inference_start_ms": percentiles(frame_age_ms),
        "queue_depth_max": max_depth[0],
        "queue_depth_mean": None,
        "producer_blocked_ms": round(producer_blocked_total[0], 3),
        "output_video": "DISABLED (benchmark; VideoWriter I/O would pollute timing)",
        "model_load": args._model_load,
        "tegrastats": parsed,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("fifo DONE processed=%d throughput=%.2ffps producer_blocked=%.1fms qdep_max=%d" %
        (processed, summary["processing_throughput_fps"], producer_blocked_total[0], max_depth[0]))
    runlog.close()
    return summary


def run_latest(args, owl, nsam, text, text_encodings, cap, in_w, in_h, n_total, fps):
    outdir = outdir_path(args, "latest")
    os.makedirs(outdir, exist_ok=True)
    runlog = open(os.path.join(outdir, "run.log"), "w")
    def log(m):
        print(m, flush=True); runlog.write(m + "\n"); runlog.flush()
    # single-slot holder; producer supersedes pending unprocessed frame
    slot = {"frame": None}
    slot_lock = threading.Lock()
    cond = threading.Condition(slot_lock)
    state = {"pending_version": 0, "consumed_version": 0, "stop": False}
    dropped = [0]
    results = {}
    results_lock = threading.Lock()
    frame_age_samples = []

    def producer():
        for sf in replay_frames(cap, args.loops, fps, n_total, 1):
            if state["stop"]:
                break
            with cond:
                if slot["frame"] is not None:
                    # pending unprocessed frame superseded -> drop old
                    old = slot["frame"]
                    old.dropped_reason = "superseded_by_newer_frame"
                    dropped[0] += 1
                    with results_lock:
                        results[old.idx] = {"sf": old, "status": "DROPPED", "dropped_reason": old.dropped_reason}
                sf.enqueue_time = time.perf_counter()
                slot["frame"] = sf
                state["pending_version"] += 1
                cond.notify()
        with cond:
            state["stop"] = True
            cond.notify_all()

    def consumer():
        while True:
            with cond:
                while slot["frame"] is None and not state["stop"]:
                    cond.wait()
                if slot["frame"] is None and state["stop"]:
                    break
                sf = slot["frame"]
                slot["frame"] = None
            if sf is None:
                break
            sf.dequeue_time = time.perf_counter()
            inf_start = time.perf_counter()
            rec, status, ok = run_one_frame(owl, nsam, sf.frame_bgr, in_w, in_h, text, text_encodings,
                                            args.threshold, args.target_label)
            cuda_sync()
            inf_end = time.perf_counter()
            completion = time.perf_counter()
            with results_lock:
                results[sf.idx] = {"sf": sf, "rec": rec, "status": status,
                                   "inf_start": inf_start, "inf_end": inf_end, "completion": completion}

    tegra = TegraSampler(os.path.join(outdir, "tegrastats.log"))
    tegra.start()
    t0_all = time.perf_counter()
    pt = threading.Thread(target=producer, daemon=True)
    ct = threading.Thread(target=consumer, daemon=True)
    pt.start(); ct.start()
    ct.join()
    with cond:
        state["stop"] = True
        cond.notify_all()
    pt.join(timeout=5)
    wall = time.perf_counter() - t0_all

    fj = open(os.path.join(outdir, "frames.jsonl"), "w")
    processing_ms = []; qwait_ms = []; frame_age_ms = []; live_lat = []
    processed = 0; detections = 0; nan_inf = 0; dropped_n = 0
    for idx in sorted(results.keys()):
        r = results[idx]
        sf = r["sf"]
        if r["status"] == "DROPPED":
            dropped_n += 1
            row = {"frame_index": sf.idx, "source_timestamp_ms": round(sf.source_timestamp_ms, 3),
                   "scheduled_arrival_ms": round(sf.scheduled_arrival_ms, 3),
                   "arrival_wall_s": round(sf.arrival_wall - t0_all, 4),
                   "enqueue_time_s": round(sf.enqueue_time - t0_all, 4) if sf.enqueue_time else None,
                   "status": "DROPPED", "dropped_reason": r.get("dropped_reason", "superseded_by_newer_frame"),
                   "processing_ms": None, "live_e2e_latency_ms": None,
                   "frame_age_at_inference_start_ms": None, "queue_wait_ms": None}
            fj.write(json.dumps(row) + "\n")
            continue
        p_ms = (r["inf_end"] - r["inf_start"]) * 1000.0
        qw = (sf.dequeue_time - sf.enqueue_time) * 1000.0 if (sf.enqueue_time and sf.dequeue_time) else 0.0
        age = (r["inf_start"] - sf.arrival_wall) * 1000.0
        latency = (r["completion"] - t0_all) * 1000.0 - sf.scheduled_arrival_ms
        processing_ms.append(p_ms); qwait_ms.append(qw); frame_age_ms.append(age); live_lat.append(latency)
        status = r["status"]
        if status == "DETECTED": detections += 1
        if status == "INFERENCE_NAN_INF": nan_inf += 1
        if status != "NO_DETECTION": processed += 1
        row = {"frame_index": sf.idx, "source_timestamp_ms": round(sf.source_timestamp_ms, 3),
               "scheduled_arrival_ms": round(sf.scheduled_arrival_ms, 3),
               "arrival_wall_s": round(sf.arrival_wall - t0_all, 4),
               "enqueue_time_s": round(sf.enqueue_time - t0_all, 4) if sf.enqueue_time else None,
               "dequeue_time_s": round(sf.dequeue_time - t0_all, 4) if sf.dequeue_time else None,
               "inference_start_s": round(r["inf_start"] - t0_all, 4),
               "inference_end_s": round(r["inf_end"] - t0_all, 4),
               "completion_s": round(r["completion"] - t0_all, 4),
               "processing_ms": round(p_ms, 3), "queue_wait_ms": round(qw, 3),
               "frame_age_at_inference_start_ms": round(age, 3),
               "live_e2e_latency_ms": round(latency, 3), "status": status, "inference": r["rec"]}
        fj.write(json.dumps(row) + "\n")
    fj.close()
    raw = tegra.stop_and_save()
    parsed = parse_tegrastats(raw)
    with open(os.path.join(outdir, "tegrastats_parsed.json"), "w") as f:
        json.dump(parsed, f, indent=2)
    total_source = n_total
    summary = {
        "mode": "latest",
        "source_fps": fps,
        "source_frames": total_source,
        "processed_frames": processed,
        "dropped_frames": dropped_n,
        "drop_rate": round(dropped_n / total_source, 4) if total_source else 0.0,
        "detection_frames": detections,
        "nan_inf_frames": nan_inf,
        "processing_throughput_fps": round(processed / wall, 4) if wall else None,
        "wall_time_s": round(wall, 3),
        "processing_ms": percentiles(processing_ms),
        "live_e2e_latency_ms": percentiles(live_lat),
        "queue_wait_ms": percentiles(qwait_ms),
        "frame_age_at_inference_start_ms": percentiles(frame_age_ms),
        "queue_capacity": 1,
        "output_video": "DISABLED (benchmark; VideoWriter I/O would pollute timing)",
        "model_load": args._model_load,
        "tegrastats": parsed,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("latest DONE processed=%d dropped=%d drop_rate=%.3f throughput=%.2ffps age_p95=%.1fms" %
        (processed, dropped_n, summary["drop_rate"], summary["processing_throughput_fps"],
         summary["frame_age_at_inference_start_ms"]["p95"]))
    runlog.close()
    return summary


def outdir_path(args, mode):
    return os.path.join(args.outdir, mode)


def main():
    ap = argparse.ArgumentParser(description="Realtime video runtime / backpressure benchmark")
    ap.add_argument("--input-video", default=DEFAULT_INPUT)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--prompts", default="an owl,a glove,a finger")
    ap.add_argument("--target-label", default="an owl")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--owl-engine", default=DEFAULT_OWL_ENGINE)
    ap.add_argument("--nanosam-enc", default=DEFAULT_NANOSAM_ENC)
    ap.add_argument("--nanosam-dec", default=DEFAULT_NANOSAM_DEC)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--loops", type=int, default=5)
    ap.add_argument("--queue-size", type=int, default=8)
    ap.add_argument("--modes", default="sequential,fifo,latest")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print("ERROR cannot open input", flush=True); return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cnt_per = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_total = cnt_per * args.loops
    print("input fps=%.2f frames/loop=%d loops=%d total=%d %dx%d" %
          (fps, cnt_per, args.loops, n_total, in_w, in_h), flush=True)

    owl, nsam, text, text_encodings, mload = load_models(args)
    args._model_load = mload
    print("models loaded %s" % json.dumps(mload), flush=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    summaries = {}
    for mode in modes:
        print("=== running mode %s ===" % mode, flush=True)
        cap2 = cv2.VideoCapture(args.input_video)
        if mode == "sequential":
            s = run_sequential(args, owl, nsam, text, text_encodings, cap2, in_w, in_h, n_total, fps)
        elif mode == "fifo":
            s = run_fifo(args, owl, nsam, text, text_encodings, cap2, in_w, in_h, n_total, fps)
        elif mode == "latest":
            s = run_latest(args, owl, nsam, text, text_encodings, cap2, in_w, in_h, n_total, fps)
        else:
            print("unknown mode %s" % mode); cap2.release(); continue
        cap2.release()
        summaries[mode] = s

    with open(os.path.join(args.outdir, "comparison.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    # comparison.md
    def g(m, *path, default="N/A"):
        cur = summaries.get(m)
        if not cur:
            return default
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur

    lines = []
    lines.append("# Realtime Video Runtime Strategy Comparison\n")
    lines.append("Input: %s (synthetic, runtime-controlled benchmark; NOT an algorithm benchmark)\n" % args.input_video)
    lines.append("Source FPS: %s, source frames: %s, loops: %d\n" % (fps, n_total, args.loops))
    lines.append("Output video writing DISABLED during benchmark (timing integrity).\n")
    lines.append("")
    lines.append("| Mode | Source FPS | Processed FPS | Processed Frames | Dropped Frames | Drop Rate | P50 Live Latency (ms) | P95 Live Latency (ms) | P99 Live Latency (ms) | P95 Queue Wait (ms) | Max Queue Depth | Peak RAM (MB) | GR3D Mean (%) | Power Mean (mW) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in modes:
        l = g(m, "live_e2e_latency_ms", "p50")
        l95 = g(m, "live_e2e_latency_ms", "p95")
        l99 = g(m, "live_e2e_latency_ms", "p99")
        qw95 = g(m, "queue_wait_ms", "p95")
        qdmax = g(m, "queue_depth_max", default="N/A")
        ram = g(m, "tegrastats", "RAM_MB", "max")
        gr3d = g(m, "tegrastats", "GR3D_pct", "mean")
        pwr = g(m, "tegrastats", "VDD_IN_mW", "mean")
        def f95(v):
            if isinstance(v, dict):
                return v.get("p95")
            return v
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            m, fps, g(m, "processing_throughput_fps"), g(m, "processed_frames"),
            g(m, "dropped_frames"), g(m, "drop_rate"), l, l95, l99, qw95, qdmax, ram, gr3d, pwr))
    lines.append("")
    with open(os.path.join(args.outdir, "comparison.md"), "w") as f:
        f.write("\n".join(lines))
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
