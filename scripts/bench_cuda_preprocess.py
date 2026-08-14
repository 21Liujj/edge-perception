# -*- coding: utf-8 -*-
"""Stage 7B standalone benchmark + numerical comparison for fused CUDA preprocess.
(A) numerical: baseline(PIL default=BICUBIC) vs CUDA(bilinear) tensor error.
(B) latency: CPU preprocess path vs CUDA path, full boundary (CPU image -> TRT-ready
    device tensor), warmup>=50 iters>=500.
"""
import os, sys, json, time, argparse, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for edge_perception
import numpy as np
import torch
import PIL.Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
OWL = os.path.join(ROOT, "third_party/nanoowl/assets/owl_glove_small.jpg")
DOGS = os.path.join(ROOT, "third_party/nanosam/assets/dogs.jpg")
SIZE = 1024
MEAN = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
STD = torch.tensor([58.395, 57.12, 57.375])[:, None, None]

def resized_dims(image, size=SIZE):
    aspect = image.width / image.height
    if aspect >= 1: rw = size; rh = int(size / aspect)
    else: rh = size; rw = int(size * aspect)
    return rh, rw

def baseline_cpu_preprocess(image, out_gpu):
    size = out_gpu.shape[-1]
    rh, rw = resized_dims(image, size)
    pr = image.resize((rw, rh))  # PIL default == BICUBIC
    arr = np.asarray(pr)
    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    tn = (t - MEAN) / STD
    staging = out_gpu.new_zeros((1, 3, size, size))
    staging[0, :, :rh, :rw] = tn
    out_gpu.copy_(staging)
    return out_gpu

def cuda_preprocess(prep, image):
    rh, rw = resized_dims(image)
    src_h, src_w = prep.prepare_source(image)
    prep.launch_fused(src_h, src_w, rh, rw)
    prep._stream.synchronize()
    return prep.image_tensor_gpu

def pct(a, p):
    a = sorted(a)
    if not a: return float("nan")
    k = (len(a) - 1) * p / 100.0
    f = int(k); c = min(f + 1, len(a) - 1)
    return a[f] + (a[c] - a[f]) * (k - f)

def bench(fn, warmup, iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"mean": statistics.mean(ts), "min": min(ts),
            "p50": pct(ts, 50), "p95": pct(ts, 95), "p99": pct(ts, 99), "max": max(ts)}

def numerical(files):
    from edge_perception.runtime.cuda_preprocess import CudaPreprocessor
    recs = {}
    prep = CudaPreprocessor(size=SIZE)
    for name, path in files.items():
        img = PIL.Image.open(path).convert("RGB")
        rh, rw = resized_dims(img)
        b = torch.zeros((1, 3, SIZE, SIZE))
        baseline_cpu_preprocess(img, b)
        src_h, src_w = prep.prepare_source(img)
        prep.launch_fused(src_h, src_w, rh, rw)
        prep._stream.synchronize()
        c = prep.image_tensor_gpu.detach().cpu()
        rb = b[0, :, :rh, :rw]; rc = c[0, :, :rh, :rw]
        d = (rb - rc).abs().flatten()
        recs[name] = {"src_size": list(img.size), "rh": rh, "rw": rw,
                      "region_max_abs": d.max().item(), "region_mean_abs": d.mean().item(),
                      "region_rmse": d.pow(2).mean().sqrt().item(),
                      "region_p50_abs": pct(d.tolist(), 50),
                      "region_p95_abs": pct(d.tolist(), 95),
                      "region_p99_abs": pct(d.tolist(), 99),
                      "pad_max_abs_cuda": c[0, :, rh:, :].abs().max().item()}
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["numerical", "benchmark", "both"], default="both")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/runtime_optimization/stage7b"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = {"owl": OWL, "dogs": DOGS}
    if args.mode in ("numerical", "both"):
        recs = numerical(files)
        obj = {"baseline_resample": "PIL_default_BICUBIC", "cuda_resample": "bilinear_half_pixel",
               "per_image": recs}
        with open(os.path.join(args.out, "preprocess_numerical_comparison.json"), "w") as f:
            json.dump(obj, f, indent=2)
        print(json.dumps(obj, indent=2))
    if args.mode in ("benchmark", "both"):
        from edge_perception.runtime.cuda_preprocess import CudaPreprocessor
        prep = CudaPreprocessor(size=SIZE)
        img = PIL.Image.open(OWL).convert("RGB")
        out_gpu = torch.zeros((1, 3, SIZE, SIZE), device="cuda")
        cpu = bench(lambda: baseline_cpu_preprocess(img, out_gpu), args.warmup, args.iters)
        cu = bench(lambda: cuda_preprocess(prep, img), args.warmup, args.iters)
        res = {"image": "owl_glove_small.jpg", "warmup": args.warmup, "iters": args.iters,
               "boundary": "CPU image -> TRT-ready device tensor (full preprocess boundary)",
               "cpu_preprocess": cpu, "cuda_preprocess": cu,
               "full_preprocess_speedup": cpu["mean"] / cu["mean"]}
        with open(os.path.join(args.out, "preprocess_benchmark.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
