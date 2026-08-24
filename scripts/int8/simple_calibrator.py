#!/usr/bin/env python3
"""Simplified NanoSAM INT8 Calibrator."""
import json, sys, time
from pathlib import Path
from PIL import Image
import torch
import tensorrt as trt

ROOT = Path("/home/nvidia/jetson-deploy-lab")
sys.path.insert(0, str(ROOT / "third_party/nanosam"))

from nanosam.utils.predictor import preprocess_image

class SimpleCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, manifest_json, batch_size, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        with open(manifest_json) as f:
            manifest = json.load(f)
        
        img_root = Path(manifest_json).parent / "train2017_512/images"
        self.images = [img_root / img["file_name"] for img in manifest["images"]]
        
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.idx = 0
        self.device_mem = None
        
        print(f"[Calibrator] {len(self.images)} images, batch={batch_size}")
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.idx >= len(self.images):
            print(f"[Calibrator] Done: {self.idx} images")
            return None
        
        batch_tensors = []
        for _ in range(self.batch_size):
            if self.idx >= len(self.images):
                break
            img_path = self.images[self.idx]
            self.idx += 1
            
            pil_img = Image.open(img_path).convert("RGB")
            tensor = preprocess_image(pil_img, size=1024).squeeze(0)  # [3,1024,1024]
            batch_tensors.append(tensor)
        
        if not batch_tensors:
            return None
        
        batch = torch.stack(batch_tensors, 0)  # [N,3,1024,1024]
        
        if self.device_mem is None:
            self.device_mem = torch.zeros((self.batch_size, 3, 1024, 1024), dtype=torch.float32, device="cuda")
        
        self.device_mem[:batch.shape[0]].copy_(batch)
        
        if self.idx % 100 == 0:
            print(f"[Calibrator] {self.idx}/{len(self.images)}")
        
        return [int(self.device_mem.data_ptr())]
    
    def read_calibration_cache(self):
        if Path(self.cache_file).exists():
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"[Calibrator] Cache written: {len(cache)} bytes")
