#!/usr/bin/env python3
"""NanoSAM INT8 Entropy Calibrator for TensorRT 10.3."""
import json
import tensorrt as trt
import torch
from pathlib import Path
from PIL import Image
import sys

# Add nanosam to path
ROOT = Path("/home/nvidia/jetson-deploy-lab")
sys.path.insert(0, str(ROOT / "third_party/nanosam"))

from nanosam.utils.predictor import preprocess_image


class NanoSAMEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """TensorRT INT8 Entropy Calibrator for NanoSAM."""
    
    def __init__(self, manifest_path, image_root, batch_size=1, cache_file=None):
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.batch_size = batch_size
        self.cache_file = cache_file
        
        # Load manifest
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        
        self.image_list = [
            self.image_root / img["file_name"]
            for img in self.manifest["images"]
        ]
        
        print(f"[Calibrator] Loaded {len(self.image_list)} images")
        print(f"[Calibrator] Batch size: {self.batch_size}")
        print(f"[Calibrator] Cache file: {self.cache_file}")
        
        self.current_index = 0
        self.device_input = None
        
        # Stats
        self.batches_consumed = 0
        self.images_consumed = 0
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        """Get next batch of preprocessed images."""
        if self.current_index >= len(self.image_list):
            print(f"[Calibrator] Calibration complete: {self.images_consumed} images, {self.batches_consumed} batches")
            return None
        
        # Prepare batch
        batch_images = []
        batch_end = min(self.current_index + self.batch_size, len(self.image_list))
        
        for i in range(self.current_index, batch_end):
            img_path = self.image_list[i]
            try:
                # Load image
                pil_img = Image.open(img_path).convert("RGB")
                
                # Use official NanoSAM preprocessing
                preprocessed = preprocess_image(pil_img, size=1024)  # [1, 3, 1024, 1024] fp32 cuda
                
                # Remove batch dimension for stacking
                preprocessed = preprocessed.squeeze(0)  # [3, 1024, 1024]
                
                batch_images.append(preprocessed)
                
            except Exception as e:
                print(f"[Calibrator] Error processing {img_path}: {e}")
                continue
        
        if not batch_images:
            return None
        
        # Stack to batch
        batch_tensor = torch.stack(batch_images, dim=0)  # [N, 3, 1024, 1024]
        
        # Allocate device buffer if needed
        if self.device_input is None:
            # Allocate for max batch size
            self.device_input = torch.zeros(
                (self.batch_size, 3, 1024, 1024),
                dtype=torch.float32,
                device="cuda"
            )
        
        # Copy batch to device buffer
        actual_batch = batch_tensor.size(0)
        self.device_input[:actual_batch].copy_(batch_tensor)
        
        self.current_index = batch_end
        self.batches_consumed += 1
        self.images_consumed += actual_batch
        
        if self.batches_consumed % 50 == 0:
            print(f"[Calibrator] Progress: {self.images_consumed}/{len(self.image_list)} images")
        
        # Return device pointer as int
        return [int(self.device_input.data_ptr())]
    
    def read_calibration_cache(self):
        """Read existing calibration cache if available."""
        if self.cache_file and Path(self.cache_file).exists():
            print(f"[Calibrator] Reading cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                data = f.read()
            print(f"[Calibrator] Cache size: {len(data)} bytes")
            return data
        return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache to file."""
        if self.cache_file:
            print(f"[Calibrator] Writing cache to {self.cache_file} ({len(cache)} bytes)")
            with open(self.cache_file, "wb") as f:
                f.write(cache)
    
    def get_stats(self):
        return {
            "images_consumed": self.images_consumed,
            "batches_consumed": self.batches_consumed,
            "total_images": len(self.image_list)
        }
