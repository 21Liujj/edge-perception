# EdgePerception - Open Vocabulary Segmentation on Jetson

**Lightweight Vision-Language Perception Pipeline on NVIDIA Jetson Orin**

Open-vocabulary image segmentation on embedded GPU: provide an image and a text prompt like `"cup"`, get back a pixel-accurate mask — no fixed category list, no retraining per object class.

## Project Status

**✅ EDGE_PERCEPTION_V1_COMPLETE**

Pipeline combining NanoOWL (text-conditioned detection) and NanoSAM (promptable segmentation) with comprehensive performance analysis and optimization experiments.

## Architecture

```
Image + Text Prompt ("a cup")
         ↓
    NanoOWL (OWL-ViT base-patch32, TensorRT FP16)
         ↓
    Bounding Box + Confidence Score
         ↓
    NanoSAM (ResNet18 distilled, TensorRT FP16/INT8)
         ↓
    Pixel-Accurate Segmentation Mask
```

**Why Two Models?**
- **NanoOWL**: Zero-shot object detection with open vocabulary (no fixed categories)
- **NanoSAM**: Promptable segmentation (needs box/point input, no detection capability)
- **Together**: Open-vocabulary instance segmentation without training

## Performance Metrics

| Component | Precision | Latency | Memory |
|-----------|-----------|---------|--------|
| NanoOWL | FP16 | ~100-150ms | ~800MB |
| NanoSAM | FP16/INT8 | ~80-120ms | ~600MB |
| **End-to-End** | Mixed | **~200-300ms** | **~1.4GB** |

**Hardware**: Jetson AGX Orin, JetPack 6.2, TensorRT 10.3.0

## Key Features

✅ **Open Vocabulary**: Segment any object described in text, no category limits  
✅ **TensorRT Optimized**: FP16 and INT8 quantization for edge deployment  
✅ **Production C++ Runtime**: Low-latency inference with minimal overhead  
✅ **Comprehensive Analysis**: Stage-by-stage performance profiling and optimization  
✅ **Scientific Validation**: Honest negative results documented (Stage G)

## Quick Start

### 1. Prerequisites

```bash
# Verify JetPack installation
dpkg-query --show nvidia-jetpack

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip build-essential cmake
pip3 install torch torchvision onnx
```

### 2. Clone and Setup

```bash
git clone https://github.com/21Liujj/edge-perception.git
cd edge-perception

# Download pre-trained models (not in repo due to size)
# NanoOWL: OWL-ViT base-patch32
# NanoSAM: ResNet18 distilled encoder + MobileSAM decoder
./scripts/download_models.sh

# Build TensorRT engines
python3 scripts/build_engines.py
```

### 3. Run Inference

```bash
# Python API
python3 edge_perception/demo.py \
  --image /path/to/image.jpg \
  --prompt "a cup" \
  --output output.png

# C++ Runtime (lower latency)
cd build
./edge_perception_demo \
  --image /path/to/image.jpg \
  --prompt "a cup"
```

## Project Structure

```
edge-perception/
├── edge_perception/            # Main Python package
│   ├── nanoowl/               # OWL-ViT detection module
│   ├── nanosam/               # SAM segmentation module
│   ├── pipeline.py            # End-to-end chained inference
│   └── demo.py                # Demo script
├── cpp_runtime/               # C++ production runtime
│   ├── include/               # Public headers
│   ├── src/                   # Implementation
│   └── examples/              # Usage examples
├── scripts/                   # Build and evaluation scripts
│   ├── build_engines.py       # TensorRT engine builder
│   ├── evaluate_pipeline.py   # Performance benchmarking
│   └── download_models.sh     # Model download helper
├── csrc/                      # CUDA preprocessing kernels
├── data/                      # Sample data and calibration sets
├── docs/                      # Technical documentation
│   ├── nanoowl_localization_improvement.md
│   └── vlm_algorithm_deep_dive.md
├── tests/                     # Unit tests
└── STAGE_G_SUMMARY.txt        # Latest experimental results
```

## Development Stages

This project progressed through multiple research and optimization stages:

| Stage | Focus | Key Results |
|-------|-------|-------------|
| A-D | Pipeline integration, TensorRT conversion | ✅ Working FP16/INT8 engines |
| E | Quantization analysis (INT8 vs FP16) | 📊 Quality/speed tradeoffs documented |
| F | C++ runtime development | ✅ Production-ready inference |
| G | Localization refinement experiments | ⚠️ Training-free methods insufficient |

### Stage G Highlights (Latest)

**Goal**: Improve NanoOWL bounding box localization accuracy through post-processing  
**Methods Tested**: 
- Method A: Multi-candidate fusion
- Method B: Spatial consensus voting

**Results**: Training-free post-processing insufficient due to lack of redundant overlapping candidates in NanoOWL's 24×24 patch grid

**Scientific Value**: Documented negative results prevent wasteful heuristic stacking; points to model-level improvements as correct next step

See `STAGE_G_SUMMARY.txt` and `docs/nanoowl_localization_improvement.md` for detailed analysis.

## API Usage

### Python API

```python
from edge_perception import EdgePerceptionPipeline

# Initialize pipeline
pipeline = EdgePerceptionPipeline(
    owl_engine_path="models/owl_encoder.engine",
    sam_encoder_path="models/sam_encoder.engine",
    sam_decoder_path="models/sam_decoder.onnx"
)

# Run inference
result = pipeline.predict(
    image_path="image.jpg",
    text_prompt="a cup"
)

# Access results
mask = result.mask          # Binary segmentation mask
bbox = result.bbox          # Detected bounding box
score = result.score        # Detection confidence
```

### C++ API

```cpp
#include <edge_perception/pipeline.h>

// Initialize
EdgePerception::Pipeline pipeline(config);

// Inference
auto result = pipeline.segment(image, "a cup");
cv::Mat mask = result.mask;
```

## Performance Analysis

### Latency Breakdown

| Stage | FP16 | INT8 | Notes |
|-------|------|------|-------|
| OWL Image Encoding | 100ms | N/A | Text encoding done once |
| SAM Image Encoding | 80ms | 65ms | 19% speedup with INT8 |
| SAM Mask Decoding | 15ms | 15ms | Unchanged (lightweight) |
| **Total** | **195ms** | **180ms** | End-to-end latency |

### Accuracy Trade-offs

INT8 quantization on NanoSAM:
- **mIoU**: 0.741 (FP16) → 0.728 (INT8) → **-1.3% degradation**
- **Latency**: 80ms → 65ms → **19% speedup**
- **Conclusion**: FP16 recommended for production (better quality/speed balance)

See `docs/vlm_algorithm_deep_dive.md` for comprehensive analysis.

## Known Limitations

1. **Localization Accuracy**: NanoOWL's 24×24 patch grid (32px per patch) causes coarse localization; errors propagate to final mask quality
2. **Single Object**: Current pipeline returns highest-confidence detection only
3. **No Tracking**: Frame-by-frame processing; no temporal consistency
4. **Fixed Resolution**: Optimal at 768×768 input (NanoOWL native resolution)

## Reproducing Experiments

```bash
# Run full evaluation pipeline
python3 scripts/evaluate_pipeline.py \
  --dataset data/val_split.json \
  --output results/evaluation/

# Stage G experiments (bbox refinement)
python3 scripts/vlm_analysis/evaluate_box_refinement.py

# Generate visualizations
python3 scripts/visualize_results.py \
  --results results/stage_g_bbox_refinement/
```

## Building from Source

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

## Technical Documentation

- **`STAGE_G_SUMMARY.txt`**: Latest experimental results and conclusions
- **`docs/nanoowl_localization_improvement.md`**: Detailed Stage G analysis
- **`docs/vlm_algorithm_deep_dive.md`**: Architecture and optimization deep dive
- **`README.md`** (original): Additional technical context

## Future Directions

Based on Stage G findings, recommended improvements require model-level changes:

1. **Fine-tune OWL-ViT detection head** on Jetson-specific error distributions
2. **Explore stronger grounding models** (e.g., Grounding DINO)
3. **Multi-object detection** support
4. **Temporal tracking** for video streams

Post-processing heuristics provide minimal gains due to insufficient candidate redundancy.

## Citation

```bibtex
@software{edgeperception2026,
  title={EdgePerception: Open-Vocabulary Segmentation on NVIDIA Jetson},
  author={Liu, Junjie},
  year={2026},
  url={https://github.com/21Liujj/edge-perception}
}
```

## License

MIT License - See LICENSE file for details

## Contact

- **Author**: Liu Junjie (21Liujj)
- **Email**: 3154288405@qq.com
- **GitHub**: [@21Liujj](https://github.com/21Liujj)

---

**Open Vocabulary | Edge Optimized | Production Ready**
