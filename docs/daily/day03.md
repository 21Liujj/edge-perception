# Day 3

## 今日目标

固定 NanoSAM 官方 upstream，完成 baseline 路径审计（架构、执行链、依赖、模型需求）；恢复并复现 NanoSAM TensorRT promptable-segmentation baseline（ONNX artifact 离线恢复 + Engine 构建 + 真实 Box/Point Prompt 分割）；工程化为可复用 CLI，做分模块性能 Benchmark，并关闭 Day 3。不下载模型（已离线转存）、不重建已有 Engine、不修改 upstream、不做 NanoOWL+NanoSAM 联合 / COCO / INT8 / ROS2 / MP4。

## Stage 3A — NanoSAM upstream pin + architecture audit

### Upstream pin

- 初始检查 third_party/nanosam 不存在。
- GitHub 此时可达（git ls-remote https://github.com/NVIDIA-AI-IOT/nanosam.git HEAD 成功）。
- 克隆官方仓库（GIT_LFS_SKIP_SMUDGE=1，仅源码）到 third_party/nanosam。
  - origin: https://github.com/NVIDIA-AI-IOT/nanosam.git
  - branch: main
  - HEAD commit: 653633614b2eb93b06ba3be9adb2aeffb117bd72（2023-09-20, "fix learning rate arg passing"）
  - git status: clean
- `python setup.py develop` 就地安装（editable，未 --user；未改上游源码、未装新依赖）。Predictor 可 import。

### Baseline 架构（以 README + 源码为准）

- Image Encoder = NanoSAM ResNet18（由 MobileSAM image encoder 蒸馏）；输入 1×3×1024×1024；输出 image_embeddings [1,256,64,64]；FP16 TRT engine。
- Mask Decoder = MobileSAM mask decoder；FP32 TRT engine；输入 image_embeddings / point_coords [1,N,2] / point_labels [1,N] / mask_input [1,1,256,256] / has_mask_input [1]；输出 iou_predictions [1,4] + low_res_masks [1,4,256,256]（4 个候选 mask，经 upscale_mask 双线性回原图）。
- point label 语义：0=背景，1=前景，2=box 左上，3=box 右下。
- 两 engine 经 torch2trt.TRTModule 加载（nanosam/utils/predictor.py）。

### Mask selection rule（读源码，非记忆）

- decoder 输出 4 个候选 mask：`low_res_masks` [1,4,256,256] 与 `iou_predictions` [1,4]（predictor.py run_mask_decoder 输出 4 候选；predict 经 upscale_mask 放大整批 4 个到 hi_res_mask [1,4,H,W]）。
- 官方 examples/basic_usage.py 选择**固定 slot 0**，非 IoU argmax：`mask = (mask[0, 0] > 0).detach().cpu().numpy()`。
- selected mask index = 0；selected predicted IoU = iou_predictions[0]。
- 本项目 CLI/benchmark 沿用此约定，并记录 selected index 与 selected predicted IoU。

### 依赖审计（仅审计，不安装）

torch 2.5.0a0+872d972e41.nv24.08 / torchvision 0.20.0a0+afc54f7 / numpy 1.23.5 / tensorrt 10.3.0 / torch2trt importable / transformers 4.46.3 / Pillow 12.3.0 / OpenCV 4.10.0 / matplotlib 3.10.3；timm MISSING（仅 vit_t 导出/本地 encoder 用，非 baseline 运行链）。setup.py install_requires=[]。全程未装/升级/降级任何包。Stage 3A status: PASS。

## Stage 3B — ONNX artifact recovery + TensorRT baseline（PASS）

### 真实 Failure Case（保留）

A. NanoSAM README 原 Google Drive artifact 已不可访问：Jetson 访问 Google Drive (drive.google.com:443) 超时；`resnet18_image_encoder.onnx` / `mobile_sam_mask_decoder.onnx` 无法在线下载（官方 Drive ID：image encoder 14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR、mask decoder 1jYNvnseTL49SNRx9PDcbkZ9DwsY8up7n）。

B. Jetson 本地 mask decoder export 因 timm 缺失停止：`nanosam/mobile_sam/modeling/tiny_vit_sam.py` 需 `timm`，本机未装；按约束未安装 timm，未从 mobile_sam.pt 重新导出，**没有污染稳定环境**。

C. 尝试 dustynv NanoSAM Docker artifact recovery：Docker Hub / CloudFront 大 layer 下载多次 EOF，因此停止拉取大镜像，未采用。

D. 最终通过可验证 artifact 来源取得两个 ONNX：User 离线转存官方提供 ONNX 到 Jetson，PC→Jetson SHA256 完全一致后恢复 Stage 3B。

### 模型来源（离线转存，非在线下载）/ SHA256

- `resnet18_image_encoder.onnx` 63,321,621 B — SHA256 (PC=Jetson) `d266fcdc9e4f0182b59946cd3cf1be331641f1d0d1de2415c4fd94e7a1c9cf0a` ✓
- `mobile_sam_mask_decoder.onnx` 16,504,875 B — SHA256 (PC=Jetson) `41e49a298099048186ce109a4518286b8972959898a02577414405efa5c3b247` ✓
- onnx.checker.check_model：两个 ONNX 均 PASS；形状解析正常。

### Engine 构建（README trtexec 参数；未改 ONNX / 未装插件 / 未升级 TensorRT；TRT 10.3.0）

- Image Encoder（FP16）：`trtexec --onnx=data/resnet18_image_encoder.onnx --saveEngine=data/resnet18_image_encoder.engine --fp16` → `&&&& PASSED`。size 34,766,700 B。SHA256 `c277e5ca445dc29b06e8be41a481e74bd806c4dd82071cbfbd47fddcf7aaa6a8`。输入 1×3×1024×1024 → 输出 [1,256,64,64]。
- Mask Decoder（FP32）：`trtexec --onnx=data/mobile_sam_mask_decoder.onnx --saveEngine=data/mobile_sam_mask_decoder.engine --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:1x1x2,point_labels:1x1 --maxShapes=point_coords:1x10x2,point_labels:1x10` → `&&&& PASSED`。size 18,891,924 B。SHA256 `afb5ac4958e6bab21ed8aa0cc65798f53a71406fe6124cf6b2e008da6fccb4f2`。输入 image_embeddings/point_coords/point_labels/mask_input/has_mask_input → 输出 iou_predictions[1,4]+low_res_masks[1,4,256,256]。
- 重载校验：Predictor(ENC,DEC) 加载成功；set_image 得 image_embeddings [1,256,64,64]。

### Box / Point Prompt 结果（assets/dogs.jpg，1180×760）

- Box Prompt（bbox=[100,100,850,759]，labels [2,3]）：PASS — iou [0.662,0.626,0.758,0.713]；hi_res [1,4,760,1180]；选 slot 0，selected IoU 0.662；非零像素 242,155，area_ratio 0.270021；无 NaN/Inf。
- Point Prompt（foreground [475,430]，label [1]）：PASS — iou [0.640,0.630,0.740,0.737]；hi_res [1,4,760,1180]；选 slot 0，selected IoU 0.640；非零像素 233,325，area_ratio 0.260175；无 NaN/Inf。

Stage 3B status: PASS。提交 `feat: reproduce NanoSAM TensorRT baseline`。

## Stage 3C — Reusable CLI + module benchmark（PASS）

### 可复用 CLI（scripts/run_nanosam_infer.py，最小整理原脚本，不新建重复实现）

参数：`--image`、`--prompt-type box|point`、`--box`、`--point`、`--point-label`、`--image-encoder-engine`、`--mask-decoder-engine`、`--output-dir`、`--mask-index`。输出：`<tag>_result.json`、`<tag>_mask.npy`（可复用 mask 数据）、`<tag>_visualization.jpg`、`<tag>_timing.json`。记录 mask selection rule（固定 slot 0，basic_usage.py）与 selected_predicted_iou。未修改 third_party/nanosam。

### 分模块 Benchmark（scripts/benchmark_nanosam.py）

固定输入 dogs.jpg，固定 Box Prompt [100,100,850,759]。所有 GPU timing 在 `torch.cuda.synchronize()` 后记录（非未同步 wall-clock）。

- model_load_ms = 626.954
- Image Encoder / set_image（warmup 20，test 100，重复同图编码实验）：mean 66.3219 / min 48.951 / max 75.6008 / P50 68.3064 / P95 74.7322 / P99 75.2667 ms。
- Box Decoder（warmup 30，test 300，复用缓存 embedding，不重复 set_image）：mean 11.0126 / min 9.4854 / max 15.6441 / P50 10.5491 / P95 13.8645 / P99 15.5847 ms；estimated prompt FPS = 1000/mean = 90.81。
- Point Decoder smoke（test 100，复用同一 embedding）：mean 10.2938 / P50 10.3027 / P95 10.4284 / P99 10.5383 ms；estimated prompt FPS = 97.15。确认不同 Prompt 类型可复用 embedding。
- full_segmentation（真实 encoder + 首个 box prompt 路径）= 52.186 ms。未将 prompt-only FPS 写成完整 NanoSAM FPS。

### tegrastats（解码负载下采样，n=121，JetPack6 `--start --interval 100 --logfile` 语法）

- RAM peak 2719 MB（last 2337）
- GR3D peak 92%（last 0%；Orin Nano/JetPack6 下空闲样本读 0%，按观测记录非猜测）
- GPU temperature peak 54.125 C（last 52.156）
- VDD_IN peak 8964 mW（last 8712）

### 正确性回归（benchmark 后重跑 baseline）

- Box regression：PASS — selected index 0，IoU 0.662，非零 242,155，无 NaN/Inf。
- Point regression：PASS — selected index 0，IoU 0.640，非零 233,325，无 NaN/Inf。

Stage 3C status: PASS。

## Day 3 Final Result

- NanoSAM commit: 653633614b2eb93b06ba3be9adb2aeffb117bd72
- 两个 ONNX SHA256：image encoder `d266fcdc…c9cf0a`；mask decoder `41e49a29…c3b247`（PC=Jetson 一致）
- Image Encoder FP16 Engine：34,766,700 B，SHA256 `c277e5ca…aaa6a8`
- Mask Decoder FP32 Engine：18,891,924 B，SHA256 `afb5ac49…ccb4f2`
- input/output shapes：encoder 1×3×1024×1024→[1,256,64,64]；decoder 见上→iou[1,4]+low_res_masks[1,4,256,256]
- Box Prompt：PASS（selected IoU 0.662，非零 242,155，area_ratio 0.270021，无 NaN/Inf）
- Point Prompt：PASS（selected IoU 0.640，非零 233,325，area_ratio 0.260175，无 NaN/Inf）
- mask selection rule：固定 slot 0（examples/basic_usage.py `mask[0,0]>0`，非 IoU argmax）；selected mask index = 0；selected predicted IoU = iou_predictions[0]
- benchmark 真实数据：model_load 626.954 ms；encoder mean 66.32 / P50 68.31 / P95 74.73 / P99 75.27 ms；box decoder mean 11.01 / P50 10.55 / P95 13.86 / P99 15.58 ms（prompt FPS 90.81）；point decoder mean 10.29 ms（FPS 97.15）；full_segmentation 52.186 ms；peak RAM 2719MB / GR3D 92% / GPU 54.125C / VDD_IN 8964mW。
- evidence 路径：`results/nanosam_baseline/`（Stage 3B baseline + regression/）、`results/nanosam_benchmark/`（benchmark.json / encoder_latencies.csv / decoder_latencies.csv / tegrastats.log / tegrastats_parsed.json / summary.md）
- 脚本：`scripts/run_nanosam_infer.py`（CLI）、`scripts/benchmark_nanosam.py`（benchmark）

## Day 3 结论

PASS — NanoSAM TensorRT promptable segmentation baseline reproduced and module-level latency characterized on Jetson Orin Nano.

## 明确未做

- NanoOWL + NanoSAM joint pipeline
- COCO accuracy / mIoU
- INT8
- MP4
- ROS2
- long-run stability

## Day 4 起点

NanoOWL → NanoSAM joint pipeline：

Image + Text Prompt → NanoOWL → Bounding Box → NanoSAM Box Prompt → Segmentation Mask
