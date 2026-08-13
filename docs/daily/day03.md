# Day 3

## 今日目标

固定 NanoSAM 官方 upstream，完成 baseline 路径审计（架构、执行链、依赖、模型需求）。不下载模型、不构建 Engine、不做 inference。

## Stage 3A — NanoSAM upstream pin + baseline audit

### Upstream pin

- 初始检查 third_party/nanosam 不存在。
- GitHub 此时可达（git ls-remote https://github.com/NVIDIA-AI-IOT/nanosam.git HEAD 成功；curl https://github.com http 200）。网络阻塞已解除，无需 OFFLINE_TRANSFER_REQUIRED。
- 克隆官方仓库（GIT_LFS_SKIP_SMUDGE=1，仅源码）到 third_party/nanosam，未用镜像、未下载第三方打包版。
  - origin: https://github.com/NVIDIA-AI-IOT/nanosam.git
  - branch: main
  - HEAD commit: 653633614b2eb93b06ba3be9adb2aeffb117bd72（2023-09-20, "fix learning rate arg passing"）
  - git status: clean

### Baseline 架构（以当前固定 commit 的 README + 源码为准）

A. Image Encoder
- backbone: NanoSAM ResNet18（由 MobileSAM image encoder 蒸馏而来；非训练阶段不涉及）。
- 输入 shape: 1x3x1024x1024（Predictor.__init__ 默认 image_encoder_size=1024；preprocess_image 将任意图像 letterbox 到该尺寸）。注：源码 preprocess_image 默认参数为 512，但 Predictor 默认传入 1024，与 README resnet18_image_encoder.engine 构建口径一致。
- 输出 embedding shape: image_embeddings（image encoder TRTModule 输出名 image_embeddings），作为 mask decoder 输入。
- 官方精度: image encoder 用 FP16（trtexec --fp16）；README 注 MobileSAM encoder 必须 FP32（FP16 会出错），但 NanoSAM ResNet18 encoder 用 FP16 无明显精度下降。
- 是否转 TensorRT: 是。Engine 构建：下载 resnet18_image_encoder.onnx 后 trtexec --onnx=data/resnet18_image_encoder.onnx --saveEngine=data/resnet18_image_encoder.engine --fp16。

B. Prompt / Mask 路径
- 支持的 Prompt: point 与 box（box 用两个点 + label 2/3 表达）。
- point label 语义（README）: 0=背景点，1=前景点，2=bounding box 左上，3=bounding box 右下。
- Mask Decoder runtime: MobileSAM mask decoder，FP32 TensorRT engine。输入 image_embeddings, point_coords, point_labels, mask_input, has_mask_input；输出 iou_predictions, low_res_masks（256x256 低分辨率 mask，经 upscale_mask 双线性插值回原图）。
- Engine 构建：从 assets/mobile_sam.pt 导出 ONNX（python3 -m nanosam.tools.export_sam_mask_decoder_onnx --model-type=vit_t --checkpoint=assets/mobile_sam.pt --output=data/mobile_sam_mask_decoder.onnx），再 trtexec --onnx=... --saveEngine=data/mobile_sam_mask_decoder.engine --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=... --maxShapes=point_coords:1x10x2,point_labels:1x10（支持最多 10 个 keypoint）。
- 两个 engine 均经 torch2trt.TRTModule 加载（nanosam/utils/predictor.py）。

### 最简 sample（Image + Box -> Mask）

- 入口: examples/basic_usage.py
- sample image: assets/dogs.jpg
- prompt: 固定 bbox [100, 100, 850, 759]（x0,y0,x1,y1）-> points [[100,100],[850,759]]，point_labels [2,3]
- 所需模型文件: data/resnet18_image_encoder.engine + data/mobile_sam_mask_decoder.engine
- 执行: python3 examples/basic_usage.py --image_encoder=data/resnet18_image_encoder.engine --mask_decoder=data/mobile_sam_mask_decoder.engine -> 输出 data/basic_usage_out.jpg

### 依赖审计（仅审计，不安装；不破坏核心包）

venv: /home/nvidia/jetson-deploy-lab/.venv (include-system-site-packages = true)

| 依赖 | 状态 | 版本 |
|---|---|---|
| torch | AVAILABLE | 2.5.0a0+872d972e41.nv24.08 |
| torchvision | AVAILABLE | 0.20.0a0+afc54f7 |
| numpy | AVAILABLE | 1.23.5 |
| tensorrt | AVAILABLE | 10.3.0 |
| torch2trt | AVAILABLE | (Day 1 install) |
| transformers | AVAILABLE | 4.46.3 |
| Pillow (PIL) | AVAILABLE | 12.3.0 |
| OpenCV (cv2) | AVAILABLE | 4.10.0 |
| matplotlib | AVAILABLE | 3.10.3 |
| timm | MISSING（OPTIONAL，仅 nanosam/models/timm_image_encoder.py 用到，非 baseline 执行链） |

- VERSION_RISK 提示：setup.py install_requires=[]，无版本锁；baseline 运行链只依赖 torch/torchvision/numpy/tensorrt/torch2trt/PIL/matplotlib，均 AVAILABLE 且与 NanoOWL 共栈一致（未改变）。
- 无依赖被安装/升级/降级；torch/torchvision/numpy/TensorRT/CUDA/transformers/Pillow/OpenCV/torch2trt 全部保持现状。

### 下一阶段（Stage 3B）真正运行 NanoSAM 所需模型/权重

| 文件 | 来源 | 作用 | 联网? | 需 TRT 构建? |
|---|---|---|---|---|
| assets/mobile_sam.pt | 仓库自带（已随 clone，~40MB） | MobileSAM 权重，导出 mask decoder ONNX 的输入 checkpoint | 否 | 否（是 ONNX 导出源） |
| resnet18_image_encoder.onnx | Google Drive（README 链接，需下载） | NanoSAM ResNet18 image encoder ONNX | 是（Google Drive，非 GitHub/HF） | 否（是 TRT 构建输入） |
| data/mobile_sam_mask_decoder.onnx | 本地从 mobile_sam.pt 导出 | mask decoder ONNX | 否（本地导出） | 否（是 TRT 构建输入） |
| data/resnet18_image_encoder.engine | trtexec --fp16 由 ONNX 构建 | image encoder TRT engine | 否 | 是 |
| data/mobile_sam_mask_decoder.engine | trtexec 由 ONNX 构建（FP32） | mask decoder TRT engine | 否 | 是 |

- 联网项：resnet18_image_encoder.onnx（Google Drive）。若 Stage 3B 时 Google Drive 不可达，需离线传入该 ONNX。
- 本仓库已自带 assets/mobile_sam.pt，mask decoder ONNX/engine 可本地离线导出+构建。

### Stage 3A 约束遵守

- 未下载模型、未构建 Engine、未做 inference、未做 NanoOWL+NanoSAM 联合、未 benchmark、未 INT8、未 ROS2、未安装新依赖、未修改 NanoSAM 上游、未改核心包。
- 本阶段只 clone 源码 + 只读审计 README/setup.py/examples/源码。

- Stage 3A status: PASS。

---

## Stage 3B — NanoSAM TensorRT baseline (model prep + Engine + prompt segmentation)

### 执行结果：OFFLINE_TRANSFER_REQUIRED（未构建 Engine、未做 inference）

- Part 一 (nanosam package import): PASS — `python setup.py develop` 就地安装（editable egg-link，未改上游源码、未装新依赖）；`Predictor` 可 import。
- Part 二 (image encoder ONNX): OFFLINE_TRANSFER_REQUIRED — 官方来源 Google Drive（file ID 14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR），Jetson 不可达（port 443 超时），未下载、未用镜像。目标路径 third_party/nanosam/data/resnet18_image_encoder.onnx。
- Part 三 (mask decoder ONNX): BLOCKED — 本地官方脚本导出失败 `ModuleNotFoundError: No module named timm`（vit_t 的 TinyViT image encoder 需要 timm）；按规则 8 不安装 timm。README 官方替代来源亦为 Google Drive（file ID 1jYNvnseTL49SNRx9PDcbkZ9DwsY8up7n），同样不可达。目标路径 third_party/nanosam/data/mobile_sam_mask_decoder.onnx。
- Part 四 (Engine 构建): 未开始（按规则，无 ONNX 不继续 Engine Build）。
- Part 五 (Box/Point Prompt inference): 未开始；Box Prompt 未运行，Point Prompt 未运行；无 NaN/Inf 可报。
- Part 六 (合法性检查): N/A（无 inference）。

### 需离线传入的官方文件（无镜像）
1. resnet18_image_encoder.onnx — Google Drive ID 14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR → third_party/nanosam/data/resnet18_image_encoder.onnx（image encoder ONNX，trtexec --fp16 输入）
2. mobile_sam_mask_decoder.onnx — Google Drive ID 1jYNvnseTL49SNRx9PDcbkZ9DwsY8up7n → third_party/nanosam/data/mobile_sam_mask_decoder.onnx（mask decoder ONNX，trtexec FP32 输入；本地导出需 timm）
   - 注：mobile_sam.pt 已在仓库（offline, ~40MB），但本地导出 mask decoder ONNX 需 timm；若允许 timm，文件2 可本地离线生成（免网络）。

### 网络可达性（实测）
- drive.google.com : http 000 (Connection timed out, port 443)
- docs.google.com : http 000 (Connection timed out)
- drive.usercontent.google.com : http 000 (Connection timed out)
- (对比：GitHub / Hugging Face 在 Stage 3A 已确认可达；Google Drive 不可达)

### 续行命令（ONNX 到位后）
- image encoder:  trtexec --onnx=data/resnet18_image_encoder.onnx --saveEngine=data/resnet18_image_encoder.engine --fp16
- mask decoder :  trtexec --onnx=data/mobile_sam_mask_decoder.onnx --saveEngine=data/mobile_sam_mask_decoder.engine --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:1x1x2,point_labels:1x1 --maxShapes=point_coords:1x10x2,point_labels:1x10
- inference     :  python3 examples/basic_usage.py --image_encoder=data/resnet18_image_encoder.engine --mask_decoder=data/mobile_sam_mask_decoder.engine（assets/dogs.jpg, box [100,100,850,759]→points [[100,100],[850,759]] labels [2,3]）

### 核心包版本（保持不变）
torch 2.5.0a0+872d972e41.nv24.08 / torchvision 0.20.0a0+afc54f7 / numpy 1.23.5 / tensorrt 10.3.0 / torch2trt importable / transformers 4.46.3 / Pillow 12.3.0 / OpenCV 4.10.0 / timm MISSING(未安装)。

### 约束遵守
- 未做 NanoOWL+NanoSAM 联合 / benchmark / INT8 / ROS2 / MP4 / 训练；未改 upstream（nanosam git status clean）；未 remote/push/GitHub 上传；未升级/降级核心包；未安装新依赖（timm 未装）；未继续 Stage 3C。

### Engine / 证据
- 未产出 *.engine / *.onnx / mask 输出；third_party/nanosam/data/ 为空。
- 证据目录：results/nanosam_baseline/blocker_report.md（含完整失败点、官方 Drive ID/目标路径/续行命令）。

---

## Day 3 / Stage 3B (Resume) - Reproduce NanoSAM TensorRT baseline - PASS

恢复时间 2026-08-13。User 离线转存两个 ONNX（PC→Jetson），续行完成模型准备、Engine 构建、真实 Prompt Segmentation。

### Failure Case 历史（真实、保留）
1. 官方 README Google Drive 离线模型不可达：Jetson 访问 Google Drive (drive.google.com:443) 超时（port 443 timeout），`resnet18_image_encoder.onnx` / `mobile_sam_mask_decoder.onnx` 无法下载。官方 Drive ID：image encoder `14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR`、mask decoder (alt) `1jYNvnseTL49SNRx9PDcbkZ9DwsY8up7n`。
2. 本地导出 mask decoder ONNX 失败：`nanosam/mobile_sam/modeling/tiny_vit_sam.py` 需 `timm`，本机未安装 timm；按约束未安装 timm，未从 `mobile_sam.pt` 重新导出。
3. Docker 镜像 CloudFront EOF 未使用：官方 `docker pull` 路径在源端 CloudFront 出现 EOF，未采用。
4. 最终方案：User 离线转存官方提供的两个 ONNX 到 Jetson，PC→Jetson SHA256 一致后恢复 Stage 3B。

### 模型来源（离线转存，非在线下载）
- `resnet18_image_encoder.onnx` 63,321,621 B
  - SHA256 (PC) `d266fcdc9e4f0182b59946cd3cf1be331641f1d0d1de2415c4fd94e7a1c9cf0a`
  - SHA256 (Jetson) `d266fcdc9e4f0182b59946cd3cf1be331641f1d0d1de2415c4fd94e7a1c9cf0a` ✓ 一致
- `mobile_sam_mask_decoder.onnx` 16,504,875 B
  - SHA256 (PC) `41e49a298099048186ce109a4518286b8972959898a02577414405efa5c3b247`
  - SHA256 (Jetson) `41e49a298099048186ce109a4518286b8972959898a02577414405efa5c3b247` ✓ 一致
- onnx.checker.check_model：两个 ONNX 均 PASS；ONNX 形状解析正常（image encoder 1×3×1024×1024→[1,256,64,64]；mask decoder 输入 image_embeddings/point_coords/point_labels/mask_input/has_mask_input，输出 iou_predictions[1,4]+low_res_masks[1,4,256,256]）。

### Engine 构建（README trtexec 参数，未改 ONNX / 未装插件 / 未升级 TensorRT）
- Image Encoder（FP16）：
  `trtexec --onnx=data/resnet18_image_encoder.onnx --saveEngine=data/resnet18_image_encoder.engine --fp16`
  → `&&&& PASSED` (TensorRT 10.3.0)。size 34,766,700 B。SHA256 `c277e5ca445dc29b06e8be41a481e74bd806c4dd82071cbfbd47fddcf7aaa6a8`。
- Mask Decoder（FP32，min/opt/max point shapes）：
  `trtexec --onnx=data/mobile_sam_mask_decoder.onnx --saveEngine=data/mobile_sam_mask_decoder.engine --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:1x1x2,point_labels:1x1 --maxShapes=point_coords:1x10x2,point_labels:1x10`
  → `&&&& PASSED` (TensorRT 10.3.0)。size 18,891,924 B。SHA256 `afb5ac4958e6bab21ed8aa0cc65798f53a71406fe6124cf6b2e008da6fccb4f2`。
- 重载校验：`nanosam.utils.predictor.Predictor(ENC, DEC)` 加载两个 engine 成功（无异常），`set_image` 得到 image_embeddings `[1,256,64,64]`。

### 真实 Prompt Segmentation（assets/dogs.jpg，原图 1180×760）
- Box Prompt（bbox=[100,100,850,759]，points [[100,100],[850,759]]，labels [2,3]）：PASS
  - iou_predictions [0.662, 0.626, 0.758, 0.713]；low_res_mask [1,4,256,256]；hi_res_mask [1,4,760,1180]
  - best mask (mask[0,0]>0) 非零像素 242,155，mask_area_ratio 0.270021；NaN/Inf 无
- Point Prompt（foreground [475,430]，label [1]，单点不调优）：PASS
  - iou_predictions [0.640, 0.630, 0.740, 0.737]；low_res_mask [1,4,256,256]；hi_res_mask [1,4,760,1180]
  - best mask 非零像素 233,325，mask_area_ratio 0.260175；NaN/Inf 无
- 两类 prompt 均得到合法非空分割掩码，可视化确认掩码覆盖犬只区域。

### 合规字段
- 核心包版本未变：torch 2.5.0a0+872d972e41.nv24.08 / torchvision 0.20.0a0+afc54f7 / numpy 1.23.5 / tensorrt 10.3.0 / torch2trt importable / transformers 4.46.3 / Pillow 12.3.0 / OpenCV 4.10.0；timm 仍缺失（未安装）。
- 未改 upstream（nanosam git status clean）；未装新依赖；未改 ONNX/源码；未装插件；未升级 TensorRT。
- 未做 NanoOWL+NanoSAM 联合 / benchmark / INT8 / ROS2 / MP4 / 训练；未 remote/push/GitHub 上传；未继续 Stage 3C。
- 第三方大文件已 gitignore（*.engine/*.onnx/*.pt 不入 git）。

### 证据目录 results/nanosam_baseline/
- onnx_sha256.txt、engine_sha256.txt
- image_encoder_build.log、mask_decoder_build.log
- inference.log、inference_summary.json
- box_prompt_result.json、box_prompt_visualization.jpg
- point_prompt_result.json、point_prompt_visualization.jpg

### 状态
- Stage 3B：PASS。
- 本地提交：`feat: reproduce NanoSAM TensorRT baseline`（不 push）。
