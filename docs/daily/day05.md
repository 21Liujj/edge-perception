# Day 5

## 今日目标

建立 COCO 2017 正式质量评测协议，并完成数据准备审计。为 Day 5 后续 precision / quality / performance ablation 建立可比较、口径冻结的评测骨架。

本阶段只建立正式 evaluation protocol 和数据准备：
- 不重建 TensorRT Engine
- 不下载第三方数据集
- 不运行大规模模型评测
- 不修改 NanoOWL / NanoSAM upstream
- 不做 INT8 / MP4 / ROS2 / stability
- 不 remote / push / GitHub 上传
- 不继续 Stage 5B

---

## Stage 5A — COCO Evaluation Protocol

详细跨 Day 协议见 `docs/evaluation_protocol.md`（口径冻结，避免散落 daily log）。本日志记录 Stage 5A 的关键结论与审计结果。

### 数据集版本

- COCO 2017 Validation set（官方 `images.cocodataset.org`，不使用第三方镜像）。
- `val2017/`（5000 images）+ `annotations/instances_val2017.json`。

#### 当前 dataset / dependency 状态（Stage 5A 审计，本地执行）

| 项 | 状态 | 证据 |
|---|---|---|
| COCO val2017 本地存在 | **NO** | 已检查 `/home/nvidia/jetson-deploy-lab`、`/home/nvidia/`、`/data/`、`/datasets/`，均无 `val2017/` / `instances_val2017.json` |
| 官方源 connectivity | **AVAILABLE** | `curl -sI` 两个官方 zip 均 HTTP 200（轻量 header 检查，未下载 GB 级数据）|
| pycocotools | **MISSING** | `python3 -c "import pycocotools"` → ModuleNotFoundError |
| torchvision CocoDetection | AVAILABLE | torchvision 0.20.0a0+afc54f7 |
| NanoSAM 官方 eval_coco.py | 已阅读 | `third_party/nanosam/nanosam/tools/eval_coco.py` |

- 结论：**协议与框架已就绪；数据与 pycocotools 待准备**。本阶段返回的状态为 protocol/framework 层 PASS；数据层为待办（见下“离线数据获取”）。

### 离线数据获取（本阶段不执行，列出所需官方文件 + Jetson 目标目录）

需从官方源获取（仅官方，不使用镜像）：

1. `http://images.cocodataset.org/zips/val2017.zip`（~1 GB，5000 images）
2. `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`（~252 MB，含 `instances_val2017.json`）

Jetson 推荐目标目录（与官方 eval_coco.py 默认一致）：

```
/home/nvidia/jetson-deploy-lab/data/coco/
    val2017/
    annotations/
        instances_val2017.json
```

下载到 Jetson 后无需改协议或脚本默认路径。下载需在后续 Stage 显式授权后进行。

### NanoSAM 正式评测协议（要点，详见 evaluation_protocol.md）

- **评测单位 = COCO instance**（per annotation），不是 image。
- 流程：COCO instance → GT box（xywh→xyxy）→ NanoSAM box prompt `points=[[x1,y1],[x2,y2]]` `labels=[2,3]` → predicted mask（slot 0）→ GT seg mask `annToMask(ann)>0` → instance IoU。
- GT box 转换：`[x,y,w,h] → [x,y,x+w,y+h]`（与官方 `eval_coco.py` 一致）。
- **predicted mask slot = 0（BASELINE 锁定，非 IoU argmax）**，与 Day 4 baseline 一致。
  - 与官方 eval_coco.py 的关键差异：官方用 `iou_preds.argmax()` 选 slot（高估），本项目锁定 slot 0（更严格公平）。所有 variant 共用 slot 0，不偷偷改 argmax。
- 同一 image 多 instance 复用 image embedding（`set_image` 仅首 instance）。

指标：instance IoU / mean IoU(mIoU) / median IoU / P10/P50/P90 / IoU≥0.50 比例 / IoU≥0.75 比例 / valid·failed 数量 / mIoU by size bucket(small·medium·large) / skipped(crowd·invalid_box·invalid_gt_mask)。

核心指标记录：**评测单位 = COCO instance；核心指标 = instance IoU + mIoU（mean IoU）**。

invalid / edge 处理（写死）：
- crowd（`iscrowd==1`）：排除，计 `skipped_crowd`。
- zero/invalid bbox（`w<=0 or h<=0`）：`INVALID_BOX`，跳过，不计有效。
- 过小实例（`area<32²`）：不排除，单独 small bucket 统计（“不只挑大目标”，小目标 IoU 偏低是真实质量信号）。
- segmentation 缺失/annToMask 全零：`INVALID_GT_MASK`，跳过。
- 推理异常/NaN/Inf：`INFERENCE_FAILED`，IoU=null，计 failed，不中断整批，不进 IoU 分母（失败率单独报告）。

### NanoOWL Detection Protocol（定义，不执行）

- 输入：COCO 80 category names（从 `instances_val2017.json` categories 表取，不自编）。
- 固定 prompt 模板：`a {category_name}`（冻结，禁止后续改）。
- 单批 80 类一次性传入 `predict(image, text, text_encodings, threshold=0.10)`。
- 固定 threshold `0.10`（与 Day1/2/4 baseline 一致）；固定 top-k = 100/image（score 降序，NanoOWL `predict` 无内置 top-k，仅 threshold 过滤，top-k 在输出 JSON 阶段裁剪）。
- 输出 COCO detection JSON：`image_id` / `category_id` / `bbox`(xywh) / `score`。
- 未来指标：COCO AP / AP50 / AP75（需 `pycocotools.cocoeval.COCOeval`，当前 MISSING）。

#### NanoOWL API 约束（从本地源码确认，非猜测）

读 `third_party/nanoowl/nanoowl/owl_predictor.py`：
- `predict(image, text: List[str], text_encodings, threshold=0.1, pad_square=True)` — text 无数量硬上限，**80 类 prompt 无 API 约束**。
- `decode(image_output, text_output, threshold=0.1)` — threshold 单值广播为 per-label same threshold；无 top-k 参数。
- 输出 xyxy 原图像素（Day 4 已验证）。
- 结论：80 类直接可用；唯一成本是 text encode 一次（warmup 前编码并复用）。

### subset policy（deterministic）

| subset | 目标 instance | 用途 |
|---|---|---|
| SMOKE | ~20 | 跑通 / sanity |
| QUICK | ~100 | 快速质量趋势 |
| STANDARD | ~500 | Day 5 主评测口径 |
| FULL | val2017 全部合法实例 | 最终 / 论文口径可比 |

selection rule（冻结）：`numpy.random.seed(2027)`；过滤 `iscrowd==0`+合法 box+可解码 seg；按 category 分组 shuffle 后**轮转采样**至目标 N（覆盖多 category）；**不按 area 排序**（不只挑大目标，大小目标等同概率）；manifest schema 固定（image_id / annotation_id / category_id / category_name / image_path / bbox / area / iscrowd + 元数据）。

**本阶段数据未到位：manifest 生成逻辑实现于 `scripts/make_coco_subset.py`，不伪造 manifest 实例内容。**

### baseline precision configuration（锁定，与 Day 4 一致）

| 模块 | 精度 |
|---|---|
| NanoOWL | FP16 |
| NanoSAM Image Encoder | FP16 |
| NanoSAM Mask Decoder | FP32 |

mask selection = slot 0。

### planned FP32 variants（只设计，不执行）

| variant | OWL | SAM-Enc | SAM-Dec |
|---|---|---|---|
| A — Current baseline | FP16 | FP16 | FP32 |
| B — Quality reference | FP32 | FP32 | FP32 |
| C — attribution: OWL only | FP32 | FP16 | FP32 |
| D — attribution: SAM-enc only | FP16 | FP32 | FP32 |

所有 variant 共用：同一 subset / 同一 prompt / 同一 preprocessing / 同一 slot 0 / 同一 metric code。重建不同精度 Engine 在后续 Stage 显式授权后进行；Stage 5A 不重建。

### 评测工具

- `scripts/eval_nanosam_coco.py`（framework only，不伪造运行结果）：
  - CLI：`--coco-root --annotations --subset smoke|quick|standard|full --manifest --image-encoder-engine --mask-decoder-engine --output-dir --max-instances`
  - 模型只加载一次；同 image 多 instance 复用 embedding；每 instance GT box prompt；slot 0；per-instance IoU；aggregate metrics。
  - 数据缺失→`DATASET_MISSING` exit 2；pycocotools 缺失→`MISSING_DEPENDENCY: pycocotools` exit 3。
- `scripts/make_coco_subset.py`：deterministic manifest 生成器（同上 gating）。

### pycocotools

- 状态：**MISSING**。
- 安全安装建议（**本阶段不执行**）：`pip install pycocotools`（与 numpy 1.23.5 兼容，不升级其他核心依赖）。
- 安装需后续 Stage 显式授权。

---

## 约束遵守

- 未重建 Engine；未下载第三方数据集；未运行大规模推理；未修改 NanoOWL/NanoSAM upstream。
- 未做 INT8 / MP4 / ROS2 / stability。
- 未 remote / push / GitHub 上传；未继续 Stage 5B。
- 仅新增文档与 framework 脚本。
- 核心环境未变（数据层未触碰；未安装 pycocotools）。

## 结论

Stage 5A：protocol/framework 就绪。COCO val2017 与 pycocotools 待准备（官方源可访问）。Day 5 后续 Stage 可在数据到位后执行 precision ablation（A/B/C/D）。
