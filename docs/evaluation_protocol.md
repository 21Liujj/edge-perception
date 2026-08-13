# EdgePerception — COCO Quality Evaluation Protocol

跨 Day 正式评测协议。本协议固化评测口径，避免散落在 daily log 中。
**本协议为 Stage 5A 定义；大规模推理在后续 Stage 执行。**

- 状态：Protocol 定义完成（Stage 5A）。大规模评测尚未执行。
- 约束：本阶段不重建 Engine、不下载第三方数据集、不运行大规模模型评测、不修改 NanoOWL/NanoSAM upstream。

---

## 0. 数据集版本

- **COCO 2017 Validation set**（官方，唯一来源 `images.cocodataset.org`，不使用第三方镜像）。
  - images: `val2017/`（5000 images）
  - annotations: `annotations/instances_val2017.json`
- 评测基于 `instances_val2017.json`（含 80 category、bbox、segmentation polygon、iscrowd、area）。

### 0.1 当前 dataset / dependency 状态（Stage 5A 审计）

| 项 | 状态 |
|---|---|
| COCO val2017 本地存在 | **NO**（已检查 `/home/nvidia/jetson-deploy-lab`、`/home/nvidia/`、`/data/`、`/datasets/`，均不存在）|
| 官方源 connectivity | **AVAILABLE**（`http://images.cocodataset.org/zips/val2017.zip` 与 `.../annotations/annotations_trainval2017.zip` 均 HTTP 200，轻量 header 检查，未下载）|
| pycocotools | **MISSING**（`import pycocotools` → ModuleNotFoundError）|
| torchvision CocoDetection | AVAILABLE（torchvision 0.20.0a0+afc54f7，含 `torchvision.datasets.CocoDetection`）|
| 官方 NanoSAM eval_coco.py 参考 | 已阅读（`third_party/nanosam/nanosam/tools/eval_coco.py`）|

### 0.2 离线数据获取要求（联网开发机 / Jetson 下载均可，本阶段不执行）

需要从官方源获取（**仅官方，不使用镜像**）：

1. `http://images.cocodataset.org/zips/val2017.zip`（~1 GB，5000 images）
2. `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`（~252 MB，含 `instances_val2017.json`）

**Jetson 推荐目标目录**（与官方 eval_coco.py 默认一致，便于复用）：

```
/home/nvidia/jetson-deploy-lab/data/coco/
    val2017/                         # 解压 val2017.zip
    annotations/
        instances_val2017.json       # 解压 annotations_trainval2017.zip
```

> 下载到 Jetson 后无需改动本协议或脚本路径默认值（`--coco-root` 默认即 `data/coco/val2017`，
> `--annotations` 默认即 `data/coco/annotations/instances_val2017.json`）。
> 下载需在 Day 5 后续 Stage 显式授权后进行；Stage 5A 不下载。

---

## 1. NanoSAM Evaluation Protocol（GT-Box-Prompt Quality）

### 1.1 评测单位

- **样本单位 = COCO instance**（per annotation），**不是 image**。
- 每个 instance 一次 box-prompt 推理 + 一次 mask IoU 计算。

### 1.2 流程

```
COCO val2017 instance
  → GT bounding box (xywh → xyxy)
  → NanoSAM Box Prompt  points=[[x1,y1],[x2,y2]]  labels=[2,3]
  → predicted mask (selected slot)
  → GT segmentation mask (annToMask polygon; RLE 仅 iscrowd 时出现，本协议排除 crowd)
  → instance IoU
```

### 1.3 GT box → NanoSAM prompt

- COCO ann `bbox` = [x, y, w, h]（xywh，原图像素）。
- 转换：`box_xyxy = [x, y, x+w, y+h]`（与官方 `eval_coco.py` 的 `box_xywh_to_xyxy` 一致）。
- NanoSAM prompt（与 Day 4 bridge 约定、与官方 eval_coco 一致）：
  - `points = [[x1,y1],[x2,y2]]`，`point_labels = [2,3]`（2=box 左上，3=box 右下）。
  - 两端均为原图像素 xyxy，**不交换 xy 顺序**。

### 1.4 GT segmentation 解码

- 非 crowd：`annToMask(ann)` 返回 polygon → 二值 mask（原图 H×W）。
- crowd：**排除**（见 §1.6）。
- GT mask 二值化：`gt = (annToMask(ann) > 0)`。

### 1.5 predicted mask slot 选择（BASELINE 锁定）

- **Baseline 必须使用 mask slot 0**（`hi_res_mask[0,0] > 0`），与 Day 4 baseline 一致。
- predicted mask = `predictor.predict(...)` 返回的第 1 个返回值 `hi_res_mask`，shape `[1,4,H,W]`，取 `hi_res_mask[0, 0] > 0`。
- selected predicted IoU = `mask_iou[0,0]`（仅记录，**不用于选 mask**）。

> **与官方 eval_coco.py 的关键差异（必须记录）：**
> 官方 `eval_coco.py.predict_box` 使用 `mask[0, iou_preds.argmax()]`（IoU argmax 选 slot），
> 会高估真实 box-prompt 质量（相当于让模型自选最优 mask）。
> 本项目 baseline 锁定 **slot 0**（与 Day 4 mask selection rule 一致：官方 `examples/basic_usage.py` 固定 slot 0），
> 是更严格、更公平的口径。**所有 variant 共用此规则，不偷偷改成 IoU argmax。**
> 若未来需对比官方口径，单独跑一次 `--mask-select argmax` 变体并明确标注，不混入 baseline。

### 1.6 invalid / edge case 处理（写死，禁止后续临时改口径）

- **crowd（`iscrowd == 1`）**：**排除**，计入 `skipped_crowd`，不进入有效样本。
- **invalid / zero-area bbox**：`w <= 0 or h <= 0` → 标记 `INVALID_BOX`，计入 `skipped_invalid_box`，跳过推理。
- **过小实例**：定义“过小” = `area < 32*32`（COCO small object 边界，area 为 COCO ann 自带像素面积）。不排除，但**单独统计** small/medium/large bucket：
  - small: `area < 32²`（1024）
  - medium: `32² <= area <= 96²`（9216）
  - large: `area > 96²`
  > 与官方 NanoSAM README 的 small/medium/large 口径一致（README 按 target object size 分别报 mIoU）。
  > 不剔除小目标（“不只挑大目标”）；小目标 IoU 偏低是真实质量信号，记录而非掩盖。
- **segmentation 缺失**：COCO instances annotation 必含 segmentation（非 crowd 时为 polygon）。
  若 `annToMask` 全零或异常 → 标记 `INVALID_GT_MASK`，计入 `skipped_invalid_gt_mask`，跳过。
- **推理异常**：单 instance 推理抛异常 → 捕获，标记 `INFERENCE_FAILED`，IoU 记 null，计入 `failed`；
  **不中断整批评测**，继续下一 instance。
- **NaN / Inf**：predicted mask 或 IoU 出现 NaN/Inf → 同 `INFERENCE_FAILED` 处理。

### 1.7 image embedding 复用

- 同一 image 的多个 instance：**仅首个 instance 调用 `predictor.set_image(image)`**，后续 instance 复用 embedding（与官方 `eval_coco.py` 的 `set_image=(j==0)` 一致，也同 Day 4 cached-embedding 思路）。
- 这不改变 per-instance IoU 的正确性（box prompt 仍 per-instance），仅避免重复编码。

---

## 2. Metrics（NanoSAM Quality）

每个有效 instance 计算一次 `instance IoU = |pred ∩ gt| / |pred ∪ gt|`（二值 mask，`union==0` 时 IoU=0）。

聚合指标：

| 指标 | 定义 |
|---|---|
| instance IoU | per-instance 原始值 |
| **mean IoU / mIoU** | 全部有效 instance IoU 的算术平均 |
| median IoU | 有效 instance IoU 中位数 |
| P10 / P50 / P90 IoU | 有效 instance IoU 的分位数 |
| IoU >= 0.50 比例 | 有效 instance 中 IoU ≥ 0.5 的占比 |
| IoU >= 0.75 比例 | 有效 instance 中 IoU ≥ 0.75 的占比 |
| valid / failed instance 数量 | valid = 进入统计的有效样本数；failed = inference 失败数 |
| mIoU by size bucket | small / medium / large 分别的 mean IoU |
| skipped count | crowd / invalid_box / invalid_gt_mask 各类别跳过数 |

- skipped 不进入 mean/median/分位分母；failed（INFERENCE_FAILED）单独计，**也不进入 IoU 统计分母**（避免把 0 当真实质量拉低均值；失败率单独作为可靠性指标报告）。

---

## 3. Subset Policy（deterministic）

为在 Jetson 上先做快速实验，定义 deterministic subset。**生成逻辑固定，不伪造 manifest 内容。**

### 3.1 通用 selection rule（所有 subset 共用）

1. 加载 `instances_val2017.json`（pycocotools 或 torchvision CocoDetection）。
2. 过滤：`iscrowd == 0` 且 bbox 合法（`w>0 and h>0`）且 segmentation 可解码（`annToMask` 非全零）。
3. 固定 random seed：`numpy.random.seed(2027)`（deterministic，可复现）。
4. 尽量覆盖多个 category：先按 category 分组，每个 category 内 shuffle 后轮转采样，直到达到目标 instance 数（覆盖广，不集中在少数类）。
5. **不只挑大目标**：在 step 4 轮转采样时不按 area 排序，等同概率采样大小目标（small/medium/large 分布接近全集）。
6. 达到目标 N 即停，记录最终所覆盖 category 数与 size bucket 分布写入 manifest 元数据。

### 3.2 subset 定义

| subset | 目标 instance 数 | 用途 |
|---|---|---|
| **SMOKE** | ~20 | 跑通 pipeline / 快速 sanity |
| **QUICK** | ~100 | 快速质量趋势 |
| **STANDARD** | ~500 | Day 5 主评测口径 |
| **FULL** | COCO val2017 全部合法实例 | 最终 / 与论文口径可比 |

- 目标 instance 数为“约”值（轮转采样到覆盖 ≥ 阈值 category 数时可能有 ±少量偏差），最终以 manifest 实际记录为准。

### 3.3 manifest schema（JSON）

固定为 JSON，每个 instance 至少记录：

```json
{
  "subset": "standard",
  "seed": 2027,
  "target_instances": 500,
  "generated_at": "<ISO8601 生成时填>",
  "num_instances": <实际>,
  "num_categories": <实际>,
  "size_bucket_counts": {"small": N, "medium": N, "large": N},
  "instances": [
    {
      "image_id": 139,
      "annotation_id": 12345,
      "category_id": 1,
      "category_name": "person",
      "image_path": "data/coco/val2017/000000000139.jpg",
      "bbox_xywh": [x, y, w, h],
      "area": 1234.5,
      "iscrowd": 0
    }
  ]
}
```

- `image_path` 相对 project root（`cooc_root` 解析后的 val2017 路径）。
- `category_name` 从 `instances_val2017.json` 的 categories 表映射（id→name）。
- manifest 文件命名：`results/coco_subsets/{subset}_manifest.json`。
- **本阶段：数据未到位，manifest 生成逻辑实现于 `scripts/make_coco_subset.py`（设计，可创建但运行需数据 + pycocotools）；不伪造 manifest 实例内容。**

---

## 4. NanoOWL Detection Protocol（定义，不执行）

### 4.1 输入

- **COCO 80 category names**（从 `instances_val2017.json` categories 表取 80 类 name，**不自行编造**）。
- 固定文本 prompt 模板：`a {category_name}`（owlvit-patch32 对 "a photo of a X" 与 "a X" 均常见；本协议固定 `a {name}` 以保证可复现；模板在 protocol 固定后禁止改）。
  - 单批 prompt：全部 80 类文本一次性传入 `predictor.encode_text(prompts)` / `predict(image, text, ...)`。
  - `text: List[str]` 数量无硬上限（OwlViT 支持任意数量 text query；80 类 OK）。

### 4.2 detection 规则（写死）

- **threshold**：固定 `0.10`（与 Day 1/2/4 NanoOWL baseline 一致）。每个 category 同一 threshold。
  - `decode(image_output, text_output, threshold=0.10)` 已对每个 label 应用相同 threshold。
- **top-k 规则**：NanoOWL `predict` 不内置 top-k（仅 threshold 过滤）。本协议固定 **per-image top-k = 100 detections**（按 score 降序取前 100，覆盖足够候选，避免极端图爆炸）。
  - top-k 在 COCO JSON 输出阶段裁剪，不改变模型行为。
- 输出 bbox 为 xyxy 原图像素坐标（Day 4 已确认），输出 COCO JSON 时转回 xywh。

### 4.3 输出 COCO detection JSON

字段：

```json
[
  {
    "image_id": 139,
    "category_id": 1,
    "bbox": [x, y, w, h],
    "score": 0.3419
  }
]
```

- `category_id` = COCO category id（与 80 类 name 映射一致）。
- `bbox` xywh 原图像素。
- 文件：`results/nanoowl_coco_detections.json`。

### 4.4 指标（未来，需 pycocotools COCOeval）

- **COCO AP**（AP @[IoU=0.50:0.95]）
- **AP50**（AP @[IoU=0.50]）
- **AP75**（AP @[IoU=0.75]）
- 依赖：`pycocotools.cocoeval.COCOeval`（**当前 MISSING，见 §0.1**）。

### 4.5 NanoOWL API 约束（从本地源码确认，非猜测）

读 `third_party/nanoowl/nanoowl/owl_predictor.py`：

- `OwlPredictor.predict(image: PIL.Image, text: List[str], text_encodings, threshold=0.1, pad_square=True)` — text 为 List[str]，**无数量硬上限**，80 类 prompt 输入无 API 约束。
- `decode(image_output, text_output, threshold=0.1)` — threshold 单值会广播为 per-label same threshold；无 top-k 参数。
- 输出 `OwlDecodeOutput(labels, scores, boxes, input_indices)`；boxes 为 xyxy 原图像素（Day 4 已验证 `_owl_box_roi_to_box_global`）。
- **结论**：80 类 prompt 直接可用，无需特殊处理；唯一约束是 prompt 数量越多 text encode 一次成本越高（warmup 前编码一次并复用，同 Day 4 benchmark 思路）。

---

## 5. Baseline Precision Configuration（锁定）

| 模块 | 当前 baseline 精度 |
|---|---|
| NanoOWL | **FP16** |
| NanoSAM Image Encoder | **FP16** |
| NanoSAM Mask Decoder | **FP32** |

- 此为 Day 4 稳定 baseline，Day 5 起所有 ablation 以此为对照基准。
- mask selection = **slot 0**（§1.5）。

---

## 6. Precision Ablation Variants（设计，不执行）

所有 variant 必须共用：**同一 COCO subset、同一 prompt、同一 preprocessing、同一 mask selection rule（slot 0）、同一 metric code**。只有精度配置不同，才能做真正 ablation。

| variant | NanoOWL | NanoSAM Encoder | NanoSAM Decoder |
|---|---|---|---|
| **A — Current baseline** | FP16 | FP16 | FP32 |
| **B — Quality reference** | FP32 | FP32 | FP32 |
| **C — attribution**: OWL only | FP32 | FP16 | FP32 |
| **D — attribution**: SAM-enc only | FP16 | FP32 | FP32 |

- B 为全 FP32 质量上界对照；C/D 用于把质量差异归因到具体模块（NanoOWL vs NanoSAM Encoder）。
- Decoder 在非 B 中保持 FP32（baseline 即 FP32），避免 decoder 精度污染归因。
- 重建不同精度的 Engine 在后续 Stage 显式授权后进行；**Stage 5A 不重建任何 Engine。**

---

## 7. Evaluator 实现要求

`scripts/eval_nanosam_coco.py`（本阶段创建 framework，不伪造运行结果）：

1. 模型只加载一次（OwlPredictor / NanoSAM Predictor 各一次）。
2. 同一 image 的多个 instance 优先复用 NanoSAM image embedding（`set_image` 仅首 instance）。
3. 每个 instance 使用 GT box prompt。
4. selected mask index = 0。
5. 计算 predicted mask vs GT mask IoU。
6. 保存 per-instance result（JSON/CSV）。
7. 统计 aggregate metrics（§2）。

CLI：

```
--coco-root        (default data/coco/val2017)
--annotations      (default data/coco/annotations/instances_val2017.json)
--subset           smoke|quick|standard|full
--manifest         (path to subset manifest JSON; full 时忽略)
--image-ncoder-engine
--mask-decoder-engine
--output-dir
--max-instances    (可选，硬上限，覆盖 subset 数)
```

- 数据缺失时：CLI 可 import / argparse 成功，运行时检测 `coco_root` 不存在 → 打印 `DATASET_MISSING` 并退出（exit 2），不伪造结果。
- pycocotools 缺失时：GT mask 解码用 `torchvision.datasets.CocoDetection` + `coco.annToMask`（不依赖 pycocotools 顶层 import；`annToMask` 来自 `pycocotools`，故仍需 pycocotools）。因此 pycocotools MISSING 时 evaluator 运行 → 打印 `MISSING_DEPENDENCY: pycocotools` 并退出（exit 3）。
- 安全安装建议（**本阶段不执行**）：`pip install pycocotools`（与现有 numpy 1.23.5 兼容；不升级其他核心依赖）。安装需在后续 Stage 显式授权。

---

## 8. 变更与口径冻结

- 本协议在 Stage 5A 冻结。后续任何口径变更（threshold/top-k/slot/invalid 规则）必须在本文档显式修改并注明 Stage 与理由，禁止在评测脚本里临时改口径。
- subset 生成逻辑冻结于 §3.1；manifest 一旦生成不可手工编辑实例列表。
