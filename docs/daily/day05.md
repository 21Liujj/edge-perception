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

---

## Stage 5B — NanoSAM COCO Quality Baseline（执行）

在 Stage 5A 协议冻结后，COCO val2017 数据与 pycocotools 已准备就绪。本阶段执行 NanoSAM box-prompt segmentation 质量基线评测，不修改协议、不重建 Engine、不跑 FULL、不继续 Stage 5C。

### 1. COCO 数据校验

PC→Jetson 传入两个官方 ZIP，SHA256 逐一确认（完全一致）：

- `data/coco/downloads/val2017.zip` → `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05`
- `data/coco/downloads/annotations_trainval2017.zip` → `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`

ZIP integrity test（`unzip -t`）：两个 ZIP 均无错误。

### 2. 解压

仅解压评测需要的数据（只保留 `instances_val2017.json`)：

- `data/coco/val2017/`（5000 images）
- `data/coco/annotations/instances_val2017.json`

原始 val2017 图片 / downloads ZIP / 大型 annotation 不进入 Git（`.gitignore` 新增规则）；manifest 进入 Git。

### 3. Dataset statistics

- val2017 图片数量 = **5000**
- pycocotools 版本 = **2.0.11**（可正常加载 instances_val2017.json）
- categories = 80（id 1..90，80 类）
- annotations = 36781（crowd 446，非 crowd 合法 36335）
- 非crowd size buckets：small 15264 / medium 12392 / large 8679

### 4. Manifests（seed=2027，Stage 5A protocol）

生成于 `data/eval_manifests/`（deterministic，进入 Git）：

| subset | instances | unique images | categories | small | medium | large |
|---|---|---|---|---|---|---|
| smoke   | 20   | 20  | 20 | 6   | 6   | 8   |
| quick   | 100  | 99  | 80 | 40  | 28  | 32  |
| standard| 500  | 455 | 80 | 175 | 166 | 159 |
| full    | 36335| 4952| 80 | 15264| 12392| 8679 |

FULL manifest 仅作为索引生成，**未用于推理**（不跑 FULL inference）。

### 5. Evaluator 修复（不改协议）

`scripts/eval_nanosam_coco.py` 在 Stage 5A 仅有框架，运行时发现两处实现缺陷并修复（均为 bug fix，非口径修改）：

1. **`_materialize` 路径丢失**：manifest instance 携带 `bbox_xywh`+`image_path`（frozen schema），原代码据 `bbox_xyxy` 判定 pre-materialized → 落入 fallback 用空 `images_root` 重建，导致 `image_path` 退化为裸文件名，全部 `IMAGE_LOAD_FAILED`。修复：据 `image_path`+`annotation_id` 判定 manifest instance，从 `bbox_xywh` 推导 `bbox_xyxy`，保留 manifest `image_path`，并补 `status=OK`。
2. **`mask_iou` 变量遮蔽**：`predictor.predict(...)` 解包出返回值命名为 `mask_iou`（iou_predictions 张量），遮蔽了同名 IoU 计算函数 → `TypeError: 'Tensor' object is not callable`。修复：返回值重命名为 `mask_iou_preds`。

未修改 evaluation protocol（评测单位 / GT box 转换 / slot 0 / invalid 规则 / metric 定义均不变）。

### 6. NanoSAM 配置（BASELINE LOCKED）

| 模块 | 精度 |
|---|---|
| Image Encoder (ResNet18) | FP16 TRT |
| Mask Decoder (MobileSAM) | FP32 TRT |
| Prompt | COCO GT Bounding Box (xyxy, labels [2,3]) |
| Mask selection | slot 0（固定，非 IoU argmax） |
| Embedding reuse | 同一 image 多 instance 复用（仅首 instance set_image） |

Engine SHA256 与 Day 3 构建一致，未重建：enc `c277e5ca...aaa6a8`（FP16）、dec `afb5ac49...ccb4f2`（FP32）。

### 7. SMOKE (~20 instances)

- **PASS**：valid=20，failed=0，mIoU=0.7555，wall=2.36s。
- Gating：execution errors=0、valid>0、IoU 无 NaN/Inf。通过，继续。

### 8. QUICK (~100 instances)

- valid=100，failed=0，mIoU=0.7213，median=0.7602，P10=0.4487，P90=0.9208。
- IoU>=0.50 = 0.880；IoU>=0.75 = 0.530。
- mIoU by size：small 0.6598(40) / medium 0.7279(28) / large 0.7925(32)。
- unique images=99，embedding reuse=1。

执行稳定（0 failed），继续 STANDARD。

### 9. STANDARD (~500 instances)

- valid=500，failed=0，mIoU=0.7181，median=0.7839，P10=0.3791，P90=0.9192。
- IoU>=0.50 = 0.846；IoU>=0.75 = 0.590。
- mIoU by size：small 0.6053(175) / medium 0.7627(166) / large 0.7958(159)。
- **unique images = 455**
- **embedding reuse 数量 = 45**（500 instances − 455 set_image calls）
- **evaluation wall time = 52.06 s**
- **instances/sec = 9.604**（离线评测吞吐，非实时 NanoSAM FPS）

### 10. Runtime / embedding reuse 备注

- instances/sec 为离线评测吞吐（含图片加载 + set_image + predict + IoU 计算），非 NanoSAM 单图实时 FPS。
- embedding reuse：同一 image 多 instance 仅首 instance 调 `set_image`，其余复用 embedding（与官方 eval_coco.py `set_image=(j==0)` 一致）。STANDARD 中 45 个 instance 复用了已缓存的 embedding。
- 未将 prompt-only decode 误写为完整 NanoSAM FPS。

### 11. Day 3 dogs.jpg Box Prompt regression（重新运行）

- 输入：`third_party/nanosam/assets/dogs.jpg`（1180×760），box prompt [100,100,850,759] labels [2,3]，slot 0。
- **PASS**：selected IoU 0.6619（与 Day 3 baseline 0.662 一致），mask 非空（242155 非零像素），无 NaN/Inf。

### 12. Evidence 路径

- SMOKE：`results/coco_nanosam_baseline/smoke/`（summary.json / per_instance.jsonl / failed_instances.json / run.log）
- QUICK：`results/coco_nanosam_baseline/quick/`（同上）
- STANDARD：`results/coco_nanosam_baseline/standard/`（同上）
- 汇总：`results/coco_nanosam_baseline/baseline_summary.md`
- Manifests：`data/eval_manifests/{smoke,quick,standard,full}_manifest.json`
- dogs regression：`results/nanosam_baseline/box_result.json` / `box_mask.npy` / `box_visualization.jpg`

### 13. Constraints 遵守

- 未重建 Engine；未做 FP32 ablation；未跑 NanoOWL COCO AP；未做 INT8 / MP4 / ROS2 / stability。
- 未修改 upstream；未 remote / push / GitHub 上传。
- **未运行 FULL 推理**。
- 未修改 evaluation protocol 改善结果。
- **未继续 Stage 5C**。
- 核心环境保持：torch 2.5.0a0+872d972e41.nv24.08 / tensorrt 10.3.0 / numpy 1.23.5 / torchvision 0.20.0a0+afc54f7 / pycocotools 2.0.11。无评测残留进程。

## Stage 5B 结论

NanoSAM COCO box-prompt 质量基线已建立：STANDARD mIoU=0.7181（500 instances，0 failed），small/medium/large mIoU = 0.6053 / 0.7627 / 0.7958，IoU>=0.50=0.846，IoU>=0.75=0.590。dogs.jpg 回归 PASS。基线可用于后续 precision ablation（A/B/C/D）对照。**不继续 Stage 5C。**
