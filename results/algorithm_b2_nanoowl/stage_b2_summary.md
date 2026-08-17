# Stage B2 — NanoOWL COCO 检测基线

## 评测范围

- 数据集：COCO 2017 val，5,000 张图像，36,781 个 GT 实例
- 模型：`google/owlvit-base-patch32`
- Engine：已验收的 NanoOWL FP16 TensorRT 图像编码器
- 提示词：80 个 COCO 类别，统一使用 `a {category_name}`
- 主策略：threshold `0.10`、每图 top-k `100`、`pad_square=False`
- 正式评测器：`pycocotools 2.0.11`、`COCOeval(iouType="bbox")`

SMOKE 20、QUICK 100、STANDARD 500 和 FULL 5,000 按顺序全部完成。STANDARD 总耗时 44.52 秒，图像循环 16.92 秒，生成结果约 4.1 MB，据此判断 FULL 在 wall time 和磁盘方面可执行。

## FULL 主指标

| 指标 | 数值 |
|---|---:|
| AP | 0.258616 |
| AP50 | 0.407885 |
| AP75 | 0.272948 |
| APsmall | 0.095263 |
| APmedium | 0.264089 |
| APlarge | 0.442186 |
| AR1 | 0.255193 |
| AR10 | 0.365058 |
| AR100 | 0.372312 |

主策略输出 44,962 个预测，5,000 张图中有 82 张零预测。所有输出框均为有限数值，已裁剪到图像边界并转换为 COCO xywh，且通过 `COCO.loadRes` 和 COCOeval 验证。

## 策略分析

threshold 消融，top-k 固定为 100：

| Threshold | 预测数 | AP | AP50 | AR100 |
|---:|---:|---:|---:|---:|
| 0.05 | 111,588 | 0.270547 | 0.430476 | 0.427018 |
| 0.10 | 44,962 | 0.258616 | 0.407885 | 0.372312 |
| 0.20 | 16,100 | 0.219814 | 0.337401 | 0.279905 |
| 0.30 | 8,171 | 0.178063 | 0.262881 | 0.209947 |

top-k 消融，threshold 固定为 0.10：

| Top-k | 预测数 | AP | AR100 |
|---:|---:|---:|---:|
| 10 | 29,700 | 0.249041 | 0.344323 |
| 50 | 44,315 | 0.258508 | 0.371937 |
| 100 | 44,962 | 0.258616 | 0.372312 |

threshold 0.05 取得本次测试中最高的 AP 和 recall，但预测数是主阈值的 2.48 倍。在 threshold 0.10 下，top-k 50 相对 100 几乎无损；top-k 10 会明显降低 recall。

## 类别与失败分析

APlarge `0.442186` 是 APsmall `0.095263` 的 4.64 倍。高 AP 类别包括 cat `0.615114`、train `0.592233`、airplane `0.575426` 和 fire hydrant `0.556048`。弱势高频类别包括 book：1,161 个 GT、AP `0.041912`；handbag：540 个 GT、AP `0.052681`；backpack：371 个 GT、AP `0.059292`；chair：1,791 个 GT、AP `0.107673`。

仅用于分析的 IoU 0.50 matcher 发现 car/truck、chair/couch、knife/fork、snowboard/skis 和 microwave/oven 等高频语义混淆。15 张可视化覆盖成功检测、FP-heavy、FN-heavy、小目标失败和语义混淆。该 matcher 不参与正式指标，以上正式质量结论均来自 COCOeval。

## 吞吐率

- 模型加载：2.20 秒
- 80 个提示词一次性文本编码：0.49 秒
- 图像循环：165.26 秒
- Evaluation throughput：30.26 images/s
- 评测器总 wall time：349.17 秒

图像循环吞吐包含图像加载、预处理、TensorRT 图像编码、NanoOWL 解码、host 传输与输出验证。该口径不同于 Day 2 steady-state inference-only FPS，也不同于 NanoOWL 到 NanoSAM 联合 Pipeline 吞吐率。

## 回归结果

- Day 2 NanoOWL：PASS，5/5 cases，owl anchor score `0.3419235944747925`
- Day 4 joint pipeline：`status=OK`，owl 框和分数复现，NanoSAM 保持 fixed slot 0，mask 有限且非空，selected predicted IoU `0.720552`

## 解释与决策

`ESTABLISH_NANOOWL_COCO_BASELINE`

FULL 结果是在统一提示词模板下得到的零样本开放词汇 COCO 基线。实际 AP、弱势类别、小目标差距和提示词相关语义混淆都是当前部署的真实限制。由于类别定义、监督方式和推理输入不同，该结果不能直接与经过 COCO 专项训练的 YOLO 类检测器对比；这一差异也不能用来回避实测质量。

Stage B2 已完成。未进行训练、微调、模型修改、Engine 重建、精度实验、CUDA/C++ 优化，也未修改任一 third-party 模型目录。
