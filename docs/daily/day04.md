# Day 4

## 今日目标

完成 NanoOWL → NanoSAM 首次自动联合 Pipeline：Image + Text Prompt → NanoOWL → Bounding Box → NanoSAM Box Prompt → Segmentation Mask。本阶段只验证联合正确性，不做正式 Benchmark / COCO / INT8 / MP4 / ROS2 / stability。

## Stage 4A 架构

- 单进程内联合（不通过 subprocess 拼两个 CLI），复用 `scripts/run_nanoowl.py` 已验证的 OwlPredictor 逻辑与官方 `nanosam.utils.predictor.Predictor`，不复制 third_party 模型核心实现，不修改 upstream。
- CLI（`scripts/run_edge_perception.py`）：`--image --prompts --threshold --target-label --nanoowl-engine --nanosam-image-encoder-engine --nanosam-mask-decoder-engine --output-dir`。
- 流程：NanoOWL 开放词汇检测 → selection → bridge（box 转换）→ NanoSAM Box Prompt → mask。

## NanoOWL → NanoSAM bridge 规则

1. NanoOWL 输出候选 detection：label / score / box [x1,y1,x2,y2]。
2. Coordinate convention（读两端源码确认，非记忆）：
   - NanoOWL：OwlPredictor.predict 返回 output.boxes 为 **xyxy，原图像素空间**（pad_square=False）。见 owl_predictor `_owl_box_roi_to_box_global` 将归一化 roi box 映射到原图 (x0,y0,wh) 像素。
   - NanoSAM：Predictor.predict(points, point_labels) 的 points 在 **原图 XY 像素空间**；preprocess_points 按 orig_image_encoder_size/max(W,H) 缩放。Box prompt 编码为 points=[[x1,y1],[x2,y2]]，labels=[2,3]（2=box 左上，3=box 右下）。
   - 两端均为原图 [x,y]/xyxy，**约定一致，无需交换 xy 顺序**：NanoOWL box [x1,y1,x2,y2] 直接映射 NanoSAM point_coords [[x1,y1],[x2,y2]]。
3. Bridge 必查并记录：原图 NanoOWL box、image W/H、box 是否越界、clipping 后 box、NanoSAM point_coords、labels [2,3]。owl case 中 box 未越界（clipped_box == original_box）。

## detection selection rule

- 若指定 `--target-label`：从 label 匹配该 target 的 detection 中选 score 最高（detections 已按 score 降序，取首个匹配）。
- 若未指定：取所有 detection 中 score 最高的一个（dets[0]）。
- 该 rule 写入输出 JSON（`nanoowl.selection_rule`）。owl case rule = "highest-score detection matching target-label an owl"。

## mask selection rule

- MobileSAM decoder 输出 4 候选 mask（low_res_masks[1,4,256,256]、iou_predictions[1,4]）。
- 官方 examples/basic_usage.py 选 **固定 slot 0**（`mask[0,0]>0`，非 IoU argmax）。本阶段沿用，selected_mask_index=0；selected_predicted_iou=iou_predictions[0]。

## owl case 结果（PASS）

- 输入：third_party/nanoowl/assets/owl_glove_small.jpg（W=756 H=499）；prompts `an owl,a glove,a finger`；target-label `an owl`；threshold 0.10。
- NanoOWL：1 detection；selected label **an owl**，score **0.3419**（与 Day1/2 baseline 完全一致），box [275.1943, 82.3545, 463.6406, 399.3462]（与 baseline 一致）。
- Bridge：box 未越界；clipped_box == original_box；point_coords [[275.1943,82.3545],[463.6406,399.3462]]；labels [2,3]。
- NanoSAM：selected_mask_index **0**；predicted_ious [0.720552, 0.733915, 0.847281, 0.790455]；selected_iou **0.720552**；mask_shape [499,756]；nonzero_pixels 34065；mask_area_ratio 0.0903；无 NaN/Inf。
- status: OK。联合可视化包含 NanoOWL box + label + score + NanoSAM mask overlay。

## NO_DETECTION case 结果（execution PASS）

- 输入：third_party/nanoowl/assets/camera.jpg（W=640 H=480）；prompts `a camera,a hand,a person`；threshold 0.1（复用 Day2 regression camera_scene case）。
- NanoOWL：0 detections → selection_rule="no detections"。
- Pipeline：返回结构化 `status=NO_DETECTION`，**不调用 NanoSAM**（bridge=null, nanosam=null），程序正常退出。
- NO_DETECTION 属于 execution PASS，不是模型/程序失败。

## Timing（correctness timing，非正式 Benchmark）

owl case（timing.json，GPU async 操作后 cuda.synchronize）：
- nanoowl_ms 462.748（含 text encode + detect）
- nanosam_set_image_ms 700.932
- nanosam_decode_ms 28.003
- pipeline_total_ms 1191.683（=nanoowl + set_image + decode；不含 model_load）

NO_DETECTION case：pipeline_total_ms 446.234（仅 NanoOWL，无 NanoSAM）。

注：以上为 Stage 4A correctness timing，**不作为正式 NanoSAM E2E baseline**；不沿用 Day 3 的 52.186 ms 作为正式 baseline。

## 约束遵守

- 未修改 third_party/nanoowl、third_party/nanosam；未重建任何 Engine；未安装新依赖。
- 未做正式 Benchmark / COCO / INT8 / MP4 / ROS2 / stability。
- 核心环境版本保持不变：torch 2.5.0a0+872d972e41.nv24.08 / torchvision 0.20.0a0+afc54f7 / numpy 1.23.5 / tensorrt 10.3.0 / transformers 4.46.3 / Pillow 12.3.0 / OpenCV 4.10.0 / torch2trt importable / timm MISSING。
- 未 remote / push / GitHub 上传；未继续 Stage 4B。

## evidence 路径

- results/joint_pipeline/owl/：pipeline_result.json、input.jpg、nanoowl_visualization.jpg、nanosam_mask.npy、joint_visualization.jpg、timing.json、inference.log
- results/joint_pipeline/no_detection/：pipeline_result.json、input.jpg、nanoowl_visualization.jpg、timing.json、inference.log
- 脚本：scripts/run_edge_perception.py

## 结论

Stage 4A：PASS — NanoOWL → NanoSAM 首次自动联合 Pipeline 正确性验证通过（owl 检测→分割正确、NO_DETECTION 优雅降级）。
