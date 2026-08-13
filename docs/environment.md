# 阶段 0：Jetson 环境基线

## 阶段目标与验收标准

阶段 0 的目标是确认目标板能够支持后续 `PyTorch -> ONNX -> ONNX Runtime -> TensorRT` 开发，并把环境状态固化为可重复生成的证据。

验收标准：

- [x] 确认硬件、OS、L4T、CUDA、cuDNN、TensorRT 和 Python 版本。
- [x] 确认 PyTorch 能实际执行 CUDA 计算，而不只是成功导入。
- [x] 确认 TensorRT Builder 能生成最小 engine。
- [x] 确认 ONNX Runtime 暴露 CUDA/TensorRT Execution Provider。
- [x] 确认 `tegrastats`、Nsight Systems、Nsight Compute 状态。
- [x] 确认功耗模式、内存、存储和基础编译工具链。
- [x] 检查 Docker 和摄像头状态，并区分主线阻塞项与可选项。
- [x] 在项目虚拟环境安装 Python `onnx`，并完成最小 ONNX -> TensorRT 端到端 smoke test。
- [x] 通过项目环境脚本配置 CUDA 工具链的 `PATH`，保证可直接调用 `nvcc`/`ncu`。

## 实机基线

采集时间：2026-08-05，设备时区 Asia/Shanghai。

| 项目 | 实测状态 | 结论 |
|---|---|---|
| 板卡 | NVIDIA Jetson Orin Nano Developer Kit Super | 目标硬件正确 |
| CPU | 6 核 ARM64 | 正常 |
| 内存 | 7.62 GiB 可见内存，约 3.81 GiB zram swap | 符合 8GB SKU；后续不能使用 `nvidia-smi` 口径描述独立显存 |
| 系统盘 | 256GB NVMe，根分区约 233GB，剩余约 185GB | 足够存放模型、COCO val 与实验结果；完整 train2017 需控制空间 |
| OS | Ubuntu 22.04.5 LTS | 正常 |
| Kernel | 5.15.148-tegra | Jetson 内核正常 |
| L4T | R36.4.3，package `36.4.3-20250107174145` | JetPack 6.x 系列底座 |
| JetPack meta-package | 未安装；APT candidate 为 6.2.1 | 不等于 SDK 不存在；关键组件已分包安装，暂不整体升级 |
| CUDA Toolkit | 12.6.68，位于 `/usr/local/cuda-12.6` | 已完整安装，包含编译器和开发库 |
| CUDA PATH | 用户全局 `PATH` 未修改；`source scripts/env.sh` 后可直接调用 `nvcc`/`ncu` | 项目级配置已关闭工具链阻塞项 |
| cuDNN | 9.3.0.75 | 已安装 runtime/dev/sample |
| TensorRT | 10.3.0.30，Python 10.3.0 | 已安装 runtime/dev/plugin/sample |
| PyTorch | 2.5.0a0 NVIDIA build，CUDA 12.6，cuDNN 9.3 | GPU 可用 |
| GPU | Orin，compute capability 8.7 | CUDA smoke test 通过 |
| ONNX Runtime | 1.20.0 | 可见 TensorRT/CUDA/CPU EP；阶段 2 仍需验证实际模型执行无 fallback |
| Python ONNX | 1.17.0，项目 `.venv` | checker、shape inference 和 ONNX 保存已验证 |
| OpenCV / NumPy | 4.10.0 / 1.23.5 | 基础前处理依赖存在 |
| 功耗模式 | 25W | 当前正常；正式 benchmark 前固定并记录 |
| 空闲温度/功耗 | GPU/SoC 约 50～52C；VDD_IN 约 4.4W | 仅作为盘点快照，不作为性能结论 |
| tegrastats | `/usr/bin/tegrastats` | 可采集 RAM/CPU/GPU/温度/rail 功耗 |
| Nsight Systems | 2024.5.4 | 可用 |
| Nsight Compute | CUDA 12.6 CLI 已安装 | 可用，命令目前需绝对路径或修正 PATH |
| C/C++ 工具链 | GCC/G++ 11.4，CMake 3.22.1 | 满足后续 C++ Runtime 基线 |
| Docker | CLI 未安装，service inactive | 阶段 1～6 不阻塞；阶段 8 前安装并验证 NVIDIA runtime |
| GStreamer Argus | `nvarguscamerasrc` 存在，daemon active | Camera 软件链存在 |
| 摄像头传感器 | 无 `/dev/video*`，media topology 无 sensor link | IMX219 尚未接入/识别；不阻塞离线主线 |

## 已验证的运行能力

### PyTorch CUDA

在 GPU 上执行了 1024x1024 矩阵乘法 smoke test：

- `torch.cuda.is_available() == True`
- 设备名：`Orin`
- Compute capability：`8.7`
- CUDA 分配成功，矩阵结果可读回

该测试只证明 CUDA 执行链可用，不记录为正式性能数据，因为没有固定时钟、热稳态、独立重复和标准 benchmark 配置。

### TensorRT Builder

使用 TensorRT Python API 创建了一个显式 batch 的 identity network，并成功执行 `build_serialized_network`。测试脚本在读取 `IHostMemory` 长度时使用了不适用于 TensorRT 10.3 Python binding 的 `len()`，因此最后抛出 `TypeError`；错误发生在 engine 已构建之后，不是 Builder/GPU 失败。正式 smoke test 将改用 `serialized.nbytes` 或直接序列化字节流，并加入一次真实 enqueue。

### ONNX -> TensorRT 端到端 smoke test

项目使用 `python3 -m venv --system-site-packages .venv` 复用 Jetson 自带的 TensorRT、PyTorch、ONNX Runtime、OpenCV 和 NumPy，只在虚拟环境增加 `onnx==1.17.0` 与 `protobuf==4.25.8`。ONNX 初次导入时曾因继承的系统 protobuf 3.12.4 缺少所需 API 而失败，固定虚拟环境 protobuf 后恢复，未覆盖 NVIDIA 提供的系统 SDK 包。

`scripts/smoke_onnx_trt.sh` 已完成以下链路：

1. 生成 opset 17、IR version 9 的固定 shape Identity ONNX；
2. 执行 ONNX checker 与 shape inference；
3. 使用 TensorRT 10.3 `trtexec` 解析 ONNX、构建并序列化 engine；
4. 重新加载 engine，在 GPU 上预热并执行推理；
5. 导出逐次耗时、profile、layer info、构建/运行日志和资产 SHA256。

构建和运行日志均包含 `&&&& PASSED TensorRT.trtexec`。本次 engine 构建墙钟时间为 3749 ms；该数值只用于确认构建时间记录链路。Identity 图会被 TensorRT 大幅消除，所有性能数字均不作为项目 benchmark 或简历结果。

保留产物：

- `results/smoke/identity.onnx`
- `results/smoke/identity.engine`
- `results/smoke/build.log`、`run.log`
- `results/smoke/build_time_ms.txt`
- `results/smoke/times.json`、`profile.json`、`layers.json`
- `results/smoke/sha256.txt`

### 性能与系统监控

`tegrastats` 已连续输出 CPU 频率、GR3D、RAM、温度和 `VDD_IN`。Nsight Systems 与 Nsight Compute CLI 均存在。`jetson_clocks --show` 需要 root，正式 benchmark 是否锁频必须作为实验条件显式记录，不默认启用。

## 阻塞项分级

## 阶段 0 关闭结论

阶段 1 前的三个主线阻塞项均已关闭：

1. 项目 `.venv` 已安装并验证 ONNX 1.17.0；
2. `scripts/env.sh` 已提供项目级 CUDA `PATH` 和 library path；
3. 最小 ONNX -> TensorRT engine 构建、反序列化和 GPU 推理链路已通过。

重新运行 `scripts/inventory.sh` 后，`results/env.json` 中 `readiness.required_missing` 与 `readiness.configuration_issues` 均为空。`docker` 和 `camera-sensor` 仍列在 `optional_missing`，分别在阶段 8 和阶段 7 处理，不阻塞阶段 1～6。

### 阶段 8 前必须关闭

1. 安装 Docker Engine 与 NVIDIA Container Runtime。
2. 验证容器内 GPU、TensorRT 版本和宿主 L4T ABI 一致。

### 非主线阻塞项

1. IMX219 传感器当前未识别。CSI/Argus 接入安排在阶段 7，主线先使用 COCO、静态图片和录制视频。
2. `nvidia-jetpack` meta-package 未安装。关键 CUDA/cuDNN/TensorRT 组件均存在，不能为了显示元包而盲目安装或升级；尤其 APT candidate 已高于当前 L4T 版本，升级前必须检查兼容性和回滚方案。

## 复现命令

在 Jetson 执行：

```bash
cd /home/nvidia/jetson-deploy-lab
bash scripts/inventory.sh
```

输出：

- `results/inventory.txt`：便于人工审阅的原始命令结果。
- `results/env.json`：供后续 benchmark 自动附加的结构化环境信息。

## 下一步阶段门

阶段 0 完成前只进行最小且可解释的环境修复：创建项目 Python 虚拟环境、安装 `onnx`、修正用户级 CUDA PATH、执行 ONNX/TensorRT smoke test。暂不安装 YOLO、COCO、Docker，也不调整系统功耗和时钟，以避免在基础链路未验证前引入额外变量。
