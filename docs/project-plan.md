# Jetson Orin Nano 端侧 AI Infra 项目第一版方案

## Context

目标不是“在板子上跑通一个模型”，而是在 1～2 个月内形成可复现、可量化、可审查、可写入简历的完整工程证据链。推荐先做视觉部署优化主项目，打通模型、图、Runtime、量化、C++、Profiling、系统工程，再做小型 VLM 扩展；不同时铺开。

已实机确认：Jetson Orin Nano Super 8GB、Ubuntu 22.04.5、L4T R36.4.3、TensorRT 10.3.0（CUDA 12.5 构建）、Python 3.10、25W、约 233GB NVMe。当前未发现 `nvcc`、Docker、`/dev/video*`，JetPack meta-package 未安装；cuDNN/CUDA runtime 与 Python 包仍需精确盘点。它不影响整体选型，但属于阶段 0 阻塞项。

---

# 1. 岗位能力拆解与项目证据

| 能力 | 岗位真正考察内容 | 项目必须留下的证据 |
|---|---|---|
| 模型层 | 结构、输入输出、前后处理、精度/速度权衡、PTQ | 模型结构说明；PyTorch 基线；COCO mAP；FP32/FP16/INT8 逐样本及分层误差报告 |
| 图优化与编译 | ONNX 导出、shape inference、算子兼容、融合、构建 profile/plugin | ONNX 图、Netron 截图、ONNX checker/ORT 对齐、TRT layer/profile 导出、不支持算子诊断与修复记录 |
| 推理 Runtime | context/binding、动态 shape、生命周期、并发、错误处理 | Python 和 C++ 同功能 Runtime；固定/动态 shape；冷/热启动；同步/异步；可重复 benchmark |
| CUDA/GPU 性能 | stream、event、H2D/D2H、pinned buffer、内存复用、同步瓶颈 | Nsight Systems 时间线；阶段耗时；同步→异步、pageable→pinned、分配→复用的消融实验 |
| LLM/VLM | 视觉编码器/projector/LLM、量化、TTFT、prefill/decode、KV cache | 扩展项目的分阶段耗时、tokens/s、TTFT、峰值内存、上下文/分辨率/量化实验；不声称机器人闭环 |
| 嵌入式 Linux | 功耗/温度/内存约束、服务、日志、故障恢复、容器 | nvpmodel/tegrastats 数据；systemd；结构化日志；异常重启；1～8h soak test；Docker 可复现环境 |
| 机器人集成 | ROS 2 消息、QoS、时间戳、离线 bag/视频、相机链路 | `image→detections` 节点、参数/QoS、离线 rosbag2 回放；摄像头接入只在设备就绪后追加 |
| 工程交付 | 配置、测试、CI、文档、制品、复现实验 | CMake、配置 schema、单元/集成/回归测试、版本清单、自动化脚本、结果 CSV/JSON、技术报告与演示 |

你的强项是 Linux/嵌入式/车端/稳定性与工程交付；项目应重点补齐图编译、TensorRT、CUDA Runtime 和定量精度证据，而不是重复做纯训练。

---

# 2. 项目选择结论

## 主项目（必须完成）

**名称：JetsonDeployLab：面向 Jetson 的目标检测多后端推理与可复现优化平台**

1. **解决的问题**：同一检测模型从 PyTorch 到 ONNX Runtime、TensorRT FP32/FP16/INT8，在 Jetson 上如何兼顾精度、延迟、吞吐、内存、功耗、温度和稳定性，并以 Python/C++ 两套 Runtime 可复现交付。
2. **技术亮点**：COCO 精度闭环；校准集可追溯；固定/动态 shape；TRT 10 Runtime；CUDA stream/event、pinned memory、buffer reuse；前/推理/后处理分解；可选 GPU NMS/plugin；Nsight+tegrastats；systemd/Docker/ROS 2。
3. **岗位覆盖**：直接覆盖模型转换、ORT/TRT、FP16/INT8、C++、CUDA、Profiling、板级资源、服务化和机器人接入，覆盖目标岗位的大多数核心项。
4. **适合背景**：你已有检测/CV、Python/C++、Linux、日志与稳定性经验，可把精力用于部署深度而非从零学习任务本身。
5. **可实现性**：YOLOv8s/同量级模型在 8GB Orin Nano 上现实；COCO val2017 规模可控；不依赖摄像头。
6. **最大风险**：项目退化成“YOLO + trtexec FPS”；或者 INT8 只测速不评精度；不同后端前后处理不一致造成无效对比。
7. **降级方案**：先完成固定 `1×3×640×640`、batch=1、CPU NMS 的端到端闭环；动态 shape、GPU NMS/plugin、双流流水线作为加分项。Plugin 必须由 profiling 证明值得做，不能为包装强加。
8. **简历最终内容**：用真实数字填写：“设计并实现 Jetson Orin Nano 多后端检测部署平台，打通 PyTorch→ONNX→ORT/TensorRT FP32/FP16/INT8；实现 C++ 异步 Runtime 与可复现 benchmark，在 COCO val2017 上量化 mAP 变化，并记录 P50/P95/P99、吞吐、内存、功耗、温度和长稳结果；通过 Nsight 定位[真实瓶颈]并使[真实指标]由 X 改善至 Y。”

## 扩展项目（主项目验收后再做）

**名称：EdgeVLM Probe：Jetson 小型 VLM 分阶段推理分析与结构化机器人任务接口**

1. **解决的问题**：在 8GB 统一内存约束下，量化小型 VLM 的视觉编码、prefill、decode、KV cache 与分辨率/上下文权衡，并输出可验证 JSON 任务建议。
2. **技术亮点**：视觉编码器/projector/LLM 分解；TTFT、tokens/s、峰值内存；上下文、图片分辨率和量化消融；HTTP/gRPC 或 ROS 2 service；JSON schema 输出。
3. **岗位覆盖**：补齐 VLM、低比特、KV cache、prefill/decode、多模态服务和具身接口叙事。
4. **适合背景**：可复用主项目的监控、benchmark、服务和 ROS 2 基础设施。
5. **可实现性**：首选 **SmolVLM2-500M** 级别；运行时生态需阶段门验证。8GB 不把 3B+ FP16 当主线。
6. **最大风险**：aarch64 量化 Runtime/算子兼容；模型虽启动但速度和内存不可用；评价变成主观截图。
7. **降级方案**：退到 Florence-2-base 或同级 0.2～0.5B 模型；以离线图片、固定 prompt、结构化任务集做评估。Qwen2.5-VL-3B INT4 仅是有余力挑战，不是交付承诺。
8. **简历最终内容**：如实写模型、量化格式和实测：“在 8GB Jetson 上部署 [模型]，分解视觉编码/TTFT/decode，评估分辨率、上下文和量化对时延、tokens/s、峰值内存及任务质量的影响，并提供 ROS 2/服务化结构化动作建议输出。”不得写“完成 VLA 闭环”。

**结论**：必须先 A 后 B。视觉主项目是完整 Infra 证据；VLM 是对目标岗位的定向扩展。

---

# 3. 主项目完整技术架构

```text
公开图片/COCO/录制视频/未来 CSI Camera/rosbag2
        │
        ▼
[CPU 输入层] 解码、时间戳、采样、配置加载
        │
        ▼
[预处理]
  基线：CPU resize/letterbox/normalize/HWC→CHW
  优化：pinned host buffer + 可选 CUDA/NPP 预处理
        │ H2D（CUDA stream）
        ▼
[模型资产流水线]
  PyTorch checkpoint
     → ONNX export/opset/shape inference/checker
     → ORT CUDA baseline
     → TensorRT builder
        ├ FP32 engine
        ├ FP16 engine
        └ INT8 engine + calibration cache + timing cache
        │
        ▼
[Runtime]
  Python：正确性参考、快速实验
  C++：TRT 10 API、RAII buffers、execution context、CUDA event/stream、异步 enqueueV3
        │ D2H / 可选 GPU 后处理
        ▼
[后处理] decode、置信度过滤、NMS、坐标还原、JSON/可视化
        │
        ├── CLI / 离线视频 / REST(可选)
        └── ROS 2：Image/CompressedImage → Detection2DArray 或自定义消息

横切能力：
  Benchmark runner → latency JSONL/CSV、accuracy、engine build/cold start
  tegrastats collector → CPU/GPU/RAM/功耗/温度
  Nsight Systems/TRT layer profiler → 时间线与层级报告
  structured logger → run_id/model/hash/config/version/error
  tests → unit/integration/numerical/regression/soak
  Docker/CMake/scripts → 构建与复现
  systemd → 启动、依赖、重启、资源限制、日志
```

- **CPU**：I/O、解码、基线前后处理、调度、日志与 ROS 2。
- **GPU**：网络计算；有证据后承担预处理/NMS。
- **CUDA**：buffer、pinned memory、H2D/D2H、stream/event、异步与可选 graph。
- **TensorRT**：图解析、tactic/precision/profile、engine 构建与执行。
- **systemd**：进程生命周期、开机启动、失败重启、环境/参数和日志，不参与模型计算。

---

# 4. 主项目阶段划分（约 6～8 周）

| 阶段 | 目标/功能 | 需掌握 | 验收标准 | 输出文件/证据 | 常见问题 | 地位 |
|---|---|---|---|---|---|---|
| 0 环境盘点 | 固化 OS/SDK/功耗/依赖；最小 CUDA/TRT smoke test | JetPack/L4T 包关系、统一内存 | 版本清单；`trtexec` 可构建并运行最小 ONNX；监控可采集 | `scripts/inventory.sh`, `results/env.json`, `docs/environment.md` | meta-package 缺失、nvcc/Python binding 缺失 | 核心基础 |
| 1 PyTorch 基线 | 数据、前后处理、单图/批量、COCO mAP 与阶段计时 | 检测输出、mAP、warmup | 官方权重精度在合理容差；固定样本输出可回归 | `src/python/pytorch_runner.py`, golden outputs, baseline JSON | 数据类别映射、letterbox 坐标错 | 核心 |
| 2 ONNX/ORT | 导出、checker、shape、CPU/CUDA EP、数值对齐 | opset、dynamic axes、图检查 | ONNX checker 通过；至少 100 样本输出/最终 mAP 与 PT 对齐；ORT CUDA 确认非 CPU fallback | ONNX、导出脚本、对齐报告、图截图 | NMS/export ops、fallback | 核心 |
| 3 TRT FP32/FP16 | builder、engine、timing cache、固定/动态 profile | TRT 10 API、precision/tactic/profile | 两种 engine 成功；精度对齐；构建时间/大小/层信息；标准 benchmark | engines（不必提交大文件）、build manifest、profile JSON | engine 不可跨版本/设备复用 | 核心 |
| 4 INT8 | 校准集、cache、代表性验证、误差定位 | PTQ、calibration、per-layer precision | 可重建 INT8 engine；COCO mAP、分层输出误差及校准规模消融；不只报告 FPS | `calibration/manifest.json`, cache hash, accuracy report | TRT 10 quantization路径差异、敏感层 | 核心亮点 |
| 5 C++ Runtime | 与 Python 同 I/O；RAII、异步、动态 shape、错误处理 | CMake/TRT/CUDA、enqueueV3 | golden sample 一致；无每帧 malloc；asan/valgrind 可行部分通过；P95 可重复 | `src/cpp/`, tests, CLI | binding/shape、生命周期、隐式同步 | 核心亮点 |
| 6 Profiling/优化 | 分解 decode/pre/H2D/infer/D2H/post；内存复用、pinned、async 消融 | Nsight、CUDA event、统计学 | 至少找到 1 个真实瓶颈；每个优化有 before/after 和同条件证据；不可只用平均值 | `.nsys-rep` 摘要、消融 CSV、火焰/时间线图 | 测量扰动、时钟/温度漂移 | 核心亮点 |
| 7 ROS 2/摄像头 | 离线 ROS 2 节点；相机就绪后接 CSI | QoS、时间戳、GStreamer/Argus | rosbag2/视频回放稳定；消息字段/时间戳正确；相机不是前置阻塞 | `ros2_ws/`, launch/params, bag test | 当前无 `/dev/video*`；CSI 常走 Argus | ROS 核心，相机加分 |
| 8 稳定性/工程化 | Docker、systemd、日志、资源故障、1～8h soak | 守护、信号、日志轮转、容器 GPU | 可一键部署；异常可诊断；长稳无持续内存增长/崩溃，保留完整资源曲线 | Dockerfile、service、soak report | 容器与 JetPack ABI、磁盘日志增长 | 核心亮点 |
| 9 展示交付 | README、架构图、结果表、报告、视频、简历 | 技术表达与复现 | 新环境按文档复现核心实验；所有数字可追溯到原始结果和 commit | README、report、video script、resume bullets | cherry-pick 最好数字、缺原始数据 | 核心交付 |

**Bonus 仅包括**：自定义 TRT Plugin/GPU NMS、CUDA Graph、双 stream pipeline、CSI Camera、QAT、ROS 2 zero-copy。必须先有 profiling 或业务理由。

---

# 5. 模型和数据集选择

推荐固定任务为 **COCO 2017 detection**：`val2017` 5k 做最终精度，校准图像从 `train2017` 按固定 seed 抽样（如 100/500/1000 做消融），严禁从 val/test 校准。另建 20～50 张 golden 小集用于每日回归。

| 候选 | 规模 | Jetson | ONNX | TRT | INT8 | 精度评估 | 后续 VLM/VLA | 含金量 | 风险 |
|---|---:|---|---|---|---|---|---|---|---|
| **YOLOv8s（推荐主线）** | 约 11M，约 29 GFLOPs@640 | 很适合 8GB | 成熟；建议导出 pre-NMS 图 | Conv 主干支持好；NMS 可 CPU 或 TRT plugin | 中低；易形成完整 PTQ 闭环 | COCO 工具成熟 | 输出结构化目标可接 VLM/ROS | 模型常见，但完整 Runtime/量化/系统实验含金量高 | 容易被看成简单复现；需以平台深度避免 |
| RT-DETR-R18 | 约 20M 量级，Transformer detector | 可运行但更重 | 中等；shape/attention 要验证 | TRT 10 基本可行，无 NMS是一优点 | 中高；attention/归一化较敏感 | COCO 方便 | Transformer 叙事更强 | 图分析和量化误差更有深度 | 导出版本差异、INT8 掉点与延迟风险；适合作为替换模型/加分实验 |
| YOLOX-S | 约 9M，约 27 GFLOPs@640 | 很适合 | 成熟，decode/NMS 可拆 | 算子友好 | 低到中 | COCO 方便 | 一般 | 能很好展示手写 decode/NMS 与 Runtime | 新颖度较低，生态维护度需评估 |
| MobileNetV3-Large + DeepLabV3（备选分割） | 轻量 | 很适合 | 较成熟 | 算子大体友好；resize需检查 | 中等，像素精度敏感 | Cityscapes/自有全景数据方便 | 与机器人场景理解相关 | 契合你的分割背景 | 数据预处理/高分辨率开销，项目主线会与检测矩阵分叉 |

**最终选择**：主线只用 YOLOv8s。RT-DETR-R18 是主项目完成后的“第二模型泛化验证”，不是并行主线。若许可证/版本或导出链不稳，降级 YOLOX-S。

---

# 6. GitHub 仓库设计

```text
jetson-deploy-lab/
├── configs/                 # 模型、数据、精度、benchmark、功耗模式 YAML
├── models/                  # 只放下载清单/hash/metadata；大权重和 engine 不入 Git
├── scripts/                 # inventory、download、export、build、deploy、bench、soak
├── src/
│   ├── python/              # PyTorch/ORT/TRT runners、校准、评估、结果 schema
│   └── cpp/                 # TRT runtime、buffers、pre/post、CLI、metrics、logging
├── include/                 # C++ 公共头文件
├── plugins/                 # 仅 profiling 证明必要后加入 plugin/CUDA kernel
├── calibration/             # 抽样脚本、manifest、cache metadata，不提交大数据
├── ros2_ws/src/             # inference node、msg/launch/params、bag regression
├── benchmarks/              # runner、metric definitions、experiment manifests
├── tests/                   # unit、golden numerical、integration、regression、soak assertions
├── docker/                  # Jetson 基础镜像、开发/运行镜像、compose
├── systemd/                 # service、env 示例、日志轮转
├── docs/                    # 架构、转换、量化、profiling、故障、复现、报告
├── assets/                  # 小型公开样例、架构图、演示素材
├── results/                 # 原始 JSONL/CSV、聚合表、图；按 device/commit/run_id 组织
├── third_party/             # 许可证清晰的依赖说明，优先 submodule/下载脚本
├── CMakeLists.txt
├── pyproject.toml
├── Makefile                 # 统一入口，不隐藏实际命令
└── README.md
```

每次结果记录：git commit、模型 hash、engine hash、SDK、power mode、clock policy、输入、batch、warmup、重复次数、环境温度/起始温度。

---

# 7. 实验矩阵与测量规范

## 核心矩阵

| 组 | 自变量 | 固定控制 | 主要指标 |
|---|---|---|---|
| 后端/精度 | PyTorch GPU、ORT CUDA、TRT FP32/FP16/INT8 | 模型权重、COCO、pre/post、640、B1、25W | mAP50-95、模型/engine大小、build time、纯推理与 E2E P50/P95/P99、FPS、RAM/功耗/温度 |
| Runtime 语言 | TRT Python vs C++ | 同一 engine、输入序列、pre/post 语义、warmup | 冷启动、CPU占用、E2E尾延迟、吞吐、RSS |
| Shape | fixed 640；dynamic profile 320/640/960；实际 320/640/960 | TRT precision、B1、数据顺序 | build/engine大小、内存、每尺寸延迟与对应 mAP |
| Batch | 1/2/4（内存允许才到8） | 640、25W、precision | 单样本延迟、batch latency、throughput、峰值内存、功耗 |
| 功耗 | 7W/15W/25W（以板上实际 mode 为准） | engine、B1、640、风扇策略 | 延迟分位、FPS/W、温度、是否降频 |
| Runtime 优化消融 | pageable/pinned；每帧分配/复用；sync/async；CPU/GPU pre/post | 同 engine、输入、clock/power | pre/H2D/infer/D2H/post 分段，E2E尾延迟、CPU、功耗 |
| INT8 校准 | 100/500/1000；随机 vs 类别覆盖 | 同 seed 规则、同 val2017 | mAP、per-class AP、输出余弦/L∞统计、build time/cache size |
| 稳定性 | 1h smoke、8h final | 固定工作负载和服务配置 | crash/error、内存斜率、P99漂移、温度、降频、掉帧 |

## 基准有效性规则

- 推理计时用 CUDA event；端到端用 monotonic wall clock，二者不可混用。
- 明确是否包含 decode/pre/H2D/infer/D2H/post/NMS；同时报告 pure inference 与 E2E。
- 每组 warmup 后至少数百/上千次，至少 3 次独立运行，保存原始样本而非只存均值。
- 固定输入序列、前后处理和阈值；确认 ORT 使用 CUDA EP，不能静默 CPU fallback。
- 稳态性能和冷启动分开；engine build 和 deserialize 分开。
- 统一内存平台不要伪称 `nvidia-smi` 显存；报告进程 RSS、tegrastats RAM/SWAP、TRT device memory/context buffer 等定义明确的数值。
- 功耗取 tegrastats rail 采样区间统计；预热到热稳态，记录风扇、功耗模式和是否锁频。锁频与非锁频结果不能混表。
- INT8 精度必须跑完整 val2017；校准集不与验证集重叠。
- RTX 5090 只作为功能/相对对照；因架构与功耗不同，不用简单“快几倍”作核心结论。TRT engine 必须在目标 Jetson 构建。

---

# 8. 登录 Jetson 后前 7 天执行计划

## Day 1：实机环境清单与风险关闭

**Jetson（只读检查）**

```bash
cat /proc/device-tree/model; echo
cat /etc/nv_tegra_release
uname -a; lsb_release -a
apt-cache policy nvidia-jetpack nvidia-l4t-core
nvcc --version || true
find /usr/local -maxdepth 2 -name version.json -o -name version.txt
/usr/src/tensorrt/bin/trtexec --help | grep 'TensorRT v' | head -1
ldconfig -p | grep -E 'cudnn|nvinfer|cudart'
dpkg-query -W | grep -E 'cuda|cudnn|nvinfer|tensorrt|onnx|docker|container'
python3 --version
python3 -c 'import tensorrt as trt; print(trt.__version__)'
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)' || true
nvpmodel -q --verbose
free -h; df -h; lsblk
which tegrastats; timeout 10s tegrastats --interval 1000
systemctl status docker --no-pager || true
docker --version || true
ls -l /dev/video* /dev/nvhost-* 2>/dev/null || true
gst-inspect-1.0 nvarguscamerasrc || true
```

需 sudo 的写操作（安装/锁频）先形成清单，不在盘点时盲装。**验收**：生成 `results/env.json`，标出 required/missing/optional。已知 Docker、`nvcc`、摄像头节点缺失，需要判断是工具链未装还是 PATH/容器方案。

## Day 2：仓库骨架、实验协议和数据

- **本地电脑**：创建 GitHub 仓库、目录、issue/milestone、LICENSE、结果 JSON schema。
- **服务器**：下载 COCO val2017/annotations；按 seed 建 train2017 校准 manifest；准备 20～50 张 golden set。
- **Jetson**：测试网络、磁盘、数据同步和 checksum；确定原生环境还是 NVIDIA L4T 容器。优先与板载 L4T ABI 匹配，不随意 pip 覆盖系统 TRT。
- **验收**：一条命令校验数据 hash；实验配置能输出 run manifest。

## Day 3：PyTorch 正确性基线

- **5090 服务器**：固定 YOLOv8s 具体版本/commit/权重 hash；实现官方参考推理、COCO mAP、golden outputs；记录 pre/post 精确定义。
- **Jetson**：若 PyTorch CUDA 环境已就绪则跑 20 张 smoke；未就绪不阻塞 ONNX 资产生成。
- **验收**：val2017 mAP 与该固定版本官方结果在预设容差内；单图、目录、视频三种输入至少先完成单图/目录。

## Day 4：ONNX 导出与数值对齐

- **5090 服务器**：导出固定 shape pre-NMS ONNX；执行 checker、shape inference、simplify（前后都校验）；输出节点/算子统计和 Netron 图。
- **服务器/Jetson**：PyTorch vs ORT CPU/CUDA 比较至少 100 张输出；最终以 mAP 而非只有余弦值验收。
- **验收**：无静默 fallback；golden 输出和 COCO 子集/全量误差在阈值内；保留失败日志和修复原因。

## Day 5：Jetson TensorRT FP32/FP16 基线

- **Jetson**：用 `trtexec` 分别构建 FP32/FP16 固定 shape engine，导出 times/profile/layer info；记录 build time、engine size、deserialize、pure inference。engine 只在 Jetson 构建。
- **验收**：两种 engine 可加载；同一 golden set 结果对齐；标准化命令可重跑。不要把 `trtexec` 吞吐当 E2E。

## Day 6：Python TensorRT 完整流水线

- **Jetson**：实现/验证 TRT Python runner：图片 decode→letterbox→H2D→enqueue→D2H→decode/NMS→可视化/JSON；加入 warmup、阶段计时、错误处理、run manifest。
- **验收**：与 PyTorch golden detections 一致；输出 pure inference 与 E2E P50/P95/P99；无每帧 engine/context 重建。

## Day 7：第一周基线报告与阶段门

- **Jetson**：跑 FP32/FP16 标准实验；同步 tegrastats；短时 30～60min soak；记录冷启动与热稳态。
- **本地电脑**：生成第一版结果表、架构图、问题清单；审查数据是否同条件。
- **验收**：`PT→ONNX→ORT→TRT FP32/FP16→Python E2E` 证据闭环；明确第 2 周 INT8 所用量化路径、校准规模和 C++ API 设计。

---

# 实施约束与后续协作

1. 每次只推进一个阶段，先列目标、验收、命令执行位置，再写可运行代码。
2. 每次修改列出文件和原因；报错先留日志、最小复现、定位层级，再换方案。
3. 性能数据全部由自动化脚本生成，原始数据、聚合逻辑和环境 manifest 同步保存。
4. 不把模型训练放在主线；5090 负责导出/校准准备/必要微调，Jetson 负责 engine、Runtime、profiling、资源和稳定性亮点。
5. 不承诺 Plugin、QAT、CSI、zero-copy 或 3B VLM；只有主线完成且 profiling/硬件条件支持才追加。
6. 最终交付：完整仓库、架构图、可复现结果表、技术报告、演示脚本、3～5 条基于实测的简历描述、面试自述和问题库。

# 端到端验证原则

最终由全新 checkout 按 README 在 Jetson 上完成：环境检查 → 资产下载/hash → ONNX 检查 → Jetson engine 构建 → golden regression → COCO accuracy → benchmark+tegrastats → C++ E2E → systemd 启动 → 离线 ROS 2 回放 → soak test。任何无法一键完成的步骤必须在报告中说明人工依赖和原因。