# Inference

VLA policy server 与 Agilex Piper 双臂推理客户端。项目支持 WebSocket / 共享内存传输、SDK/CAN 直连控制、采集服务对接，以及无硬件 Mock 测试。

[English](README.md)

## 支持范围

- Piper 双臂：每侧 6 个关节和 1 个夹爪，共 14 维动作
- 推理模式：同步、naive async、temporal smoothing、temporal ensembling、RTC、Legato、TTRTC、VLASH
- Policy 后端：JAX checkpoint、TensorRT engine
- 传输方式：跨机器 WebSocket、同机共享内存
- 运行记录：模型 I/O、视频、动作、延迟与事件日志

## 目录

```text
client/      推理客户端、采集对接与工具
server/      Policy server 与模型代码
scripts/     启动、CAN 配置和调试脚本
docs/        接口与集成文档
test/        测试与 Mock 数据
```

## 安装

要求 Python 3.11 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --python 3.11
. .venv/bin/activate
```

运行机器人客户端时还需安装硬件依赖：

```bash
uv pip install -r client/requirements_inference.txt
sudo apt update
sudo apt install -y can-utils ethtool
```

## 快速开始

### 1. 配置并启动服务端

编辑 `server/config.yaml`：

```yaml
python_bin: .venv/bin/python
transport: websocket
port: 8000
default_prompt: fold the sleeve

policy:
  type: checkpoint
  config: pi05_flatten_fold_normal
  dir: /path/to/checkpoint
  asset_id: OpenDriveLab-org/Kai0
```

`policy.config` 必须与训练时的模型结构和 transforms 一致。`asset_id` 对应：

```text
/path/to/checkpoint/assets/<asset_id>/norm_stats.json
```

启动服务：

```bash
./scripts/run_server.sh --config server/config.yaml
```

可先检查启动命令：

```bash
./scripts/run_server.sh --dry-run
```

### 2. 配置并启动客户端

在 `client/config_agilex.yaml` 中设置服务地址、CAN 设备、相机序列号、初始姿态和推理模式。最常用的字段如下：

```yaml
python_bin: .venv/bin/python

server:
  host: 127.0.0.1
  port: 8000

inference:
  execution_mode: async
  async_mode: temporal_smoothing
  chunk_size: 50
  publish_rate: 30
  observation_rate: 30
  inference_rate: 3
  prompt: fold the sleeve
```

连接 CAN 与 RealSense 后先执行硬件自检，再启动推理：

```bash
./scripts/run_client.sh --config client/config_agilex.yaml --check-hardware
./scripts/run_client.sh --config client/config_agilex.yaml --log-level INFO
```

机器人运行存在安全风险。首次部署请降低速度、确认急停可用，并在机械臂工作空间外观察。

### 3. 无硬件测试

仓库包含小型测试数据，可启动 Mock action server 和客户端：

```bash
./scripts/run_client_mock.sh
```

运行测试：

```bash
uv run pytest test client/tests server/openpi packages
```

## 常用配置

异步模式通过 `inference.async_mode` 选择：

```text
naive
temporal_smoothing
temporal_ensembling
rtc
legato
ttrtc
vlash
```

模式参数位于 `inference.modes.async.<mode>`。Legato 和 TTRTC 还需要分别在服务端开启 `use_legato_inference` 或 `use_ttrtc_inference`。RTC、Legato、TTRTC 使用匹配的模型配置和 checkpoint。

同机部署时可将两端切换为共享内存：

```yaml
transport: shared_memory
shared_memory_socket_path: /tmp/openpi_policy.sock
```

启动脚本默认通过 `taskset` 绑定 CPU 核。部署到不同机器前，请根据硬件修改或移除脚本中的 `TASKSET_PREFIX`。

## 记录与日志

客户端默认输出到 `client/inference_records/`，可在 `recording` 配置段控制模型 I/O、视频、动作 CSV 和运行事件。服务端日志路径由 `server/config.yaml` 中的 `log_file` 与 `event_log_file` 设置。

## 采集服务对接

采集架构使用 `client/run_inference_service.py`：

```bash
cd client
python run_inference_service.py --config config_agilex.yaml
python run_inference_service.py --list-modes
```

详见：

- [采集对接说明](client/integration/README.md)
- [Inference Service TCP API](docs/INFERENCE_SERVICE_TCP_API.md)
- [Piper XH 时间轴集成](docs/PIPER_XH_TIMEAXIS_INTEGRATION.md)

## TensorRT

将 `policy.type` 改为 `tensorrt`，并配置 `engine`、`assets_dir`、`asset_id`、`device` 和 `precision`。已有 ONNX 模型可通过以下命令构建 engine：

```bash
.venv/bin/python scripts/build_trt_engine.py \
  --onnx /path/to/model.onnx \
  --engine /path/to/model_fp16.engine
```

运行环境需自行安装与 CUDA 匹配的 TensorRT。
