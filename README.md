# Inference

VLA policy server and Agilex Piper dual-arm inference client. The project supports WebSocket and shared-memory transports, direct SDK/CAN control, data-collection integration, and hardware-free mock testing.

[中文](README_zh-CN.md)

## Features

- Dual Piper arms with 14-dimensional actions
- Sync, naive async, temporal smoothing, temporal ensembling, RTC, Legato, TTRTC, and VLASH modes
- JAX checkpoint and TensorRT policy backends
- WebSocket for remote deployment and shared memory for same-host deployment
- Model I/O, video, action, latency, and event recording

## Installation

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --python 3.11
. .venv/bin/activate
```

Robot clients also require:

```bash
uv pip install -r client/requirements_inference.txt
sudo apt update
sudo apt install -y can-utils ethtool
```

## Quick start

### Server

Edit `server/config.yaml`:

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

`policy.config` must match the model architecture and transforms used for training. Norm stats are loaded from:

```text
/path/to/checkpoint/assets/<asset_id>/norm_stats.json
```

Start the server:

```bash
./scripts/run_server.sh --config server/config.yaml
```

Use `./scripts/run_server.sh --dry-run` to inspect the launch command first.

### Client

Set the server address, CAN devices, camera serial numbers, initial pose, and inference mode in `client/config_agilex.yaml`:

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

After connecting CAN and RealSense devices, run a hardware check and start inference:

```bash
./scripts/run_client.sh --config client/config_agilex.yaml --check-hardware
./scripts/run_client.sh --config client/config_agilex.yaml --log-level INFO
```

Robot operation is hazardous. For the first deployment, reduce speed, verify the emergency stop, and stay outside the robot workspace.

### Mock test

Run the bundled hardware-free smoke test:

```bash
./scripts/run_client_mock.sh
```

Run the test suite with:

```bash
uv run pytest test client/tests server/openpi packages
```

## Configuration notes

Select an asynchronous mode with `inference.async_mode`: `naive`, `temporal_smoothing`, `temporal_ensembling`, `rtc`, `legato`, `ttrtc`, or `vlash`. Mode-specific values live under `inference.modes.async.<mode>`.

Legato and TTRTC must also be enabled on the server with `use_legato_inference` and `use_ttrtc_inference`. RTC, Legato, and TTRTC require matching model configurations and checkpoints.

For same-host deployment, configure both sides with:

```yaml
transport: shared_memory
shared_memory_socket_path: /tmp/openpi_policy.sock
```

The launch scripts pin CPU cores with `taskset`. Adjust or remove `TASKSET_PREFIX` for the target machine.

## Logs and integration

Client recordings are written to `client/inference_records/` by default. Use the `recording` section to control model I/O, video, action CSV, and runtime event output. Server log paths are configured with `log_file` and `event_log_file`.

For data-collection integration, use `client/run_inference_service.py`. See:

- [Collection integration](client/integration/README.md)
- [Inference Service TCP API](docs/INFERENCE_SERVICE_TCP_API.md)
- [Piper XH timeline integration](docs/PIPER_XH_TIMEAXIS_INTEGRATION.md)

## TensorRT

Set `policy.type` to `tensorrt` and configure `engine`, `assets_dir`, `asset_id`, `device`, and `precision`. Build an engine from an existing ONNX model with:

```bash
.venv/bin/python scripts/build_trt_engine.py \
  --onnx /path/to/model.onnx \
  --engine /path/to/model_fp16.engine
```

Install a TensorRT version compatible with the target CUDA runtime separately.
