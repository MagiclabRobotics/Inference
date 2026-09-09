# 采集对接层（integration）

与 `client/inference/`（策略 Runtime、mode、Policy 调用）**解耦**。  
策略或 `inference.mode` 变更时，通常只改 `config_agilex.yaml` 的 `inference` 段；采集协议稳定则本包无需改动。

## 模块

| 文件 | 职责 |
|------|------|
| `collector_contract.py` | 版本号、IMAGE_KEYS、协议类型 |
| `collector_tcp.py` | piperserver TCP `get_joint_state` |
| `ros_image_source.py` | 推理进程内 ROS 三路图 + 时间轴 deque |
| `collector_robot_io.py` | 实现 `RobotIO`；`apply_action` 为空 |
| `host_runtime.py` | `run_embedded(execute_actions=False)` |
| `runtime_builder.py` | yaml + 采集 `start` 会话参数合并 |
| `inference_service.py` | TCP 9001：`start` / `pop_action` / `stop` |

## 启动

```bash
cd client
python run_inference_service.py --config config_agilex.yaml
python run_inference_service.py --mode vlash_async   # 覆盖 yaml 默认策略
python run_inference_service.py --list-modes         # 打印 available_modes
```

## 配置分工

| 配置段 | 维护方 | 内容 |
|--------|--------|------|
| `inference.execution_mode` / `async_mode` / `modes` | 策略 | temporal_smoothing、vlash_async、rtc… |
| `collector.*` | 对接 | 采集 host/port、ROS topic |
| `inference_service.*` | 对接 | 本服务监听 9001 |
| piperserver `config.json` | 采集 | `use_inference_service`、`inference_service_port` |

## 采集侧前置条件

1. `agilex_inference.use_inference_service: true`（否则 9000 不提供 `get_joint_state`）
2. 采集已启动且双臂在线
3. 本服务与采集同机或可访问采集 `listen_port`（默认 9000）

## 换策略

1. **推荐**：启动命令加 `--mode <name>`（名称见 `inference.available_modes`）
2. 或改 `config_agilex.yaml` 中 `inference.execution_mode`、`async_mode` 及 `modes.*`

重启本服务即可；**无需改 piperserver**。
