# piperserver-master_xh × inference-timeaxis_alignment 接入说明

> **采集仓**：`piperserver-master_xh`  
> **推理策略仓**：`inference-timeaxis_alignment`（`InferenceRuntime` + `config_agilex.yaml`）  
> **模型（Policy）**：远程 WebSocket 服务，由策略侧配置 `host/port`，非本仓库代码  

**版本**：策略契约 `INFERENCE_SDK_VERSION = 1.0.0`（见策略仓 `robot_io_contract.py`）

---

## 0. 先看结论：xh 现状 vs 接入目标

| 项目 | `piperserver-master_xh` **当前** | 接入 **timeaxis** 之后 |
|------|----------------------------------|-------------------------|
| 推理策略 | **自带简化版**：`InferencePacketServer` + 本地 `StreamActionBuffer` + `agilex_inference_client` | **改用** 策略仓 `InferenceRuntime`（mode/chunk/延迟等以 yaml 为准） |
| 观测 | 三路图用 **latest**，关节 `getCurrentJointsWithGripper()`，**无时间轴对齐** | `PiperRosRobotIO.get_observation()`：图+CAN 时间戳对齐 |
| 调模型 | `build_inference_payload` → `policy.infer`（约 `inference_rate` Hz） | Runtime 内 `_build_payload` → `policy.infer`（yaml `inference_rate`） |
| 执行 | `_run_dispatch_loop` 30Hz `JointCtrl` | Runtime `_control_loop` 30Hz `apply_action` |
| TCP 9000 | `set_dagger_mode` → `start_infer_loop` | 建议保持命令名；内部改为拉起 `run_embedded(runtime)` |

**重要**：xh 源码里 **尚未** 引用 `inference-timeaxis_alignment`，要接入需按 **§6 改造**（或合并 `piperserver-master` 的 `infer_2` 模块）。

---

## 1. 三层分工（对接时怎么说）

```
上位机/DAgger
    │  TCP 9000（采集协议，§3）
    ▼
piperserver-master_xh          ← 采集 + 执行 + 挂载策略
    │  Python：RobotIO（§4） + InferenceRuntime（策略仓）
    │  WebSocket msgpack（§5）
    ▼
Policy 模型服务                 ← 你们维护的「模型」，不是推理策略
```

| 层 | 谁维护 | 发什么 / 回什么 |
|----|--------|-----------------|
| TCP 9000 | 采集（xh） | 上位机发 `cmd` → xh 回 `status/message/data`（§3） |
| RobotIO | 采集实现，策略定义字段 | 策略 **调** `get_observation()` ← 采集 **回** 观测；策略 **调** `apply_action` ← 采集 **执行**（§4） |
| Policy infer | 模型服务 | 策略 **发** payload（§5）→ 模型 **回** `actions` |

---

## 2. 部署目录

```text
工作目录/
├── piperserver-master_xh/          # 在此 python main.py
│   ├── config.json
│   ├── main.py
│   ├── PiperWrapper.py             # 接入需补 CAN 时间戳接口（§6.2）
│   └── integration/                # 接入时新增（§6.1）
└── inference-timeaxis_alignment/   # 策略仓，与 xh 同级或子目录
    ├── client/config_agilex.yaml
    ├── client/inference/           # runtime.py, robot_io_factory.py, host_runtime.py
    └── packages/openpi-client/
```

`inference_root` 建议填 **绝对路径** 或 `./../inference-timeaxis_alignment`。

---

## 3. 采集对外接口：TCP 9000（上位机 ↔ xh）

- 传输：TCP，**一行一个 JSON**，UTF-8，以 `\n` 结尾。
- 监听：`config.json` → `agilex_inference.listen_host` / `listen_port`（默认 `0.0.0.0:9000`）。
- Policy 地址：`agilex_inference.host` / `port`（模型，非 9000）。

### 3.1 命令一览（xh 当前已实现）

| cmd | params | xh 行为 | 响应（成功时典型） |
|-----|--------|---------|-------------------|
| `start` | — | 握手 | `{"status":true,"message":"inference packet server ready"}` |
| `set_dagger_mode` | `enable: bool` | `true` → `start_infer_loop()`；`false` → `stop_infer_loop()` | `{"status":true,"message":"dageer mode changed"}` |
| `set_dagger_config` | `ip`, `port`, `prompt` | 更新 runner + 写回 `config.json` | `{"status":true,"message":"set inference config done"}` |
| `infer` | `prompt?` | **单步**推理，不启持续循环 | `{"status":true,"data":{...}}` |
| `latest` | — | 最近一次推理结果 | `{"status":true,"data":...}` |
| `queue` | — | 动作队列快照 | `{"status":true,"data":...}` |
| `stop` | — | 停服务 | `{"status":true,"message":"stopping"}` |

失败：`{"status":false,"message":"..."}`。

### 3.2 接入 timeaxis 后建议

- **命令表保持不变**，避免改 DAgger。
- `set_dagger_mode(true)` 内部改为：起 `InferenceSdkService` / `OpenPiRuntimePacketServer`，`run_embedded(InferenceRuntime)`。
- `infer` 单步：可保留（策略仓单步组观测 + 一次 `policy.infer`），或文档标明「连续推理请用 set_dagger_mode」。

### 3.3 xh 当前持续推理在做什么（未接 timeaxis）

`start_infer_loop` 后：

1. **推理线程**（`inference_rate`，默认 3Hz）：`qpos` + 三路 **latest 图** → `run_policy_inference` → `actions` → 本地 `StreamActionBuffer.integrate_new_chunk`。
2. **下发线程**（`publish_rate`，默认 30Hz）：`pop` → `_send_action_frame_direct` → Piper 从臂。

参数来自 `config.json` 的 `agilex_inference`（如 `inference_rate`、`publish_rate`、`latency_k`），**不是** `config_agilex.yaml`。

---

## 4. 采集 ↔ 推理策略：RobotIO（接入 timeaxis 后的核心）

策略仓通过 `robot_io_factory.create_robot_io(cfg)` 加载采集实现的 IO。  
推荐 entry_point（采集新增 `integration/piperserver_bridge.py`）：

```text
integration.piperserver_bridge:create_piper_ros_robot_io
```

创建前采集须 `push_bridge_context(image_node, left_wrapper, right_wrapper, packet_cfg)`（与 `architecture_workspace/piperserver` 相同）。

### 4.1 策略 → 采集：调用与“响应”

| 策略调用 | 采集须做什么 | 返回 / 结果 |
|----------|--------------|-------------|
| `start()` | 起 ROS 采图（已有）+ 双臂状态采样线程（~200Hz，CAN 时间戳） | 无 |
| `get_observation()` | 按 §4.2 组一帧 | **观测 dict**（下表） |
| `apply_action(action14)` | 14 维关节+夹爪下发从臂（同 xh `_send_action_frame_direct`） | 无；失败抛异常 |
| `close()` | 停线程 | 无 |

### 4.2 观测 dict（采集 “回” 给策略）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `qpos` | `float32[14]` | ✓ | 左6+左夹+右6+右夹，弧度；顺序同 xh `_split_action_frame` |
| `images` | `dict` | ✓ | 键固定：`top_head`, `hand_right`, `hand_left`，BGR `uint8` |
| `sync_frame_time` | `float` | ✓ | 秒；三路图时间锚点（§4.3） |
| `state_timestamp` | `float` | 推荐 | 双臂对齐后状态时间 |
| `image_timestamp` | `float` | 推荐 | 三路对齐后最小图像时间 |
| `image_timestamps` | `dict[str,float]` | 推荐 | 每路实际取用帧时间 |

### 4.3 时间轴（采集实现时必须满足）

与策略仓 `robot_io.AgilexRobotIO` / `integration/piper_ros_robot_io.py` 一致：

1. 每路相机维护 `(stamp, image)` deque；`latest_frame_time` = 三路**最新 stamp 的最小值**。
2. 每路取 **stamp ≥ frame_time** 的最早一帧。
3. 关节：`PiperWrapper.read_slave_joint_state_stamped()` → `(state[7], can_time_stamp)`，deque 对齐到 `frame_time`。

**xh 当前缺口**：`ImageCollector` 无 deque/对齐；`PiperWrapper` 无 `read_slave_joint_state_stamped`（§6.2）。

### 4.4 动作 `action14`（策略 “发” 给采集执行）

| 索引 | 含义 |
|------|------|
| 0–5 | 左臂关节 rad |
| 6 | 左夹爪 |
| 7–12 | 右臂关节 rad |
| 13 | 右夹爪 |

夹爪偏移：`left_gripper_offset` / `right_gripper_offset`（config 可配，默认右 0.003）。

---

## 5. 推理策略 ↔ 模型：Policy infer（你们模型侧对齐）

策略仓 `InferenceRuntime` 周期性调用（频率 = yaml `inference_rate`，非 30Hz）：

### 5.1 策略发给模型（请求）

**基础字段**（`temporal_smoothing` 等 Base mode）：

```python
{
  "images": {
    "top_head":  ndarray,  # shape (3, 224, 224), RGB, CHW
    "hand_right": ndarray,
    "hand_left": ndarray,
  },
  "state": ndarray,         # shape (14,), float, 当前关节
  "prompt": str,
}
```

**扩展字段**（按 yaml `inference.mode` 由策略自动加，模型须支持时才开 mode）：

| mode | 额外请求字段 |
|------|----------------|
| `vlash_async` | `state` 可能为 delay 步上的“未来”动作；策略内部估 delay |
| `rtc` | `inference_delay`, `execute_horizon`, `enable_rtc`, `prev_action_chunk`, … |
| `legato_async` | `inference_delay`, `execute_horizon`, `ramp_down`, `prev_action_chunk_model`, … |

xh **当前** 单步请求（`agilex_inference_client.build_inference_payload`）只有基础三项，且 `state` 即 `qpos`，**无 mode 扩展**。

### 5.2 模型回给策略（响应）

| 字段 | 必填 | 说明 |
|------|------|------|
| `actions` | ✓ | `numpy` `[T, 14]`，T 通常 = yaml `chunk_size`（如 50） |
| `actions_model` | legato 等 | 下一轮策略可能带回 |
| `policy_timing.infer_ms` | 建议 | 策略打日志、估延迟 |

策略收到后：**排队、平滑、30Hz 逐步** `apply_action`，**不会**把整段 chunk 一次下发。

### 5.3 与 xh 当前 payload 的差异

| 项 | xh `build_inference_payload` | timeaxis `_base_payload` |
|----|-------------------------------|---------------------------|
| 图像键名 | 同左 | 同左 |
| resize 224 | ✓ | ✓ |
| `state` | `qpos` | 同，但 mode 可能改写 |
| chunk | xh 单步常 `chunk_size:1` | yaml 通常 50 |

接入 timeaxis 后，**以策略仓 + yaml 为准**，模型需按 **chunk** 返回。

---

## 6. xh 接入 timeaxis 改造清单（采集侧工程）

### 6.1 建议从 `piperserver-master` 或 `architecture_workspace/piperserver` 拷贝/对齐

| 新增/修改 | 作用 |
|-----------|------|
| `integration/agilex_image_collector.py` | 带 deque + `latest_frame_time` / `read_images_at_or_after` |
| `integration/piper_ros_robot_io.py` | 实现 RobotIO |
| `integration/piperserver_bridge.py` | `create_piper_ros_robot_io` + bridge 上下文 |
| `integration/inference_sdk_host.py` | 挂载 `inference_root`，`run_embedded` |
| `main.py` | `agilex_inference.backend == "sdk"` 时走 `InferenceSdkService` |

参考配置：`architecture_workspace/piperserver/config.sdk.example.json`。

### 6.2 必改 `PiperWrapper.py`

增加（V3/timeaxis 已有）：

```python
def read_slave_joint_state_stamped(self) -> tuple[list[float], float]:
    # 7 维关节+夹爪（rad）+ CAN time_stamp（秒）
```

无此方法则 **§4.3 关节对齐无法实现**。

### 6.3 `config.json` 示例（接入后）

在现有 `agilex_inference` 上增加：

```json
"agilex_inference": {
  "enable": true,
  "backend": "sdk",
  "host": "<Policy_IP>",
  "port": 8000,
  "prompt": "fold the sleeve",
  "listen_host": "0.0.0.0",
  "listen_port": 9000,
  "camera_names": ["top_head", "hand_right", "hand_left"],
  "image_topics": [
    "/sensor/cam_front/color_jpg",
    "/sensor/cam_right/color_jpg",
    "/sensor/cam_left/color_jpg"
  ],
  "inference_root": "/abs/path/to/inference-timeaxis_alignment",
  "openpi_runtime_config": "/abs/path/to/inference-timeaxis_alignment/client/config_agilex.yaml",
  "robot_io_entry_point": "integration.piperserver_bridge:create_piper_ros_robot_io",
  "skip_move_to_init": true,
  "wait_images_at_startup": false,
  "state_poll_hz": 200,
  "state_deque_maxlen": 300,
  "sync_frame_timeout_s": 2.0,
  "image_frame_deque_maxlen": 300
}
```

**推理策略参数**（mode、chunk、30/3Hz）只改 **`config_agilex.yaml`**，不要仍在 xh 里用 `inference_rate` 当唯一来源。

### 6.4 策略仓无需为 xh 单独改 runtime

保证 checkout 含：

- `client/inference/runtime.py`
- `robot_io_factory.py`, `host_runtime.py`, `robot_io_contract.py`

`config_agilex.yaml` 中 `inference.mode` 与真机一致（如 `temporal_smoothing`）。

---

## 7. 启动与验证

1. 起 CAN、ROS 相机。  
2. 起 **Policy**（`host:port` 可达）。  
3. `cd piperserver-master_xh && python main.py`。  
4. 日志：`backend=sdk`、`INFERENCE_SDK_VERSION=1.0.0`、`mode=... publish_hz=30 infer_hz=3`。  
5. TCP：`{"cmd":"set_dagger_mode","params":{"enable":true}}`。  
6. 臂连续运动；无 `TimeoutError`（图/CAN 对齐）。

---

## 8. 责任边界（给采集同事一句话）

- **采集**：ROS、双臂、TCP 9000、RobotIO 实现、部署路径。  
- **推理策略仓**：`InferenceRuntime` + yaml（怎么问模型、怎么执行 chunk）。  
- **模型**：WebSocket `infer` 的输入输出（§5）。

---

## 9. 相关文档

| 文档 | 位置 |
|------|------|
| RobotIO 全文 | `inference-timeaxis_alignment/docs/INFERENCE_COLLECTOR_INTERFACE.md` |
| sdk 配置示例 | `architecture_workspace/piperserver/config.sdk.example.json` |
| infer_2 参考实现 | `piperserver-master/openpi_runtime_backend.py`（已内嵌 timeaxis 的完整宿主） |

---

## 10. 变更记录

| 日期 | 说明 |
|------|------|
| 2026-05 | 首版：基于 `piperserver-master_xh` 现状 + `inference-timeaxis_alignment` 策略契约 |
