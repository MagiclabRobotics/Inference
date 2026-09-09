# Inference Service TCP API

本文档定义采集侧 `piperserver-master` 与策略服务之间的 TCP 9001 接口。

## 传输协议

- 传输方式：TCP 长连接。
- 编码格式：UTF-8 JSON。
- 分帧方式：每个请求和响应均以 `\n` 结尾，一行一个 JSON。
- 策略服务不直接控臂，只负责观测、推理、动作缓冲和按需返回动作。
- `piperserver-master` 负责 30Hz 拉取动作并执行控臂。

## 责任边界

策略服务只提供客观运行数据，不判断任务成功失败。

策略服务不提供：

- `success`
- `error_reason`

这些字段应由 `piperserver-master` 或上层客户端根据采集结果判断。

策略服务提供：

- 切入会话开始/结束时间
- 推理次数
- 动作拉取次数
- 最近一次推理耗时
- 平均推理往返耗时
- 平均模型耗时
- 平均传输耗时
- 推理步数配置
- `request_id` / `chunk_id` / `chunk_step_index`

## start

启动一次切入会话。策略服务开始采图、观测、推理，但不直接控臂。

请求：

```json
{
  "cmd": "start",
  "params": {
    "openpi_runtime_config": "config_agilex.yaml",
    "policy_host": "127.0.0.1",
    "policy_port": 8000,
    "prompt": "fold the sleeve",
    "sync_frame_timeout_s": 2.0,
    "state_poll_hz": 200.0
  }
}
```

字段说明：

- `openpi_runtime_config`: 策略服务使用的 runtime YAML。
- `policy_host`: OpenPI policy server 地址。
- `policy_port`: OpenPI policy server 端口。
- `prompt`: 本次会话 prompt。
- `sync_frame_timeout_s`: 图像帧与关节状态同步超时。
- `state_poll_hz`: 采集侧关节状态查询采样率。

响应：

```json
{
  "status": true,
  "message": "inference service started",
  "sdk_version": "..."
}
```

## status

查询策略服务状态。

请求：

```json
{"cmd": "status"}
```

响应：

```json
{
  "status": true,
  "running": true,
  "pending_actions": 12,
  "mode": "base",
  "policy_host": "127.0.0.1",
  "policy_port": 8000,
  "policy_ready": true,
  "start_error": null,
  "last_session_summary": null,
  "sdk_version": "..."
}
```

字段说明：

- `running`: 当前是否存在运行中的策略会话。
- `pending_actions`: 策略服务 action buffer 中待消费动作数量。
- `mode`: 当前推理模式，例如 `base`、`vlash_async`、`rtc`、`legato_async`。
- `policy_host` / `policy_port`: 当前 policy server 地址。
- `policy_ready`: 策略服务是否已经连接 policy server。
- `start_error`: 策略服务启动或后台 runtime 异常信息；仅表示系统异常，不表示任务失败原因。
- `last_session_summary`: 上一次 `stop` 生成的会话统计；没有则为 `null`。

## pop_action

采集侧 30Hz 调用，非阻塞拉取一帧动作。

请求：

```json
{"cmd": "pop_action"}
```

无动作时响应：

```json
{
  "status": true,
  "action": null
}
```

有动作时响应：

```json
{
  "status": true,
  "action": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.06, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.06],
  "chunk_id": 123,
  "chunk_step_index": 0
}
```

字段说明：

- `action`: 固定 14 维，左右臂各 7 维。
- `chunk_id`: 当前 action chunk 标识，优先对齐 policy server request id。
- `chunk_step_index`: 当前 action 在 chunk 内的步号。

异常响应示例：

```json
{
  "status": false,
  "message": "invalid action dim 13"
}
```

## stop

结束一次切入会话，并返回本次会话统计 JSON。

请求：

```json
{"cmd": "stop"}
```

响应：

```json
{
  "status": true,
  "message": "stopped",
  "session_summary": {
    "type": "inference_session_summary",
    "schema_version": 1,
    "session_start_timestamp_sec": 1717830000.123,
    "session_end_timestamp_sec": 1717830030.456,
    "test_time": {
      "start_timestamp_sec": 1717830000.123,
      "end_timestamp_sec": 1717830030.456,
      "duration_s": 30.333
    },
    "policy_host": "127.0.0.1",
    "policy_port": 8000,
    "transport": "websocket",
    "mode": "base",
    "smooth_method": "temporal",
    "prompt": "fold the sleeve",
    "num_steps": null,
    "inference_count": 90,
    "action_pop_count": 850,
    "pending_actions": 4,
    "latest_inference": {
      "timestamp_sec": 1717830030.123,
      "request_id": 123,
      "roundtrip_latency_ms": 115.2,
      "model_infer_latency_ms": 98.4,
      "transport_latency_ms": 16.8,
      "payload_latency_ms": 4.1,
      "postprocess_latency_ms": 0.2,
      "infer_frequency_hz": 3.0,
      "raw_chunk_len": 50,
      "processed_chunk_len": 50,
      "num_steps": null,
      "client_timing": {
        "transport": "websocket",
        "request_id": 123,
        "pack_ms": 2.1,
        "send_ms": 0.4,
        "wait_response_ms": 110.0,
        "unpack_ms": 1.3,
        "roundtrip_ms": 114.0
      }
    },
    "avg_roundtrip_latency_ms": 118.0,
    "avg_model_infer_latency_ms": 100.5,
    "avg_transport_latency_ms": 17.5
  }
}
```

`session_summary` 字段说明：

- `type`: 固定为 `inference_session_summary`。
- `schema_version`: 当前统计 JSON schema 版本。
- `test_time`: 本次切入到切出的时间范围。
- `policy_host` / `policy_port`: 本次连接的 policy server。
- `transport`: 策略服务到 policy server 的传输方式，例如 `websocket`。
- `mode`: 推理模式。
- `smooth_method`: action chunk 后处理方式，例如 `temporal` 或 `raw`。
- `prompt`: 本次会话 prompt。
- `num_steps`: 本次配置的推理步数；未配置时为 `null`。
- `inference_count`: 本次会话内策略推理请求次数。
- `action_pop_count`: 本次会话内 `pop_action` 成功弹出动作次数。
- `pending_actions`: 切出时 action buffer 中剩余动作数量。
- `latest_inference`: 最近一次推理的详细耗时。
- `avg_roundtrip_latency_ms`: 本次会话平均推理往返耗时。
- `avg_model_infer_latency_ms`: 本次会话平均模型耗时。
- `avg_transport_latency_ms`: 本次会话平均传输耗时。

## piperserver-master 切出返回

`piperserver-master` 在收到 `set_dagger_mode(false)` 后会调用策略服务 `stop`，并将 `session_summary` 放入自己的响应中。

请求：

```json
{
  "cmd": "set_dagger_mode",
  "params": {
    "enable": false
  }
}
```

响应：

```json
{
  "status": true,
  "message": "dagger mode changed",
  "inference_mode": "service",
  "cut_in": false,
  "session_summary": {
    "type": "inference_session_summary",
    "schema_version": 1
  }
}
```

`session_summary` 内容与策略服务 `stop` 响应中的字段一致。

## set_policy_config

EvaluationKit 在加载评测任务后，会把模型配置里的 `client` 段发给策略服务，用于替换默认 runtime YAML。

请求：

```json
{
  "cmd": "set_policy_config",
  "params": {
    "config": "{\"server\":{\"host\":\"127.0.0.1\",\"port\":8000},\"inference\":{\"mode\":\"base\",\"prompt\":\"fold the cloth\"}}"
  }
}
```

字段说明：

- `config`: EvaluationKit `model_config.client` 的 JSON 字符串；结构与 `config_agilex.yaml` 一致，也可直接传 JSON 对象。
- `server.host` / `server.port`: 策略服务会忽略客户端下发的这两个字段；远程 Policy 地址只使用 evaluationserver `start` 时传入的 `policy_host` / `policy_port`。

响应：

```json
{
  "status": true,
  "message": "policy config saved",
  "config_path": "/abs/path/to/client/runtime_config_client.yaml",
  "ignored_client_server_fields": {
    "server.host": "127.0.0.1",
    "server.port": 8000
  },
  "modified_fields": [
    {
      "field": "inference.prompt",
      "old": "fold the sleeve",
      "new": "Flatten and fold the cloth."
    }
  ],
  "session_restarted": false,
  "session_stopped": false
}
```

行为说明：

- 客户端通常只下发 `inference` 等部分字段；策略服务会与启动时的 bootstrap 配置（如 `config_agilex.yaml`）做 **deep merge**，不会整文件覆盖。
- 客户端下发的 `server.host` / `server.port` 不参与 merge，会记录到 `ignored_client_server_fields`；远程 Policy 地址由 evaluationserver 的 `policy_host` / `policy_port` 决定。
- 合并结果落盘为 `runtime_config_client.yaml`。
- `modified_fields` 会列出相对 bootstrap 配置真正改动的字段路径、旧值和新值。
- 后续 `start` 会优先使用该文件，而不是 `config_agilex.yaml` 或 `openpi_runtime_config`。
- 保存后会校验 `inference.mode` 是否在 `available_modes` 中。
- 若当前已有活跃会话，会先 `stop`，新配置在下次切入时生效。

## get_infer_result

兼容评测客户端的推理结果查询接口。该接口只返回策略服务侧的客观推理指标，不返回任务成功失败判断。

请求：

```json
{"cmd": "get_infer_result"}
```

响应：

```json
{
  "status": true,
  "data": {
    "inference_time": 30.333,
    "infer_roundtrip": 115.2,
    "model_time": 98.4,
    "transport_time": 16.8,
    "inference_steps": 850
  }
}
```

字段说明：

- `inference_time`: 本次切入会话已持续时间，单位秒；切出后为该次会话总时长。
- `infer_roundtrip`: 最近一次推理往返耗时，单位 ms；切出后优先返回最近一次推理耗时，没有则返回会话平均值。
- `model_time`: 最近一次模型耗时，单位 ms；切出后优先返回最近一次模型耗时，没有则返回会话平均值。
- `transport_time`: 最近一次传输耗时，单位 ms；切出后优先返回最近一次传输耗时，没有则返回会话平均值。
- `inference_steps`: 本轮动作执行步数，对应 `session_summary.action_pop_count`。

无可用推理结果时响应：

```json
{
  "status": true,
  "data": {
    "inference_time": null,
    "infer_roundtrip": null,
    "model_time": null,
    "transport_time": null,
    "inference_steps": null
  }
}
```
