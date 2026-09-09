"""GPU Policy Server 进程入口。

启动流程是：解析配置 -> 加载 checkpoint/TensorRT Policy -> 选择传输层 ->
持续接受客户端的 ``infer(observation)`` 请求。设备采集、Action Buffer 和机械臂
控制都位于客户端，不属于本进程职责。
"""

import dataclasses
import enum
import socket

from loguru import logger
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import server_logging
from openpi.training import config as _config
import tyro


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"


class TransportMode(str, enum.Enum):
    WEBSOCKET = "websocket"
    MULTI_WEBSOCKET = "multi_websocket"
    SHARED_MEMORY = "shared_memory"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    # Optional norm-stats asset id. If omitted, the config's data asset id is used.
    asset_id: str | None = None


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Tensorrt:
    """Load a TensorRT engine exported from the PyTorch sample_actions path."""

    # Training config name used for transforms and norm stats.
    config: str
    # TensorRT engine path, typically an FP16 .engine built from the exported ONNX.
    engine: str
    # Optional assets root containing <asset_id>/norm_stats.json. If omitted, the config's assets path is used.
    assets_dir: str | None = None
    # Optional norm-stats asset id. If omitted, the config's data asset id is used.
    asset_id: str | None = None
    # CUDA device used for TensorRT execution.
    device: str = "cuda"
    # RNG seed used only when requests do not provide an explicit diffusion noise tensor.
    seed: int | None = None
    # TensorRT output tensor name.
    output_name: str = "actions"
    # Metadata label for the loaded engine precision.
    precision: str = "fp16"


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Transport used to serve policy requests.
    transport: TransportMode = TransportMode.WEBSOCKET
    # Unix domain socket used when transport=shared_memory.
    shared_memory_socket_path: str = "/tmp/openpi_policy.sock"
    # Number of workers used by transport=multi_websocket.
    websocket_workers: int = 2
    # Shared request queue size used by transport=multi_websocket.
    websocket_queue_size: int = 16
    # Enable Legato inference path if the loaded model supports `legato_sample_actions`.
    use_legato_inference: bool = False
    # Enable inference-only TTRTC clean-prefix sampling for a compatible JAX checkpoint.
    use_ttrtc_inference: bool = False
    # Enable SnapFlow 1-NFE inference path if the loaded model supports `sample_actions_one_step`.
    use_snapflow_inference: bool = False
    # Record the policy's behavior for debugging.
    record: bool = False

    # Human-readable log level for console and text log output.
    log_level: str = "INFO"
    # Text log file. Set to "" to disable file logging.
    log_file: str = "logs/server.log"
    # Structured JSONL event log file. Set to "" to disable event logging.
    event_log_file: str = "logs/server_events.jsonl"
    # Emit one terminal inference summary every N requests. JSONL events are still recorded for every request.
    request_log_every_n: int = 1

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Tensorrt | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """根据配置创建统一的 Policy 接口，隐藏 checkpoint/TRT 的加载差异。"""
    sample_kwargs = {
        "use_legato_inference": args.use_legato_inference,
        "use_ttrtc_inference": args.use_ttrtc_inference,
        "use_snapflow_inference": args.use_snapflow_inference,
    }
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            if args.policy.asset_id is not None:
                train_config = dataclasses.replace(
                    train_config,
                    data=dataclasses.replace(
                        train_config.data,
                        assets=dataclasses.replace(train_config.data.assets, asset_id=args.policy.asset_id),
                    ),
                )
            return _policy_config.create_trained_policy(
                train_config,
                args.policy.dir,
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs,
            )
        case Tensorrt():
            if args.use_ttrtc_inference:
                raise ValueError("TTRTC inference is not available for TensorRT engines; use a JAX checkpoint.")
            from openpi.policies import tensorrt_policy

            return tensorrt_policy.TensorRTPolicy(
                engine_path=args.policy.engine,
                config_name=args.policy.config,
                assets_dir=args.policy.assets_dir,
                asset_id=args.policy.asset_id,
                default_prompt=args.default_prompt,
                device=args.policy.device,
                seed=args.policy.seed,
                output_name=args.policy.output_name,
                precision=args.policy.precision,
                use_legato_inference=args.use_legato_inference,
            )
        case Default():
            if checkpoint := DEFAULT_CHECKPOINT.get(args.env):
                return _policy_config.create_trained_policy(
                    _config.get_config(checkpoint.config),
                    checkpoint.dir,
                    default_prompt=args.default_prompt,
                    sample_kwargs=sample_kwargs,
                )
            raise ValueError(f"Unsupported environment mode: {args.env}")


def main(args: Args) -> None:
    server_logging.configure_server_logging(
        level=args.log_level,
        log_file=args.log_file or None,
        event_log_file=args.event_log_file or None,
    )

    # 模型和 norm stats 在监听端口前完成加载，避免客户端连上后才发现权重错误。
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info(
        "creating server host={} ip={} port={} transport={} shared_memory_socket={} "
        "websocket_workers={} websocket_queue_size={} policy_metadata_keys={} log_file={} event_log_file={}",
        hostname,
        local_ip,
        args.port,
        args.transport.value,
        args.shared_memory_socket_path,
        args.websocket_workers,
        args.websocket_queue_size,
        sorted(policy_metadata.keys()),
        args.log_file or None,
        args.event_log_file or None,
    )

    # 三种传输层最终都调用同一个 policy.infer；这里只决定请求如何到达 GPU。
    if args.transport == TransportMode.SHARED_MEMORY:
        from openpi.serving import shared_memory_policy_server

        server = shared_memory_policy_server.SharedMemoryPolicyServer(
            policy=policy,
            socket_path=args.shared_memory_socket_path,
            metadata=policy_metadata,
            request_log_every_n=args.request_log_every_n,
        )
    elif args.transport == TransportMode.MULTI_WEBSOCKET:
        from openpi.serving import multi_websocket_policy_server

        server = multi_websocket_policy_server.MultiWebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
            request_log_every_n=args.request_log_every_n,
            worker_count=args.websocket_workers,
            queue_size=args.websocket_queue_size,
        )
    else:
        from openpi.serving import websocket_policy_server

        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
            request_log_every_n=args.request_log_every_n,
        )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
