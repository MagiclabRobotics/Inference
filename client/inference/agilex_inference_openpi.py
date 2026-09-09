"""Agilex/Piper 直连推理客户端入口。

本文件只负责组装对象和管理生命周期，核心数据流在 ``InferenceRuntime`` 中：

    RobotIO 采集观测 -> Policy Client 请求 GPU 服务 -> Action Buffer -> RobotIO 执行动作

真实硬件和 LeRobot mock 数据共用同一个 Runtime，只在 ``_make_robot_io`` 处选择不同
的 RobotIO 实现。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import signal
import sys
import time

from keyboard_control import KeyboardEpisodeController

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
OPENPI_CLIENT_SRC = REPO_ROOT / "packages" / "openpi-client" / "src"
CLIENT_TOOLS_DIR = REPO_ROOT / "client" / "tools"
if str(OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))
if str(CLIENT_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_TOOLS_DIR))


def _build_runtime_config(cfg) -> dict:
    from config import section

    profile = cfg.runtime_options()
    server = section(cfg, "server")
    profile.setdefault("host", server.get("host", "localhost"))
    profile.setdefault("port", server.get("port", 8000))
    profile.setdefault("transport", server.get("transport", "websocket"))
    profile.setdefault("shared_memory_socket_path", server.get("shared_memory_socket_path", "/tmp/openpi_policy.sock"))
    if server.get("connect_timeout_s") is not None:
        profile.setdefault("connect_timeout_s", server.get("connect_timeout_s"))
    for key in (
        "endpoints",
        "servers",
        "connections_per_endpoint",
        "max_in_flight",
        "result_timeout_s",
        "first_result_timeout_s",
        "connect_retry_s",
    ):
        if server.get(key) is not None:
            profile.setdefault(key, server.get(key))

    profile.setdefault("state_dim", 14)

    execution_mode = profile.get("execution_mode", "sync")
    if execution_mode == "async":
        async_mode = profile.get("async_mode", "")
        if async_mode == "legato" and profile.get("delay_clip_max") is None:
            profile["delay_clip_max"] = int(profile.get("chunk_size", 50)) - 1
        if "delay_clip_max" in profile and int(profile.get("delay_clip_max", -1)) < 0:
            profile["delay_clip_max"] = int(profile.get("chunk_size", 50)) - 1
    return profile


def _recording_config(cfg, *, run_subdir: str | None = None) -> dict:
    from config import section

    recording = dict(section(cfg, "recording"))
    unique_run_dir = bool(recording.pop("unique_run_dir", False))
    if unique_run_dir and not (recording.get("root_dir") or recording.get("record_dir")):
        recording["root_dir"] = "client/inference_records"
    for key in ("root_dir", "record_dir"):
        if not recording.get(key):
            continue
        path = Path(str(recording[key])).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        if unique_run_dir:
            if run_subdir is None:
                raise ValueError("recording.unique_run_dir requires one run_subdir per client process")
            path /= run_subdir
        recording[key] = str(path)
    return recording


def _make_robot_io(cfg, telemetry):
    from config import section
    from robot_io import AgilexRobotIO
    from robot_io import MockLeRobotRobotIO

    robot_io_cfg = dict(section(cfg, "robot_io"))
    io_type = str(robot_io_cfg.get("type", robot_io_cfg.get("kind", "agilex"))).replace("-", "_").lower()
    if io_type in {"mock", "mock_lerobot", "lerobot"}:
        return MockLeRobotRobotIO.from_config(robot_io_cfg)
    if io_type != "agilex":
        raise ValueError(f"Unsupported robot_io.type: {io_type}")
    return AgilexRobotIO(
        camera_cfg=section(cfg, "camera"),
        arm_cfg=section(cfg, "arm"),
        can_cfg=section(cfg, "can"),
        telemetry=telemetry,
    )


def _run_keyboard_episodes(controller, io, runtime_cls, runtime_cfg, recording_cfg, arm_cfg, telemetry) -> bool:
    while controller.wait_for_start():
        runtime = runtime_cls(io=io, cfg=dict(runtime_cfg), recording_cfg=dict(recording_cfg), telemetry=telemetry)
        episode_idx = runtime.output_manager.episode_idx
        started = time.monotonic()
        try:
            controller.set_runtime(runtime)
            logging.info("[episode control] episode_%d STARTED", episode_idx)
            runtime.run()
        finally:
            controller.set_runtime(None)
            runtime.close(hold_position=True)
        logging.info("[episode control] episode_%d SAVED (duration %.1fs)", episode_idx, time.monotonic() - started)
        if runtime.signal_shutdown_received or controller.quit_requested:
            return True
        io.move_to_init(arm_cfg)
    return controller.quit_requested


def _interrupt_wait(signum, frame) -> None:
    del signum, frame
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config_agilex.yaml", help="inference yaml config")
    parser.add_argument("--check-hardware", action="store_true", help="Read one state/image observation and exit")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument(
        "--keyboard-control", action=argparse.BooleanOptionalAction, default=None,
        help="Override keyboard_control.enabled (s: start, space: finish episode, q: quit)",
    )
    args = parser.parse_args()

    from config import load_config
    from config import section
    from multi_runtime import MultiServerInferenceRuntime
    from robot_io import check_hardware
    from runtime import InferenceRuntime
    from realtime_plot import RealtimePlotConfig
    from realtime_plot import TelemetryPublisher

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    # YAML 同时描述设备、远程 Policy Server、推理模式和记录选项。
    cfg = load_config(args.config)
    keyboard_cfg = dict(section(cfg, "keyboard_control"))
    keyboard_enabled = bool(keyboard_cfg.get("enabled", False))
    if args.keyboard_control is not None:
        keyboard_enabled = args.keyboard_control
    keyboard_enabled = keyboard_enabled and not args.check_hardware
    if keyboard_enabled and not sys.stdin.isatty():
        raise RuntimeError("keyboard control requires an interactive TTY; use --no-keyboard-control for automatic runs")
    keyboard_controller = KeyboardEpisodeController(keyboard_cfg) if keyboard_enabled else None
    recording_run_subdir = None
    if bool(section(cfg, "recording").get("unique_run_dir", False)):
        recording_run_subdir = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
    recording_cfg = _recording_config(cfg, run_subdir=recording_run_subdir)
    telemetry = TelemetryPublisher(RealtimePlotConfig.from_config(section(cfg, "telemetry_plot")))
    telemetry.start_viewer()
    io = _make_robot_io(cfg, telemetry)
    runtime = None
    full_shutdown_requested = False
    previous_sigterm = None
    io_started = False
    try:
        # 设备必须先启动，Runtime 随后才会创建观测/推理/控制循环。
        io.start()
        io_started = True
        if args.check_hardware:
            check_hardware(io)
            return
        io.move_to_init(section(cfg, "arm"))
        runtime_cfg = _build_runtime_config(cfg)
        # multi_websocket 使用独立的并发 Runtime；其余传输复用标准 Runtime。
        runtime_cls = (
            MultiServerInferenceRuntime
            if str(runtime_cfg.get("transport", "websocket")).replace("-", "_").lower()
            in {"multi_websocket", "websocket_multi"}
            else InferenceRuntime
        )
        if keyboard_controller is None:
            runtime = runtime_cls(io=io, cfg=runtime_cfg, recording_cfg=recording_cfg, telemetry=telemetry)
            runtime.run()
        else:
            previous_sigterm = signal.signal(signal.SIGTERM, _interrupt_wait)
            keyboard_controller.start()
            full_shutdown_requested = _run_keyboard_episodes(
                keyboard_controller, io, runtime_cls, runtime_cfg, recording_cfg, section(cfg, "arm"), telemetry,
            )
    except KeyboardInterrupt:
        full_shutdown_requested = True
        logging.info("keyboard interrupt received; shutting down")
    finally:
        # 无论正常结束还是异常退出，都先停止 Runtime，再释放机器人和相机。
        try:
            if keyboard_controller is not None:
                keyboard_controller.close()
            if runtime is not None:
                runtime.close()
                full_shutdown_requested = full_shutdown_requested or runtime.signal_shutdown_received
            if full_shutdown_requested and io_started:
                try:
                    io.move_to_shutdown(section(cfg, "arm"))
                except Exception as exc:
                    logging.warning("shutdown zero-position move failed: %s", exc)
        finally:
            try:
                io.close()
            finally:
                telemetry.close()
                if previous_sigterm is not None:
                    signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
