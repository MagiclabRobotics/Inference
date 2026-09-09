"""机器人侧实时推理调度。

异步模式下有三条不同频率的数据流：

* observation thread：以 ``observation_rate`` 采集最新图像和关节状态；
* inference thread：以 ``inference_rate`` 请求一个 action chunk；
* control loop：以 ``publish_rate`` 从 Action Buffer 取一步动作并执行。

Action Buffer 将低频模型输出解耦为高频控制指令，并负责新旧 chunk 的时间对齐和平滑。
同步模式则在一个循环内依次执行“观测 -> 推理 -> 执行完整 chunk”。
"""

from __future__ import annotations

from collections import deque
import json
import logging
import signal
import threading
import time
from typing import Any

import cv2
import numpy as np
from openpi_client import image_tools

from async_inference_recorder import AsyncInferenceRecorder
from async_policy_rollout_recorder import AsyncPolicyRolloutRecorder
from async_runtime_logger import AsyncRuntimeLogger
from async_runtime_logger import EpisodeOutputManager
import cv2
import numpy as np
from openpi_client import image_tools
from openpi_client import shared_memory_client_policy
from robot_io import AgilexRobotIO
from action_buffers import create_action_buffer
from inference_modes import (
    bounded_int,
    build_inference_mode,
    validate_actions_model,
    stream_smooth_method,
    execution_mode
)
from policy_clients import create_policy_client


logger = logging.getLogger(__name__)


def _sleep_rate(last_t: float, rate_hz: float) -> float:
    period = 1.0 / max(float(rate_hz), 1e-6)
    now = time.monotonic()
    sleep_s = period - (now - last_t)
    if sleep_s > 0:
        time.sleep(sleep_s)
    return time.monotonic()


def _configured_num_denoising_steps(cfg: dict[str, Any]) -> int | None:
    for key in ("num_denoising_steps", "denoising_steps", "num_steps"):
        if key in cfg and cfg[key] is not None:
            return bounded_int(key, cfg[key], min_value=1)
    return None


def _action_step_trace(action_step: dict[str, Any]) -> Any:
    return action_step.get("action_trace", action_step["action"])


def _action_step_value(action_step: dict[str, Any]) -> np.ndarray:
    action = _action_step_trace(action_step)
    return np.asarray(getattr(action, "value", action), dtype=float)


def _action_step_value_source(action_step: dict[str, Any]) -> Any:
    return getattr(_action_step_trace(action_step), "value_source", ())


class InferenceRuntime:
    """连接 RobotIO、远程 Policy 和 Action Buffer 的客户端运行时。"""

    def __init__(
        self,
        io: AgilexRobotIO,
        cfg: dict[str, Any],
        recording_cfg: dict[str, Any],
        telemetry: Any | None = None,
    ):
        self.io = io
        self.cfg = cfg
        self.recording_cfg = recording_cfg
        self.telemetry = telemetry
        self.output_manager = self._make_output_manager()
        self.shutdown = threading.Event()
        self.episode_stop_requested = False
        self._action_lock = threading.RLock()
        self.observation_lock = threading.Lock()
        self.latest_observation: dict[str, Any] | None = None
        self.delay_lock = threading.Lock()
        self.delay_rtt_buffer = deque(maxlen=max(1, int(cfg.get("delay_window", 20))))
        self.pred_delay_steps = 0
        self.pending_actions_model: np.ndarray | None = None
        # Snapshot captured immediately before a request. TTRTC uses the same
        # snapshot both for prefix selection and reply-time buffer alignment.
        self.request_progress_before: dict[str, int] | None = None
        # Buffer 类型由 action_buffer 和 smooth_method 共同决定。控制循环只依赖
        # integrate_new_chunk/pop_next_action 这组统一接口。
        self.stream_buffer = create_action_buffer(
            self.cfg,
            max_chunks=int(cfg.get("buffer_max_chunks", 10)),
            state_dim=int(cfg.get("state_dim", 14)),
            smooth_method=stream_smooth_method(cfg),
            ensemble_new_weight=float(cfg.get("ensemble_new_weight", cfg.get("temporal_ensemble_new_weight", 0.6))),
        )
        # Mode Handler 只负责模式特有的请求字段和响应处理，例如 RTC/Legato 的
        # 历史 chunk 与延迟参数；通用的网络调用仍由 Runtime 完成。
        self.mode_handler = build_inference_mode(self)
        self.runtime_logger = self._make_runtime_logger()
        # [collection] 延迟连接策略服务：start_session 先返回，policy 在 embedded 线程首次访问时连接；
        # 同时保留 self._policy 供 inference_service.status() → piperserver policy_ready 检测。
        self._policy = None
        self._policy_host = str(cfg.get("host", "localhost"))
        self._policy_port = int(cfg.get("port", 8000))
        self.recorder = self._make_recorder()
        self.policy_rollout_recorder = self._make_policy_rollout_recorder()
        self.threads: list[threading.Thread] = []
        self.last_infer_finish_monotonic: float | None = None
        self.action_rate_window_start: float | None = None
        self.action_rate_window_steps = 0
        self._suppress_mode_payload_log = False
        self.first_action_publish_monotonic: float | None = None
        self.signal_shutdown_received = False
        # [collection] 会话级指标累计，供 TCP 推理服务 build_session_summary/build_infer_result。
        # 纯推理入口不读取这些字段，故为零开销加法。
        self.session_start_walltime = time.time()
        self.session_start_monotonic = time.monotonic()
        self.inference_count = 0
        self.action_pop_count = 0
        self._roundtrip_latency_total_ms = 0.0
        self._model_infer_latency_total_ms = 0.0
        self._model_infer_latency_count = 0
        self._transport_latency_total_ms = 0.0
        self._transport_latency_count = 0
        self._latest_inference: dict[str, Any] | None = None

    @property
    def policy(self):
        """首次使用时才连接 Policy Server，便于嵌入式会话先完成初始化。"""
        if self._policy is None:
            if self.shutdown.is_set():
                raise RuntimeError("runtime already closed; cannot connect policy")
            self._policy = self._make_policy_client()
        return self._policy

    def _make_output_manager(self) -> EpisodeOutputManager:
        root = self.recording_cfg.get("root_dir") or self.recording_cfg.get("record_dir") or "client/inference_records"
        return EpisodeOutputManager(str(root))

    def _make_runtime_logger(self) -> AsyncRuntimeLogger | None:
        if not bool(self.recording_cfg.get("record_runtime_events", True)) and not bool(
            self.recording_cfg.get("record_action_steps", True)
        ):
            return None
        logger_obj = AsyncRuntimeLogger(
            action_csv_path=self.output_manager.action_steps_path,
            high_follow_csv_path=self.output_manager.high_follow_commands_path,
            event_log_path=self.output_manager.runtime_events_path,
            queue_size=int(self.recording_cfg.get("queue_size", 4096)),
        )
        logger_obj.start()
        set_runtime_logger = getattr(self.io, "set_runtime_logger", None)
        if callable(set_runtime_logger):
            set_runtime_logger(logger_obj)
        return logger_obj

    def _make_recorder(self) -> AsyncInferenceRecorder | None:
        if not bool(self.recording_cfg.get("record_model_io", self.recording_cfg.get("record_inference", False))):
            return None
        recorder = AsyncInferenceRecorder(
            root_dir=str(self.output_manager.model_io_dir),
            output_dir=str(self.output_manager.model_io_dir),
            episode_idx=self.output_manager.episode_idx,
            camera_names=["cam_high", "cam_right_wrist", "cam_left_wrist"],
            fps=int(self.recording_cfg.get("fps", 10)),
            queue_size=int(self.recording_cfg.get("queue_size", 512)),
            video_codec=str(self.recording_cfg.get("video_codec", "mp4v")),
        )
        recorder.start()
        return recorder

    def _make_policy_rollout_recorder(self) -> AsyncPolicyRolloutRecorder | None:
        if not bool(self.recording_cfg.get("record_policy_rollout", False)):
            return None
        recorder = AsyncPolicyRolloutRecorder(
            output_dir=self.output_manager.policy_rollout_dir,
            episode_idx=self.output_manager.episode_idx,
            fps=int(self.recording_cfg.get("policy_rollout_fps", self.recording_cfg.get("fps", 30))),
            queue_size=int(self.recording_cfg.get("policy_rollout_queue_size", self.recording_cfg.get("queue_size", 4096))),
            video_codec=str(self.recording_cfg.get("policy_rollout_video_codec", self.recording_cfg.get("video_codec", "mp4v"))),
        )
        recorder.start()
        return recorder

    def _make_policy_client(self):
        return create_policy_client(self.cfg)

    def log_event(self, event: dict[str, Any]) -> None:
        if self.runtime_logger is not None:
            self.runtime_logger.log_event(event)

    def log_mode_payload(self, payload_event: dict[str, Any]) -> None:
        if self._suppress_mode_payload_log:
            return
        event = {
            "event": "mode_payload",
            "execution_mode": str(self.cfg.get("execution_mode", "async")),
            "method": str(self.cfg.get("method", "")),
        }
        event.update(payload_event)
        self.log_event(event)

    def get_delay_steps(self) -> int:
        with self.delay_lock:
            return int(max(0, self.pred_delay_steps))

    def get_delay_ref_latency_ms(self) -> float | None:
        with self.delay_lock:
            if not self.delay_rtt_buffer:
                return None
            arr = np.asarray(self.delay_rtt_buffer, dtype=float)
            ref_sec = (
                float(np.percentile(arr, 90))
                if self.cfg.get("delay_stat", "median") == "p90"
                else float(np.median(arr))
            )
            return ref_sec * 1000.0

    def update_delay_steps(self, rtt_sec: float) -> None:
        rate = float(self.cfg.get("publish_rate", 30))
        with self.delay_lock:
            self.delay_rtt_buffer.append(float(rtt_sec))
            arr = np.asarray(self.delay_rtt_buffer, dtype=float)
            ref = float(np.percentile(arr, 90)) if self.cfg.get("delay_stat", "median") == "p90" else float(np.median(arr))
            delay = int(np.ceil(ref * rate)) + int(self.cfg.get("delay_margin_steps", 0))
            clip_max = self.cfg.get("delay_clip_max")
            if clip_max is None:
                new_pred_delay_steps = max(0, delay)
            else:
                new_pred_delay_steps = int(np.clip(delay, 0, max(0, int(clip_max))))
            if new_pred_delay_steps != self.pred_delay_steps:
                logger.info(
                    "updated predicted delay steps: %d -> %d (ref=%.1fms rtt=%.1fms)",
                    self.pred_delay_steps,
                    new_pred_delay_steps,
                    ref * 1000.0,
                    rtt_sec * 1000.0,
                )
            self.pred_delay_steps = new_pred_delay_steps

    def _observation_thread(self) -> None:
        """持续刷新最新观测；推理线程只消费最近一帧，不积压旧帧。"""
        last_t = time.monotonic()
        while not self.shutdown.is_set():
            t0 = time.monotonic()
            try:
                obs = self.io.get_observation()
                with self.observation_lock:
                    self.latest_observation = obs
                self.log_event(
                    {
                        "event": "observation",
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                        "state_timestamp": obs.get("state_timestamp"),
                        "image_timestamp": obs.get("image_timestamp"),
                    }
                )
            except Exception as exc:
                logger.warning("observation failed: %s", exc)
                time.sleep(0.005)
            last_t = _sleep_rate(last_t, float(self.cfg.get("observation_rate", self.cfg.get("publish_rate", 30))))

    def _get_latest_observation(self) -> dict[str, Any] | None:
        with self.observation_lock:
            return None if self.latest_observation is None else dict(self.latest_observation)

    def _base_payload(self, obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
        """构造所有推理模式共有的图像、语言和 proprio 输入。

        RobotIO 输出 BGR/HWC 图像；网络请求使用 RGB/CHW，并在客户端先缩放到
        224x224，以免传输原始 640x480 图像。模式相关字段由 ``_build_payload``
        随后通过 Mode Handler 注入。
        """
        images = obs["images"]
        top = images["top_head"]
        right = images["hand_right"]
        left = images["hand_left"]
        image_arrs_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in (top, right, left)]
        raw_images = {
            "top_head": image_arrs_rgb[0].copy(),
            "hand_right": image_arrs_rgb[1].copy(),
            "hand_left": image_arrs_rgb[2].copy(),
        }
        image_arrs = image_tools.resize_with_pad(np.asarray(image_arrs_rgb), 224, 224)
        payload = {
            "images": {
                "top_head": image_arrs[0].transpose(2, 0, 1),
                "hand_right": image_arrs[1].transpose(2, 0, 1),
                "hand_left": image_arrs[2].transpose(2, 0, 1),
            },
            "prompt": str(self.cfg.get("prompt", "fold the sleeve")),
        }
        num_steps = _configured_num_denoising_steps(self.cfg)
        if num_steps is not None:
            payload["num_steps"] = num_steps
        return payload, raw_images, np.asarray(obs["qpos"], dtype=float)

    def _build_payload(self, obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        payload, raw_images, proprio = self._base_payload(obs)
        # Base 模式只加入当前 state；RTC/Legato/VLASH 还会加入历史动作或延迟信息。
        return self.mode_handler.build_payload(payload, proprio), raw_images

    def _warmup_inference(self) -> None:
        """Run one blocking inference before publishing actions, matching the legacy ROS startup path."""
        try:
            obs = self.io.get_observation()
            self._suppress_mode_payload_log = True
            try:
                payload, _ = self._build_payload(obs)
                out = self.policy.infer(payload)
                continuation_payload = self._build_warmup_continuation_payload(obs, out)
                if continuation_payload is not None:
                    _ = self.policy.infer(continuation_payload)
            finally:
                self._suppress_mode_payload_log = False
        except Exception as exc:
            logger.warning("startup warmup inference failed: %s", exc)

    def _build_warmup_continuation_payload(self, obs: dict[str, Any], out: Any) -> dict[str, Any] | None:
        if not isinstance(out, dict):
            return None
        async_mode = str(self.cfg.get("async_mode"))
        payload, _, proprio = self._base_payload(obs)
        if async_mode == "legato":
            actions_model = out.get("actions_model")
            actions = out.get("actions")
            if actions_model is None or actions is None:
                return None
            try:
                prev_action_chunk_model = validate_actions_model(actions_model, len(actions))
            except Exception as exc:
                logger.warning("[Legato] warmup continuation skipped: %s", exc)
                return None
            chunk_size = int(self.cfg.get("chunk_size", len(prev_action_chunk_model)))
            payload.update(
                {
                    "state": proprio,
                    "inference_delay": 0,
                    "execute_horizon": 0,
                    "ramp_down": bounded_int(
                        "ramp_down",
                        self.cfg.get("ramp_down_steps", 22),
                        min_value=0,
                        max_value=chunk_size,
                    ),
                    "prev_action_chunk_model": prev_action_chunk_model,
                }
            )
            return payload
        if async_mode == "rtc":
            actions = out.get("actions")
            if actions is None or len(actions) == 0:
                return None
            try:
                prev_model = validate_actions_model(out.get("actions_model"), len(actions))
            except ValueError as exc:
                logger.warning("[RTC] warmup continuation skipped: %s", exc)
                return None
            if prev_model is None or not np.isfinite(prev_model).all():
                return None
            chunk_size = int(self.cfg.get("chunk_size", len(actions)))
            execute_horizon = self.cfg.get("execute_horizon", chunk_size)
            payload.update(
                {
                    "state": proprio,
                    "execute_horizon": max(1, min(int(execute_horizon), chunk_size)),
                    "enable_rtc": True,
                    "mask_prefix_delay": bool(self.cfg.get("mask_prefix_delay", False)),
                    "prefix_attention_schedule": str(self.cfg.get("prefix_attention_schedule", "exp")),
                    "max_guidance_weight": float(self.cfg.get("max_guidance_weight", 0.5)),
                    "inference_delay": 0,
                    "prev_action_chunk_model": prev_model.copy(),
                }
            )
            return payload
        return None

    def _run_inference_once(self, obs: dict[str, Any]) -> dict[str, Any]:
        """完成一次“构造请求 -> 远程推理 -> 合并新 chunk”的闭环。"""
        self.pending_actions_model = None
        progress_before = self.stream_buffer.get_chunk_progress()
        self.request_progress_before = progress_before

        # 1. 将 RobotIO 的原始观测变成 Policy Server 接受的 payload。
        payload_start = time.monotonic()
        payload, raw_images = self._build_payload(obs)
        payload_latency_ms = (time.monotonic() - payload_start) * 1000.0

        # 2. infer 是阻塞式 RPC。异步指的是它运行在独立推理线程中，不阻塞
        #    30 Hz 控制循环，而不是单次网络调用本身异步。
        t0 = time.monotonic()
        out = self.policy.infer(payload)
        infer_finish = time.monotonic()
        if self.shutdown.is_set():
            return out
        rtt_sec = infer_finish - t0
        roundtrip_latency_ms = rtt_sec * 1000.0
        model_infer_latency_ms = _extract_model_infer_latency_ms(out)
        request_id = _extract_server_request_id(out)
        client_timing = _extract_client_timing(out)
        transport_latency_ms = None
        if model_infer_latency_ms is not None:
            transport_latency_ms = max(0.0, roundtrip_latency_ms - model_infer_latency_ms)

        # 3. 模式处理器解析响应，并更新 RTC/Legato/VLASH 所需的历史状态。
        actions = self.mode_handler.handle_result(out, rtt_sec)
        post_start = time.monotonic()
        raw_chunk_len = int(len(actions)) if actions is not None else 0
        processed_chunk_len = 0
        if actions is not None and len(actions) > 0:
            processed_chunk_len = int(len(actions))
            # 4. 新 chunk 不会直接下发。Buffer 会先补偿已流逝的时间，再按配置
            #    直接切换、线性平滑或 temporal ensemble。
            switch_info = self.stream_buffer.integrate_new_chunk(
                actions,
                max_k=int(self.cfg.get("latency_k", 0)),
                min_m=int(self.cfg.get("min_smooth_steps", 8)),
                actions_model_chunk=self.pending_actions_model,
                chunk_id=request_id,
                **self.mode_handler.integration_kwargs(progress_before),
            )
            if switch_info is not None:
                event = {
                    "event": "chunk_switch",
                    "pred_delay_steps": self.get_delay_steps(),
                    "executed_steps_at_trigger": int(progress_before["executed_steps"]),
                }
                event.update(switch_info)
                self.log_event(event)
        post_latency_ms = (time.monotonic() - post_start) * 1000.0

        infer_frequency_hz = None
        if self.last_infer_finish_monotonic is not None:
            dt = infer_finish - self.last_infer_finish_monotonic
            if dt > 0:
                infer_frequency_hz = 1.0 / dt
        self.last_infer_finish_monotonic = infer_finish

        self._record_inference_metrics(
            roundtrip_latency_ms=roundtrip_latency_ms,
            model_infer_latency_ms=model_infer_latency_ms,
            transport_latency_ms=transport_latency_ms,
            request_id=request_id,
            infer_frequency_hz=infer_frequency_hz,
        )

        self.log_event(
            {
                "event": "inference",
                "execution_mode": str(self.cfg.get("execution_mode", "async")),
                "async_mode": str(self.cfg.get("async_mode")),
                "smooth_method": self.stream_buffer.smooth_method,
                "request_id": request_id,
                "payload_latency_ms": payload_latency_ms,
                "infer_latency_ms": roundtrip_latency_ms,
                "roundtrip_latency_ms": roundtrip_latency_ms,
                "model_infer_latency_ms": model_infer_latency_ms,
                "transport_latency_ms": transport_latency_ms,
                "client_timing": client_timing,
                "infer_frequency_hz": infer_frequency_hz,
                "pred_delay_steps": self.get_delay_steps(),
                "delay_ref_latency_ms": self.get_delay_ref_latency_ms(),
                "chunk_id_before_infer": int(progress_before["chunk_id"]),
                "executed_steps_at_trigger": int(progress_before["executed_steps"]),
            }
        )
        self.log_event(
            {
                "event": "chunk_postprocess",
                "execution_mode": str(self.cfg.get("execution_mode", "async")),
                "async_mode": str(self.cfg.get("async_mode")),
                "smooth_method": self.stream_buffer.smooth_method,
                "postprocess_latency_ms": post_latency_ms,
                "raw_chunk_len": raw_chunk_len,
                "processed_chunk_len": processed_chunk_len,
                "actions_model_present": self.pending_actions_model is not None,
            }
        )
        tracer = self.recording_cfg.get("inference_frame_tracer")
        if tracer is not None:
            tracer.record_inference(obs, request_id=request_id, wall_time_sec=time.time())
        if self.recorder is not None:
            self.recorder.record_model_io(
                payload=payload,
                model_output_actions=actions,
                timestamp_sec=time.time(),
                raw_images=raw_images,
            )
        logger.info(
            "infer smooth=%s request_id=%s roundtrip=%.1fms model=%s transport=%s pending=%d",
            self.stream_buffer.smooth_method,
            request_id if request_id is not None else "n/a",
            roundtrip_latency_ms,
            _format_optional_ms(model_infer_latency_ms),
            _format_optional_ms(transport_latency_ms),
            self.stream_buffer.pending_count(),
        )

    def _inference_thread(self) -> None:
        """按 inference_rate 使用最新观测请求 action chunk。"""
        last_t = time.monotonic()
        while not self.shutdown.is_set():
            obs = self._get_latest_observation()
            if obs is None:
                time.sleep(0.005)
                continue
            try:
                self._run_inference_once(obs)
            except Exception as exc:
                logger.warning("inference failed: %s", exc)
                time.sleep(0.01)
            last_t = _sleep_rate(last_t, float(self.cfg.get("inference_rate", 3)))

    def _control_loop(self) -> None:
        """按 publish_rate 消费 Buffer；这是实际驱动 RobotIO 的唯一异步循环。"""
        max_steps = int(self.cfg.get("max_publish_step", 10000))
        last_t = time.monotonic()
        step = 0
        while step < max_steps and not self.shutdown.is_set():
            action_step = self.stream_buffer.pop_next_action()
            if action_step is None:
                time.sleep(0.001)
                continue
            if str(self.cfg.get("ctrl_type", "joint")) != "joint":
                raise ValueError("SDK runtime currently supports ctrl_type=joint only")
            action = _action_step_value(action_step)
            if self.first_action_publish_monotonic is None:
                self.first_action_publish_monotonic = time.monotonic()
            if self.telemetry is not None:
                self.telemetry.publish("cmd_vla_30hz", action)
            if not self._apply_action_if_running(action, action_step=action_step):
                break
            self._record_policy_rollout(action, action_step)
            self._log_action_step(action, action_step)
            step += 1
            self._maybe_log_action_publish_rate()
            if step % max(1, int(self.cfg.get("log_every_steps", 30))) == 0:
                elapsed_s = (
                    time.monotonic() - self.first_action_publish_monotonic
                    if self.first_action_publish_monotonic is not None
                    else 0.0
                )
                logger.info(
                    "published step=%d pending=%d action_execution_elapsed=%.3fs",
                    step,
                    self.stream_buffer.pending_count(),
                    elapsed_s,
                )
            last_t = _sleep_rate(last_t, float(self.cfg.get("publish_rate", 30)))
        self.shutdown.set()

    def _sync_loop(self) -> None:
        """同步模式：每次获得一个 chunk 后完整执行，再发起下一次推理。"""
        max_steps = int(self.cfg.get("max_publish_step", 10000))
        publish_rate = float(self.cfg.get("publish_rate", 30))
        last_publish_t = time.monotonic()
        step = 0
        while step < max_steps and not self.shutdown.is_set():
            try:
                obs_start = time.monotonic()
                obs = self.io.get_observation()
                self.log_event(
                    {
                        "event": "observation",
                        "latency_ms": (time.monotonic() - obs_start) * 1000.0,
                        "state_timestamp": obs.get("state_timestamp"),
                        "image_timestamp": obs.get("image_timestamp"),
                    }
                )

                self._run_inference_once(obs)

                while step < max_steps and not self.shutdown.is_set():
                    action_step = self.stream_buffer.pop_next_action()
                    if action_step is None:
                        break
                    if str(self.cfg.get("ctrl_type", "joint")) != "joint":
                        raise ValueError("SDK runtime currently supports ctrl_type=joint only")
                    action = _action_step_value(action_step)
                    if self.first_action_publish_monotonic is None:
                        self.first_action_publish_monotonic = time.monotonic()
                    if not self._apply_action_if_running(action):
                        break
                    self._log_action_step(action, action_step)
                    step += 1
                    self._maybe_log_action_publish_rate()
                    if step % max(1, int(self.cfg.get("log_every_steps", 30))) == 0:
                        elapsed_s = (
                            time.monotonic() - self.first_action_publish_monotonic
                            if self.first_action_publish_monotonic is not None
                            else 0.0
                        )
                        logger.info(
                            "sync published step=%d pending=%d action_execution_elapsed=%.3fs",
                            step,
                            self.stream_buffer.pending_count(),
                            elapsed_s,
                        )
                    last_publish_t = _sleep_rate(last_publish_t, publish_rate)

            except Exception as exc:
                logger.warning("sync inference failed: %s", exc)
                time.sleep(0.01)
        self.shutdown.set()

    def _record_policy_rollout(self, action: np.ndarray, action_step: dict[str, Any]) -> None:
        if self.policy_rollout_recorder is None:
            return
        obs = self._get_latest_observation()
        if obs is None:
            return
        self.policy_rollout_recorder.record(obs=obs, action=action, action_step=action_step)

    def _log_action_step(self, action: np.ndarray, action_step: dict[str, Any]) -> None:
        if self.runtime_logger is None or not bool(self.recording_cfg.get("record_action_steps", True)):
            return
        if action.size < 14:
            return
        row = {
            "timestamp_sec": time.time(),
            "monotonic_sec": time.monotonic(),
            "chunk_id": int(action_step["chunk_id"]),
            "chunk_step_index": int(action_step["chunk_step_index"]),
            "left_j1": float(action[0]),
            "left_j2": float(action[1]),
            "left_j3": float(action[2]),
            "left_j4": float(action[3]),
            "left_j5": float(action[4]),
            "left_j6": float(action[5]),
            "left_gripper": float(action[6]),
            "right_j1": float(action[7]),
            "right_j2": float(action[8]),
            "right_j3": float(action[9]),
            "right_j4": float(action[10]),
            "right_j5": float(action[11]),
            "right_j6": float(action[12]),
            "right_gripper": float(action[13]),
            "action_value_source": json.dumps(_action_step_value_source(action_step)),
        }
        self.runtime_logger.log_action_step(row)

    def _maybe_log_action_publish_rate(self) -> None:
        now = time.monotonic()
        if self.action_rate_window_start is None:
            self.action_rate_window_start = now
            self.action_rate_window_steps = 0
            return
        self.action_rate_window_steps += 1
        window_steps = int(self.cfg.get("action_rate_log_window_steps", 100))
        if self.action_rate_window_steps < max(1, window_steps):
            return
        elapsed = now - self.action_rate_window_start
        actual_hz = self.action_rate_window_steps / elapsed if elapsed > 0 else None
        self.log_event(
            {
                "event": "action_publish_rate",
                "target_hz": float(self.cfg.get("publish_rate", 30)),
                "actual_hz": actual_hz,
                "window_duration_ms": elapsed * 1000.0,
                "window_steps": int(self.action_rate_window_steps),
            }
        )
        self.action_rate_window_start = now
        self.action_rate_window_steps = 0

    def _handle_signal(self, signum, _frame) -> None:
        logger.info("received signal %s, shutting down", signum)
        self.signal_shutdown_received = True
        self.shutdown.set()
        self._hold_robot_position()

    def _apply_action_if_running(self, action, *, action_step=None) -> bool:
        with self._action_lock:
            if self.shutdown.is_set():
                return False
            if action_step is None:
                self.io.apply_action(action)
            else:
                self.io.apply_action(action, action_step=action_step)
            return True

    def _hold_robot_position(self) -> None:
        hold = getattr(self.io, "hold_current_position", None)
        if callable(hold):
            try:
                with self._action_lock:
                    hold()
            except Exception as exc:
                logger.warning("failed to hold robot position: %s", exc)

    def request_episode_stop(self) -> None:
        """Finish this episode while retaining the shared robot connection."""
        if self.shutdown.is_set():
            return
        self.episode_stop_requested = True
        self.shutdown.set()
        self._hold_robot_position()
        self.log_event({"event": "episode_stop_requested"})

    # ------------------------------------------------------------------
    # [collection] 采集推理服务接口：指标累计 / 按步取动作 / 会话汇总
    # ------------------------------------------------------------------
    def _record_inference_metrics(
        self,
        *,
        roundtrip_latency_ms: float,
        model_infer_latency_ms: float | None,
        transport_latency_ms: float | None,
        request_id: int | None = None,
        infer_frequency_hz: float | None = None,
    ) -> None:
        self.inference_count += 1
        self._roundtrip_latency_total_ms += float(roundtrip_latency_ms)
        if model_infer_latency_ms is not None:
            self._model_infer_latency_total_ms += float(model_infer_latency_ms)
            self._model_infer_latency_count += 1
        if transport_latency_ms is not None:
            self._transport_latency_total_ms += float(transport_latency_ms)
            self._transport_latency_count += 1
        self._latest_inference = {
            "roundtrip_latency_ms": float(roundtrip_latency_ms),
            "model_infer_latency_ms": None if model_infer_latency_ms is None else float(model_infer_latency_ms),
            "transport_latency_ms": None if transport_latency_ms is None else float(transport_latency_ms),
            "request_id": request_id,
            "infer_frequency_hz": infer_frequency_hz,
        }

    def pop_action_step(self) -> dict[str, Any] | None:
        """采集 30Hz 取一步动作（不经 io.apply_action，由外部采集执行）。"""
        step = self.stream_buffer.pop_next_action()
        if step is not None:
            self.action_pop_count += 1
        return step

    def _avg_latencies(self) -> tuple[float | None, float | None, float | None]:
        avg_round = (self._roundtrip_latency_total_ms / self.inference_count) if self.inference_count else None
        avg_model = (
            self._model_infer_latency_total_ms / self._model_infer_latency_count
            if self._model_infer_latency_count
            else None
        )
        avg_transport = (
            self._transport_latency_total_ms / self._transport_latency_count
            if self._transport_latency_count
            else None
        )
        return avg_round, avg_model, avg_transport

    def build_session_summary(self) -> dict[str, Any]:
        avg_round, avg_model, avg_transport = self._avg_latencies()
        return {
            "test_time": {
                "start_walltime": self.session_start_walltime,
                "end_walltime": time.time(),
                "duration_s": max(0.0, time.monotonic() - self.session_start_monotonic),
            },
            "inference_count": int(self.inference_count),
            "action_pop_count": int(self.action_pop_count),
            "avg_roundtrip_latency_ms": avg_round,
            "avg_model_infer_latency_ms": avg_model,
            "avg_transport_latency_ms": avg_transport,
            "latest_inference": self._latest_inference,
            "mode": str(self.cfg.get("mode", self.cfg.get("async_mode", ""))),
        }

    def build_infer_result(self) -> dict[str, Any]:
        latest = self._latest_inference or {}
        avg_round, avg_model, avg_transport = self._avg_latencies()
        return {
            "inference_time": max(0.0, time.monotonic() - self.session_start_monotonic),
            "infer_roundtrip": latest.get("roundtrip_latency_ms") or avg_round,
            "model_time": latest.get("model_infer_latency_ms") or avg_model,
            "transport_time": latest.get("transport_latency_ms") or avg_transport,
            "inference_steps": int(self.action_pop_count),
        }

    def run(self) -> None:
        """启动所选执行模式并阻塞到任务结束或收到退出信号。"""
        previous_sigint = signal.getsignal(signal.SIGINT)
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        try:
            if self.shutdown.is_set():
                return
            logger.info("server metadata: %s", self.policy.get_server_metadata())
            self._warmup_inference()
            if self.shutdown.is_set():
                return
            if execution_mode(self.cfg) == "sync":
                self._sync_loop()
                return
            self.threads = [
                threading.Thread(target=self._observation_thread, name="observation", daemon=True),
                threading.Thread(target=self._inference_thread, name="inference", daemon=True),
            ]
            for thread in self.threads:
                thread.start()
            self._control_loop()
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)

    def close(self, *, hold_position: bool = False) -> None:
        if self.first_action_publish_monotonic is not None:
            logger.info("action execution elapsed=%.3fs", time.monotonic() - self.first_action_publish_monotonic)
        self.shutdown.set()
        for thread in self.threads:
            thread.join(timeout=1.0)
        if hold_position or self.episode_stop_requested or self.signal_shutdown_received:
            self._hold_robot_position()
        if self.recorder is not None:
            self.recorder.stop()
        if self.policy_rollout_recorder is not None:
            self.policy_rollout_recorder.stop()
        if self.runtime_logger is not None:
            set_runtime_logger = getattr(self.io, "set_runtime_logger", None)
            if callable(set_runtime_logger):
                set_runtime_logger(None)
            self.runtime_logger.stop()
        if self._policy is not None:
            close = getattr(self._policy, "close", None)
            if close is not None:
                close()
            self._policy = None


def _extract_model_infer_latency_ms(out: Any) -> float | None:
    return out.get("policy_timing", {}).get("infer_ms")


def _extract_server_request_id(out: Any) -> int | None:
    if not isinstance(out, dict):
        return None
    timing = out.get("server_timing", {})
    if not isinstance(timing, dict):
        return None
    value = timing.get("request_id", timing.get("request_index"))
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _extract_client_timing(out: Any) -> dict[str, Any] | None:
    if not isinstance(out, dict):
        return None
    timing = out.get("client_timing")
    return timing if isinstance(timing, dict) else None


def _format_optional_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}ms"
