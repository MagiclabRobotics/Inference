"""异步推理模式适配层。

Runtime 负责通用 RPC 和 Action Buffer；本模块只处理不同算法的请求/响应差异：

* Base：发送当前机器人状态；
* VLASH：用旧 chunk 中的未来动作近似网络延迟后的状态；
* RTC：把上一 chunk、执行窗口和估计延迟交给 RTC 模型；
* Legato：把模型空间中的上一 chunk 交给去噪过程做连续性引导。
* TTRTC：把上一模型空间 chunk 的未执行前缀固定在新一轮去噪中。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


def bounded_int(name: str, value, *, min_value: int, max_value: int | None = None) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be convertible to int, got {value!r}") from exc
    if out < min_value:
        out = min_value
    if max_value is not None and out > max_value:
        out = max_value
    return out


def validate_actions_model(actions_model, expected_horizon: int) -> np.ndarray | None:
    if actions_model is None:
        return None
    arr = np.asarray(actions_model, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"actions_model must have shape [H, D], got {arr.shape}")
    if arr.shape[0] != int(expected_horizon):
        raise ValueError(f"actions_model must have horizon {expected_horizon}, got {arr.shape[0]}")
    return arr


class BaseInferenceMode:
    """标准模式：以当前 proprio 为条件，直接使用服务端返回的动作。"""

    def __init__(self, runtime: Any):
        self.runtime = runtime

    @property
    def cfg(self) -> dict[str, Any]:
        return self.runtime.cfg

    @property
    def buffer(self):
        return self.runtime.stream_buffer

    def build_payload(self, base_payload: dict[str, Any], proprio: np.ndarray) -> dict[str, Any]:
        base_payload["state"] = proprio
        return base_payload

    def handle_result(self, out: dict[str, Any], rtt_sec: float) -> np.ndarray | None:
        actions = out.get("actions", None) if isinstance(out, dict) else None
        if actions is None or len(actions) == 0:
            return None
        return np.asarray(actions, dtype=float)

    def integration_kwargs(self, progress_before: dict[str, int]) -> dict[str, Any]:
        del progress_before
        return {}



class VlashMode(BaseInferenceMode):
    """用预计请求返回时刻的未来状态代替当前状态，减小观测延迟。"""

    def build_payload(self, base_payload: dict[str, Any], proprio: np.ndarray) -> dict[str, Any]:
        delay_steps = self.runtime.get_delay_steps()
        # Buffer 中的未来动作可视作短期状态预测；没有可用预测时安全回退到实测状态。
        future = self.buffer.peek_future_action(delay_steps)
        use_future = bool(self.cfg.get("enable_future_state_injection", True)) and future is not None
        base_payload["state"] = future if use_future else proprio
        self.runtime.log_mode_payload(
            {
                "used_future_state": bool(use_future),
                "pred_delay_steps": int(delay_steps),
            }
        )
        return base_payload

    def handle_result(self, out: dict[str, Any], rtt_sec: float) -> np.ndarray | None:
        self.runtime.update_delay_steps(rtt_sec)
        return super().handle_result(out, rtt_sec)


class RtcMode(BaseInferenceMode):
    """用已接纳 chunk 的模型空间动作引导 RTC，并对齐请求时刻。"""

    def build_payload(self, base_payload: dict[str, Any], proprio: np.ndarray) -> dict[str, Any]:
        chunk_size = int(self.cfg.get("chunk_size", 50))
        progress = self.runtime.request_progress_before or self.buffer.get_chunk_progress()
        prev = self.buffer.get_prev_action_chunk_model()
        start = 0
        if prev is not None:
            chunk_size = len(prev)
            # remaining_steps 同时反映之前丢弃的前缀与已发布动作，避免重放旧 chunk 的开头。
            start = max(0, chunk_size - int(progress["remaining_steps"]))
            indices = np.minimum(start + np.arange(chunk_size), chunk_size - 1)
            base_payload["prev_action_chunk_model"] = prev[indices].copy()
        execute_horizon = self.cfg.get("execute_horizon", chunk_size)
        execute_horizon = max(1, min(int(execute_horizon), chunk_size))
        base_payload.update(
            {
                "state": proprio,
                "execute_horizon": execute_horizon,
                "enable_rtc": True,
                "mask_prefix_delay": bool(self.cfg.get("mask_prefix_delay", False)),
                "prefix_attention_schedule": str(self.cfg.get("prefix_attention_schedule", "exp")),
                "max_guidance_weight": float(self.cfg.get("max_guidance_weight", 0.5)),
                "inference_delay": int(max(0, self.runtime.get_delay_steps())),
            }
        )
        self.runtime.log_mode_payload(
            {
                "execute_horizon": int(execute_horizon),
                "inference_delay": int(self.runtime.get_delay_steps()),
                "has_prev_action_chunk": prev is not None,
                "executed_steps_at_trigger": start,
            }
        )
        return base_payload

    def handle_result(self, out: dict[str, Any], rtt_sec: float) -> np.ndarray | None:
        self.runtime.update_delay_steps(rtt_sec)
        actions = super().handle_result(out, rtt_sec)
        self.runtime.pending_actions_model = None
        if actions is not None:
            model_actions = validate_actions_model(out.get("actions_model"), len(actions))
            if model_actions is None or not np.isfinite(model_actions).all():
                raise ValueError("RTC requires finite actions_model from the policy server; update the RTC server.")
            self.runtime.pending_actions_model = model_actions.copy()
        return actions

    def integration_kwargs(self, progress_before: dict[str, int]) -> dict[str, Any]:
        # 新预测以请求发出时刻为起点，跳过 RPC 期间实际消费的旧动作步数。
        return {"drop_reference_remaining": int(progress_before["remaining_steps"])}


class LegatoMode(BaseInferenceMode):
    """让新 chunk 在模型去噪阶段参考上一 chunk，而非仅在客户端后处理。"""

    def build_payload(self, base_payload: dict[str, Any], proprio: np.ndarray) -> dict[str, Any]:
        chunk_size = int(self.cfg.get("chunk_size", 50))
        progress = self.buffer.get_chunk_progress()
        inference_delay = bounded_int(
            "inference_delay",
            self.runtime.get_delay_steps(),
            min_value=0,
            max_value=chunk_size,
        )
        execute_horizon = bounded_int(
            "execute_horizon",
            # 请求发出前已执行的步数，加上请求往返期间预计继续执行的步数。
            int(progress["executed_steps"]) + inference_delay,
            min_value=0,
            max_value=chunk_size,
        )
        ramp_down = bounded_int(
            "ramp_down",
            self.cfg.get("ramp_down_steps", 22),
            min_value=0,
            max_value=chunk_size,
        )
        base_payload.update(
            {
                "state": proprio,
                "inference_delay": inference_delay,
                "execute_horizon": execute_horizon,
                "ramp_down": ramp_down,
            }
        )
        prev_action_chunk_model = self.buffer.get_prev_action_chunk_model()
        if prev_action_chunk_model is not None and len(prev_action_chunk_model) > 0:
            # 必须使用归一化后的模型空间动作，服务端会直接把它送入去噪引导。
            base_payload["prev_action_chunk_model"] = prev_action_chunk_model
        self.runtime.log_mode_payload(
            {
                "inference_delay": int(inference_delay),
                "execute_horizon": int(execute_horizon),
                "ramp_down": int(ramp_down),
                "executed_steps_at_trigger": int(progress["executed_steps"]),
            }
        )
        return base_payload

    def handle_result(self, out: dict[str, Any], rtt_sec: float) -> np.ndarray | None:
        self.runtime.update_delay_steps(rtt_sec)
        actions = super().handle_result(out, rtt_sec)
        if actions is None:
            return None
        actions_model = out.get("actions_model", None) if isinstance(out, dict) else None
        try:
            self.runtime.pending_actions_model = validate_actions_model(actions_model, len(actions))
        except Exception as exc:
            self.runtime.pending_actions_model = None
            logger.warning("[Legato] actions_model ignored: %s", exc)
        return actions


class TtrtcMode(BaseInferenceMode):
    """Use an already TTRTC-trained model to condition on a clean action prefix."""

    def build_payload(self, base_payload: dict[str, Any], proprio: np.ndarray) -> dict[str, Any]:
        chunk_size = int(self.cfg.get("chunk_size", 50))
        progress = self.runtime.request_progress_before or self.buffer.get_chunk_progress()
        prev_action_chunk_model = self.buffer.get_prev_action_chunk_model()
        action_horizon = (
            int(len(prev_action_chunk_model))
            if prev_action_chunk_model is not None and len(prev_action_chunk_model) > 0
            else chunk_size
        )
        inference_delay = bounded_int(
            "inference_delay",
            self.runtime.get_delay_steps(),
            min_value=0,
            max_value=action_horizon,
        )
        execute_horizon = bounded_int(
            "execute_horizon",
            action_horizon - int(progress["remaining_steps"]),
            min_value=0,
            max_value=action_horizon,
        )
        base_payload.update(
            {
                "state": proprio,
                "enable_ttrtc": True,
                "inference_delay": inference_delay,
                "execute_horizon": execute_horizon,
            }
        )
        sample_num_steps = self.cfg.get("sample_num_steps")
        if sample_num_steps is not None:
            base_payload["num_steps"] = bounded_int(
                "sample_num_steps",
                sample_num_steps,
                min_value=1,
            )
        if prev_action_chunk_model is not None and len(prev_action_chunk_model) > 0:
            # TTRTC conditions inside the denoiser, so robot-space actions are invalid here.
            base_payload["prev_action_chunk_model"] = prev_action_chunk_model
        self.runtime.log_mode_payload(
            {
                "inference_delay": int(inference_delay),
                "execute_horizon": int(execute_horizon),
                "action_horizon": int(action_horizon),
                "executed_steps_at_trigger": int(progress["executed_steps"]),
                "remaining_steps_at_trigger": int(progress["remaining_steps"]),
                "has_prev_action_chunk_model": prev_action_chunk_model is not None,
            }
        )
        return base_payload

    def handle_result(self, out: dict[str, Any], rtt_sec: float) -> np.ndarray | None:
        self.runtime.update_delay_steps(rtt_sec)
        actions = super().handle_result(out, rtt_sec)
        if actions is None:
            return None
        actions_model = out.get("actions_model", None) if isinstance(out, dict) else None
        try:
            self.runtime.pending_actions_model = validate_actions_model(actions_model, len(actions))
        except Exception as exc:
            self.runtime.pending_actions_model = None
            logger.warning("[TTRTC] actions_model ignored: %s", exc)
        return actions

    def integration_kwargs(self, progress_before: dict[str, int]) -> dict[str, Any]:
        # The control loop keeps consuming actions while RPC is blocked.  Let the
        # buffer resolve the exact elapsed count under its lock when the reply lands.
        return {"drop_reference_remaining": int(progress_before["remaining_steps"])}


def _norm(value: Any, default: str = "") -> str:
    text = str(default if value is None else value).strip()
    return text.replace("-", "_").lower()


def stream_smooth_method(cfg: dict[str, Any]) -> str:
    smooth_method = _norm(cfg.get("smooth_method")) if cfg.get("smooth_method") is not None else None
    execution_mode = cfg.get("execution_mode")

    if execution_mode == "sync":
         return "raw"

    async_mode = _norm(cfg.get("async_mode"))
    if async_mode == "temporal_smoothing":
        return "temporal_smoothing"
    if async_mode == "temporal_ensembling":
        return "temporal_ensembling"

    if smooth_method is not None:
        if smooth_method in {"raw", "temporal_smoothing", "temporal_ensembling"}:
            return smooth_method
        raise ValueError(
            "smooth_method must be one of 'raw', 'temporal_smoothing', or 'temporal_ensembling', "
            f"got {smooth_method!r}"
        )
    return "raw"


def build_inference_mode(runtime: Any) -> BaseInferenceMode:
    """根据 execution_mode/async_mode 创建请求协议适配器。"""
    exec_mode = execution_mode(runtime.cfg)
    async_mode = _norm(runtime.cfg.get("async_mode")) if exec_mode == "async" else None
    runtime.cfg["execution_mode"] = exec_mode
    if async_mode == "vlash":
        return VlashMode(runtime)
    if async_mode == "rtc":
        return RtcMode(runtime)
    if async_mode == "legato":
        return LegatoMode(runtime)
    if async_mode == "ttrtc":
        return TtrtcMode(runtime)
    return BaseInferenceMode(runtime)


def execution_mode(cfg: dict[str, Any]) -> str:
    value = cfg.get("execution_mode")
    if value is None:
        return "sync"
    if value not in {"sync", "async"}:
        raise ValueError(f"execution_mode must be 'sync' or 'async', got {value!r}")
    return value
