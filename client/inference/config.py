from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _norm(value: Any, default: str = "") -> str:
    text = str(default if value is None else value).strip()
    return text.replace("-", "_").lower()


_TEMPORAL_ASYNC_MODES = {"temporal_smoothing", "temporal_ensembling"}


def _temporal_from_async_mode(async_mode: str) -> str:
    return async_mode if async_mode in _TEMPORAL_ASYNC_MODES else ""


def _split_mode(inference: dict[str, Any]) -> tuple[str, str, str]:
    execution_mode = _norm(
        inference.get("execution_mode")
    )
    if execution_mode not in {"sync", "async"}:
        raise ValueError(f"execution_mode must be 'sync' or 'async', got {execution_mode!r}")

    if execution_mode == "sync":
        return execution_mode, "", ""
    else:
        async_mode = _norm(inference.get("async_mode", ""), "")
        if not async_mode:
            async_mode = "temporal_smoothing" # default
        

        if async_mode not in (
            "naive",
            "temporal_smoothing",
            "temporal_ensembling",
            "rtc",
            "legato",
            "ttrtc",
            "vlash",
        ):
            raise ValueError(f"Unsupported async_mode: {async_mode!r}")

        temporal_method = _norm(
            inference.get("smooth_method"),
            "",
        )   
        if execution_mode == "sync":
            temporal_method = ""
        elif not temporal_method:
            temporal_method = _temporal_from_async_mode(async_mode)

        return execution_mode, async_mode, temporal_method


def _merge_mapping(dst: dict[str, Any], src: Any) -> None:
    if isinstance(src, dict):
        dst.update(src)

@dataclass
class InferenceConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def inference(self) -> dict[str, Any]:
        return self.raw.setdefault("inference", {})

    @property
    def mode(self) -> str:
        return str(self.inference.get("mode", "base")).replace("-", "_").lower()

    def available_modes(self) -> list[str]:
        """[collection] 供 ``--mode`` / set_policy_config 校验的可选模式名。

        优先取 yaml ``inference.available_modes``，否则回退到默认模式集合。
        """
        listed = self.inference.get("available_modes")
        if isinstance(listed, (list, tuple)) and listed:
            return [_norm(item) for item in listed]
        return list(_DEFAULT_AVAILABLE_MODES)

    def runtime_options(self) -> dict[str, Any]:
        inference = dict(self.inference)
        modes = dict(inference.pop("modes", {}) or {})
        execution_mode, async_mode, smooth_method = _split_mode(inference)
        if execution_mode == "sync":
            inference.pop("async_mode", None)
            inference.pop("smooth_method", None)

        _merge_mapping(inference, modes.get(execution_mode))
        execution_options = modes.get(execution_mode)
        if isinstance(execution_options, dict):
            if execution_mode == "async":
                _merge_mapping(inference, execution_options.get(async_mode))
            if smooth_method:
                _merge_mapping(inference, execution_options.get(smooth_method))

        inference["execution_mode"] = execution_mode
        if execution_mode == "async":
            inference["async_mode"] = async_mode
        
        if async_mode in _TEMPORAL_ASYNC_MODES:
            inference["smooth_method"] = async_mode
        # if smooth_method:
            # inference["temporal_method"] = smooth_method
        return inference


def load_config(path: str | Path) -> InferenceConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    return InferenceConfig(raw=raw, path=config_path)


def section(cfg: InferenceConfig, name: str) -> dict[str, Any]:
    value = cfg.raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section {name!r} must be a mapping")
    return value


def camera_serial(value: str) -> str:
    serial = str(value or "").strip()
    return serial[1:] if serial.startswith("_") else serial


# ---------------------------------------------------------------------------
# [collection] 采集推理服务的模式选择辅助
#
# master 的运行时按 ``execution_mode``(sync/async) + ``async_mode`` 驱动；为兼容采集
# 服务/启动脚本沿用的“单一模式名”习惯，这里把友好名映射回 master 的字段，
# 既能让 ``--mode`` 生效，又不改动纯推理入口 agilex_inference_openpi.py 的行为。
# 模式名必须在映射表中，避免未知模式沿用旧配置继续运行。
# ---------------------------------------------------------------------------

# mode -> (execution_mode, async_mode, smooth_method)
_MODE_TO_EXECUTION: dict[str, tuple[str, str | None, str | None]] = {
    "sync": ("sync", None, None),
    "base": ("async", "naive", None),
    "naive": ("async", "naive", None),
    "naive_async": ("async", "naive", None),
    "temporal_smoothing": ("async", "temporal_smoothing", None),
    "temporal_ensembling": ("async", "temporal_ensembling", None),
    "vlash": ("async", "vlash", "raw"),
    "vlash_async": ("async", "vlash", "raw"),
    "rtc": ("async", "rtc", "raw"),
    "rtc_async": ("async", "rtc", "raw"),
    "legato": ("async", "legato", "raw"),
    "legato_async": ("async", "legato", "raw"),
    "ttrtc": ("async", "ttrtc", "raw"),
    "ttrtc_async": ("async", "ttrtc", "raw"),
}

_DEFAULT_AVAILABLE_MODES: tuple[str, ...] = (
    "sync",
    "base",
    "naive_async",
    "temporal_smoothing",
    "temporal_ensembling",
    "vlash_async",
    "rtc",
    "legato_async",
    "ttrtc_async",
)

_MODE_DESCRIPTIONS: dict[str, str] = {
    "sync": "同步：每次推理后逐步执行整块动作",
    "base": "异步 naive：异步推理，不做时间平滑",
    "naive_async": "异步 naive：异步推理，不做时间平滑",
    "temporal_smoothing": "异步 + 时间平滑（默认推荐）",
    "temporal_ensembling": "异步 + 时间集成（多块加权融合）",
    "vlash_async": "异步 VLASH：未来状态注入",
    "rtc": "异步 RTC：实时块切换",
    "legato_async": "异步 Legato：渐入渐出衔接",
    "ttrtc_async": "异步 TTRTC：模型内干净动作前缀续接（需服务端 JAX TTRTC）",
}


def apply_mode_override(cfg: InferenceConfig, mode: str | None) -> str:
    """[collection] 把单一模式名写回 ``inference``，并映射到 master 的执行字段。

    - 传入 ``None`` 时不改动，返回当前 ``cfg.mode``。
    - 已知友好名：同时设置 ``execution_mode`` / ``async_mode`` / ``smooth_method``。
    - 未知名：抛出 ``ValueError``，保持原配置不变。
    """
    if not mode:
        return cfg.mode
    canonical = _norm(mode)
    mapping = _MODE_TO_EXECUTION.get(canonical)
    if mapping is None:
        raise ValueError(f"Unsupported inference mode: {canonical!r}")
    inference = cfg.inference
    inference["mode"] = canonical
    execution_mode, async_mode, smooth_method = mapping
    inference["execution_mode"] = execution_mode
    if execution_mode == "async" and async_mode:
        inference["async_mode"] = async_mode
    else:
        inference.pop("async_mode", None)
    if smooth_method:
        inference["smooth_method"] = smooth_method
    return canonical


def format_mode_list(cfg: InferenceConfig) -> str:
    """[collection] ``--list-modes`` 打印可选模式与简要说明。"""
    current = cfg.mode
    lines = ["可选推理模式 (available inference modes):"]
    for name in cfg.available_modes():
        marker = "  <- 当前" if name == current else ""
        desc = _MODE_DESCRIPTIONS.get(name, "")
        suffix = f"：{desc}" if desc else ""
        lines.append(f"  - {name}{suffix}{marker}")
    return "\n".join(lines)
