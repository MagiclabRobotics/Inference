"""
[xh-integration] TCP 推理服务（默认 9001）：供 piperserver-master_xh InferenceServiceClient 调用。

- start：加载 config_agilex.yaml；策略 mode 默认来自 yaml，可由启动脚本 ``--mode`` 覆盖；可覆盖 policy_host/port/prompt
- pop_action：从 InferenceRuntime.stream_buffer 弹一步，不经 io.apply_action
- stop：采集切出或 TCP 断开时释放 Runtime

入口：client/run_inference_service.py
采集改动见 piperserver-master_xh/main.py 内 [xh-integration] 注释
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from integration.collector_contract import ACTION_DIM, INFERENCE_SDK_VERSION
from integration.host_runtime import run_embedded
from integration.inference_frame_tracer import InferenceFrameTracer, should_prefer_save_path
from integration.keyboard_lcm_listener import CommandType, KeyboardLcmListener, TagType
from integration.runtime_builder import (
    build_recording_config,
    build_runtime_profile,
    load_inference_config,
    resolve_config_path,
)


logger = logging.getLogger(__name__)

CLIENT_RUNTIME_CONFIG_NAME = "runtime_config_client.yaml"
POLICY_CONFIG_DEBUG_PREFIX = "[客户端配置]"
IGNORED_CLIENT_SERVER_FIELDS = ("host", "port", "endpoints", "servers")
STAGE_LABELS = {
    "received": "收到客户端配置",
    "saved_and_validated": "保存并校验完成",
    "start_session": "启动推理会话",
    "failed": "处理失败",
}
FIELD_LABELS = {
    "server.host": "远程模型地址",
    "server.port": "远程模型端口",
    "server.transport": "传输方式",
    "inference.mode": "推理模式",
    "inference.prompt": "任务描述(prompt)",
    "inference.chunk_size": "动作块大小",
    "inference.inference_rate": "推理频率(Hz)",
    "inference.smooth_method": "平滑方式",
    "collector.host": "采集服务地址",
    "collector.port": "采集服务端口",
    "collector.camera_names": "相机名称",
    "collector.image_topics": "图像话题",
    "robot_io.entry_point": "RobotIO入口",
    "profile.host": "实际远程模型地址",
    "profile.port": "实际远程模型端口",
    "profile.transport": "实际传输方式",
    "profile.mode": "实际推理模式",
    "profile.prompt": "实际任务描述",
    "profile.chunk_size": "实际动作块大小",
    "profile.inference_rate": "实际推理频率",
    "profile.smooth_method": "实际平滑方式",
}
SESSION_OVERRIDE_LABELS = {
    "policy_host": "远程模型地址(policy_host)",
    "policy_port": "远程模型端口(policy_port)",
    "prompt": "任务描述(prompt)",
    "sync_frame_timeout_s": "图关节同步超时(秒)",
}
EXPECTED_POLICY_CONFIG_SECTIONS = (
    "server",
    "inference",
    "collector",
    "camera",
    "arm",
    "can",
    "recording",
    "robot_io",
    "inference_service",
)


def _deep_merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base, ensure_ascii=False))
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_raw(path: str | Path) -> dict[str, Any]:
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _strip_client_policy_server_fields(cfg_data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cleaned = json.loads(json.dumps(cfg_data, ensure_ascii=False))
    ignored: dict[str, Any] = {}
    server = cleaned.get("server")
    if not isinstance(server, dict):
        return cleaned, ignored
    for key in IGNORED_CLIENT_SERVER_FIELDS:
        if key in server:
            ignored[f"server.{key}"] = server.pop(key)
    if not server:
        cleaned.pop("server", None)
    return cleaned, ignored


def _collect_config_changes(before: Any, after: Any, *, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            field = f"{prefix}.{key}" if prefix else str(key)
            changes.extend(_collect_config_changes(before.get(key), after.get(key), prefix=field))
        return changes
    if before == after:
        return []
    return [{"field": prefix, "old": before, "new": after}]


def _field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field)


def _format_modified_fields_zh(changes: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in changes:
        field = str(item.get("field", ""))
        old = item.get("old")
        new = item.get("new")
        lines.append(f"  - {_field_label(field)}：{old!r} → {new!r}")
    return lines


def _format_key_fields_zh(extracted: dict[str, Any]) -> list[str]:
    order = (
        "server.host",
        "server.port",
        "inference.mode",
        "inference.prompt",
        "inference.inference_rate",
        "inference.chunk_size",
        "collector.host",
        "collector.port",
    )
    lines: list[str] = []
    for key in order:
        value = extracted.get(key)
        if value is not None:
            lines.append(f"  - {_field_label(key)}：{value!r}")
    sections = extracted.get("top_level_sections")
    if sections:
        lines.append(f"  - 配置段：{', '.join(sections)}")
    return lines


def _policy_config_preview(cfg_data: dict[str, Any], *, max_chars: int = 4000) -> str:
    text = json.dumps(cfg_data, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _mode_mapping_ok(source_mode: Any, profile: dict[str, Any]) -> bool | None:
    if source_mode is None:
        return None
    source = str(source_mode).replace("-", "_").lower()
    target_mode = str(profile.get("mode", "")).replace("-", "_").lower()
    target_smooth = str(profile.get("smooth_method", ""))
    if source in {"sync", "naive_async"}:
        return target_mode == "base" and target_smooth == "raw"
    if source in {"temporal_smoothing", "temporal_ensembling"}:
        return target_mode == "base" and target_smooth == "temporal"
    return source == target_mode


def _extract_policy_config_fields(cfg_data: dict[str, Any]) -> dict[str, Any]:
    server = cfg_data.get("server") if isinstance(cfg_data.get("server"), dict) else {}
    inference = cfg_data.get("inference") if isinstance(cfg_data.get("inference"), dict) else {}
    collector = cfg_data.get("collector") if isinstance(cfg_data.get("collector"), dict) else {}
    return {
        "top_level_sections": sorted(str(key) for key in cfg_data),
        "server.host": server.get("host"),
        "server.port": server.get("port"),
        "server.transport": server.get("transport"),
        "inference.mode": inference.get("mode"),
        "inference.prompt": inference.get("prompt"),
        "inference.chunk_size": inference.get("chunk_size"),
        "inference.inference_rate": inference.get("inference_rate"),
        "inference.smooth_method": inference.get("smooth_method"),
        "collector.host": collector.get("host"),
        "collector.port": collector.get("port"),
        "collector.camera_names": collector.get("camera_names"),
        "collector.image_topics": collector.get("image_topics"),
        "robot_io.entry_point": (
            cfg_data.get("robot_io", {}).get("entry_point")
            if isinstance(cfg_data.get("robot_io"), dict)
            else None
        ),
    }


def _build_runtime_mapping_report(
    cfg_data: dict[str, Any],
    profile: dict[str, Any],
    *,
    session_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session_params = dict(session_params or {})
    extracted = _extract_policy_config_fields(cfg_data)
    mapped = {
        "profile.host": profile.get("host"),
        "profile.port": profile.get("port"),
        "profile.transport": profile.get("transport"),
        "profile.mode": profile.get("mode"),
        "profile.prompt": profile.get("prompt"),
        "profile.chunk_size": profile.get("chunk_size"),
        "profile.inference_rate": profile.get("inference_rate"),
        "profile.smooth_method": profile.get("smooth_method"),
    }
    field_pairs = (
        ("server.host", "profile.host"),
        ("server.port", "profile.port"),
        ("server.transport", "profile.transport"),
        ("inference.mode", "profile.mode"),
        ("inference.prompt", "profile.prompt"),
        ("inference.chunk_size", "profile.chunk_size"),
        ("inference.inference_rate", "profile.inference_rate"),
        ("inference.smooth_method", "profile.smooth_method"),
    )
    matches: dict[str, bool | None] = {}
    for source_key, target_key in field_pairs:
        source_val = extracted.get(source_key)
        target_val = mapped.get(target_key)
        if source_val is None:
            matches[source_key] = None
        elif source_key == "inference.mode":
            matches[source_key] = _mode_mapping_ok(source_val, profile)
        else:
            matches[source_key] = source_val == target_val
    missing_sections = [
        section for section in EXPECTED_POLICY_CONFIG_SECTIONS if section not in cfg_data
    ]
    return {
        "received_fields": extracted,
        "mapped_profile_fields": mapped,
        "field_matches": matches,
        "missing_sections": missing_sections,
        "session_overrides": {
            key: session_params.get(key)
            for key in ("policy_host", "policy_port", "prompt", "sync_frame_timeout_s")
            if session_params.get(key) is not None
        },
    }


def _log_policy_config_mapping(
    *,
    stage: str,
    cfg_data: dict[str, Any],
    profile: dict[str, Any] | None = None,
    session_params: dict[str, Any] | None = None,
    config_path: str | None = None,
    success: bool | None = None,
    error: str | None = None,
) -> dict[str, Any] | None:
    report = None
    if profile is not None:
        report = _build_runtime_mapping_report(cfg_data, profile, session_params=session_params)
    stage_label = STAGE_LABELS.get(stage, stage)
    logger.info("%s ===== %s =====", POLICY_CONFIG_DEBUG_PREFIX, stage_label)
    if config_path is not None:
        logger.info("%s 使用配置文件：%s", POLICY_CONFIG_DEBUG_PREFIX, config_path)
    if success is True:
        logger.info("%s 结果：成功", POLICY_CONFIG_DEBUG_PREFIX)
    elif success is False:
        logger.error("%s 结果：失败", POLICY_CONFIG_DEBUG_PREFIX)
    if error:
        logger.error("%s 失败原因：%s", POLICY_CONFIG_DEBUG_PREFIX, error)

    extracted = _extract_policy_config_fields(cfg_data)
    if stage == "received":
        logger.info("%s 客户端下发内容：", POLICY_CONFIG_DEBUG_PREFIX)
        for line in _format_key_fields_zh(extracted):
            logger.info("%s%s", POLICY_CONFIG_DEBUG_PREFIX, line)
        return report

    if report is not None:
        logger.info("%s 配置文件中的关键值：", POLICY_CONFIG_DEBUG_PREFIX)
        for key in (
            "server.host",
            "server.port",
            "inference.mode",
            "inference.prompt",
            "inference.inference_rate",
            "collector.host",
            "collector.port",
        ):
            value = extracted.get(key)
            if value is not None:
                logger.info("%s  - %s：%r", POLICY_CONFIG_DEBUG_PREFIX, _field_label(key), value)

        logger.info("%s 最终实际生效值：", POLICY_CONFIG_DEBUG_PREFIX)
        for key, value in report["mapped_profile_fields"].items():
            if value is not None:
                logger.info("%s  - %s：%r", POLICY_CONFIG_DEBUG_PREFIX, _field_label(key), value)

        if report["session_overrides"]:
            logger.info("%s 采集服务(evaluationserver)传入覆盖参数：", POLICY_CONFIG_DEBUG_PREFIX)
            for key, value in report["session_overrides"].items():
                logger.info(
                    "%s  - %s：%r",
                    POLICY_CONFIG_DEBUG_PREFIX,
                    SESSION_OVERRIDE_LABELS.get(key, key),
                    value,
                )
            logger.info(
                "%s 说明：远程模型地址和任务描述以采集服务传入值为准，配置文件中的 server/inference 可被覆盖",
                POLICY_CONFIG_DEBUG_PREFIX,
            )

        overridden_by_session = {
            "server.host": "policy_host",
            "server.port": "policy_port",
            "inference.prompt": "prompt",
        }
        mismatches = []
        for field_key, matched in report["field_matches"].items():
            if matched is not False:
                continue
            override_key = overridden_by_session.get(field_key)
            if override_key and override_key in report["session_overrides"]:
                cfg_val = extracted.get(field_key)
                actual_val = report["mapped_profile_fields"].get(
                    field_key.replace("server.", "profile.").replace("inference.", "profile.")
                )
                logger.info(
                    "%s 字段被采集服务覆盖：%s  配置=%r  实际=%r",
                    POLICY_CONFIG_DEBUG_PREFIX,
                    _field_label(field_key),
                    cfg_val,
                    actual_val,
                )
                continue
            mismatches.append(field_key)

        if mismatches:
            logger.warning(
                "%s 以下字段与配置文件不一致且未被采集服务说明覆盖：%s",
                POLICY_CONFIG_DEBUG_PREFIX,
                "，".join(_field_label(key) for key in mismatches),
            )
        elif report["session_overrides"]:
            logger.info("%s 配置校验：关键字段已按预期生效（含采集服务覆盖）", POLICY_CONFIG_DEBUG_PREFIX)
        else:
            logger.info("%s 配置校验：关键字段与配置文件一致", POLICY_CONFIG_DEBUG_PREFIX)

        if report["missing_sections"] and stage != "start_session":
            logger.info(
                "%s 客户端未下发的配置段（将使用默认配置补全）：%s",
                POLICY_CONFIG_DEBUG_PREFIX,
                "，".join(report["missing_sections"]),
            )
    return report


class InferenceService:
    def __init__(self, default_config_path: str, startup_mode: str | None = None):
        self._bootstrap_default_config_path = str(default_config_path)
        self._default_config_path = str(default_config_path)
        self._client_runtime_config_path: str | None = None
        self._startup_mode = str(startup_mode).strip() if startup_mode else None
        self._lock = threading.Lock()
        self._runtime = None
        self._io = None
        self._runtime_thread: threading.Thread | None = None
        self._running = False
        self._session_profile: dict[str, Any] | None = None
        self._start_error: str | None = None
        self._last_session_summary: dict[str, Any] | None = None
        self._pop_action_seq = 0
        self._inference_frame_tracer = InferenceFrameTracer(
            fallback_root=self._inference_frames_fallback_root(),
        )
        self._keyboard_lcm_listener: KeyboardLcmListener | None = None
        self._start_keyboard_lcm_listener()

    def _inference_frames_fallback_root(self) -> Path:
        kb_cfg = self._keyboard_lcm_config()
        configured = str(kb_cfg.get("fallback_save_root", "") or "").strip()
        if configured:
            return Path(configured).expanduser()
        try:
            cfg = load_inference_config(resolve_config_path(self._bootstrap_default_config_path))
            raw = getattr(cfg, "raw", {}) or {}
            recording = dict(raw.get("recording") or {})
            root = recording.get("root_dir") or recording.get("record_dir")
            if root:
                return Path(str(root)).expanduser() / "inference_frame_traces"
        except Exception:
            pass
        return Path("/tmp/inference_frame_traces")

    def _keyboard_lcm_config(self) -> dict[str, Any]:
        try:
            cfg = load_inference_config(resolve_config_path(self._bootstrap_default_config_path))
            raw = getattr(cfg, "raw", {}) or {}
            collector = dict(raw.get("collector") or {})
            return dict(collector.get("keyboard_lcm") or {})
        except Exception:
            return {}

    def _start_keyboard_lcm_listener(self) -> None:
        kb_cfg = self._keyboard_lcm_config()
        if not bool(kb_cfg.get("enabled", True)):
            logger.info("keyboard LCM listener disabled by config")
            return
        try:
            self._keyboard_lcm_listener = KeyboardLcmListener(
                on_command=self._on_keyboard_command,
                channel=str(kb_cfg.get("channel", "keyboard_event")),
                lcm_url=str(kb_cfg.get("lcm_url", "")),
                handle_timeout_ms=int(kb_cfg.get("handle_timeout_ms", 50)),
            )
            self._keyboard_lcm_listener.start()
        except Exception as exc:
            logger.warning("keyboard LCM listener not started: %s", exc)
            self._keyboard_lcm_listener = None

    def _apply_lcm_save_path(
        self,
        save_path: str,
        *,
        record_type: int,
        tag: int,
        is_model: bool,
        timestamp_ns: int,
        reset_frames: bool,
    ) -> None:
        tracer = self._inference_frame_tracer
        if reset_frames or not tracer.is_active():
            tracer.begin_session(
                save_path=save_path,
                session_id=Path(save_path).name,
                started_at_ns=timestamp_ns or None,
            )
            logger.info(
                "inference frame tracer begin_session record_type=%s save_path=%s",
                record_type,
                save_path,
            )
            return

        current = tracer.get_save_path()
        if current == save_path:
            return
        if current and should_prefer_save_path(save_path, current):
            tracer.update_save_path(save_path, session_id=Path(save_path).name)
            logger.info(
                "inference frame tracer save_path refined record_type=%s %s -> %s",
                record_type,
                current,
                save_path,
            )
            return
        if current and should_prefer_save_path(current, save_path):
            logger.info(
                "inference frame tracer ignored less-specific save_path record_type=%s %s (keeping %s)",
                record_type,
                save_path,
                current,
            )
            return

        tracer.begin_session(
            save_path=save_path,
            session_id=Path(save_path).name,
            started_at_ns=timestamp_ns or None,
        )
        logger.info(
            "inference frame tracer restarted session record_type=%s save_path=%s",
            record_type,
            save_path,
        )

    def _on_keyboard_command(self, msg) -> None:
        save_path = str(getattr(msg, "save_path", "") or "").strip()
        record_type = int(getattr(msg, "record_type", 0) or 0)
        tag = int(getattr(msg, "tag", 0) or 0)
        is_model = bool(getattr(msg, "is_model", False))
        timestamp_ns = int(getattr(msg, "timestamp", 0) or 0)
        logger.info(
            "keyboard_event record_type=%s save_path=%r is_model=%s tag=%s timestamp=%s active_save_path=%r",
            record_type,
            save_path,
            is_model,
            tag,
            timestamp_ns,
            self._inference_frame_tracer.get_save_path(),
        )

        if record_type == CommandType["Discard_Record"]:
            self._inference_frame_tracer.discard_session()
            return

        if record_type == CommandType["Save_Record"]:
            flush_path = save_path or self._inference_frame_tracer.get_save_path()
            if not flush_path:
                logger.warning("keyboard_event Save_Record missing save_path, nothing to flush")
                return
            self._inference_frame_tracer.flush(save_path=flush_path)
            return

        if not save_path:
            return

        if record_type == CommandType["Start_Record"]:
            current = self._inference_frame_tracer.get_save_path()
            if (
                self._inference_frame_tracer.is_active()
                and current
                and should_prefer_save_path(save_path, current)
            ):
                self._inference_frame_tracer.update_save_path(
                    save_path,
                    session_id=Path(save_path).name,
                )
                logger.info(
                    "inference frame tracer save_path refined on Start_Record %s -> %s",
                    current,
                    save_path,
                )
                return
            self._apply_lcm_save_path(
                save_path,
                record_type=record_type,
                tag=tag,
                is_model=is_model,
                timestamp_ns=timestamp_ns,
                reset_frames=True,
            )
            return

        if record_type == CommandType["None"] and (is_model or tag == TagType["Dagger_Inference"]):
            self._apply_lcm_save_path(
                save_path,
                record_type=record_type,
                tag=tag,
                is_model=is_model,
                timestamp_ns=timestamp_ns,
                reset_frames=not self._inference_frame_tracer.is_active(),
            )
            return

    def _attach_inference_frame_tracer(self, recording_cfg: dict[str, Any]) -> dict[str, Any]:
        out = dict(recording_cfg)
        out["inference_frame_tracer"] = self._inference_frame_tracer
        return out

    def _client_runtime_config_save_path(self) -> Path:
        return resolve_config_path(self._bootstrap_default_config_path).parent / CLIENT_RUNTIME_CONFIG_NAME

    def _resolve_session_config_path(self, params: dict[str, Any]) -> str:
        if self._client_runtime_config_path:
            return self._client_runtime_config_path
        return str(params.get("openpi_runtime_config") or self._default_config_path)

    @staticmethod
    def _parse_policy_config_payload(params: dict[str, Any]) -> dict[str, Any]:
        raw = params.get("config")
        if raw is None:
            raise ValueError("set_policy_config requires params.config")
        if isinstance(raw, str):
            if not raw.strip():
                raise ValueError("params.config must be non-empty")
            parsed = json.loads(raw)
        elif isinstance(raw, dict):
            parsed = raw
        else:
            raise ValueError("params.config must be a JSON string or object")
        if not isinstance(parsed, dict):
            raise ValueError("params.config must decode to a JSON object")
        return parsed

    def set_policy_config(self, params: dict[str, Any] | None) -> dict[str, Any]:
        params = dict(params or {})
        try:
            client_overlay = self._parse_policy_config_payload(params)
            _log_policy_config_mapping(stage="received", cfg_data=client_overlay)
            client_overlay, ignored_server_fields = _strip_client_policy_server_fields(client_overlay)
            if ignored_server_fields:
                logger.info(
                    "%s 已忽略客户端下发的远程模型地址（改由采集服务 start 时传入）：%s",
                    POLICY_CONFIG_DEBUG_PREFIX,
                    "，".join(f"{k}={v!r}" for k, v in ignored_server_fields.items()),
                )

            bootstrap_path = resolve_config_path(self._bootstrap_default_config_path)
            bootstrap_raw = _load_config_raw(bootstrap_path)
            merged_cfg = _deep_merge_config(bootstrap_raw, client_overlay)
            modified_fields = _collect_config_changes(bootstrap_raw, merged_cfg)
            logger.info("%s 配置合并方式：在默认配置基础上覆盖客户端字段", POLICY_CONFIG_DEBUG_PREFIX)
            logger.info("%s 默认配置文件：%s", POLICY_CONFIG_DEBUG_PREFIX, bootstrap_path)
            logger.info(
                "%s 客户端覆盖的配置段：%s",
                POLICY_CONFIG_DEBUG_PREFIX,
                "，".join(sorted(client_overlay)) or "无",
            )

            save_path = self._client_runtime_config_save_path()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as fp:
                yaml.safe_dump(merged_cfg, fp, allow_unicode=True, sort_keys=False)

            resolved_path = str(save_path.resolve())
            logger.info("%s 写入配置文件：%s", POLICY_CONFIG_DEBUG_PREFIX, resolved_path)
            if modified_fields:
                logger.info(
                    "%s 相对默认配置共修改 %d 项：",
                    POLICY_CONFIG_DEBUG_PREFIX,
                    len(modified_fields),
                )
                for line in _format_modified_fields_zh(modified_fields):
                    logger.info("%s%s", POLICY_CONFIG_DEBUG_PREFIX, line)
            else:
                logger.info("%s 相对默认配置无字段变化", POLICY_CONFIG_DEBUG_PREFIX)
            loaded_cfg = load_inference_config(save_path)
            active_mode = loaded_cfg.mode
            available_modes = loaded_cfg.available_modes()
            if active_mode not in available_modes:
                raise ValueError(
                    f"inference.mode {active_mode!r} not in available_modes {available_modes}"
                )
            profile = build_runtime_profile(loaded_cfg, {})
            mapping_report = _log_policy_config_mapping(
                stage="saved_and_validated",
                cfg_data=merged_cfg,
                profile=profile,
                config_path=resolved_path,
                success=True,
            )

            with self._lock:
                self._client_runtime_config_path = resolved_path
                self._default_config_path = resolved_path
                had_active_session = self.session_active
            if had_active_session:
                self.stop_session()
                logger.info("%s 检测到正在运行的会话，已先停止，新配置将在下次切入时生效", POLICY_CONFIG_DEBUG_PREFIX)
            logger.info("%s 配置已生效，下次 start 将使用：%s", POLICY_CONFIG_DEBUG_PREFIX, resolved_path)
            return {
                "status": True,
                "message": "policy config saved",
                "config_path": resolved_path,
                "bootstrap_config_path": str(bootstrap_path),
                "ignored_client_server_fields": ignored_server_fields,
                "modified_fields": modified_fields,
                "overlay_sections": sorted(client_overlay),
                "merged_sections": sorted(merged_cfg),
                "inference_mode": active_mode,
                "available_modes": available_modes,
                "session_restarted": False,
                "session_stopped": had_active_session,
                "config_mapping": mapping_report,
            }
        except Exception as exc:
            preview_data: dict[str, Any]
            try:
                preview_data = self._parse_policy_config_payload(params)
            except Exception:
                preview_data = {"raw_config": params.get("config")}
            _log_policy_config_mapping(
                stage="failed",
                cfg_data=preview_data,
                success=False,
                error=str(exc),
            )
            raise

    @property
    def session_active(self) -> bool:
        return self._runtime is not None and self._runtime_thread is not None and self._runtime_thread.is_alive()

    def start_session(self, params: dict[str, Any] | None) -> None:
        params = dict(params or {})
        if self.session_active:
            logger.info("session already active, restarting to clear action buffer")
            self.stop_session()
        with self._lock:
            if self.session_active:
                return
            self._start_error = None
            config_path = self._resolve_session_config_path(params)
            cfg = load_inference_config(resolve_config_path(str(config_path)))
            if self._startup_mode is not None:
                from config import apply_mode_override

                apply_mode_override(cfg, self._startup_mode)
            profile = build_runtime_profile(cfg, params)
            self._session_profile = dict(profile)
            mapping_report = _log_policy_config_mapping(
                stage="start_session",
                cfg_data=dict(cfg.raw),
                profile=profile,
                session_params=params,
                config_path=str(config_path),
                success=True,
            )
            if self._client_runtime_config_path:
                logger.info("%s 本次使用客户端下发并保存的配置", POLICY_CONFIG_DEBUG_PREFIX)
            else:
                logger.info("%s 本次使用默认/采集服务指定的配置文件", POLICY_CONFIG_DEBUG_PREFIX)

            from robot_io_factory import create_robot_io
            from runtime import InferenceRuntime

            self._io = create_robot_io(cfg)
            self._io.start()
            self._runtime = InferenceRuntime(
                io=self._io,
                cfg=profile,
                recording_cfg=self._attach_inference_frame_tracer(build_recording_config(cfg)),
            )
            self._running = True
            self._runtime_thread = threading.Thread(
                target=self._run_embedded_session,
                args=(self._runtime,),
                name="inference_runtime_embedded",
                daemon=True,
            )
            self._runtime_thread.start()
            logger.info(
                "%s Piper 推理会话已启动：模式=%s，远程模型=%s:%s，任务描述=%r",
                POLICY_CONFIG_DEBUG_PREFIX,
                profile.get("mode"),
                profile.get("host"),
                profile.get("port"),
                profile.get("prompt"),
            )

    def _run_embedded_session(self, runtime) -> None:
        try:
            run_embedded(runtime, execute_actions=False)
        except Exception as exc:
            self._start_error = str(exc)
            logger.exception("embedded runtime failed")
        finally:
            self._running = False

    def stop_session(self) -> dict[str, Any] | None:
        with self._lock:
            if self._runtime is None and not self.session_active:
                # 已 stop 或仅短连接探活断开；勿用 None 覆盖既有 session 统计。
                return self._last_session_summary
            session_summary = None
            if self._runtime is not None:
                build_summary = getattr(self._runtime, "build_session_summary", None)
                if callable(build_summary):
                    try:
                        session_summary = build_summary()
                    except Exception as exc:
                        logger.warning("build session summary: %s", exc)
                self._runtime.shutdown.set()
                self._runtime.close()
            if self._runtime_thread is not None:
                self._runtime_thread.join(timeout=5.0)
            self._runtime_thread = None
            self._runtime = None
            if self._io is not None:
                try:
                    self._io.close()
                except Exception as exc:
                    logger.warning("io close: %s", exc)
            self._io = None
            if self._inference_frame_tracer.is_active():
                try:
                    self._inference_frame_tracer.flush()
                except Exception as exc:
                    logger.warning("inference frame tracer flush on stop_session: %s", exc)
            self._running = False
            self._session_profile = None
            self._start_error = None
            self._pop_action_seq = 0
            if session_summary is not None:
                self._last_session_summary = session_summary
            return session_summary if session_summary is not None else self._last_session_summary

    def _maybe_log_popped_action(
        self,
        action: np.ndarray,
        step: dict[str, Any],
    ) -> None:
        if self._runtime is None:
            return
        cfg = getattr(self._runtime, "cfg", {}) or {}
        if not bool(cfg.get("log_popped_actions", False)):
            return
        self._pop_action_seq += 1
        log_every = max(1, int(cfg.get("log_popped_action_every", cfg.get("log_every_steps", 1))))
        if self._pop_action_seq % log_every != 0:
            return
        logger.info(
            "pop_action #%d chunk_id=%s chunk_step=%s action=%s",
            self._pop_action_seq,
            int(step.get("chunk_id", 0)),
            int(step.get("chunk_step_index", 0)),
            np.round(action, 6).tolist(),
        )

    def pop_action(self) -> dict[str, Any]:
        if self._runtime is None:
            return {"status": True, "action": None}
        step = self._runtime.pop_action_step()
        if step is None:
            return {"status": True, "action": None}
        action = _action_from_step(step)
        if action.size != ACTION_DIM:
            return {"status": False, "message": f"invalid action dim {action.size}, expected {ACTION_DIM}"}
        self._maybe_log_popped_action(action, step)
        return {
            "status": True,
            "action": action.tolist(),
            "chunk_id": int(step.get("chunk_id", 0)),
            "chunk_step_index": int(step.get("chunk_step_index", 0)),
        }

    def status(self) -> dict[str, Any]:
        pending = 0
        mode = None
        policy_host = None
        policy_port = None
        if self._session_profile is not None:
            policy_host = self._session_profile.get("host")
            policy_port = self._session_profile.get("port")
        if self._runtime is not None:
            pending = self._runtime.stream_buffer.pending_count()
            mode = str(self._runtime.cfg.get("mode", ""))
            policy_host = getattr(self._runtime, "_policy_host", policy_host)
            policy_port = getattr(self._runtime, "_policy_port", policy_port)
        policy_ready = _policy_ready(self._runtime)
        return {
            "status": True,
            "running": self.session_active,
            "pending_actions": pending,
            "mode": mode,
            "policy_host": policy_host,
            "policy_port": policy_port,
            "policy_ready": policy_ready,
            "start_error": self._start_error,
            "last_session_summary": self._last_session_summary,
            "runtime_config_path": self._client_runtime_config_path or self._default_config_path,
            "sdk_version": INFERENCE_SDK_VERSION,
        }

    def get_infer_result(self) -> dict[str, Any]:
        if self._runtime is not None:
            build_result = getattr(self._runtime, "build_infer_result", None)
            if callable(build_result):
                return {"status": True, "data": build_result()}
        return {"status": True, "data": _infer_result_from_summary(self._last_session_summary)}

    def handle_request(self, request: dict) -> dict:
        cmd = str(request.get("cmd", ""))
        params = request.get("params") or {}
        try:
            if cmd == "start":
                self.start_session(params if isinstance(params, dict) else {})
                return {
                    "status": True,
                    "message": "inference service started",
                    "sdk_version": INFERENCE_SDK_VERSION,
                }
            if cmd == "stop":
                session_summary = self.stop_session()
                return {"status": True, "message": "stopped", "session_summary": session_summary}
            if cmd == "pop_action":
                return self.pop_action()
            if cmd == "status":
                return self.status()
            if cmd == "get_infer_result":
                return self.get_infer_result()
            if cmd == "set_policy_config":
                try:
                    return self.set_policy_config(params if isinstance(params, dict) else {})
                except Exception as exc:
                    return {"status": False, "message": str(exc)}
            return {"status": False, "message": f"unknown cmd: {cmd}"}
        except Exception as exc:
            logger.exception("handle_request %s", cmd)
            return {"status": False, "message": str(exc)}


def _policy_ready(runtime: Any) -> bool:
    if runtime is None:
        return False
    if getattr(runtime, "_policy", None) is not None:
        return True
    try:
        policy = getattr(runtime, "policy", None)
        return policy is not None
    except Exception:
        return False


def _action_from_step(step: dict[str, Any]) -> np.ndarray:
    action = step.get("action")
    trace = step.get("action_trace", action)
    value = getattr(trace, "value", trace)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _infer_result_from_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {
            "inference_time": None,
            "infer_roundtrip": None,
            "model_time": None,
            "transport_time": None,
            "inference_steps": None,
        }
    latest = summary.get("latest_inference")
    if not isinstance(latest, dict):
        latest = {}
    test_time = summary.get("test_time")
    duration_s = test_time.get("duration_s") if isinstance(test_time, dict) else None
    return {
        "inference_time": duration_s,
        "infer_roundtrip": latest.get("roundtrip_latency_ms") or summary.get("avg_roundtrip_latency_ms"),
        "model_time": latest.get("model_infer_latency_ms") or summary.get("avg_model_infer_latency_ms"),
        "transport_time": latest.get("transport_latency_ms") or summary.get("avg_transport_latency_ms"),
        "inference_steps": summary.get("action_pop_count"),
    }


class InferenceServiceTcpServer:
    def __init__(self, host: str, port: int, service: InferenceService):
        self.host = str(host)
        self.port = int(port)
        self.service = service
        self._running = False

    def serve_forever(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(8)
        sock.settimeout(1.0)
        self._running = True
        logger.info("InferenceService TCP listening on %s:%s", self.host, self.port)
        try:
            while self._running:
                try:
                    conn, addr = sock.accept()
                except socket.timeout:
                    continue
                threading.Thread(
                    target=self._serve_client,
                    args=(conn, addr),
                    daemon=True,
                ).start()
        finally:
            sock.close()
            self.service.stop_session()

    def _serve_client(self, conn: socket.socket, addr) -> None:
        buffer = ""
        try:
            conn.settimeout(30.0)
            while self._running:
                try:
                    data = conn.recv(4096)
                    if not data:
                        break
                    buffer += data.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.strip():
                            continue
                        request = json.loads(line)
                        resp = self.service.handle_request(request)
                        conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
                except (ConnectionResetError, BrokenPipeError):
                    break
                except socket.timeout:
                    continue
                except json.JSONDecodeError as exc:
                    resp = {"status": False, "message": f"invalid json: {exc}"}
                    conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                except Exception as exc:
                    logger.warning("client %s: %s", addr, exc)
                    break
        finally:
            conn.close()
            logger.info("client %s disconnected, stopping inference session", addr)
            self.service.stop_session()
