"""从 yaml + 采集 start 会话参数构造 InferenceRuntime（策略层复用逻辑）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_config_path(path: str | Path) -> Path:
    """解析 yaml 路径。相对路径依次尝试 cwd、client 目录、包根目录。"""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()

    client_dir = Path(__file__).resolve().parent.parent
    candidates = [
        Path.cwd() / p,
        client_dir / p.name,
        client_dir / p,
        repo_root_from_here() / p,
        repo_root_from_here() / "client" / p.name,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return (Path.cwd() / p).resolve()


def apply_session_overrides(profile: dict[str, Any], session: dict[str, Any] | None) -> dict[str, Any]:
    """采集 ``start`` 可覆盖 Policy 地址与 prompt；``inference.mode`` 仅来自 yaml。"""
    if not session:
        return profile
    out = dict(profile)
    if session.get("policy_host") is not None:
        out["host"] = str(session["policy_host"])
    if session.get("policy_port") is not None:
        out["port"] = int(session["policy_port"])
    if session.get("prompt") is not None:
        out["prompt"] = str(session["prompt"])
    if session.get("sync_frame_timeout_s") is not None:
        out["sync_frame_timeout_s"] = float(session["sync_frame_timeout_s"])
    return out


def _derive_mode_label(profile: dict[str, Any]) -> str:
    explicit = profile.get("mode")
    if explicit:
        return str(explicit).replace("-", "_").lower()
    if str(profile.get("execution_mode", "")).replace("-", "_").lower() == "sync":
        return "sync"
    async_mode = str(profile.get("async_mode") or "base").replace("-", "_").lower()
    return async_mode


def build_runtime_profile(cfg: Any, session: dict[str, Any] | None = None) -> dict[str, Any]:
    from config import section

    profile = cfg.runtime_options()
    server = section(cfg, "server")
    profile.setdefault("host", server.get("host", "localhost"))
    profile.setdefault("port", server.get("port", 8000))
    for key in (
        "transport",
        "shared_memory_socket_path",
        "connections_per_endpoint",
        "max_in_flight",
        "result_timeout_s",
        "first_result_timeout_s",
        "connect_timeout_s",
        "connect_retry_s",
        "endpoints",
        "servers",
    ):
        if key in server and key not in profile:
            profile[key] = server[key]
    profile.setdefault("state_dim", 14)
    profile = apply_session_overrides(profile, session)
    profile.setdefault("mode", getattr(cfg, "mode", None) or _derive_mode_label(profile))
    if str(profile.get("mode", "")) == "legato_async" and profile.get("delay_clip_max") is None:
        profile["delay_clip_max"] = int(profile.get("chunk_size", 50)) - 1
    if "delay_clip_max" in profile and int(profile.get("delay_clip_max", -1)) < 0:
        profile["delay_clip_max"] = int(profile.get("chunk_size", 50)) - 1
    return profile


def build_recording_config(cfg: Any) -> dict[str, Any]:
    from config import section

    recording = dict(section(cfg, "recording"))
    root = repo_root_from_here()
    for key in ("root_dir", "record_dir"):
        if not recording.get(key):
            continue
        path = Path(str(recording[key])).expanduser()
        if not path.is_absolute():
            path = root / path
        recording[key] = str(path)
    return recording


def load_inference_config(config_path: str | Path):
    from config import load_config

    return load_config(resolve_config_path(config_path))
