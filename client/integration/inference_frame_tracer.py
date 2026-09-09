"""Record per-inference image timestamps for later alignment with collector parquet."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

INFERENCE_FRAMES_FILENAME = "inference_frames.json"
_SESSION_DIR_RE = re.compile(r"^\d{8}_\d{6}$")
_COARSE_DAY_DIR_RE = re.compile(r"^[a-zA-Z]+_\d{8}$")


def save_path_rank(path: Path) -> tuple[int, int, int]:
    parts = path.parts
    round_parts = sum(1 for part in parts if part.startswith("round_"))
    return (round_parts, len(parts), len(str(path)))


def should_prefer_save_path(new: str | Path, old: str | Path) -> bool:
    new_path = Path(str(new)).expanduser()
    old_path = Path(str(old)).expanduser()
    if new_path == old_path:
        return False
    try:
        new_path.relative_to(old_path)
        return True
    except ValueError:
        pass
    try:
        old_path.relative_to(new_path)
        return False
    except ValueError:
        pass
    return save_path_rank(new_path) > save_path_rank(old_path)


def _normalize_image_timestamps(obs: dict[str, Any]) -> dict[str, float]:
    raw = obs.get("image_timestamps") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): float(value) for key, value in raw.items() if value is not None}


def _is_writable_directory(path: Path) -> bool:
    try:
        if path.exists():
            return os.access(path, os.W_OK)
        parent = path.parent
        if parent.exists():
            return os.access(parent, os.W_OK)
        parent.mkdir(parents=True, exist_ok=True)
        return os.access(parent, os.W_OK)
    except OSError:
        return False


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


class InferenceFrameTracer:
    def __init__(self, *, fallback_root: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._fallback_root = (
            Path(str(fallback_root)).expanduser() if fallback_root is not None else None
        )
        self._active = False
        self._session_id: str | None = None
        self._save_path: Path | None = None
        self._started_at_ns: int | None = None
        self._seq = 0
        self._frames: list[dict[str, Any]] = []

    def set_fallback_root(self, fallback_root: str | Path | None) -> None:
        with self._lock:
            self._fallback_root = (
                Path(str(fallback_root)).expanduser() if fallback_root is not None else None
            )

    def is_active(self) -> bool:
        with self._lock:
            return self._active and self._save_path is not None

    def get_save_path(self) -> str | None:
        with self._lock:
            return None if self._save_path is None else str(self._save_path)

    def begin_session(
        self,
        *,
        save_path: str | Path,
        session_id: str | None = None,
        started_at_ns: int | None = None,
    ) -> None:
        path = Path(str(save_path)).expanduser()
        with self._lock:
            self._active = True
            self._save_path = path
            self._session_id = session_id or path.name or str(int(time.time()))
            self._started_at_ns = started_at_ns
            self._seq = 0
            self._frames = []
        if not _is_writable_directory(path):
            logger.warning(
                "inference frame tracer: save_path is not writable by current user (%s); "
                "frames will be written to fallback root on flush if needed",
                path,
            )
        logger.info("inference frame tracer session started save_path=%s session_id=%s", path, self._session_id)

    def update_save_path(self, save_path: str | Path, *, session_id: str | None = None) -> None:
        path = Path(str(save_path)).expanduser()
        with self._lock:
            if not self._active:
                self._active = True
                self._seq = 0
                self._frames = []
            self._save_path = path
            if session_id:
                self._session_id = session_id
            elif path.name:
                self._session_id = path.name
        logger.info("inference frame tracer save_path updated to %s session_id=%s", path, self._session_id)

    def discard_session(self) -> None:
        with self._lock:
            self._active = False
            self._save_path = None
            self._session_id = None
            self._started_at_ns = None
            self._seq = 0
            self._frames = []
        logger.info("inference frame tracer session discarded")

    def record_inference(
        self,
        obs: dict[str, Any],
        *,
        request_id: int | None = None,
        wall_time_sec: float | None = None,
    ) -> None:
        with self._lock:
            if not self._active:
                return
            self._seq += 1
            image_timestamps = _normalize_image_timestamps(obs)
            image_timestamp = obs.get("image_timestamp")
            if image_timestamp is None and image_timestamps:
                image_timestamp = min(image_timestamps.values())
            frame = {
                "seq": int(self._seq),
                "request_id": request_id,
                "image_timestamp": None if image_timestamp is None else float(image_timestamp),
                "image_timestamps": image_timestamps,
                "state_timestamp": None
                if obs.get("state_timestamp") is None
                else float(obs["state_timestamp"]),
                "wall_time_sec": float(time.time() if wall_time_sec is None else wall_time_sec),
            }
            self._frames.append(frame)

    def _resolve_fallback_path(self, intended_dir: Path) -> Path | None:
        if self._fallback_root is None:
            return None
        session_key = self._session_id or intended_dir.name or str(int(time.time()))
        safe_key = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(session_key))
        return self._fallback_root / safe_key / INFERENCE_FRAMES_FILENAME

    def flush(self, *, save_path: str | Path | None = None) -> str | None:
        with self._lock:
            if not self._frames:
                logger.info("inference frame tracer flush skipped: no frames recorded")
                return None
            intended_dir = (
                Path(str(save_path or self._save_path)).expanduser()
                if (save_path or self._save_path)
                else None
            )
            if intended_dir is None:
                logger.warning("inference frame tracer flush skipped: save_path is empty")
                return None

            payload: dict[str, Any] = {
                "session_id": self._session_id,
                "save_path": str(intended_dir),
                "started_at_ns": self._started_at_ns,
                "frame_count": len(self._frames),
                "inference_frames": list(self._frames),
            }
            frames = len(self._frames)
            primary_path = intended_dir / INFERENCE_FRAMES_FILENAME

        try:
            if _is_writable_directory(intended_dir):
                _write_json(primary_path, payload)
                logger.info("wrote %s (%d frames)", primary_path, frames)
                with self._lock:
                    self._active = False
                return str(primary_path.resolve())
        except OSError as exc:
            logger.warning("inference frame tracer primary write failed: %s (%s)", primary_path, exc)

        with self._lock:
            fallback_path = self._resolve_fallback_path(intended_dir)
            if fallback_path is None:
                logger.error(
                    "inference frame tracer flush failed: cannot write %s and no fallback_save_root configured",
                    intended_dir,
                )
                return None
            payload["intended_save_path"] = str(intended_dir)
            payload["write_fallback"] = True
            payload["written_path"] = str(fallback_path)

        try:
            _write_json(fallback_path, payload)
            logger.warning(
                "inference frame tracer wrote fallback file %s (%d frames); intended save_path=%s",
                fallback_path,
                frames,
                intended_dir,
            )
            with self._lock:
                self._active = False
            return str(fallback_path.resolve())
        except OSError as exc:
            logger.error(
                "inference frame tracer flush failed for intended=%s fallback=%s: %s",
                intended_dir,
                fallback_path,
                exc,
            )
            return None
