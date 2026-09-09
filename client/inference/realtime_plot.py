from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class RealtimePlotConfig:
    enabled: bool = False
    auto_start: bool = True
    window_sec: float = 10.0
    queue_size: int = 4096
    default_arms: tuple[str, ...] = ("left",)
    default_topics: tuple[str, ...] = ("cmd_high_follow_200hz", "state_200hz")
    publish_vla_command: bool = True
    publish_high_follow_command: bool = True
    publish_state: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "RealtimePlotConfig":
        raw = dict(cfg or {})
        publish = dict(raw.get("publish") or {})
        default_arms = _tuple_str(raw.get("default_arms", ("left",)))
        default_topics = _tuple_str(raw.get("default_topics", ("cmd_high_follow_200hz", "state_200hz")))
        return cls(
            enabled=bool(raw.get("enabled", False)),
            auto_start=bool(raw.get("auto_start", True)),
            window_sec=max(1.0, float(raw.get("window_sec", 10.0))),
            queue_size=max(1, int(raw.get("queue_size", 4096))),
            default_arms=default_arms,
            default_topics=default_topics,
            publish_vla_command=bool(publish.get("vla_command", True)),
            publish_high_follow_command=bool(publish.get("high_follow_command", True)),
            publish_state=bool(publish.get("state", True)),
        )


class TelemetryPublisher:
    def __init__(self, cfg: RealtimePlotConfig):
        self.cfg = cfg
        self.enabled = bool(cfg.enabled)
        self._ctx = mp.get_context("spawn")
        self._queue: mp.Queue | None = self._ctx.Queue(maxsize=cfg.queue_size) if self.enabled else None
        self._process: mp.Process | None = None
        self._dropped = 0

    def start_viewer(self) -> None:
        if not self.enabled or not self.cfg.auto_start or self._queue is None:
            return
        if self._process is not None and self._process.is_alive():
            return
        self._process = self._ctx.Process(
            target=_run_viewer,
            args=(self._queue, self.cfg.window_sec, self.cfg.default_arms, self.cfg.default_topics),
            daemon=True,
            name="realtime-curve-viewer",
        )
        try:
            self._process.start()
            logger.info("realtime plot viewer started pid=%s", self._process.pid)
        except Exception as exc:
            logger.warning("failed to start realtime plot viewer: %s", exc)

    def publish(self, topic: str, values: np.ndarray | list[float], *, monotonic_sec: float | None = None) -> None:
        if not self.enabled or self._queue is None:
            return
        if topic == "cmd_vla_30hz" and not self.cfg.publish_vla_command:
            return
        if topic == "cmd_high_follow_200hz" and not self.cfg.publish_high_follow_command:
            return
        if topic == "state_200hz" and not self.cfg.publish_state:
            return
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size < 14:
            return
        item = {
            "topic": str(topic),
            "monotonic_sec": float(monotonic_sec if monotonic_sec is not None else time.monotonic()),
            "wall_time_sec": time.time(),
            "values": arr[:14].astype(float, copy=False).tolist(),
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._dropped += 1
        except Exception as exc:
            self.enabled = False
            logger.warning("realtime plot telemetry disabled after publish failure: %s", exc)

    def close(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._process = None
        if self._dropped:
            logger.info("realtime plot dropped telemetry samples=%d", self._dropped)


def _tuple_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(x) for x in value)
    except Exception:
        return tuple()


def _run_viewer(telemetry_queue, window_sec: float, default_arms: tuple[str, ...], default_topics: tuple[str, ...]) -> None:
    from realtime_curve_viewer import run_viewer

    run_viewer(
        telemetry_queue=telemetry_queue,
        window_sec=float(window_sec),
        default_arms=tuple(default_arms),
        default_topics=tuple(default_topics),
    )
