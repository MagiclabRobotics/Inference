from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

ACTION_STEP_FIELDS = [
    "timestamp_sec",
    "monotonic_sec",
    "chunk_id",
    "chunk_step_index",
    "action_value_source",
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_j6",
    "left_gripper",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_j6",
    "right_gripper",
]
HIGH_FOLLOW_COMMAND_FIELDS = [
    *ACTION_STEP_FIELDS,
    "high_follow_step_index",
    "high_follow_segment_steps",
    "high_follow_phase",
    "high_follow_interpolator",
    "high_follow_reached",
]


class EpisodeOutputManager:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir).expanduser()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.episode_idx = self._next_episode_idx()
        self.episode_dir = self.root_dir / f"episode_{self.episode_idx}"
        self.model_io_dir = self.episode_dir / "model_io"
        self.policy_rollout_dir = self.episode_dir / "policy_rollout"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.model_io_dir.mkdir(parents=True, exist_ok=True)
        self.policy_rollout_dir.mkdir(parents=True, exist_ok=True)
        self.action_steps_path = self.episode_dir / "action_steps.csv"
        self.high_follow_commands_path = self.episode_dir / "high_follow_commands.csv"
        self.runtime_events_path = self.episode_dir / "runtime_events.jsonl"

    def _next_episode_idx(self) -> int:
        max_idx = 0
        for name in os.listdir(self.root_dir):
            if not name.startswith("episode_"):
                continue
            path = self.root_dir / name
            if not path.is_dir():
                continue
            try:
                idx = int(name.split("_", 1)[1])
            except Exception:
                continue
            max_idx = max(max_idx, idx)
        return max_idx + 1


class AsyncRuntimeLogger:
    def __init__(
        self,
        action_csv_path: str | Path,
        event_log_path: str | Path,
        queue_size: int = 4096,
        high_follow_csv_path: str | Path | None = None,
    ):
        self.action_csv_path = Path(action_csv_path)
        self.high_follow_csv_path = Path(high_follow_csv_path) if high_follow_csv_path is not None else None
        self.event_log_path = Path(event_log_path)
        self._action_queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._high_follow_queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._event_queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._stats_lock = threading.Lock()
        self._dropped_action_steps = 0
        self._dropped_high_follow_commands = 0
        self._dropped_events = 0
        self._written_action_steps = 0
        self._written_high_follow_commands = 0
        self._written_events = 0

    def start(self) -> None:
        self.action_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.high_follow_csv_path is not None:
            self.high_follow_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._threads = [
            threading.Thread(target=self._action_worker, daemon=True, name="action-step-logger"),
            threading.Thread(target=self._high_follow_worker, daemon=True, name="high-follow-command-logger"),
            threading.Thread(target=self._event_worker, daemon=True, name="runtime-event-logger"),
        ]
        for thread in self._threads:
            thread.start()

    def log_action_step(self, row: dict[str, Any]) -> None:
        try:
            self._action_queue.put_nowait(row)
        except queue.Full:
            with self._stats_lock:
                self._dropped_action_steps += 1

    def log_high_follow_command(self, row: dict[str, Any]) -> None:
        if self.high_follow_csv_path is None:
            return
        try:
            self._high_follow_queue.put_nowait(row)
        except queue.Full:
            with self._stats_lock:
                self._dropped_high_follow_commands += 1

    def log_event(self, event: dict[str, Any]) -> None:
        item = dict(event)
        item.setdefault("timestamp_sec", time.time())
        item.setdefault("monotonic_sec", time.monotonic())
        try:
            self._event_queue.put_nowait(item)
        except queue.Full:
            with self._stats_lock:
                self._dropped_events += 1

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=join_timeout)

    def get_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "action_csv_path": str(self.action_csv_path),
                "high_follow_csv_path": str(self.high_follow_csv_path) if self.high_follow_csv_path is not None else None,
                "event_log_path": str(self.event_log_path),
                "dropped_action_steps": self._dropped_action_steps,
                "dropped_high_follow_commands": self._dropped_high_follow_commands,
                "dropped_events": self._dropped_events,
                "written_action_steps": self._written_action_steps,
                "written_high_follow_commands": self._written_high_follow_commands,
                "written_events": self._written_events,
            }

    def _action_worker(self) -> None:
        with self.action_csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=ACTION_STEP_FIELDS, extrasaction="ignore")
            writer.writeheader()
            pending_since_flush = 0
            while True:
                if self._stop_event.is_set() and self._action_queue.empty():
                    break
                try:
                    row = self._action_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                writer.writerow({field: row.get(field, "") for field in ACTION_STEP_FIELDS})
                pending_since_flush += 1
                with self._stats_lock:
                    self._written_action_steps += 1
                if pending_since_flush >= 100:
                    fp.flush()
                    pending_since_flush = 0
                self._action_queue.task_done()
            fp.flush()

    def _high_follow_worker(self) -> None:
        if self.high_follow_csv_path is None:
            return
        with self.high_follow_csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=HIGH_FOLLOW_COMMAND_FIELDS, extrasaction="ignore")
            writer.writeheader()
            pending_since_flush = 0
            while True:
                if self._stop_event.is_set() and self._high_follow_queue.empty():
                    break
                try:
                    row = self._high_follow_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                writer.writerow({field: row.get(field, "") for field in HIGH_FOLLOW_COMMAND_FIELDS})
                pending_since_flush += 1
                with self._stats_lock:
                    self._written_high_follow_commands += 1
                if pending_since_flush >= 100:
                    fp.flush()
                    pending_since_flush = 0
                self._high_follow_queue.task_done()
            fp.flush()

    def _event_worker(self) -> None:
        with self.event_log_path.open("w", encoding="utf-8") as fp:
            pending_since_flush = 0
            while True:
                if self._stop_event.is_set() and self._event_queue.empty():
                    break
                try:
                    event = self._event_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                fp.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")
                pending_since_flush += 1
                with self._stats_lock:
                    self._written_events += 1
                if pending_since_flush >= 100:
                    fp.flush()
                    pending_since_flush = 0
                self._event_queue.task_done()
            fp.flush()


def _json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)
