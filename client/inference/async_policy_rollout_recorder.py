from __future__ import annotations

import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAMERA_KEYS = ("top_head", "hand_right", "hand_left")


def _to_uint8_rgb(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim != 3:
        return None
    if arr.dtype != np.uint8:
        if arr.size > 0 and arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


class AsyncPolicyRolloutRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        episode_idx: int,
        fps: int = 30,
        queue_size: int = 4096,
        video_codec: str = "mp4v",
    ):
        self.output_dir = Path(output_dir).expanduser()
        self.video_dir = self.output_dir / "videos"
        self.hdf5_path = self.output_dir / f"episode_{int(episode_idx)}.hdf5"
        self.fps = int(fps)
        self.video_codec = str(video_codec)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        max_queue = max(1, int(queue_size))
        self._step_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self._video_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._video_writers: dict[str, cv2.VideoWriter] = {}
        self._video_shapes: dict[str, tuple[int, int]] = {}
        self._h5py = None

        self._stats_lock = threading.Lock()
        self._dropped_steps = 0
        self._dropped_video_frames = 0
        self._written_steps = 0
        self._written_video_frames = 0

    def start(self) -> None:
        try:
            import h5py as _h5py
        except ModuleNotFoundError as exc:
            raise RuntimeError("h5py is required for policy rollout recording but is not installed.") from exc
        self._h5py = _h5py
        self._threads = [
            threading.Thread(target=self._hdf5_worker, daemon=True, name="policy-rollout-hdf5"),
            threading.Thread(target=self._video_worker, daemon=True, name="policy-rollout-video"),
        ]
        for thread in self._threads:
            thread.start()

    def record(self, obs: dict[str, Any] | None, action: np.ndarray, action_step: dict[str, Any]) -> None:
        if obs is None:
            return
        images = obs.get("images") or {}
        rgb_images = {
            key: cv2.cvtColor(images[key], cv2.COLOR_BGR2RGB).copy()
            for key in CAMERA_KEYS
            if key in images and images[key] is not None
        }
        image_timestamps = obs.get("image_timestamps") or {}
        item = {
            "timestamp_sec": time.time(),
            "monotonic_sec": time.monotonic(),
            "qpos": np.asarray(obs.get("qpos", np.zeros(14, dtype=np.float32)), dtype=np.float32).copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "state_timestamp": obs.get("state_timestamp"),
            "sync_frame_time": obs.get("sync_frame_time"),
            "image_timestamp": obs.get("image_timestamp"),
            "image_timestamps": dict(image_timestamps),
            "chunk_id": int(action_step.get("chunk_id", -1)),
            "chunk_step_index": int(action_step.get("chunk_step_index", -1)),
            "images_rgb": rgb_images,
        }
        try:
            self._step_queue.put_nowait(item)
        except queue.Full:
            with self._stats_lock:
                self._dropped_steps += 1

        try:
            self._video_queue.put_nowait(item)
        except queue.Full:
            with self._stats_lock:
                self._dropped_video_frames += 1

    def stop(self, join_timeout: float = 10.0) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=join_timeout)
        self._close_video_writers()

    def get_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "output_dir": str(self.output_dir),
                "hdf5_path": str(self.hdf5_path),
                "dropped_steps": self._dropped_steps,
                "dropped_video_frames": self._dropped_video_frames,
                "written_steps": self._written_steps,
                "written_video_frames": self._written_video_frames,
            }

    def _should_exit(self) -> bool:
        return self._stop_event.is_set() and self._step_queue.empty() and self._video_queue.empty()

    def _hdf5_worker(self) -> None:
        with self._h5py.File(self.hdf5_path, "w", rdcc_nbytes=2 * 1024 * 1024) as root:
            root.attrs["sim"] = False
            root.attrs["compress"] = False
            root.attrs["created_at"] = datetime.now().isoformat()
            root.attrs["image_format"] = "RGB"
            root.attrs["record_type"] = "policy_rollout"

            obs_group = root.create_group("observations")
            d_qpos = obs_group.create_dataset("qpos", shape=(0, 14), maxshape=(None, 14), dtype=np.float32)
            d_action = root.create_dataset("action", shape=(0, 14), maxshape=(None, 14), dtype=np.float32)
            d_ts = root.create_dataset("timestamp_sec", shape=(0,), maxshape=(None,), dtype=np.float64)
            d_mono = root.create_dataset("monotonic_sec", shape=(0,), maxshape=(None,), dtype=np.float64)
            d_state_ts = root.create_dataset("state_timestamp", shape=(0,), maxshape=(None,), dtype=np.float64)
            d_sync = root.create_dataset("sync_frame_time", shape=(0,), maxshape=(None,), dtype=np.float64)
            d_image_ts = root.create_dataset("image_timestamp", shape=(0,), maxshape=(None,), dtype=np.float64)
            d_chunk = root.create_dataset("chunk_id", shape=(0,), maxshape=(None,), dtype=np.int64)
            d_chunk_step = root.create_dataset("chunk_step_index", shape=(0,), maxshape=(None,), dtype=np.int64)
            image_ts_group = root.create_group("image_timestamps")
            d_cam_ts = {
                key: image_ts_group.create_dataset(key, shape=(0,), maxshape=(None,), dtype=np.float64)
                for key in CAMERA_KEYS
            }

            flush_every = 100
            while True:
                if self._should_exit():
                    break
                try:
                    item = self._step_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                idx = d_qpos.shape[0]
                for dataset in (d_qpos, d_action):
                    dataset.resize(idx + 1, axis=0)
                for dataset in (d_ts, d_mono, d_state_ts, d_sync, d_image_ts, d_chunk, d_chunk_step, *d_cam_ts.values()):
                    dataset.resize(idx + 1, axis=0)

                qpos = np.asarray(item["qpos"], dtype=np.float32).reshape(-1)
                action = np.asarray(item["action"], dtype=np.float32).reshape(-1)
                d_qpos[idx] = self._fit_vector(qpos, 14)
                d_action[idx] = self._fit_vector(action, 14)
                d_ts[idx] = float(item["timestamp_sec"])
                d_mono[idx] = float(item["monotonic_sec"])
                d_state_ts[idx] = self._float_or_nan(item.get("state_timestamp"))
                d_sync[idx] = self._float_or_nan(item.get("sync_frame_time"))
                d_image_ts[idx] = self._float_or_nan(item.get("image_timestamp"))
                d_chunk[idx] = int(item["chunk_id"])
                d_chunk_step[idx] = int(item["chunk_step_index"])
                image_timestamps = item.get("image_timestamps") or {}
                for key, dataset in d_cam_ts.items():
                    dataset[idx] = self._float_or_nan(image_timestamps.get(key))

                with self._stats_lock:
                    self._written_steps += 1
                if (idx + 1) % flush_every == 0:
                    root.flush()
                self._step_queue.task_done()
            root.flush()

    @staticmethod
    def _fit_vector(values: np.ndarray, size: int) -> np.ndarray:
        out = np.zeros(size, dtype=np.float32)
        n = min(int(values.size), size)
        if n > 0:
            out[:n] = values[:n]
        return out

    @staticmethod
    def _float_or_nan(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    def _video_worker(self) -> None:
        while True:
            if self._should_exit():
                break
            try:
                item = self._video_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            frames = {
                key: _to_uint8_rgb((item.get("images_rgb") or {}).get(key))
                for key in CAMERA_KEYS
            }
            for cam, frame in frames.items():
                if frame is None:
                    continue
                if cam not in self._video_writers:
                    h, w = frame.shape[:2]
                    output_path = self.video_dir / f"{cam}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
                    writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (w, h))
                    if not writer.isOpened():
                        continue
                    self._video_writers[cam] = writer
                    self._video_shapes[cam] = (h, w)
                h, w = self._video_shapes[cam]
                if frame.shape[0] != h or frame.shape[1] != w:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
                self._video_writers[cam].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                with self._stats_lock:
                    self._written_video_frames += 1
            self._video_queue.task_done()
        self._close_video_writers()

    def _close_video_writers(self) -> None:
        for writer in self._video_writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self._video_writers = {}

