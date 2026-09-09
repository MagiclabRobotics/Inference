import logging
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np


logger = logging.getLogger(__name__)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim != 3:
        return None
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def _payload_img_chw_to_hwc_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        return None
    # payload images are expected CHW
    if arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


class AsyncInferenceRecorder:
    def __init__(
        self,
        root_dir: str,
        camera_names,
        fps: int = 30,
        queue_size: int = 512,
        video_codec: str = "mp4v",
        output_dir: str | None = None,
        episode_idx: int | None = None,
    ):
        os.makedirs(root_dir, exist_ok=True)
        self.episode_idx = int(episode_idx) if episode_idx is not None else self._next_episode_idx(root_dir)
        self.output_dir = output_dir if output_dir is not None else os.path.join(root_dir, f"episode_{self.episode_idx}")
        self.video_dir = os.path.join(self.output_dir, "videos")
        self.hdf5_path = os.path.join(self.output_dir, f"episode_{self.episode_idx}.hdf5")
        self.camera_names = list(camera_names)
        self.fps = int(fps)
        self.video_codec = video_codec

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        self._step_queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._video_queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = threading.Event()
        self._threads = []
        self._video_writers = {}
        self._video_shapes = {}
        self._h5py = None
        self._first_frame_saved = False

        self._stats_lock = threading.Lock()
        self._dropped_steps = 0
        self._dropped_video_frames = 0
        self._written_steps = 0
        self._written_video_frames = 0

    @staticmethod
    def _next_episode_idx(root_dir: str) -> int:
        max_idx = 0
        try:
            for name in os.listdir(root_dir):
                if not name.startswith("episode_"):
                    continue
                path = os.path.join(root_dir, name)
                if not os.path.isdir(path):
                    continue
                try:
                    idx = int(name.split("_", 1)[1])
                except Exception:
                    continue
                if idx > max_idx:
                    max_idx = idx
        except FileNotFoundError:
            return 1
        return max_idx + 1

    def start(self):
        try:
            import h5py as _h5py
        except ModuleNotFoundError as e:
            raise RuntimeError("h5py is required for --record_inference but is not installed.") from e
        self._h5py = _h5py
        h5_thread = threading.Thread(target=self._hdf5_worker, daemon=True, name="hdf5-recorder")
        video_thread = threading.Thread(target=self._video_worker, daemon=True, name="video-recorder")
        self._threads = [h5_thread, video_thread]
        for th in self._threads:
            th.start()

    def record_model_io(self, payload: dict, model_output_actions, timestamp_sec=None, raw_images: dict | None = None):
        if payload is None:
            return
        images = payload.get("images", {})
        raw_images = raw_images or {}
        # Keep main-thread work minimal: only enqueue references/timestamp.
        record = {
            "state": payload.get("state", None),
            "actions": model_output_actions,
            "timestamp": float(time.time() if timestamp_sec is None else timestamp_sec),
            "top_head_chw": images.get("top_head", None),
            "hand_right_chw": images.get("hand_right", None),
            "hand_left_chw": images.get("hand_left", None),
            "top_head_raw_rgb": raw_images.get("top_head", None),
            "hand_right_raw_rgb": raw_images.get("hand_right", None),
            "hand_left_raw_rgb": raw_images.get("hand_left", None),
        }
        try:
            self._step_queue.put_nowait(record)
        except queue.Full:
            with self._stats_lock:
                self._dropped_steps += 1

        try:
            self._video_queue.put_nowait(record)
        except queue.Full:
            with self._stats_lock:
                self._dropped_video_frames += 1

    def stop(self, join_timeout: float = 10.0):
        self._stop_event.set()
        for th in self._threads:
            th.join(timeout=join_timeout)
        self._close_video_writers()

    def get_stats(self):
        with self._stats_lock:
            return {
                "output_dir": self.output_dir,
                "hdf5_path": self.hdf5_path,
                "dropped_steps": self._dropped_steps,
                "dropped_video_frames": self._dropped_video_frames,
                "written_steps": self._written_steps,
                "written_video_frames": self._written_video_frames,
            }

    def _should_exit(self):
        return self._stop_event.is_set() and self._step_queue.empty() and self._video_queue.empty()

    def _hdf5_worker(self):
        with self._h5py.File(self.hdf5_path, "w", rdcc_nbytes=2 * 1024 * 1024) as root:
            root.attrs["sim"] = False
            root.attrs["compress"] = False
            root.attrs["created_at"] = datetime.now().isoformat()
            inp = root.create_group("model_input")
            out = root.create_group("model_output")
            d_state = inp.create_dataset("state", shape=(0, 14), maxshape=(None, 14), dtype=np.float32)
            d_ts = root.create_dataset("timestamp_sec", shape=(0,), maxshape=(None,), dtype=np.float64)
            d_actions = None

            flush_every = 100
            while True:
                if self._should_exit():
                    break
                try:
                    item = self._step_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                state = np.asarray(
                    item["state"] if item["state"] is not None else np.zeros(14, dtype=np.float32),
                    dtype=np.float32,
                )
                actions = np.asarray(item["actions"], dtype=np.float32)

                idx = d_state.shape[0]
                d_state.resize(idx + 1, axis=0)
                d_ts.resize(idx + 1, axis=0)
                if d_actions is None:
                    action_shape = actions.shape
                    d_actions = out.create_dataset(
                        "actions",
                        shape=(0, *action_shape),
                        maxshape=(None, *action_shape),
                        dtype=np.float32,
                    )
                d_actions.resize(idx + 1, axis=0)

                d_state[idx] = state
                d_actions[idx] = actions
                d_ts[idx] = item["timestamp"]

                with self._stats_lock:
                    self._written_steps += 1

                if (idx + 1) % flush_every == 0:
                    root.flush()
                self._step_queue.task_done()

            root.flush()

    def _video_worker(self):
        while True:
            if self._should_exit():
                break
            try:
                item = self._video_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Raw images are treated as RGB tensors from the pre-resize path.
            top_head_raw = _to_uint8_rgb(item.get("top_head_raw_rgb"))
            hand_right_raw = _to_uint8_rgb(item.get("hand_right_raw_rgb"))
            hand_left_raw = _to_uint8_rgb(item.get("hand_left_raw_rgb"))
            frame_dict = {
                "top_head": top_head_raw if top_head_raw is not None else _payload_img_chw_to_hwc_uint8_rgb(
                    item.get("top_head_chw")
                ),
                "hand_right": hand_right_raw if hand_right_raw is not None else _payload_img_chw_to_hwc_uint8_rgb(
                    item.get("hand_right_chw")
                ),
                "hand_left": hand_left_raw if hand_left_raw is not None else _payload_img_chw_to_hwc_uint8_rgb(
                    item.get("hand_left_chw")
                ),
            }
            frame_dict = {k: v for k, v in frame_dict.items() if v is not None}
            if not frame_dict:
                self._video_queue.task_done()
                continue

            if not self._first_frame_saved and "top_head" in frame_dict:
                self._save_first_frame(frame_dict["top_head"])

            for cam, frame in frame_dict.items():
                if cam not in self._video_writers:
                    h, w = frame.shape[:2]
                    output_path = os.path.join(self.video_dir, f"{cam}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
                    writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
                    if not writer.isOpened():
                        continue
                    self._video_writers[cam] = writer
                    self._video_shapes[cam] = (h, w)

                h, w = self._video_shapes[cam]
                if frame.shape[0] != h or frame.shape[1] != w:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self._video_writers[cam].write(bgr_frame)
                with self._stats_lock:
                    self._written_video_frames += 1

            self._video_queue.task_done()

        self._close_video_writers()

    def _save_first_frame(self, frame_rgb):
        """Save the first recorded top-head image for reproducing the initial scene."""
        output_path = os.path.join(self.output_dir, "first_frame_top_head.png")
        try:
            saved = cv2.imwrite(output_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            logger.warning("could not save first frame to %s: %s", output_path, exc)
            return
        if saved:
            self._first_frame_saved = True
        else:
            logger.warning("could not save first frame to %s", output_path)

    def _close_video_writers(self):
        for writer in self._video_writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self._video_writers = {}
