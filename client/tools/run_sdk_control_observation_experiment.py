#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
INFERENCE_DIR = REPO_ROOT / "client" / "inference"
OPENPI_CLIENT_SRC = REPO_ROOT / "packages" / "openpi-client" / "src"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))
if str(OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


class JsonlExperimentLogger:
    def __init__(self):
        self.path: Path | None = None
        self.lock = threading.Lock()
        self.fp = None

    def open_for_runtime(self, runtime) -> None:
        if self.fp is not None:
            return
        self.path = Path(runtime.output_manager.episode_dir) / "experiment_events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("a", encoding="utf-8")
        self.write(
            {
                "event": "experiment_start",
                "control_backend": "sdk",
                "timestamp_sec": time.time(),
                "monotonic_sec": time.monotonic(),
                "episode_dir": str(runtime.output_manager.episode_dir),
            }
        )
        print(f"[experiment] sdk event log: {self.path}", flush=True)

    def write(self, event: dict[str, Any]) -> None:
        if self.fp is None:
            return
        item = dict(event)
        item.setdefault("timestamp_sec", time.time())
        item.setdefault("monotonic_sec", time.monotonic())
        with self.lock:
            self.fp.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")
            self.fp.flush()

    def close(self):
        if self.fp is not None:
            self.write({"event": "experiment_stop"})
            self.fp.close()
            self.fp = None


LOGGER = JsonlExperimentLogger()
_OBS_SEQ = 0


def install_patches() -> None:
    import runtime as runtime_mod
    import robot_io as robot_io_mod

    original_init = runtime_mod.InferenceRuntime.__init__
    original_observation_thread = runtime_mod.InferenceRuntime._observation_thread
    original_base_payload = runtime_mod.InferenceRuntime._base_payload
    original_log_action_step = runtime_mod.InferenceRuntime._log_action_step
    original_read_images = robot_io_mod.RealSenseRig.read_images
    original_get_observation = robot_io_mod.AgilexRobotIO.get_observation

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        LOGGER.open_for_runtime(self)

    def patched_read_images(self):
        if not self.enabled:
            self._experiment_last_camera_stamps = {}
            return {}, time.time()
        images = {}
        timestamps = []
        per_camera_stamps = {}
        per_model_stamps = {}
        for name in ("front", "right", "left"):
            frame = self.cameras[name].read()
            model_key = self.model_keys[name]
            stamp = float(frame["color_timestamp"])
            images[model_key] = frame["color_image"]
            timestamps.append(stamp)
            per_camera_stamps[name] = stamp
            per_model_stamps[model_key] = stamp
        self._experiment_last_camera_stamps = {
            "by_camera": per_camera_stamps,
            "by_model_key": per_model_stamps,
        }
        return images, min(timestamps) if timestamps else time.time()

    def patched_get_observation(self):
        obs = original_get_observation(self)
        stamps = getattr(self.rig, "_experiment_last_camera_stamps", {})
        if obs.get("image_timestamps"):
            camera_stamps = dict(obs.get("image_timestamps") or {})
            model_stamps = {
                self.rig.model_keys.get(name, name): stamp
                for name, stamp in camera_stamps.items()
            }
        else:
            camera_stamps = dict(stamps.get("by_camera", {}))
            model_stamps = dict(stamps.get("by_model_key", {}))
        obs["image_timestamps"] = model_stamps
        obs["camera_timestamps"] = camera_stamps
        obs["used_stamps"] = {
            "front_image": obs["camera_timestamps"].get("front"),
            "right_image": obs["camera_timestamps"].get("right"),
            "left_image": obs["camera_timestamps"].get("left"),
            "left_state": obs.get("state_timestamp"),
            "right_state": obs.get("state_timestamp"),
        }
        # SDK get_observation actively reads one fresh frame per camera and one
        # fresh dual-arm state, so the used messages are also the latest messages
        # visible to this inference path at packaging time.
        obs["latest_stamps"] = dict(obs["used_stamps"])
        return obs

    def patched_observation_thread(self):
        original_observation_thread(self)

    def patched_base_payload(self, obs):
        global _OBS_SEQ
        package_start = time.time()
        package_start_mono = time.monotonic()
        payload, raw_images, proprio = original_base_payload(self, obs)
        package_end = time.time()
        _OBS_SEQ += 1
        used_stamps = dict(obs.get("used_stamps") or {})
        latest_stamps = dict(obs.get("latest_stamps") or used_stamps)
        LOGGER.write(
            {
                "event": "observation_packaged",
                "control_backend": "sdk",
                "obs_id": _OBS_SEQ,
                "package_start_sec": package_start,
                "package_start_monotonic_sec": package_start_mono,
                "package_end_sec": package_end,
                "package_end_monotonic_sec": time.monotonic(),
                "state_timestamp": obs.get("state_timestamp"),
                "image_timestamp": obs.get("image_timestamp"),
                "sync_frame_time": obs.get("sync_frame_time"),
                "camera_timestamps": obs.get("camera_timestamps", {}),
                "image_timestamps": obs.get("image_timestamps", {}),
                "used_stamps": used_stamps,
                "latest_stamps": latest_stamps,
                "state": np.asarray(proprio, dtype=float),
            }
        )
        LOGGER.write(
            {
                "event": "model_observation_timing",
                "control_backend": "sdk",
                "obs_id": _OBS_SEQ,
                "infer_request_start_sec": package_end,
                "infer_request_start_monotonic_sec": time.monotonic(),
                "package_latency_sec": package_end - package_start,
                "used_stamps": used_stamps,
                "latest_stamps": latest_stamps,
            }
        )
        return payload, raw_images, proprio

    def patched_log_action_step(self, action, action_step):
        original_log_action_step(self, action, action_step)
        LOGGER.write(
            {
                "event": "action_publish",
                "control_backend": "sdk",
                "chunk_id": int(action_step.get("chunk_id", -1)),
                "chunk_step_index": int(action_step.get("chunk_step_index", -1)),
                "action": np.asarray(action, dtype=float),
            }
        )

    runtime_mod.InferenceRuntime.__init__ = patched_init
    runtime_mod.InferenceRuntime._observation_thread = patched_observation_thread
    runtime_mod.InferenceRuntime._base_payload = patched_base_payload
    runtime_mod.InferenceRuntime._log_action_step = patched_log_action_step
    robot_io_mod.RealSenseRig.read_images = patched_read_images
    robot_io_mod.AgilexRobotIO.get_observation = patched_get_observation


def main() -> int:
    install_patches()
    try:
        import agilex_inference_openpi

        agilex_inference_openpi.main()
    finally:
        LOGGER.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
