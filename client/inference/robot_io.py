from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any
import math

from config import camera_serial
import cv2
import numpy as np

try:
    import polars as pl
except Exception:  # pragma: no cover - optional local-test dependency
    pl = None

try:
    import pyrealsense2 as rs
except Exception:  # pragma: no cover - hardware dependency
    rs = None

try:
    from piper_sdk import C_PiperInterface

    try:
        from piper_sdk import C_PiperInterface_V2
    except Exception:  # pragma: no cover - SDK version dependency
        C_PiperInterface_V2 = C_PiperInterface
except Exception:  # pragma: no cover - hardware dependency
    C_PiperInterface = None
    C_PiperInterface_V2 = None

try:
    import polars as pl
except Exception:  # pragma: no cover - optional mock dependency
    pl = None


logger = logging.getLogger(__name__)

JOINT_RAW_TO_RAD = 0.017444 / 1000.0
JOINT_RAD_TO_RAW = 57324.840764
_HIGH_FOLLOW_ARM_JOINT_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    dtype=np.int64,
)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _project_tangents_monotone(
    start_point: np.ndarray,
    target_point: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    indices: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    projected_m0 = np.array(m0, dtype=float, copy=True)
    projected_m1 = np.array(m1, dtype=float, copy=True)
    start = np.asarray(start_point, dtype=float).reshape(-1)
    target = np.asarray(target_point, dtype=float).reshape(-1)

    for idx in np.asarray(indices, dtype=np.int64).reshape(-1):
        d = float(target[idx] - start[idx])
        if abs(d) < eps:
            projected_m0[idx] = 0.0
            projected_m1[idx] = 0.0
            continue

        r0 = max(0.0, float(projected_m0[idx]) / d)
        r1 = max(0.0, float(projected_m1[idx]) / d)
        tangent_sum = r0 + r1
        if tangent_sum > 3.0:
            scale = 3.0 / tangent_sum
            r0 *= scale
            r1 *= scale

        projected_m0[idx] = r0 * d
        projected_m1[idx] = r1 * d

    return projected_m0, projected_m1


def _array_from_config(value: Any, default: list[float], *, size: int, name: str) -> np.ndarray:
    arr = np.asarray(default if value is None else value, dtype=float).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} must contain {size} values, got {arr.size}")
    return arr


class RealSenseCamera:
    def __init__(self, name: str, serial: str, width: int, height: int, fps: int, *, enable_depth: bool = False):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not available")
        self.name = name
        self.serial = camera_serial(serial)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.enable_depth = bool(enable_depth)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._lock = threading.Lock()
        self._frames = deque(maxlen=2000)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self.pipeline.start(self.config)
        self._running = True

        def _loop() -> None:
            while self._running:
                try:
                    frame = self.pipeline.wait_for_frames(timeout_ms=2000)
                except Exception as exc:
                    logger.warning("camera %s frame read failed: %s", self.name, exc)
                    continue
                with self._lock:
                    self._frames.append(frame)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def read(self, timeout_s: float = 2.0) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout_s)
        frame = None
        while time.monotonic() < deadline:
            with self._lock:
                if self._frames:
                    frame = self._frames.pop()
                    self._frames.clear()
            if frame is not None:
                break
            time.sleep(0.002)
        if frame is None:
            raise TimeoutError(f"camera {self.name} produced no frame")
        return self._convert_frame(frame)

    @staticmethod
    def _frame_timestamp(frame) -> float | None:
        color_frame = frame.get_color_frame()
        if color_frame is None:
            return None
        return float(color_frame.get_timestamp() / 1000.0)

    def latest_timestamp(self) -> float | None:
        with self._lock:
            if not self._frames:
                return None
            return self._frame_timestamp(self._frames[-1])

    def read_at_or_after(self, frame_time: float, timeout_s: float = 2.0) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout_s)
        frame = None
        while time.monotonic() < deadline:
            with self._lock:
                while self._frames:
                    stamp = self._frame_timestamp(self._frames[0])
                    if stamp is None or stamp < frame_time:
                        self._frames.popleft()
                        continue
                    frame = self._frames.popleft()
                    break
            if frame is not None:
                break
            time.sleep(0.002)
        if frame is None:
            raise TimeoutError(f"camera {self.name} produced no synced frame")
        return self._convert_frame(frame)

    def _convert_frame(self, frame) -> dict[str, Any]:
        color_frame = frame.get_color_frame()
        if color_frame is None:
            raise RuntimeError(f"camera {self.name} color frame is missing")
        color = np.asanyarray(color_frame.get_data())
        result = {
            "color_image": color,
            "color_timestamp": float(color_frame.get_timestamp() / 1000.0),
        }
        if self.enable_depth:
            depth_frame = frame.get_depth_frame()
            if depth_frame is not None:
                result["depth_image"] = np.asanyarray(depth_frame.get_data())
                result["depth_timestamp"] = float(depth_frame.get_timestamp() / 1000.0)
        return result

    def close(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.pipeline.stop()


class RealSenseRig:
    def __init__(self, camera_cfg: dict[str, Any]):
        self.enabled = bool(camera_cfg.get("enable", True))
        self.width = int(camera_cfg.get("width", 640))
        self.height = int(camera_cfg.get("height", 480))
        self.fps = int(camera_cfg.get("fps", 60))
        self.use_depth = bool(camera_cfg.get("use_depth", False))
        self.cameras: dict[str, RealSenseCamera] = {}
        self.model_keys: dict[str, str] = {}
        if self.enabled:
            for name in ("front", "right", "left"):
                cfg = camera_cfg.get(name) or {}
                serial = cfg.get("serial", "")
                if not serial:
                    raise ValueError(f"camera.{name}.serial is required")
                self.model_keys[name] = str(cfg.get("model_key", name))
                self.cameras[name] = RealSenseCamera(
                    name=name,
                    serial=serial,
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    enable_depth=self.use_depth,
                )

    def start(self) -> None:
        for camera in self.cameras.values():
            camera.start()

    def read_images(self) -> tuple[dict[str, np.ndarray], float]:
        if not self.enabled:
            return {}, time.time()
        images: dict[str, np.ndarray] = {}
        timestamps = []
        for name in ("front", "right", "left"):
            frame = self.cameras[name].read()
            images[self.model_keys[name]] = frame["color_image"]
            timestamps.append(float(frame["color_timestamp"]))
        return images, min(timestamps) if timestamps else time.time()

    def latest_frame_time(self) -> float | None:
        if not self.enabled:
            return time.time()
        timestamps = []
        for name in ("front", "right", "left"):
            stamp = self.cameras[name].latest_timestamp()
            if stamp is None:
                return None
            timestamps.append(stamp)
        return min(timestamps)

    def read_images_at_or_after(self, frame_time: float) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        if not self.enabled:
            return {}, {}
        images: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}
        for name in ("front", "right", "left"):
            frame = self.cameras[name].read_at_or_after(frame_time)
            images[self.model_keys[name]] = frame["color_image"]
            timestamps[name] = float(frame["color_timestamp"])
        return images, timestamps

    def close(self) -> None:
        for camera in self.cameras.values():
            camera.close()


class PiperArm:
    def __init__(
        self,
        can_name: str,
        auto_enable: bool = True,
        gripper_exist: bool = True,
        gripper_multiple: float = 1.0,
        prefer_v2: bool = True,
    ):
        iface_cls = C_PiperInterface_V2 if prefer_v2 and C_PiperInterface_V2 is not None else C_PiperInterface
        if iface_cls is None:
            raise RuntimeError("piper_sdk is not available")
        self.can_name = str(can_name)
        self.auto_enable = bool(auto_enable)
        self.gripper_exist = bool(gripper_exist)
        self.gripper_multiple = float(gripper_multiple)
        self._piper = iface_cls(can_name=self.can_name)
        self._lock = threading.Lock()
        self._enabled = False
        self._high_follow_mode = False

    def connect(self) -> None:
        self._piper.ConnectPort()
        if self.auto_enable:
            self.enable(timeout_s=5.0)

    def is_enabled(self) -> bool:
        low = self._piper.GetArmLowSpdInfoMsgs()
        return bool(
            low.motor_1.foc_status.driver_enable_status
            and low.motor_2.foc_status.driver_enable_status
            and low.motor_3.foc_status.driver_enable_status
            and low.motor_4.foc_status.driver_enable_status
            and low.motor_5.foc_status.driver_enable_status
            and low.motor_6.foc_status.driver_enable_status
        )

    def enable(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.is_enabled():
                self._enabled = True
                return
            self._piper.EnableArm(7)
            if self.gripper_exist:
                self._piper.GripperCtrl(0, 1000, 0x01, 0)
            time.sleep(0.5)
        self._enabled = self.is_enabled()
        if not self._enabled:
            logger.warning("arm %s auto-enable timed out", self.can_name)

    @staticmethod
    def _msg_timestamp(msg) -> float | None:
        stamp = getattr(msg, "time_stamp", None)
        if stamp is None:
            return None
        try:
            stamp = float(stamp)
        except (TypeError, ValueError):
            return None
        return stamp if stamp > 0 else None

    def read_state(self) -> tuple[np.ndarray, float]:
        joint_msg = self._piper.GetArmJointMsgs()
        high_spd_msg = self._piper.GetArmHighSpdInfoMsgs()
        gripper_msg = self._piper.GetArmGripperMsgs()
        joint = joint_msg.joint_state
        gripper = gripper_msg.gripper_state
        state = np.asarray(
            [
                joint.joint_1 * JOINT_RAW_TO_RAD,
                joint.joint_2 * JOINT_RAW_TO_RAD,
                joint.joint_3 * JOINT_RAW_TO_RAD,
                joint.joint_4 * JOINT_RAW_TO_RAD,
                joint.joint_5 * JOINT_RAW_TO_RAD,
                joint.joint_6 * JOINT_RAW_TO_RAD,
                gripper.grippers_angle / 1000000.0,
            ],
            dtype=float,
        )
        stamps = [self._msg_timestamp(joint_msg), self._msg_timestamp(high_spd_msg)]
        stamp = max((s for s in stamps if s is not None), default=time.time())
        return state, stamp

    def apply(self, action7: np.ndarray, velocity: float = 100.0, gripper_effort: float = 1000.0) -> None:
        action = np.asarray(action7, dtype=float).reshape(-1)
        if action.size < 7:
            raise ValueError(f"arm action must have 7 values, got {action.size}")
        joints = [round(float(v) * JOINT_RAD_TO_RAW) for v in action[:6]]
        gripper = round(abs(float(action[6])) * 1000 * 1000 * self.gripper_multiple)
        effort = round(_clip(float(gripper_effort), 0.5, 3000.0))
        speed = int(_clip(float(velocity), 1.0, 100.0))
        with self._lock:
            if self.auto_enable and not self._enabled:
                self.enable(timeout_s=1.0)
            self._piper.MotionCtrl_2(0x01, 0x01, speed)
            self._piper.JointCtrl(*joints)
            if self.gripper_exist:
                self._piper.GripperCtrl(gripper, effort, 0x01, 0)
            self._high_follow_mode = False

    def enter_high_follow_mode(self) -> None:
        motion_ctrl = getattr(self._piper, "MotionCtrl_2", None)
        if not callable(motion_ctrl):
            raise RuntimeError("piper_sdk interface does not provide MotionCtrl_2 for high follow mode")
        with self._lock:
            if self.auto_enable and not self._enabled:
                self.enable(timeout_s=1.0)
            motion_ctrl(0x01, 0x01, 100, 0xAD)
            self._high_follow_mode = True

    def apply_high_follow(
        self,
        q_ref: np.ndarray,
        gripper: float,
        gripper_effort: float = 1000.0,
    ) -> None:
        q = np.asarray(q_ref, dtype=float).reshape(-1)
        if q.size < 6:
            raise ValueError("high follow command requires 6 joint values")
        joints = [round(float(v) * JOINT_RAD_TO_RAW) for v in q[:6]]
        gripper_raw = round(abs(float(gripper)) * 1000 * 1000 * self.gripper_multiple)
        effort = round(_clip(float(gripper_effort), 0.5, 3000.0))
        with self._lock:
            if not self._high_follow_mode:
                self._piper.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
                self._high_follow_mode = True
            self._piper.JointCtrl(*joints)
            if self.gripper_exist:
                self._piper.GripperCtrl(gripper_raw, effort, 0x01, 0)

    def close(self) -> None:
        close = getattr(self._piper, "DisconnectPort", None)
        if callable(close):
            close()


@dataclass
class PiperHighFollowConfig:
    enabled: bool = False
    control_hz: float = 200.0
    max_queue_size: int = 32
    max_joint_vel: np.ndarray = None  # type: ignore[assignment]
    interpolator: str = "linear"
    tangent_mode: str = "uniform"
    tangent_alpha: float = 0.5
    monotone_projection: bool = False
    monotone_projection_dims: str = "arm"
    duration_scale: float = 1.0
    min_duration_s: float = 0.0
    max_duration_s: float = 0.0

    @classmethod
    def from_config(cls, arm_cfg: dict[str, Any]) -> "PiperHighFollowConfig":
        mode = str(arm_cfg.get("control_mode", arm_cfg.get("mode", "position"))).lower()
        if mode not in {"position", "high_follow"}:
            raise ValueError(f"unsupported arm.control_mode={mode!r}; expected 'position' or 'high_follow'")
        high_follow_cfg = arm_cfg.get("high_follow") or {}
        interpolator_cfg = high_follow_cfg.get("interpolator") or {}
        interpolator = str(interpolator_cfg.get("type", "linear")).lower()
        if interpolator not in {"linear", "waypoint_cubic"}:
            raise ValueError(
                "unsupported arm.high_follow.interpolator.type="
                f"{interpolator!r}; expected 'linear' or 'waypoint_cubic'"
            )
        tangent_mode = str(interpolator_cfg.get("tangent_mode", "uniform")).lower()
        if tangent_mode not in {"uniform", "centripetal"}:
            raise ValueError(
                "unsupported arm.high_follow.interpolator.tangent_mode="
                f"{tangent_mode!r}; expected 'uniform' or 'centripetal'"
            )
        monotone_projection_dims = str(interpolator_cfg.get("monotone_projection_dims", "arm")).lower()
        if monotone_projection_dims not in {"arm", "all"}:
            raise ValueError(
                "unsupported arm.high_follow.interpolator.monotone_projection_dims="
                f"{monotone_projection_dims!r}; expected 'arm' or 'all'"
            )
        max_joint_vel = _array_from_config(
            high_follow_cfg.get("max_joint_vel"),
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            size=6,
            name="arm.high_follow.max_joint_vel",
        )
        if np.any(max_joint_vel > 3.0):
            raise ValueError("arm.high_follow.max_joint_vel must not exceed the official Piper 3.0 rad/s limit")
        return cls(
            enabled=mode == "high_follow",
            control_hz=float(high_follow_cfg.get("control_hz", 200.0)),
            max_queue_size=max(1, int(high_follow_cfg.get("max_queue_size", 32))),
            max_joint_vel=max_joint_vel,
            interpolator=interpolator,
            tangent_mode=tangent_mode,
            tangent_alpha=float(interpolator_cfg.get("tangent_alpha", 0.5)),
            monotone_projection=bool(interpolator_cfg.get("monotone_projection", False)),
            monotone_projection_dims=monotone_projection_dims,
            duration_scale=max(1e-6, float(interpolator_cfg.get("duration_scale", 1.0))),
            min_duration_s=max(0.0, float(interpolator_cfg.get("min_duration_s", 0.0))),
            max_duration_s=max(0.0, float(interpolator_cfg.get("max_duration_s", 0.0))),
        )


@dataclass
class PiperDualArm:
    left: PiperArm
    right: PiperArm
    high_follow_config: PiperHighFollowConfig
    right_offset: float = 0.0
    command_velocity: float = 100.0
    gripper_effort: float = 1000.0
    state_poll_hz: float = 200.0
    telemetry: Any | None = None
    runtime_logger: Any | None = None

    @classmethod
    def from_config(
        cls,
        arm_cfg: dict[str, Any],
        can_cfg: dict[str, Any],
        telemetry: Any | None = None,
        runtime_logger: Any | None = None,
    ) -> "PiperDualArm":
        left_can = (can_cfg.get("left") or {}).get("name", "can_left_slave")
        right_can = (can_cfg.get("right") or {}).get("name", "can_right_slave")
        auto_enable = bool(arm_cfg.get("auto_enable", True))
        gripper_exist = bool(arm_cfg.get("gripper_exist", True))
        gripper_multiple = float(arm_cfg.get("gripper_multiple", 1.0))
        high_follow_config = PiperHighFollowConfig.from_config(arm_cfg)
        return cls(
            left=PiperArm(
                left_can,
                auto_enable=auto_enable,
                gripper_exist=gripper_exist,
                gripper_multiple=gripper_multiple,
                prefer_v2=high_follow_config.enabled,
            ),
            right=PiperArm(
                right_can,
                auto_enable=auto_enable,
                gripper_exist=gripper_exist,
                gripper_multiple=gripper_multiple,
                prefer_v2=high_follow_config.enabled,
            ),
            high_follow_config=high_follow_config,
            right_offset=float(arm_cfg.get("right_offset", 0.003)),
            command_velocity=float(arm_cfg.get("command_velocity", 100.0)),
            gripper_effort=float(arm_cfg.get("gripper_effort", 1000.0)),
            state_poll_hz=float(arm_cfg.get("state_poll_hz", arm_cfg.get("state_rate", 200.0))),
            telemetry=telemetry,
            runtime_logger=runtime_logger,
        )

    def __post_init__(self) -> None:
        self._state_lock = threading.Lock()
        self._left_state_deque = deque(maxlen=2000)
        self._right_state_deque = deque(maxlen=2000)
        self._state_running = False
        self._state_thread: threading.Thread | None = None
        self._high_follow_lock = threading.Lock()
        self._high_follow_running = False
        self._high_follow_thread: threading.Thread | None = None
        self._high_follow_queue: deque[np.ndarray] = deque(maxlen=max(1, int(self.high_follow_config.max_queue_size)))
        self._high_follow_metadata_queue: deque[dict[str, Any] | None] = deque(
            maxlen=max(1, int(self.high_follow_config.max_queue_size))
        )
        self._high_follow_last_ref: np.ndarray | None = None
        self._high_follow_prev_waypoint: np.ndarray | None = None
        self._high_follow_anchor_waypoint: np.ndarray | None = None
        self._high_follow_segment_mode: str | None = None
        self._high_follow_segment_metadata: dict[str, Any] | None = None
        self._high_follow_segment_prev: np.ndarray | None = None
        self._high_follow_segment_start: np.ndarray | None = None
        self._high_follow_segment_target: np.ndarray | None = None
        self._high_follow_segment_next: np.ndarray | None = None
        self._high_follow_segment_steps: int = 0
        self._high_follow_segment_index: int = 0

    def connect(self) -> None:
        self.left.connect()
        self.right.connect()
        if self.high_follow_config.enabled:
            self.left.enter_high_follow_mode()
            self.right.enter_high_follow_mode()
            logger.info(
                "high follow control enabled: sdk=JointCtrl control_hz=%.1f interpolator=%s",
                self.high_follow_config.control_hz,
                self.high_follow_config.interpolator,
            )
            self.start_high_follow_control()

    def set_runtime_logger(self, runtime_logger: Any | None) -> None:
        self.runtime_logger = runtime_logger

    def start_state_stream(self) -> None:
        if self._state_running:
            return
        self._state_running = True

        def _loop() -> None:
            period = 1.0 / max(float(self.state_poll_hz), 1e-6)
            next_t = time.monotonic()
            while self._state_running:
                try:
                    left_state, left_stamp = self.left.read_state()
                    right_state, right_stamp = self.right.read_state()
                    state_monotonic = time.monotonic()
                    with self._state_lock:
                        self._left_state_deque.append((left_stamp, left_state))
                        self._right_state_deque.append((right_stamp, right_state))
                    if self.telemetry is not None:
                        state = np.concatenate([left_state, right_state], axis=0)
                        self.telemetry.publish(
                            "state_200hz",
                            state,
                            monotonic_sec=state_monotonic,
                        )
                except Exception as exc:
                    logger.warning("arm state read failed: %s", exc)
                    time.sleep(0.005)
                next_t += period
                sleep_s = next_t - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.monotonic()

        self._state_thread = threading.Thread(target=_loop, daemon=True)
        self._state_thread.start()

    def get_state(self) -> tuple[np.ndarray, float]:
        left_state, left_stamp = self.left.read_state()
        right_state, right_stamp = self.right.read_state()
        state = np.concatenate([left_state, right_state], axis=0)
        return state, max(left_stamp, right_stamp)

    def get_synced_state(self, frame_time: float, timeout_s: float = 2.0) -> tuple[np.ndarray, float]:
        deadline = time.monotonic() + float(timeout_s)
        selected_left: tuple[float, np.ndarray] | None = None
        selected_right: tuple[float, np.ndarray] | None = None
        while time.monotonic() < deadline:
            with self._state_lock:
                while self._left_state_deque:
                    stamp, state = self._left_state_deque[0]
                    if stamp < frame_time:
                        self._left_state_deque.popleft()
                        continue
                    selected_left = self._left_state_deque.popleft()
                    break
                while self._right_state_deque:
                    stamp, state = self._right_state_deque[0]
                    if stamp < frame_time:
                        self._right_state_deque.popleft()
                        continue
                    selected_right = self._right_state_deque.popleft()
                    break
            if selected_left is not None and selected_right is not None:
                state = np.concatenate([selected_left[1], selected_right[1]], axis=0)
                return state.copy(), max(float(selected_left[0]), float(selected_right[0]))
            time.sleep(0.002)
        raise TimeoutError("arm state stream produced no synced state")

    def apply_action(self, action14: np.ndarray, action_step: dict[str, Any] | None = None) -> None:
        action = np.asarray(action14, dtype=float).reshape(-1)
        if action.size < 14:
            raise ValueError(f"dual arm action must have 14 values, got {action.size}")
        left_action = action[:7].copy()
        right_action = action[7:14].copy()
        if left_action[6]<self.right_offset:
            left_action[6]=0
        if right_action[6]<self.right_offset:
            right_action[6]=0
        if self.high_follow_config.enabled:
            self._enqueue_high_follow_action(
                np.concatenate([left_action, right_action], axis=0),
                replace=False,
                metadata=self._high_follow_metadata_from_action_step(action_step),
            )
            return
        self.left.apply(left_action, velocity=self.command_velocity, gripper_effort=self.gripper_effort)
        self.right.apply(right_action, velocity=self.command_velocity, gripper_effort=self.gripper_effort)

    def start_high_follow_control(self) -> None:
        if self._high_follow_running:
            return
        self._high_follow_running = True
        self._high_follow_thread = threading.Thread(target=self._high_follow_control_loop, daemon=True)
        self._high_follow_thread.start()

    def hold_current_position(self) -> None:
        """Discard pending waypoints and hold the latest commanded pose."""
        if not self.high_follow_config.enabled:
            return
        self._high_follow_running = False
        thread = self._high_follow_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
            if thread.is_alive():
                raise RuntimeError("high-follow worker did not stop; cannot reset pending waypoints")
        self._high_follow_thread = None
        with self._high_follow_lock:
            hold_ref = None if self._high_follow_last_ref is None else self._high_follow_last_ref.copy()
            self._high_follow_queue.clear()
            self._high_follow_metadata_queue.clear()
            self._reset_high_follow_segment_locked(reset_anchor=False)
            self._high_follow_prev_waypoint = None
            self._high_follow_anchor_waypoint = hold_ref
        self.start_high_follow_control()
        logger.info("high follow queue cleared; holding current commanded position")

    def _current_high_follow_ref(self) -> np.ndarray:
        with self._high_follow_lock:
            if self._high_follow_last_ref is not None:
                return self._high_follow_last_ref.copy()
        try:
            state, _ = self.get_state()
            return np.asarray(state[:14], dtype=float).copy()
        except Exception as exc:
            logger.warning("high follow state fallback failed: %s", exc)
            return np.zeros(14, dtype=float)

    @staticmethod
    def _high_follow_metadata_from_action_step(action_step: dict[str, Any] | None) -> dict[str, Any] | None:
        if action_step is None:
            return None
        metadata: dict[str, Any] = {}
        if "chunk_id" in action_step:
            metadata["chunk_id"] = int(action_step["chunk_id"])
        if "chunk_step_index" in action_step:
            metadata["chunk_step_index"] = int(action_step["chunk_step_index"])
        return metadata or None

    def _enqueue_high_follow_action(
        self,
        action14: np.ndarray,
        *,
        replace: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        action = np.asarray(action14, dtype=float).reshape(-1)
        if action.size < 14:
            raise ValueError(f"high follow action must have 14 values, got {action.size}")
        with self._high_follow_lock:
            if replace:
                self._high_follow_queue.clear()
                self._high_follow_metadata_queue.clear()
                self._reset_high_follow_segment_locked(reset_anchor=True)
            self._high_follow_queue.append(action[:14].copy())
            self._high_follow_metadata_queue.append(None if metadata is None else dict(metadata))

    def _reset_high_follow_segment_locked(self, *, reset_anchor: bool = False) -> None:
        if reset_anchor:
            self._high_follow_prev_waypoint = None
            self._high_follow_anchor_waypoint = (
                None if self._high_follow_last_ref is None else self._high_follow_last_ref.copy()
            )
        self._high_follow_segment_mode = None
        self._high_follow_segment_metadata = None
        self._high_follow_segment_prev = None
        self._high_follow_segment_target = None
        self._high_follow_segment_start = None
        self._high_follow_segment_next = None
        self._high_follow_segment_steps = 0
        self._high_follow_segment_index = 0

    def _compute_linear_steps(self, start: np.ndarray, target: np.ndarray) -> int:
        cfg = self.high_follow_config
        max_delta_per_step = np.maximum(cfg.max_joint_vel, 1e-6) / max(float(cfg.control_hz), 1e-6)
        current = np.asarray(start, dtype=float).reshape(-1)[:14]
        target = np.asarray(target, dtype=float).reshape(-1)[:14]
        diff = target - current
        left_ratio = np.abs(diff[:6]) / max_delta_per_step
        right_ratio = np.abs(diff[7:13]) / max_delta_per_step
        max_ratio = float(max(np.max(left_ratio), np.max(right_ratio), 1.0))
        return max(1, int(math.ceil(max_ratio)))

    def _segment_duration_steps_from_seconds(self, duration_s: float) -> int:
        cfg = self.high_follow_config
        duration_s = max(float(duration_s) * float(cfg.duration_scale), float(cfg.min_duration_s))
        if cfg.max_duration_s > 0:
            duration_s = min(duration_s, float(cfg.max_duration_s))
        return max(1, int(math.ceil(duration_s * float(cfg.control_hz))))

    @staticmethod
    def _hermite_position(s: float, p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray) -> np.ndarray:
        h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h10 = s**3 - 2.0 * s**2 + s
        h01 = -2.0 * s**3 + 3.0 * s**2
        h11 = s**3 - s**2
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    @staticmethod
    def _hermite_derivative(s: float, p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray) -> np.ndarray:
        dh00 = 6.0 * s**2 - 6.0 * s
        dh10 = 3.0 * s**2 - 4.0 * s + 1.0
        dh01 = -6.0 * s**2 + 6.0 * s
        dh11 = 3.0 * s**2 - 2.0 * s
        return dh00 * p0 + dh10 * m0 + dh01 * p1 + dh11 * m1

    @staticmethod
    def _high_follow_arm_distance(a: np.ndarray, b: np.ndarray) -> float:
        delta = a[_HIGH_FOLLOW_ARM_JOINT_INDICES] - b[_HIGH_FOLLOW_ARM_JOINT_INDICES]
        return float(np.linalg.norm(delta))

    def _segment_tangents(
        self,
        prev_point: np.ndarray | None,
        start_point: np.ndarray,
        target_point: np.ndarray,
        next_point: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        baseline = target_point - start_point
        cfg = self.high_follow_config
        if prev_point is None:
            m0 = baseline.copy()
        elif cfg.tangent_mode == "centripetal":
            l_prev = self._high_follow_arm_distance(start_point, prev_point)
            l_curr = self._high_follow_arm_distance(target_point, start_point)
            a = max(l_prev, 1e-8) ** float(cfg.tangent_alpha)
            b = max(l_curr, 1e-8) ** float(cfg.tangent_alpha)
            m0 = (b / (a + b)) * (target_point - prev_point)
        else:
            m0 = 0.5 * (target_point - prev_point)

        if next_point is None:
            m1 = baseline.copy()
        elif cfg.tangent_mode == "centripetal":
            l_curr = self._high_follow_arm_distance(target_point, start_point)
            l_next = self._high_follow_arm_distance(next_point, target_point)
            b = max(l_curr, 1e-8) ** float(cfg.tangent_alpha)
            c = max(l_next, 1e-8) ** float(cfg.tangent_alpha)
            m1 = (b / (b + c)) * (next_point - start_point)
        else:
            m1 = 0.5 * (next_point - start_point)
        if cfg.monotone_projection:
            if cfg.monotone_projection_dims == "arm":
                indices = _HIGH_FOLLOW_ARM_JOINT_INDICES
            else:
                indices = np.arange(len(start_point), dtype=np.int64)
            m0, m1 = _project_tangents_monotone(start_point, target_point, m0, m1, indices)
        return m0, m1

    def _compute_waypoint_cubic_steps(
        self,
        prev_point: np.ndarray | None,
        start_point: np.ndarray,
        target_point: np.ndarray,
        next_point: np.ndarray | None,
    ) -> int:
        cfg = self.high_follow_config
        m0, m1 = self._segment_tangents(prev_point, start_point, target_point, next_point)
        samples = np.linspace(0.0, 1.0, num=21, dtype=float)
        max_dqds = np.zeros(12, dtype=float)
        for s in samples:
            dqds = self._hermite_derivative(float(s), start_point, target_point, m0, m1)
            joint_dqds = np.concatenate([dqds[:6], dqds[7:13]], axis=0)
            max_dqds = np.maximum(max_dqds, np.abs(joint_dqds))
        joint_vel_limit = np.concatenate([cfg.max_joint_vel, cfg.max_joint_vel], axis=0)
        min_duration_s = float(np.max(max_dqds / np.maximum(joint_vel_limit, 1e-6)))
        return self._segment_duration_steps_from_seconds(min_duration_s)

    def _build_high_follow_segment_locked(self, current_ref: np.ndarray) -> bool:
        if self._high_follow_segment_target is not None or not self._high_follow_queue:
            return self._high_follow_segment_target is not None
        anchor = current_ref.copy() if self._high_follow_anchor_waypoint is None else self._high_follow_anchor_waypoint.copy()
        prev_point = None if self._high_follow_prev_waypoint is None else self._high_follow_prev_waypoint.copy()
        target = self._high_follow_queue[0].copy()
        metadata = self._high_follow_metadata_queue[0] if self._high_follow_metadata_queue else None
        next_point = self._high_follow_queue[1].copy() if len(self._high_follow_queue) > 1 else None
        mode = self.high_follow_config.interpolator
        steps = self._compute_linear_steps(anchor, target)
        if mode == "waypoint_cubic":
            if prev_point is not None and next_point is not None:
                steps = self._compute_waypoint_cubic_steps(prev_point, anchor, target, next_point)
            else:
                mode = "linear"
        self._high_follow_anchor_waypoint = anchor.copy()
        self._high_follow_segment_mode = mode
        self._high_follow_segment_metadata = None if metadata is None else dict(metadata)
        self._high_follow_segment_prev = prev_point
        self._high_follow_segment_start = anchor
        self._high_follow_segment_target = target
        self._high_follow_segment_next = next_point
        self._high_follow_segment_steps = max(1, int(steps))
        self._high_follow_segment_index = 0
        return True

    def _step_high_follow_ref(self, current: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool, dict[str, Any]]:
        current = np.asarray(current, dtype=float).reshape(-1)[:14]
        target = np.asarray(target, dtype=float).reshape(-1)[:14]
        with self._high_follow_lock:
            if not self._build_high_follow_segment_locked(current):
                return current.copy(), False, {}
            mode = str(self._high_follow_segment_mode or "linear")
            metadata = None if self._high_follow_segment_metadata is None else dict(self._high_follow_segment_metadata)
            start = self._high_follow_segment_start.copy()
            prev_point = None if self._high_follow_segment_prev is None else self._high_follow_segment_prev.copy()
            next_point = None if self._high_follow_segment_next is None else self._high_follow_segment_next.copy()
            total_steps = max(1, int(self._high_follow_segment_steps))
            step_index = self._high_follow_segment_index + 1
            self._high_follow_segment_index = min(step_index, total_steps)
        phase = float(step_index) / float(total_steps)
        if mode == "waypoint_cubic":
            m0, m1 = self._segment_tangents(prev_point, start, target, next_point)
            next_ref = self._hermite_position(phase, start, target, m0, m1)
        else:
            alpha = _clip(phase, 0.0, 1.0)
            next_ref = start + (target - start) * alpha
        reached = step_index >= total_steps
        if reached:
            next_ref = target.copy()
            with self._high_follow_lock:
                completed_start = self._high_follow_segment_start.copy()
                completed_target = self._high_follow_segment_target.copy()
                self._high_follow_prev_waypoint = completed_start
                self._high_follow_anchor_waypoint = completed_target
                self._reset_high_follow_segment_locked()
        return next_ref, reached, {
            "metadata": metadata,
            "step_index": step_index,
            "segment_steps": total_steps,
            "phase": phase,
            "interpolator": mode,
        }

    def _log_high_follow_command(
        self,
        command: np.ndarray,
        timestamp_sec: float,
        monotonic_sec: float,
        reached: bool,
        step_info: dict[str, Any],
    ) -> None:
        if self.runtime_logger is None:
            return
        log_command = getattr(self.runtime_logger, "log_high_follow_command", None)
        if not callable(log_command):
            return
        values = np.asarray(command, dtype=float).reshape(-1)
        if values.size < 14:
            return
        metadata = step_info.get("metadata") or {}
        row = {
            "timestamp_sec": float(timestamp_sec),
            "monotonic_sec": float(monotonic_sec),
            "chunk_id": metadata.get("chunk_id", ""),
            "chunk_step_index": metadata.get("chunk_step_index", ""),
            "left_j1": float(values[0]),
            "left_j2": float(values[1]),
            "left_j3": float(values[2]),
            "left_j4": float(values[3]),
            "left_j5": float(values[4]),
            "left_j6": float(values[5]),
            "left_gripper": float(values[6]),
            "right_j1": float(values[7]),
            "right_j2": float(values[8]),
            "right_j3": float(values[9]),
            "right_j4": float(values[10]),
            "right_j5": float(values[11]),
            "right_j6": float(values[12]),
            "right_gripper": float(values[13]),
            "high_follow_step_index": step_info.get("step_index", ""),
            "high_follow_segment_steps": step_info.get("segment_steps", ""),
            "high_follow_phase": step_info.get("phase", ""),
            "high_follow_interpolator": step_info.get("interpolator", ""),
            "high_follow_reached": bool(reached),
        }
        log_command(row)

    def _high_follow_control_loop(self) -> None:
        cfg = self.high_follow_config
        period = 1.0 / max(float(cfg.control_hz), 1e-6)
        next_t = time.monotonic()
        while self._high_follow_running:
            current_ref = self._current_high_follow_ref()
            with self._high_follow_lock:
                target = self._high_follow_queue[0].copy() if self._high_follow_queue else None
            if target is None:
                next_ref = current_ref
                reached = False
                step_info: dict[str, Any] = {}
            else:
                next_ref, reached, step_info = self._step_high_follow_ref(current_ref, target)
            try:
                command_monotonic = time.monotonic()
                command_timestamp = time.time()
                self.left.apply_high_follow(
                    q_ref=next_ref[:6],
                    gripper=float(next_ref[6]),
                    gripper_effort=self.gripper_effort,
                )
                self.right.apply_high_follow(
                    q_ref=next_ref[7:13],
                    gripper=float(next_ref[13]),
                    gripper_effort=self.gripper_effort,
                )
                with self._high_follow_lock:
                    self._high_follow_last_ref = next_ref.copy()
                    if reached and self._high_follow_queue:
                        self._high_follow_queue.popleft()
                        if self._high_follow_metadata_queue:
                            self._high_follow_metadata_queue.popleft()
                if self.telemetry is not None:
                    self.telemetry.publish("cmd_high_follow_200hz", next_ref, monotonic_sec=command_monotonic)
                if step_info:
                    self._log_high_follow_command(next_ref, command_timestamp, command_monotonic, reached, step_info)
            except Exception as exc:
                logger.warning("high follow command failed: %s", exc)
                time.sleep(0.01)
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()

    def move_to(self, left: list[float], right: list[float], steps: int = 100, sleep_s: float = 0.01) -> None:
        if self.high_follow_config.enabled:
            self.move_to_high_follow(left=left, right=right, timeout_s=max(2.0, int(steps) * float(sleep_s) + 1.0))
            return
        current, _ = self.get_state()
        target = np.concatenate([np.asarray(left, dtype=float), np.asarray(right, dtype=float)], axis=0)
        for i in range(max(1, int(steps))):
            alpha = float(i + 1) / float(max(1, int(steps)))
            self.apply_action(current + (target - current) * alpha)
            if sleep_s > 0:
                time.sleep(float(sleep_s))

    def move_to_high_follow(
        self,
        left: list[float],
        right: list[float],
        timeout_s: float = 5.0,
        position_tolerance: float = 0.03,
        gripper_tolerance: float = 0.01,
    ) -> None:
        target = np.concatenate([np.asarray(left, dtype=float), np.asarray(right, dtype=float)], axis=0)
        if target.size < 14:
            raise ValueError(f"dual arm init target must have 14 values, got {target.size}")
        queued_target = target.copy()
        queued_target[6] = max(0.0, queued_target[6] - self.right_offset)
        queued_target[13] = max(0.0, queued_target[13] - self.right_offset)
        self._enqueue_high_follow_action(queued_target, replace=True)
        deadline = time.monotonic() + float(timeout_s)
        last_error = math.inf
        while time.monotonic() < deadline:
            state, _ = self.get_state()
            joint_error = float(np.max(np.abs(state[:6] - target[:6])))
            joint_error = max(joint_error, float(np.max(np.abs(state[7:13] - target[7:13]))))
            gripper_error = max(abs(float(state[6] - target[6])), abs(float(state[13] - target[13])))
            last_error = max(joint_error, gripper_error)
            if joint_error <= float(position_tolerance) and gripper_error <= float(gripper_tolerance):
                logger.info(
                    "high follow init target reached: joint_error=%.4f gripper_error=%.4f",
                    joint_error,
                    gripper_error,
                )
                return
            time.sleep(0.02)
        raise TimeoutError(f"high follow init target timed out: max_error={last_error:.4f} timeout_s={float(timeout_s):.2f}")

    def close(self) -> None:
        self._high_follow_running = False
        if self._high_follow_thread is not None:
            self._high_follow_thread.join(timeout=1.0)
            self._high_follow_thread = None
        self._state_running = False
        if self._state_thread is not None:
            self._state_thread.join(timeout=1.0)
            self._state_thread = None
        self.left.close()
        self.right.close()


class AgilexRobotIO:
    def __init__(
        self,
        camera_cfg: dict[str, Any],
        arm_cfg: dict[str, Any],
        can_cfg: dict[str, Any],
        telemetry: Any | None = None,
    ):
        self.rig = RealSenseRig(camera_cfg)
        self.telemetry = telemetry
        self.arms = PiperDualArm.from_config(arm_cfg, can_cfg, telemetry=telemetry)

    def set_runtime_logger(self, runtime_logger: Any | None) -> None:
        self.arms.set_runtime_logger(runtime_logger)

    def start(self) -> None:
        self.arms.connect()
        self.arms.start_state_stream()
        self.rig.start()

    def get_observation(self) -> dict[str, Any]:
        frame_time = None
        while frame_time is None:
            frame_time = self.rig.latest_frame_time()
            if frame_time is None:
                time.sleep(0.002)
        images, image_timestamps = self.rig.read_images_at_or_after(frame_time)
        state, state_ts = self.arms.get_synced_state(frame_time)
        return {
            "qpos": state,
            "state_timestamp": state_ts,
            "image_timestamp": min(image_timestamps.values()) if image_timestamps else frame_time,
            "image_timestamps": image_timestamps,
            "sync_frame_time": frame_time,
            "images": images,
        }

    def apply_action(self, action14: np.ndarray, action_step: dict[str, Any] | None = None) -> None:
        self.arms.apply_action(action14, action_step=action_step)

    def hold_current_position(self) -> None:
        self.arms.hold_current_position()

    def move_to_init(self, arm_cfg: dict[str, Any]) -> None:
        init_action = arm_cfg.get("init_action") or {}
        left = init_action.get("left")
        right = init_action.get("right")
        if left is None or right is None:
            return
        self.arms.move_to(
            left=left,
            right=right,
            steps=int(arm_cfg.get("init_steps", 100)),
            sleep_s=float(arm_cfg.get("init_sleep_s", 0.01)),
        )

    def move_to_shutdown(self, arm_cfg: dict[str, Any]) -> None:
        shutdown_action = arm_cfg.get("shutdown_action") or {}
        if not bool(shutdown_action.get("enabled", True)):
            return
        left = shutdown_action.get("left", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07])
        right = shutdown_action.get("right", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07])
        logger.info("moving arms to shutdown zero position before exit")
        self.arms.move_to(
            left=left,
            right=right,
            steps=int(shutdown_action.get("steps", arm_cfg.get("init_steps", 200))),
            sleep_s=float(shutdown_action.get("sleep_s", arm_cfg.get("init_sleep_s", 0.01))),
        )

    def close(self) -> None:
        self.rig.close()
        self.arms.close()


class MockLeRobotRobotIO:
    """Robot IO backed by a LeRobot-style dataset, with arm commands recorded instead of executed."""

    DEFAULT_MODEL_IMAGE_KEYS = ("top_head", "hand_right", "hand_left")
    DEFAULT_STATE_KEY = "observation.state"

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        episode_index: int | None = None,
        state_key: str = DEFAULT_STATE_KEY,
        image_keys: dict[str, str] | None = None,
        loop: bool = True,
        allow_missing_images: bool = False,
    ):
        if pl is None:
            raise RuntimeError("polars is required to load LeRobot parquet data")
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.episode_index = None if episode_index is None else int(episode_index)
        self.state_key = str(state_key)
        self.loop = bool(loop)
        self.allow_missing_images = bool(allow_missing_images)
        self._frames = self._load_frames()
        if not self._frames:
            raise ValueError(f"No LeRobot frames found under {self.dataset_root}")
        columns = set(self._frames[0])
        self.image_keys = dict(image_keys or self._infer_image_keys(columns))
        self._index = 0
        self._running = False
        self.applied_actions: list[np.ndarray] = []
        self._video_cache: dict[Path, cv2.VideoCapture] = {}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> MockLeRobotRobotIO:
        root = cfg.get("dataset_root", cfg.get("root", cfg.get("repo_id", cfg.get("path"))))
        if root is None:
            raise ValueError("robot_io.dataset_root is required for MockLeRobotRobotIO")
        return cls(
            root,
            episode_index=cfg.get("episode_index"),
            state_key=str(cfg.get("state_key", cls.DEFAULT_STATE_KEY)),
            image_keys=cfg.get("image_keys"),
            loop=bool(cfg.get("loop", True)),
            allow_missing_images=bool(cfg.get("allow_missing_images", False)),
        )

    def start(self) -> None:
        self._running = True

    def get_observation(self) -> dict[str, Any]:
        row = self._next_row()
        state = self._row_array(row, self.state_key)
        images = self._row_images(row)
        timestamp = float(row.get("timestamp", time.time()))
        return {
            "qpos": state,
            "state_timestamp": timestamp,
            "image_timestamp": timestamp,
            "images": images,
        }

    def apply_action(self, action14: np.ndarray, action_step: dict[str, Any] | None = None) -> None:
        del action_step
        action = np.asarray(action14, dtype=float).reshape(-1).copy()
        self.applied_actions.append(action)
        print(f"[MOCK ARM] action14={np.array2string(action, precision=4, separator=', ')}", flush=True)

    def hold_current_position(self) -> None:
        print("[MOCK ARM] hold_current_position", flush=True)

    def move_to_init(self, arm_cfg: dict[str, Any]) -> None:
        init_action = arm_cfg.get("init_action") if isinstance(arm_cfg, dict) else None
        print(f"[MOCK ARM] move_to_init skipped init_action={init_action!r}", flush=True)

    def move_to_shutdown(self, arm_cfg: dict[str, Any]) -> None:
        shutdown_action = arm_cfg.get("shutdown_action") if isinstance(arm_cfg, dict) else None
        print(f"[MOCK ARM] move_to_shutdown skipped shutdown_action={shutdown_action!r}", flush=True)

    def set_runtime_logger(self, logger_obj) -> None:
        del logger_obj

    def close(self) -> None:
        self._running = False
        for capture in self._video_cache.values():
            capture.release()
        self._video_cache.clear()

    def _load_frames(self) -> list[dict[str, Any]]:
        parquet_files = _find_lerobot_parquet_files(self.dataset_root, self.episode_index)
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {self.dataset_root}")
        frames = []
        for parquet_file in parquet_files:
            table = pl.read_parquet(parquet_file)
            if self.episode_index is not None and "episode_index" in table.columns:
                table = table.filter(pl.col("episode_index") == self.episode_index)
            frames.extend(table.to_dicts())
        return sorted(frames, key=_lerobot_sort_key)

    def _infer_image_keys(self, columns: set[str]) -> dict[str, str]:
        image_columns = sorted(column for column in columns if column.startswith("observation.images."))
        if not image_columns:
            image_columns = _find_lerobot_video_features(self.dataset_root)
        aliases = {
            "top_head": ("observation.images.top_head", "observation.images.cam_high", "observation.images.front"),
            "hand_right": (
                "observation.images.hand_right",
                "observation.images.right_wrist",
                "observation.images.cam_right_wrist",
            ),
            "hand_left": (
                "observation.images.hand_left",
                "observation.images.left_wrist",
                "observation.images.cam_left_wrist",
            ),
        }
        out: dict[str, str] = {}
        for model_key, candidates in aliases.items():
            match = next((candidate for candidate in candidates if candidate in columns), None)
            if match is not None:
                out[model_key] = match
        for model_key, image_column in zip(self.DEFAULT_MODEL_IMAGE_KEYS, image_columns, strict=False):
            out.setdefault(model_key, image_column)
        return out

    def _next_row(self) -> dict[str, Any]:
        if self._index >= len(self._frames):
            self._index = 0 if self.loop else len(self._frames) - 1
        row = self._frames[self._index]
        if self.loop or self._index < len(self._frames) - 1:
            self._index += 1
        return row

    def _row_array(self, row: dict[str, Any], key: str) -> np.ndarray:
        if key not in row:
            raise KeyError(f"LeRobot row has no {key!r}; available keys include {sorted(row)[:10]}")
        return np.asarray(row[key], dtype=float).reshape(-1)

    def _row_images(self, row: dict[str, Any]) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for model_key, source_key in self.image_keys.items():
            image = self._load_image(row, str(source_key))
            if image is None:
                if not self.allow_missing_images:
                    raise FileNotFoundError(f"Could not load image for {model_key!r} from {source_key!r}")
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            images[str(model_key)] = image
        if not images and not self.allow_missing_images:
            raise ValueError("No image keys configured or discovered for MockLeRobotRobotIO")
        return images

    def _load_image(self, row: dict[str, Any], source_key: str) -> np.ndarray | None:
        value = row.get(source_key)
        if value is not None:
            image = _decode_lerobot_image_value(self.dataset_root, value)
            if image is not None:
                return image
        return self._load_video_frame(row, source_key)

    def _load_video_frame(self, row: dict[str, Any], source_key: str) -> np.ndarray | None:
        episode_idx = int(row.get("episode_index", self.episode_index or 0))
        frame_idx = int(row.get("frame_index", row.get("index", self._index)))
        video_path = _find_lerobot_video_file(self.dataset_root, source_key, episode_idx)
        if video_path is None:
            return None
        capture = self._video_cache.get(video_path)
        if capture is None:
            capture = cv2.VideoCapture(str(video_path))
            self._video_cache[video_path] = capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        return frame if ok else None


def _find_lerobot_parquet_files(root: Path, episode_index: int | None) -> list[Path]:
    if root.is_file() and root.suffix == ".parquet":
        return [root]
    if episode_index is not None:
        expected = f"episode_{int(episode_index):06d}.parquet"
        matches = sorted(root.glob(f"data/**/{expected}"))
        if matches:
            return matches
    matches = sorted(root.glob("data/**/*.parquet"))
    if matches:
        return matches
    return sorted(root.glob("**/*.parquet"))


def _lerobot_sort_key(row: dict[str, Any]) -> tuple[int, int, float]:
    return (
        int(row.get("episode_index", 0)),
        int(row.get("frame_index", row.get("index", 0))),
        float(row.get("timestamp", 0.0)),
    )


def _decode_lerobot_image_value(root: Path, value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value.astype(np.uint8, copy=False)
    if isinstance(value, list):
        arr = np.asarray(value)
        if arr.ndim >= 2:
            return arr.astype(np.uint8, copy=False)
    if isinstance(value, dict):
        for key in ("path", "image", "bytes"):
            if key in value:
                return _decode_lerobot_image_value(root, value[key])
        return None
    if isinstance(value, bytes | bytearray | memoryview):
        arr = np.frombuffer(bytes(value), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if isinstance(value, str):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        return cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None


def _find_lerobot_video_file(root: Path, feature_key: str, episode_index: int) -> Path | None:
    expected = f"episode_{int(episode_index):06d}.mp4"
    candidates = sorted((root / "videos").glob(f"**/{feature_key}/{expected}"))
    if candidates:
        return candidates[0]
    candidates = sorted((root / "videos").glob(f"**/{expected}"))
    return candidates[0] if candidates else None


def _find_lerobot_video_features(root: Path) -> list[str]:
    video_root = root / "videos"
    if not video_root.exists():
        return []
    features = {path.parent.name for path in video_root.glob("**/episode_*.mp4")}
    return sorted(feature for feature in features if feature.startswith("observation.images."))


def check_hardware(io: AgilexRobotIO | MockLeRobotRobotIO) -> None:
    obs = io.get_observation()
    qpos = np.asarray(obs["qpos"], dtype=float)
    print(f"[CHECK] qpos shape={qpos.shape} first={np.round(qpos[:7], 4).tolist()}")
    for name, image in sorted((obs.get("images") or {}).items()):
        print(f"[CHECK] image {name}: shape={tuple(image.shape)} dtype={image.dtype}")
