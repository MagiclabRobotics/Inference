"""
[xh-integration] RobotIO 适配层：对接 piperserver-master_xh + 本仓 InferenceRuntime。

- 图像：本进程 ROS（ros_image_source），与采集 service 模式分工一致
- 关节：collector_tcp → 采集 TCP 9000 cmd=get_joint_state
- apply_action：故意 no-op；真机控臂由采集 30Hz pop_action → execute_action

配置：config_agilex.yaml → robot_io.entry_point 指向本模块 create_from_config
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from integration.collector_contract import ACTION_DIM, IMAGE_KEYS
from integration.collector_tcp import CollectorTcpClient
from integration.ros_image_source import RosImageSource, rclpy


logger = logging.getLogger(__name__)


class CollectorRobotIO:
    def __init__(
        self,
        *,
        joint_client: CollectorTcpClient,
        image_topics: list[str],
        camera_names: list[str],
        sync_frame_timeout_s: float = 2.0,
        frame_deque_maxlen: int = 300,
    ):
        self._joint = joint_client
        self._sync_frame_timeout_s = float(sync_frame_timeout_s)
        self._image_topics = list(image_topics)
        self._camera_names = list(camera_names)
        self._frame_deque_maxlen = int(frame_deque_maxlen)
        self._image_node: RosImageSource | None = None
        self._spin_thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        if rclpy is None:
            raise RuntimeError("rclpy 不可用，无法启动 ROS 图像订阅")
        try:
            rclpy.init()
        except Exception:
            pass
        self._image_node = RosImageSource(
            self._image_topics,
            self._camera_names,
            frame_deque_maxlen=self._frame_deque_maxlen,
        )
        self._running = True

        def _spin() -> None:
            while self._running and self._image_node is not None:
                try:
                    rclpy.spin_once(self._image_node, timeout_sec=0.1)
                except Exception as exc:
                    logger.warning("ROS spin: %s", exc)
                    time.sleep(0.05)

        self._spin_thread = threading.Thread(target=_spin, name="ros_image_spin", daemon=True)
        self._spin_thread.start()
        logger.info("CollectorRobotIO: ROS 图像订阅已启动 cameras=%s", self._camera_names)

    def get_observation(self) -> dict[str, Any]:
        if self._image_node is None:
            raise RuntimeError("CollectorRobotIO 未 start")
        frame_time = None
        deadline = time.monotonic() + self._sync_frame_timeout_s
        while frame_time is None and time.monotonic() < deadline:
            frame_time = self._image_node.latest_frame_time()
            if frame_time is None:
                time.sleep(0.002)
        if frame_time is None:
            raise TimeoutError("等待 ROS 图像超时")

        images, image_timestamps = self._image_node.read_images_at_or_after(
            frame_time, timeout_s=self._sync_frame_timeout_s
        )
        for key in IMAGE_KEYS:
            if key not in images or images[key] is None:
                raise RuntimeError(f"观测缺少图像 key={key!r}，当前 keys={list(images.keys())}")

        js = self._joint.get_joint_state_at_or_after(float(frame_time))
        qpos = np.asarray(js["qpos"], dtype=np.float32)
        if qpos.size != ACTION_DIM:
            raise ValueError(f"qpos 必须是 {ACTION_DIM} 维，收到 {qpos.size}")

        image_ts_min = (
            min(float(v) for v in image_timestamps.values()) if image_timestamps else float(frame_time)
        )
        return {
            "qpos": qpos,
            "state_timestamp": float(js.get("state_timestamp", frame_time)),
            "image_timestamp": image_ts_min,
            "image_timestamps": dict(image_timestamps),
            "sync_frame_time": float(frame_time),
            "images": images,
        }

    def apply_action(self, action14: np.ndarray) -> None:
        del action14  # 采集侧 execute_action；此处 intentionally no-op

    def move_to_init(self, arm_cfg: dict[str, Any]) -> None:
        del arm_cfg
        raise NotImplementedError("推理服务模式请机械臂先就位")

    def close(self) -> None:
        self._running = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
            self._spin_thread = None
        if self._image_node is not None:
            self._image_node.stop()
            self._image_node = None
        self._joint.close()


def create_from_config(cfg: Any, *, section, **kwargs: Any) -> CollectorRobotIO:
    """``robot_io.entry_point`` 工厂：从 yaml ``collector`` 段构造。"""
    from config import section as cfg_section

    raw = getattr(cfg, "raw", None)
    if not isinstance(raw, dict):
        raise TypeError("create_from_config 需要 InferenceConfig")
    coll = dict(raw.get("collector") or {})
    coll.update({k: v for k, v in kwargs.items() if v is not None})

    host = str(coll.get("host", "127.0.0.1"))
    port = int(coll.get("port", 9000))
    topics = list(coll.get("image_topics") or [])
    names = list(coll.get("camera_names") or list(IMAGE_KEYS))
    if len(topics) != 3:
        raise ValueError("collector.image_topics 需要 3 路")
    if len(names) != 3:
        raise ValueError("collector.camera_names 需要 3 路")

    inf = cfg_section(cfg, "inference") if callable(section) else {}
    timeout = float(
        coll.get(
            "sync_frame_timeout_s",
            inf.get("sync_frame_timeout_s", kwargs.get("sync_frame_timeout_s", 2.0)),
        )
    )
    return CollectorRobotIO(
        joint_client=CollectorTcpClient(host, port),
        image_topics=topics,
        camera_names=names,
        sync_frame_timeout_s=timeout,
        frame_deque_maxlen=int(coll.get("frame_deque_maxlen", 300)),
    )
