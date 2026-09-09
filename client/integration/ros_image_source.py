"""ROS 压缩图订阅 + 按时间戳 deque（推理服务自维护，不依赖采集采图）。"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

try:
    import cv2
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
except Exception:
    cv2 = None
    rclpy = None
    Node = object
    CompressedImage = None


class RosImageSource(Node):
    def __init__(
        self,
        image_topics: list[str],
        camera_names: list[str],
        *,
        frame_deque_maxlen: int = 300,
    ):
        super().__init__("inference_ros_image_source")
        if cv2 is None or CompressedImage is None:
            raise RuntimeError("cv2 / sensor_msgs 不可用，无法订阅图像")
        self.camera_names = list(camera_names)
        self.image_topics = list(image_topics)
        if len(self.camera_names) != len(self.image_topics):
            raise ValueError("camera_names 与 image_topics 数量须一致")
        self._frame_deque_maxlen = max(8, int(frame_deque_maxlen))
        self._locks = {name: threading.Lock() for name in self.camera_names}
        self._frame_deques = {
            name: deque(maxlen=self._frame_deque_maxlen) for name in self.camera_names
        }
        for name, topic in zip(self.camera_names, self.image_topics):
            self.create_subscription(CompressedImage, topic, self._make_callback(name), 10)

    def _make_callback(self, camera_name: str):
        def _callback(msg) -> None:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warning(f"decode failed: {camera_name}")
                return
            stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            with self._locks[camera_name]:
                self._frame_deques[camera_name].append((stamp, img))

        return _callback

    def latest_frame_time(self) -> float | None:
        stamps: list[float] = []
        for name in self.camera_names:
            with self._locks[name]:
                dq = self._frame_deques[name]
                if not dq:
                    return None
                stamps.append(float(dq[-1][0]))
        return min(stamps)

    def read_images_at_or_after(
        self, frame_time: float, *, timeout_s: float = 2.0
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        deadline = time.monotonic() + float(timeout_s)
        images: dict[str, np.ndarray] = {}
        stamps_out: dict[str, float] = {}
        for name in self.camera_names:
            while time.monotonic() < deadline:
                with self._locks[name]:
                    dq = self._frame_deques[name]
                    while dq and float(dq[0][0]) < float(frame_time):
                        dq.popleft()
                    if dq:
                        st, im = dq[0]
                        if float(st) >= float(frame_time):
                            dq.popleft()
                            images[name] = im
                            stamps_out[name] = float(st)
                            break
                time.sleep(0.002)
            else:
                raise TimeoutError(
                    f"相机 {name!r} 在 {timeout_s}s 内未得到 stamp>={frame_time} 的图像"
                )
        return images, stamps_out

    def stop(self) -> None:
        try:
            self.destroy_node()
        except Exception:
            pass
