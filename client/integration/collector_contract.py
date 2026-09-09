"""
采集 ↔ 推理 稳定契约（版本化）。

采集实现细节（Piper/CAN/TCP 报文）仅出现在 integration 层；
``client/inference/runtime.py`` 与策略 mode 不依赖具体采集仓库。
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

INFERENCE_SDK_VERSION = "1.0.0"
OBS_SCHEMA_VERSION = "1"
ACTION_DIM = 14
IMAGE_KEYS = ("top_head", "hand_right", "hand_left")


class CollectorJointClient(Protocol):
    """采集侧关节时间轴查询（如 piperserver TCP 9000 ``get_joint_state``）。"""

    def get_joint_state_at_or_after(self, frame_time: float) -> dict[str, Any]: ...


class RobotIO(Protocol):
    """与 ``client/inference/robot_io_factory.RobotIO`` 相同语义。"""

    def start(self) -> None: ...

    def get_observation(self) -> dict[str, Any]: ...

    def apply_action(self, action14: np.ndarray) -> None: ...

    def move_to_init(self, arm_cfg: dict[str, Any]) -> None: ...

    def close(self) -> None: ...
