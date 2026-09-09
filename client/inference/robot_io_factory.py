"""
可插拔 Robot IO：默认 AgilexRobotIO（RealSense + CAN）；也可通过 yaml 注入外部数据源。

infer_2 / 其它宿主进程在 sys.path 就绪后，可实现与 `AgilexRobotIO` 相同契约的类，并通过
`robot_io.entry_point` 指向 ``module.path:callable``，由 ``callable(cfg, *, section, **kwargs)``
返回实例；后续 ``InferenceRuntime`` 仍走同一套帧对齐与策略逻辑。

约定（与 ``AgilexRobotIO`` 一致）：
  - ``start() / close()``
  - ``get_observation()`` → dict，至少含 ``qpos``、``images``（BGR），及可选 ``image_timestamp`` 等
  - ``apply_action(action14)``
  - ``move_to_init(arm_cfg)``
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol

import numpy as np


class RobotIO(Protocol):
    """与 ``robot_io.AgilexRobotIO`` 相同的运行时契约（duck typing）。"""

    def start(self) -> None: ...

    def get_observation(self) -> dict[str, Any]: ...

    def apply_action(self, action14: np.ndarray) -> None: ...

    def move_to_init(self, arm_cfg: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


def create_robot_io(cfg: Any) -> RobotIO:
    """
    根据 yaml 根键 ``robot_io`` 构造 IO；未配置或 ``entry_point`` 为 default 时用内置 ``AgilexRobotIO``。

    外部工厂签名::

        def build_io(cfg: InferenceConfig, *, section, **kwargs) -> RobotIO: ...

    ``cfg`` 为 ``config.load_config`` 的返回值；``section`` 为 ``config.section``，便于读 ``camera``/``arm``/``can``。
    ``robot_io.kwargs``（可选 dict）以关键字形式传入工厂。
    """
    from config import section as cfg_section

    raw = getattr(cfg, "raw", None)
    if not isinstance(raw, dict):
        raise TypeError("create_robot_io 需要 InferenceConfig（带 raw 映射）")

    robot_io_cfg = dict(raw.get("robot_io") or {})
    entry = str(robot_io_cfg.get("entry_point", "") or "").strip()
    extra = robot_io_cfg.get("kwargs")
    extra = dict(extra) if isinstance(extra, dict) else {}

    if not entry or entry.lower() in ("default", "agilex", "agilex_sdk"):
        from robot_io import AgilexRobotIO

        return AgilexRobotIO(
            camera_cfg=cfg_section(cfg, "camera"),
            arm_cfg=cfg_section(cfg, "arm"),
            can_cfg=cfg_section(cfg, "can"),
        )

    if ":" not in entry:
        raise ValueError(
            f'robot_io.entry_point 须为 "package.module:callable" 或 default/agilex，收到: {entry!r}'
        )
    mod_name, _, qual = entry.partition(":")
    mod_name, qual = mod_name.strip(), qual.strip()
    if not mod_name or not qual:
        raise ValueError(f"robot_io.entry_point 格式错误: {entry!r}")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, qual)
    if not callable(fn):
        raise TypeError(f"robot_io.entry_point 不可调用: {entry!r} -> {type(fn).__name__}")
    return fn(cfg, section=cfg_section, **extra)
