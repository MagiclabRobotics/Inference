from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "client" / "inference"))

from robot_io import PiperDualArm, PiperHighFollowConfig, _project_tangents_monotone  # noqa: E402


def _points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prev = np.zeros(14, dtype=float)
    start = np.ones(14, dtype=float)
    target = np.full(14, 1.1, dtype=float)
    next_point = np.full(14, 5.0, dtype=float)
    return prev, start, target, next_point


def _dual_arm(config: PiperHighFollowConfig) -> PiperDualArm:
    return PiperDualArm(left=None, right=None, high_follow_config=config)  # type: ignore[arg-type]


def test_segment_tangents_default_behavior_unchanged() -> None:
    prev, start, target, next_point = _points()
    cfg = PiperHighFollowConfig(interpolator="waypoint_cubic")
    m0, m1 = _dual_arm(cfg)._segment_tangents(prev, start, target, next_point)

    np.testing.assert_allclose(m0, 0.5 * (target - prev))
    np.testing.assert_allclose(m1, 0.5 * (next_point - start))


def test_project_tangents_monotone_zeroes_wrong_direction_tangent() -> None:
    start = np.zeros(14, dtype=float)
    target = np.zeros(14, dtype=float)
    start[0] = 1.0
    target[0] = 1.1
    m0 = np.zeros(14, dtype=float)
    m1 = np.zeros(14, dtype=float)
    m0[0] = -0.2
    m1[0] = 0.1

    projected_m0, projected_m1 = _project_tangents_monotone(start, target, m0, m1, np.array([0]))

    assert projected_m0[0] == pytest.approx(0.0)
    assert projected_m1[0] == pytest.approx(0.1)


def test_project_tangents_monotone_scales_large_tangents() -> None:
    start = np.zeros(14, dtype=float)
    target = np.zeros(14, dtype=float)
    start[0] = 1.0
    target[0] = 1.1
    m0 = np.zeros(14, dtype=float)
    m1 = np.zeros(14, dtype=float)
    m0[0] = 0.15
    m1[0] = 0.55

    projected_m0, projected_m1 = _project_tangents_monotone(start, target, m0, m1, np.array([0]))

    assert projected_m0[0] == pytest.approx(0.0642857142857143)
    assert projected_m1[0] == pytest.approx(0.23571428571428574)


def test_project_tangents_monotone_zeroes_near_static_dimension() -> None:
    start = np.zeros(14, dtype=float)
    target = np.zeros(14, dtype=float)
    start[0] = 1.0
    target[0] = 1.0 + 1e-10
    m0 = np.ones(14, dtype=float)
    m1 = np.ones(14, dtype=float)

    projected_m0, projected_m1 = _project_tangents_monotone(start, target, m0, m1, np.array([0]))

    assert projected_m0[0] == pytest.approx(0.0)
    assert projected_m1[0] == pytest.approx(0.0)


def test_monotone_projection_dims_arm_leaves_grippers_unprojected() -> None:
    prev, start, target, next_point = _points()
    cfg = PiperHighFollowConfig(
        interpolator="waypoint_cubic",
        monotone_projection=True,
        monotone_projection_dims="arm",
    )
    m0, m1 = _dual_arm(cfg)._segment_tangents(prev, start, target, next_point)

    # Gripper dimensions 6 and 13 keep the raw Catmull-Rom tangent.
    assert m1[6] == pytest.approx(0.5 * (next_point[6] - start[6]))
    assert m1[13] == pytest.approx(0.5 * (next_point[13] - start[13]))
    # Arm dimensions are projected to the monotone feasible boundary.
    assert m0[0] / (target[0] - start[0]) + m1[0] / (target[0] - start[0]) == pytest.approx(3.0)


def test_projected_hermite_derivative_is_monotone_on_projected_dimension() -> None:
    start = np.zeros(14, dtype=float)
    target = np.zeros(14, dtype=float)
    start[0] = 1.0
    target[0] = 1.1
    m0 = np.zeros(14, dtype=float)
    m1 = np.zeros(14, dtype=float)
    m0[0] = -0.2
    m1[0] = 0.55
    projected_m0, projected_m1 = _project_tangents_monotone(start, target, m0, m1, np.array([0]))
    d = target[0] - start[0]

    for s in np.linspace(0.0, 1.0, num=101):
        dqds = PiperDualArm._hermite_derivative(float(s), start, target, projected_m0, projected_m1)
        assert d * dqds[0] >= -1e-12


def test_monotone_projection_dims_validation() -> None:
    with pytest.raises(ValueError, match="monotone_projection_dims"):
        PiperHighFollowConfig.from_config(
            {
                "control_mode": "high_follow",
                "high_follow": {
                    "interpolator": {
                        "type": "waypoint_cubic",
                        "monotone_projection_dims": "wrist",
                    }
                },
            }
        )
