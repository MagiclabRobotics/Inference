from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "client" / "inference"))

from async_runtime_logger import AsyncRuntimeLogger


def test_high_follow_command_csv_is_written(tmp_path) -> None:
    logger = AsyncRuntimeLogger(
        action_csv_path=tmp_path / "action_steps.csv",
        high_follow_csv_path=tmp_path / "high_follow_commands.csv",
        event_log_path=tmp_path / "runtime_events.jsonl",
    )
    logger.start()
    logger.log_high_follow_command(
        {
            "timestamp_sec": 1.0,
            "monotonic_sec": 2.0,
            "chunk_id": 3,
            "chunk_step_index": 4,
            "left_j1": 0.1,
            "right_gripper": 0.2,
            "high_follow_step_index": 5,
            "high_follow_segment_steps": 6,
            "high_follow_phase": 5.0 / 6.0,
            "high_follow_interpolator": "waypoint_cubic",
            "high_follow_reached": False,
        }
    )
    logger.stop()

    with (tmp_path / "high_follow_commands.csv").open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    assert len(rows) == 1
    assert rows[0]["monotonic_sec"] == "2.0"
    assert rows[0]["chunk_id"] == "3"
    assert rows[0]["chunk_step_index"] == "4"
    assert rows[0]["high_follow_interpolator"] == "waypoint_cubic"
