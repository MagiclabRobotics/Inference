from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_SRC = REPO_ROOT / "scripts"

if str(SCRIPTS_SRC) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_SRC))

from visualize_action_steps import JOINT_FIELDS  # noqa: E402
from visualize_action_steps import collect_chunk_ids  # noqa: E402
from visualize_action_steps import load_action_steps  # noqa: E402
from visualize_action_steps import plot_action_steps  # noqa: E402


def _write_action_steps_csv(path: Path) -> None:
    fieldnames = ["timestamp_sec", "monotonic_sec", "chunk_id", "chunk_step_index", "action_value_source", *JOINT_FIELDS]
    rows = []
    for row_idx in range(2):
        action_values = {joint: 100.0 * row_idx + dim for dim, joint in enumerate(JOINT_FIELDS)}
        sources = [
            [[5, row_idx, 1000.0 + dim, 0.4], [6, row_idx + 10, 2000.0 + dim, 0.6]]
            for dim, _joint in enumerate(JOINT_FIELDS)
        ]
        rows.append(
            {
                "timestamp_sec": str(1.0 + row_idx),
                "monotonic_sec": str(10.0 + row_idx),
                "chunk_id": str(5 + row_idx),
                "chunk_step_index": str(row_idx),
                "action_value_source": json.dumps(sources),
                **{joint: str(value) for joint, value in action_values.items()},
            }
        )
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_action_steps_parses_joint_values_and_sources(tmp_path):
    csv_path = tmp_path / "action_steps.csv"
    _write_action_steps_csv(csv_path)

    series = load_action_steps(csv_path)

    assert series.actions.shape == (2, 14)
    np.testing.assert_allclose(series.actions[:, 0], [0.0, 100.0])
    assert series.sources_by_joint[0][0].chunk_id == 5
    assert series.sources_by_joint[0][0].source_step_index == 0
    assert series.sources_by_joint[0][0].value == 1000.0
    assert series.sources_by_joint[0][0].weight == 0.4
    assert series.sources_by_joint[0][1].chunk_id == 6
    assert series.sources_by_joint[0][1].action_index == 0
    assert collect_chunk_ids(series) == [5, 6]


def test_plot_action_steps_writes_png(tmp_path):
    csv_path = tmp_path / "action_steps.csv"
    output_path = tmp_path / "action_steps.png"
    _write_action_steps_csv(csv_path)
    series = load_action_steps(csv_path)

    result = plot_action_steps(series, output_path=output_path, title="test episode", dpi=80)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
