from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

JOINT_FIELDS = [
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_j6",
    "left_gripper",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_j6",
    "right_gripper",
]


@dataclass(frozen=True)
class SourcePoint:
    action_index: int
    joint_index: int
    chunk_id: int
    source_step_index: int
    value: float
    weight: float = 1.0


@dataclass(frozen=True)
class ActionStepSeries:
    csv_path: Path
    joint_names: tuple[str, ...]
    action_indices: np.ndarray
    actions: np.ndarray
    sources_by_joint: tuple[tuple[SourcePoint, ...], ...]


def load_action_steps(csv_path: str | Path) -> ActionStepSeries:
    path = Path(csv_path).expanduser()
    action_rows: list[list[float]] = []
    sources_by_joint: list[list[SourcePoint]] = [[] for _ in JOINT_FIELDS]

    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        _validate_fields(reader.fieldnames, path)
        for action_index, row in enumerate(reader):
            action_rows.append([float(row[joint]) for joint in JOINT_FIELDS])
            raw_sources = json.loads(row.get("action_value_source") or "[]")
            if len(raw_sources) != len(JOINT_FIELDS):
                raise ValueError(
                    f"{path}: row {action_index} action_value_source must have {len(JOINT_FIELDS)} joints, "
                    f"got {len(raw_sources)}"
                )
            for joint_index, joint_sources in enumerate(raw_sources):
                for source in joint_sources:
                    sources_by_joint[joint_index].append(_parse_source_point(source, action_index, joint_index, path))

    actions = np.asarray(action_rows, dtype=float)
    if actions.ndim == 1:
        actions = actions.reshape(0, len(JOINT_FIELDS))
    return ActionStepSeries(
        csv_path=path,
        joint_names=tuple(JOINT_FIELDS),
        action_indices=np.arange(actions.shape[0], dtype=int),
        actions=actions,
        sources_by_joint=tuple(tuple(points) for points in sources_by_joint),
    )


def collect_chunk_ids(series: ActionStepSeries) -> list[int]:
    seen = set()
    chunk_ids = []
    for joint_sources in series.sources_by_joint:
        for source in joint_sources:
            if source.chunk_id in seen:
                continue
            seen.add(source.chunk_id)
            chunk_ids.append(source.chunk_id)
    return chunk_ids


def plot_action_steps(
    series: ActionStepSeries,
    *,
    output_path: str | Path,
    title: str | None = None,
    dpi: int = 150,
) -> Path:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    chunk_ids = collect_chunk_ids(series)
    colors = _chunk_colors(chunk_ids, plt)
    fig, axes = plt.subplots(7, 2, figsize=(16, 20), sharex=True)
    axes_flat = axes.reshape(-1)

    for joint_index, (axis, joint_name) in enumerate(zip(axes_flat, series.joint_names, strict=True)):
        axis.plot(
            series.action_indices,
            series.actions[:, joint_index],
            color="#111827",
            linewidth=1.2,
            label="executed action",
            zorder=3,
        )
        _plot_sources_for_joint(axis, series.sources_by_joint[joint_index], colors)
        axis.set_title(joint_name, fontsize=10)
        axis.grid(visible=True, alpha=0.25, linewidth=0.6)
        axis.set_ylabel("value")

    for axis in axes_flat[-2:]:
        axis.set_xlabel("action step")

    legend_handles = [Line2D([0], [0], color="#111827", linewidth=1.5, label="executed action")]
    legend_handles.extend(
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[chunk_id], markersize=6, label=str(chunk_id))
        for chunk_id in chunk_ids
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(max(1, len(legend_handles)), 12),
        title="source chunk_id",
        fontsize=8,
    )
    fig.suptitle(title or f"action steps: {series.csv_path}", fontsize=14)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    fig.savefig(output, dpi=int(dpi))
    plt.close(fig)
    return output


def _validate_fields(fieldnames: Sequence[str] | None, path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"{path}: missing CSV header")
    missing = [field for field in [*JOINT_FIELDS, "action_value_source"] if field not in fieldnames]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")


def _parse_source_point(source, action_index: int, joint_index: int, path: Path) -> SourcePoint:
    if len(source) == 3:
        chunk_id, source_step_index, value = source
        weight = 1.0
    elif len(source) == 4:
        chunk_id, source_step_index, value, weight = source
    else:
        raise ValueError(f"{path}: source item must be [chunk_id, step_index, value, weight], got {source!r}")
    return SourcePoint(
        action_index=int(action_index),
        joint_index=int(joint_index),
        chunk_id=int(chunk_id),
        source_step_index=int(source_step_index),
        value=float(value),
        weight=float(weight),
    )


def _chunk_colors(chunk_ids: Sequence[int], plt) -> dict[int, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20", max(1, min(20, len(chunk_ids))))
    return {chunk_id: cmap(idx % 20) for idx, chunk_id in enumerate(chunk_ids)}


def _plot_sources_for_joint(axis, sources: Sequence[SourcePoint], colors: dict[int, tuple[float, float, float, float]]) -> None:
    by_chunk: dict[int, list[SourcePoint]] = {}
    for source in sources:
        by_chunk.setdefault(source.chunk_id, []).append(source)
    for chunk_id, points in by_chunk.items():
        axis.scatter(
            [point.action_index for point in points],
            [point.value for point in points],
            color=colors[chunk_id],
            s=14,
            alpha=0.8,
            linewidths=0,
            zorder=4,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize action_steps.csv joint values and source values.")
    parser.add_argument("csv_path", type=Path, help="Path to action_steps.csv")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--title", default=None, help="Figure title")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    series = load_action_steps(args.csv_path)
    output = args.output or args.csv_path.with_name("action_steps_visualization.png")
    result = plot_action_steps(series, output_path=output, title=args.title, dpi=args.dpi)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
