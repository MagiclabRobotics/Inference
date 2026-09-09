#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = REPO_ROOT / "packages" / "openpi-client" / "src"
if str(OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))

from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402


VIDEO_KEYS = {
    "top_head": "observation.images.top_head",
    "hand_left": "observation.images.hand_left",
    "hand_right": "observation.images.hand_right",
}


def _load_info(dataset_root: Path) -> dict[str, Any]:
    with (dataset_root / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _episode_chunk(info: dict[str, Any], episode_index: int) -> int:
    return episode_index // int(info.get("chunks_size", 1000))


def _format_dataset_path(template: str, info: dict[str, Any], episode_index: int, **extra: Any) -> str:
    values = {
        "episode_chunk": _episode_chunk(info, episode_index),
        "episode_index": episode_index,
        **extra,
    }
    return template.format(**values)


def _read_episode_row(dataset_root: Path, info: dict[str, Any], episode_index: int) -> dict[str, Any]:
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    for path in episode_files:
        rows = pl.read_parquet(path).filter(pl.col("episode_index") == episode_index)
        if rows.height:
            return rows.row(0, named=True)

    episodes_jsonl = dataset_root / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        with episodes_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if int(row["episode_index"]) == episode_index:
                    row["data_path"] = _format_dataset_path(info["data_path"], info, episode_index)
                    return row

    if not episode_files:
        raise FileNotFoundError(
            "No episode metadata found. Expected either "
            f"{dataset_root / 'meta' / 'episodes'}/chunk-*/file-*.parquet or {episodes_jsonl}"
        )
    raise ValueError(f"episode_index={episode_index} was not found in {dataset_root}")


def _read_data_row(dataset_root: Path, episode_row: dict[str, Any], frame_index: int | None) -> dict[str, Any]:
    if "data_path" in episode_row:
        data_path = dataset_root / str(episode_row["data_path"])
    else:
        data_path = (
            dataset_root
            / "data"
            / f"chunk-{int(episode_row['data/chunk_index']):03d}"
            / f"file-{int(episode_row['data/file_index']):03d}.parquet"
        )
    if frame_index is None:
        frame_index = int(episode_row["length"]) // 2
    if frame_index < 0 or frame_index >= int(episode_row["length"]):
        raise ValueError(f"frame_index must be in [0, {int(episode_row['length']) - 1}], got {frame_index}")

    rows = pl.read_parquet(data_path).filter(
        (pl.col("episode_index") == int(episode_row["episode_index"])) & (pl.col("frame_index") == frame_index)
    )
    if rows.height != 1:
        raise ValueError(f"Expected one row in {data_path}, got {rows.height}")
    return rows.row(0, named=True)


def _read_video_frame(
    dataset_root: Path,
    info: dict[str, Any],
    episode_row: dict[str, Any],
    video_key: str,
    frame_index: int,
) -> np.ndarray:
    if f"videos/{video_key}/chunk_index" in episode_row:
        chunk_index = int(episode_row[f"videos/{video_key}/chunk_index"])
        file_index = int(episode_row[f"videos/{video_key}/file_index"])
        from_timestamp = float(episode_row[f"videos/{video_key}/from_timestamp"])
        video_path = dataset_root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
    else:
        from_timestamp = 0.0
        video_path = dataset_root / _format_dataset_path(
            info["video_path"],
            info,
            int(episode_row["episode_index"]),
            video_key=video_key,
        )
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        absolute_frame = int(round((from_timestamp * fps) + frame_index))
        cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to read frame {absolute_frame} from {video_path}")
        return frame_bgr
    finally:
        cap.release()


def build_payload(dataset_root: Path, episode_index: int, frame_index: int | None, prompt: str) -> tuple[dict, dict]:
    info = _load_info(dataset_root)
    episode_row = _read_episode_row(dataset_root, info, episode_index)
    data_row = _read_data_row(dataset_root, episode_row, frame_index)
    frame_index = int(data_row["frame_index"])

    frames_bgr = {
        client_key: _read_video_frame(dataset_root, info, episode_row, video_key, frame_index)
        for client_key, video_key in VIDEO_KEYS.items()
    }
    image_arrs = [cv2.cvtColor(frames_bgr[key], cv2.COLOR_BGR2RGB) for key in ("top_head", "hand_right", "hand_left")]
    image_arrs = image_tools.resize_with_pad(np.asarray(image_arrs), 224, 224)

    payload = {
        "images": {
            "top_head": image_arrs[0].transpose(2, 0, 1),
            "hand_right": image_arrs[1].transpose(2, 0, 1),
            "hand_left": image_arrs[2].transpose(2, 0, 1),
        },
        "state": np.asarray(data_row["observation.state"], dtype=np.float32),
        "prompt": prompt,
    }
    info = {
        "dataset_root": str(dataset_root),
        "codebase_version": info.get("codebase_version"),
        "episode_index": int(episode_index),
        "frame_index": frame_index,
        "timestamp": float(data_row["timestamp"]),
        "state_shape": list(payload["state"].shape),
        "image_shapes": {key: list(value.shape) for key, value in payload["images"].items()},
        "prompt": prompt,
    }
    return payload, info


def _summarize_output(output: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in output.items():
        if isinstance(value, np.ndarray):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(np.nanmin(value)) if value.size else None,
                "max": float(np.nanmax(value)) if value.size else None,
                "mean": float(np.nanmean(value)) if value.size else None,
            }
        else:
            summary[key] = value
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an OpenPI websocket request from a LeRobot sample.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--episode-index", default=0, type=int)
    parser.add_argument("--frame-index", type=int, help="Episode-local frame index. Defaults to the middle frame.")
    parser.add_argument("--prompt", default="Flatten and fold the cloth.")
    parser.add_argument("--build-only", action="store_true", help="Only build and validate the payload; do not call server.")
    args = parser.parse_args()

    payload, info = build_payload(args.dataset_root, args.episode_index, args.frame_index, args.prompt)
    result: dict[str, Any] = {"request": info}

    if not args.build_only:
        policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
        result["server_metadata"] = policy.get_server_metadata()
        result["response"] = _summarize_output(policy.infer(payload))

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
