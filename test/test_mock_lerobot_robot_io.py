from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENT_INFERENCE_SRC = REPO_ROOT / "client" / "inference"

if str(CLIENT_INFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(CLIENT_INFERENCE_SRC))

from robot_io import MockLeRobotRobotIO  # noqa: E402


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 10, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_lerobot_like_dataset(root: Path) -> None:
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    image_root = root / "images"
    rows = []
    for frame_idx in range(2):
        image_paths = {}
        for key, value in (("top_head", 10), ("hand_right", 20), ("hand_left", 30)):
            image_path = image_root / key / f"{frame_idx:06d}.png"
            _write_image(image_path, value + frame_idx)
            image_paths[key] = image_path.relative_to(root).as_posix()
        rows.append(
            {
                "episode_index": 0,
                "frame_index": frame_idx,
                "timestamp": frame_idx * 0.1,
                "observation.state": [float(frame_idx + i) for i in range(14)],
                "observation.images.top_head": image_paths["top_head"],
                "observation.images.hand_right": image_paths["hand_right"],
                "observation.images.hand_left": image_paths["hand_left"],
            }
        )
    pl.DataFrame(rows).write_parquet(data_dir / "episode_000000.parquet")


def test_mock_lerobot_robot_io_reads_observation_and_prints_actions(tmp_path, capsys):
    _write_lerobot_like_dataset(tmp_path)
    io = MockLeRobotRobotIO.from_config(
        {
            "dataset_root": str(tmp_path),
            "loop": False,
            "image_keys": {
                "top_head": "observation.images.top_head",
                "hand_right": "observation.images.hand_right",
                "hand_left": "observation.images.hand_left",
            },
        }
    )

    io.start()
    first = io.get_observation()
    second = io.get_observation()
    io.apply_action(np.arange(14, dtype=float))

    np.testing.assert_allclose(first["qpos"], np.arange(14, dtype=float))
    np.testing.assert_allclose(second["qpos"], np.arange(1, 15, dtype=float))
    assert set(first["images"]) == {"top_head", "hand_right", "hand_left"}
    assert first["images"]["top_head"].shape == (8, 10, 3)
    assert first["images"]["top_head"][0, 0, 0] == 10
    assert io.applied_actions[-1].shape == (14,)
    assert "[MOCK ARM] action14=" in capsys.readouterr().out
    io.close()


def test_mock_lerobot_robot_io_repeats_last_frame_when_loop_disabled(tmp_path):
    _write_lerobot_like_dataset(tmp_path)
    io = MockLeRobotRobotIO.from_config({"dataset_root": str(tmp_path), "loop": False})

    first = io.get_observation()
    second = io.get_observation()
    third = io.get_observation()

    np.testing.assert_allclose(first["qpos"], np.arange(14, dtype=float))
    np.testing.assert_allclose(second["qpos"], np.arange(1, 15, dtype=float))
    np.testing.assert_allclose(third["qpos"], np.arange(1, 15, dtype=float))
