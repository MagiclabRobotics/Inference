from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


CLIENT_INFERENCE_SRC = Path(__file__).resolve().parents[1] / "client" / "inference"
if str(CLIENT_INFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(CLIENT_INFERENCE_SRC))

from async_inference_recorder import AsyncInferenceRecorder  # noqa: E402


class FakeVideoWriter:
    def isOpened(self):
        return True

    def write(self, frame):
        pass

    def release(self):
        pass


def _write_queued_frames(tmp_path, monkeypatch, items):
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: FakeVideoWriter())
    recorder = AsyncInferenceRecorder(str(tmp_path), camera_names=["top_head"])
    for item in items:
        recorder._video_queue.put(item)
    recorder._stop_event.set()
    recorder._video_worker()
    return recorder, Path(recorder.output_dir) / "first_frame_top_head.png"


def test_first_frame_uses_raw_rgb_and_is_not_overwritten(tmp_path, monkeypatch):
    first = np.full((12, 20, 3), [201, 31, 7], dtype=np.uint8)
    next_frame = np.full((12, 20, 3), [2, 3, 4], dtype=np.uint8)
    recorder, output = _write_queued_frames(
        tmp_path,
        monkeypatch,
        [
            {"top_head_raw_rgb": first, "top_head_chw": np.zeros((3, 4, 4), dtype=np.uint8)},
            {"top_head_raw_rgb": next_frame},
        ],
    )
    assert recorder._first_frame_saved
    np.testing.assert_array_equal(cv2.imread(str(output)), first[:, :, ::-1])
    assert recorder.get_stats()["written_video_frames"] == 2


def test_first_frame_falls_back_to_model_input(tmp_path, monkeypatch):
    chw = np.full((3, 8, 9), 0, dtype=np.uint8)
    chw[0] = 123
    _, output = _write_queued_frames(tmp_path, monkeypatch, [{"top_head_chw": chw}])
    expected_bgr = chw.transpose(1, 2, 0)[:, :, ::-1]
    np.testing.assert_array_equal(cv2.imread(str(output)), expected_bgr)


def test_first_frame_waits_for_top_head_image(tmp_path, monkeypatch):
    frame = np.full((8, 9, 3), 37, dtype=np.uint8)
    _, output = _write_queued_frames(
        tmp_path, monkeypatch, [{"hand_left_raw_rgb": frame}, {"top_head_raw_rgb": frame}]
    )
    np.testing.assert_array_equal(cv2.imread(str(output)), frame)


@pytest.mark.parametrize("raises", [False, True])
def test_first_frame_write_failure_keeps_video_recording_alive(tmp_path, monkeypatch, raises):
    def fail_write(path, frame):
        if raises:
            raise OSError("write failed")
        return False

    monkeypatch.setattr(cv2, "imwrite", fail_write)
    frame = np.zeros((8, 9, 3), dtype=np.uint8)
    recorder, output = _write_queued_frames(tmp_path, monkeypatch, [{"top_head_raw_rgb": frame}])
    assert not recorder._first_frame_saved
    assert not output.exists()
    assert recorder.get_stats()["written_video_frames"] == 1
