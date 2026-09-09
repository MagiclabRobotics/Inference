from __future__ import annotations

import json
from pathlib import Path

from integration.inference_frame_tracer import InferenceFrameTracer, should_prefer_save_path
from integration.inference_service import InferenceService
from integration.keyboard_lcm_listener import CommandType
from integration.lcm_types.keyboard.keyboard_command_t import keyboard_command_t


def test_inference_frame_tracer_writes_json(tmp_path: Path) -> None:
    tracer = InferenceFrameTracer()
    tracer.begin_session(save_path=tmp_path, session_id="round_1", started_at_ns=123)
    tracer.record_inference(
        {
            "image_timestamp": 1781664657.1458614,
            "image_timestamps": {
                "top_head": 1781664657.1791713,
                "hand_right": 1781664657.2124844,
                "hand_left": 1781664657.1458614,
            },
            "state_timestamp": 1781664657.152848,
        },
        request_id=2,
        wall_time_sec=1781664657.2972364,
    )
    out = tracer.flush()
    assert out is not None
    data = json.loads(Path(out).read_text(encoding="utf-8"))
    assert data["session_id"] == "round_1"
    assert data["save_path"] == str(tmp_path)
    assert len(data["inference_frames"]) == 1
    frame = data["inference_frames"][0]
    assert frame["seq"] == 1
    assert frame["request_id"] == 2
    assert frame["image_timestamp"] == 1781664657.1458614


def test_inference_frame_tracer_uses_fallback_when_primary_not_writable(tmp_path: Path) -> None:
    tracer = InferenceFrameTracer(fallback_root=tmp_path / "fallback")
    blocked = Path("/root/data_output/mcap_source/teleop_test")
    tracer.begin_session(save_path=blocked, session_id="teleop_test")
    tracer.record_inference(
        {"image_timestamp": 1.0, "image_timestamps": {"top_head": 1.0}, "state_timestamp": 1.1},
        request_id=1,
    )
    out = tracer.flush()
    assert out is not None
    data = json.loads(Path(out).read_text(encoding="utf-8"))
    assert data["write_fallback"] is True
    assert data["intended_save_path"] == str(blocked)
    assert Path(data["written_path"]).exists()


def test_should_prefer_round_path_over_teleop_day_path() -> None:
    teleop = "/root/data_output/mcap_source/teleop_20260622"
    round_path = "/root/data_output/mcap_source/20260622_193947/round_1"
    assert should_prefer_save_path(round_path, teleop)
    assert not should_prefer_save_path(teleop, round_path)


def test_tracer_refines_save_path_without_resetting_frames(tmp_path: Path) -> None:
    tracer = InferenceFrameTracer()
    coarse = tmp_path / "teleop_20260622"
    fine = tmp_path / "20260622_193947" / "round_1"
    tracer.begin_session(save_path=coarse, session_id="teleop_20260622")
    tracer.record_inference({"image_timestamp": 1.0, "image_timestamps": {"top_head": 1.0}, "state_timestamp": 1.1})
    tracer.update_save_path(fine, session_id="round_1")
    tracer.record_inference({"image_timestamp": 2.0, "image_timestamps": {"top_head": 2.0}, "state_timestamp": 2.1})
    out = tracer.flush()
    data = json.loads(Path(out).read_text(encoding="utf-8"))
    assert data["save_path"] == str(fine)
    assert data["frame_count"] == 2


def test_keyboard_command_handler_starts_and_flushes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(InferenceService, "_start_keyboard_lcm_listener", lambda self: None)
    service = InferenceService(default_config_path=str(tmp_path / "missing.yaml"))
    round_dir = tmp_path / "20260622_193947" / "round_1"
    start_msg = keyboard_command_t()
    start_msg.timestamp = 999
    start_msg.save_path = str(round_dir)
    start_msg.record_type = CommandType["Start_Record"]
    start_msg.is_model = False
    start_msg.tag = 0
    service._on_keyboard_command(start_msg)
    assert service._inference_frame_tracer.is_active()

    service._inference_frame_tracer.record_inference(
        {"image_timestamp": 1.0, "image_timestamps": {"top_head": 1.0}, "state_timestamp": 1.1},
        request_id=1,
    )
    save_msg = keyboard_command_t()
    save_msg.save_path = str(round_dir)
    save_msg.record_type = CommandType["Save_Record"]
    service._on_keyboard_command(save_msg)
    assert (round_dir / "inference_frames.json").exists()


def test_keyboard_command_refines_path_on_start_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(InferenceService, "_start_keyboard_lcm_listener", lambda self: None)
    service = InferenceService(default_config_path=str(tmp_path / "missing.yaml"))
    coarse = tmp_path / "teleop_20260622"
    fine = tmp_path / "20260622_193947" / "round_1"

    infer_msg = keyboard_command_t()
    infer_msg.save_path = str(coarse)
    infer_msg.record_type = 0
    infer_msg.is_model = True
    infer_msg.tag = 2
    service._on_keyboard_command(infer_msg)
    assert service._inference_frame_tracer.get_save_path() == str(coarse)

    start_msg = keyboard_command_t()
    start_msg.save_path = str(fine)
    start_msg.record_type = CommandType["Start_Record"]
    service._on_keyboard_command(start_msg)
    assert service._inference_frame_tracer.get_save_path() == str(fine)
