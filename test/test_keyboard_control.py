from __future__ import annotations

from pathlib import Path
import os
import signal
import sys
import termios
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "client" / "inference", REPO_ROOT / "packages" / "openpi-client" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import agilex_inference_openpi as client  # noqa: E402
from config import InferenceConfig  # noqa: E402
from keyboard_control import KeyboardEpisodeController, _control_key  # noqa: E402
from robot_io import PiperDualArm, PiperHighFollowConfig  # noqa: E402
from runtime import InferenceRuntime  # noqa: E402


def _wait_until(predicate):
    deadline = time.monotonic() + 2.0
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for keyboard event"
        time.sleep(0.005)


@pytest.mark.parametrize(("value", "expected"), [(None, "s"), ("S", "s"), ("space", " "), (" ", " ")])
def test_control_key_normalization(value, expected):
    assert _control_key(value, "s") == expected


@pytest.mark.parametrize("cfg", [{"start_key": "ss"}, {"quit_key": "S"}, {"stop_key": "回"}])
def test_control_keys_reject_ambiguous_bindings(cfg):
    with pytest.raises(ValueError):
        KeyboardEpisodeController(cfg)


def test_keyboard_pty_start_stop_quit_and_terminal_restore(monkeypatch):
    master, slave = os.openpty()
    original_attrs = termios.tcgetattr(slave)
    stdin = os.fdopen(slave)
    controller = KeyboardEpisodeController({})
    starts = []
    waiter = threading.Thread(target=lambda: starts.append(controller.wait_for_start()))
    stopped = threading.Event()
    runtime = SimpleNamespace(request_episode_stop=stopped.set)
    try:
        monkeypatch.setattr(sys, "stdin", stdin)
        controller.start()
        waiter.start()
        _wait_until(lambda: controller._accept_start)
        assert starts == []
        os.write(master, b"s")
        waiter.join(timeout=2.0)
        assert starts == [True]
        controller.set_runtime(runtime)
        os.write(master, b" ")
        assert stopped.wait(timeout=2.0)
        os.write(master, b"q")
        _wait_until(lambda: controller.quit_requested)
        assert controller.wait_for_start() is False
    finally:
        controller.close()
        if waiter.ident is not None:
            waiter.join(timeout=2.0)
        assert termios.tcgetattr(slave) == original_attrs
        stdin.close()
        os.close(master)


def test_start_key_is_ignored_during_episode_creation_and_cleanup():
    controller = KeyboardEpisodeController({})
    controller._handle_key("s")
    assert not controller._start_requested.is_set()
    controller.set_runtime(SimpleNamespace())
    controller._handle_key("s")
    assert not controller._start_requested.is_set()
    controller.set_runtime(None)
    controller._handle_key("s")
    assert not controller._start_requested.is_set()


def test_quit_during_episode_creation_stops_runtime_on_attach():
    controller = KeyboardEpisodeController({})
    controller._handle_key("q")
    stopped = threading.Event()
    controller.set_runtime(SimpleNamespace(request_episode_stop=stopped.set))
    assert stopped.is_set()
    assert controller.wait_for_start() is False


def test_start_unpaused_only_auto_starts_the_first_episode():
    controller = KeyboardEpisodeController({"start_paused": False})
    assert controller.wait_for_start() is True
    assert not controller._auto_start
    assert not controller._start_requested.is_set()
    controller.close()
    assert controller.wait_for_start() is False


def test_stop_during_episode_creation_stops_runtime_on_attach():
    controller = KeyboardEpisodeController({"start_paused": False})
    assert controller.wait_for_start() is True
    controller._handle_key(" ")
    stopped = threading.Event()
    controller.set_runtime(SimpleNamespace(request_episode_stop=stopped.set))
    assert stopped.is_set()


def test_keyboard_without_tty_fails_before_robot_creation(monkeypatch):
    cfg = InferenceConfig(raw={"keyboard_control": {"enabled": True}}, path=Path("unused.yaml"))
    monkeypatch.setattr("config.load_config", lambda path: cfg)
    monkeypatch.setattr(sys, "argv", ["client", "--config", "unused.yaml"])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(client, "_make_robot_io", lambda *args: pytest.fail("robot must not be created"))
    with pytest.raises(RuntimeError, match="interactive TTY"):
        client.main()


@pytest.mark.parametrize("flag", ["--no-keyboard-control", "--check-hardware"])
def test_noninteractive_automatic_run_and_hardware_check_remain_available(tmp_path, monkeypatch, flag):
    cfg = InferenceConfig(
        raw={"keyboard_control": {"enabled": True}, "inference": {"execution_mode": "sync"},
             "recording": {"root_dir": str(tmp_path)}},
        path=tmp_path / "config.yaml",
    )
    events = []
    io = SimpleNamespace(
        start=lambda: events.append("io_start"), close=lambda: events.append("io_close"),
        move_to_init=lambda cfg: events.append("init"),
    )
    telemetry = SimpleNamespace(start_viewer=lambda: None, close=lambda: events.append("telemetry_close"))
    runtime = SimpleNamespace(
        run=lambda: events.append("run"), close=lambda: events.append("runtime_close"),
        signal_shutdown_received=False,
    )
    monkeypatch.setattr("config.load_config", lambda path: cfg)
    monkeypatch.setattr(sys, "argv", ["client", flag])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(client, "_make_robot_io", lambda *args: io)
    monkeypatch.setattr("realtime_plot.TelemetryPublisher", lambda cfg: telemetry)
    monkeypatch.setattr("runtime.InferenceRuntime", lambda **kwargs: runtime)
    monkeypatch.setattr("robot_io.check_hardware", lambda io: events.append("check"))
    client.main()
    if flag == "--check-hardware":
        assert events == ["io_start", "check", "io_close", "telemetry_close"]
    else:
        assert events == ["io_start", "init", "run", "runtime_close", "io_close", "telemetry_close"]


class FakeIO:
    def __init__(self):
        self.actions = []
        self.holds = 0
        self.init_moves = 0

    def apply_action(self, action, **kwargs):
        self.actions.append(action.copy())

    def hold_current_position(self):
        self.holds += 1

    def move_to_init(self, cfg):
        self.init_moves += 1


def _runtime(tmp_path, io=None):
    return InferenceRuntime(
        io=io or FakeIO(),
        cfg={"execution_mode": "sync"},
        recording_cfg={
            "root_dir": str(tmp_path),
            "record_runtime_events": False,
            "record_action_steps": False,
            "record_model_io": False,
        },
    )


def test_episode_stop_rejects_late_action_and_keeps_robot_connected(tmp_path):
    io = FakeIO()
    runtime = _runtime(tmp_path, io)
    action = np.zeros(14)
    assert runtime._apply_action_if_running(action)
    runtime.request_episode_stop()
    assert runtime.episode_stop_requested
    assert runtime.shutdown.is_set()
    assert not runtime.signal_shutdown_received
    assert not runtime._apply_action_if_running(action)
    assert len(io.actions) == 1
    assert io.holds == 1
    runtime.close()
    assert io.holds == 2


def test_stop_during_inference_discards_late_response(tmp_path):
    runtime = _runtime(tmp_path)
    runtime._build_payload = lambda obs: ({}, {})

    def infer(payload):
        runtime.request_episode_stop()
        return {"actions": np.ones((2, 14))}

    runtime._policy = SimpleNamespace(infer=infer)
    runtime._run_inference_once({})
    assert runtime.stream_buffer.pending_count() == 0
    assert runtime.inference_count == 0
    runtime.close()


@pytest.mark.parametrize("fail", [False, True])
def test_runtime_restores_signal_handlers_between_episodes(tmp_path, monkeypatch, fail):
    runtime = _runtime(tmp_path)
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    runtime._policy = SimpleNamespace(get_server_metadata=lambda: {})
    runtime._warmup_inference = lambda: None

    def run_episode():
        if fail:
            raise RuntimeError("episode failed")
        runtime.request_episode_stop()

    runtime._sync_loop = run_episode
    try:
        if fail:
            with pytest.raises(RuntimeError, match="episode failed"):
                runtime.run()
        else:
            runtime.run()
        assert {sig: signal.getsignal(sig) for sig in before} == before
    finally:
        runtime.close()


def test_keyboard_sessions_create_distinct_episode_directories(tmp_path):
    io = FakeIO()
    controller = KeyboardEpisodeController({})
    remaining = iter([True, True, False])
    controller.wait_for_start = lambda: next(remaining)
    episodes = []

    class TestRuntime(InferenceRuntime):
        def run(self):
            episodes.append(self.output_manager.episode_dir)
            self.request_episode_stop()

    result = client._run_keyboard_episodes(
        controller, io, TestRuntime, {"execution_mode": "sync"},
        {"root_dir": str(tmp_path), "record_model_io": False, "record_action_steps": False,
         "record_runtime_events": False}, {}, None,
    )
    assert result is False
    assert [p.name for p in episodes] == ["episode_1", "episode_2"]
    assert all(p.is_dir() for p in episodes)
    assert io.init_moves == 2
    assert io.holds == 4
    assert controller._runtime is None


def test_recording_run_directory_is_shared_by_episodes():
    cfg = InferenceConfig(
        raw={"recording": {"root_dir": "client/inference_records", "unique_run_dir": True}},
        path=Path("config.yaml"),
    )
    first = client._recording_config(cfg, run_subdir="run_example")
    second = client._recording_config(cfg, run_subdir="run_example")
    assert first == second
    assert first["root_dir"] == str(REPO_ROOT / "client/inference_records/run_example")
    with pytest.raises(ValueError, match="run_subdir"):
        client._recording_config(cfg)


def test_hold_clears_pending_high_follow_segment_without_changing_pose(monkeypatch):
    arms = PiperDualArm(left=None, right=None, high_follow_config=PiperHighFollowConfig(enabled=True))
    hold = np.arange(14, dtype=float)
    arms._high_follow_last_ref = hold.copy()
    arms._high_follow_queue.append(hold + 1)
    arms._high_follow_metadata_queue.append({"chunk_id": 1})
    arms._high_follow_segment_target = hold + 1
    restarted = []
    monkeypatch.setattr(arms, "start_high_follow_control", lambda: restarted.append(True))
    arms.hold_current_position()
    assert not arms._high_follow_queue
    assert not arms._high_follow_metadata_queue
    assert arms._high_follow_segment_target is None
    np.testing.assert_array_equal(arms._high_follow_anchor_waypoint, hold)
    np.testing.assert_array_equal(arms._high_follow_last_ref, hold)
    assert restarted == [True]


def test_hold_does_not_restart_worker_that_failed_to_stop(monkeypatch):
    arms = PiperDualArm(left=None, right=None, high_follow_config=PiperHighFollowConfig(enabled=True))
    arms._high_follow_thread = SimpleNamespace(join=lambda timeout: None, is_alive=lambda: True)
    monkeypatch.setattr(arms, "start_high_follow_control", lambda: pytest.fail("must not create a second worker"))
    with pytest.raises(RuntimeError, match="did not stop"):
        arms.hold_current_position()
