from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = REPO_ROOT / "packages" / "openpi-client" / "src"
CLIENT_INFERENCE_SRC = REPO_ROOT / "client" / "inference"

for path in (OPENPI_CLIENT_SRC, CLIENT_INFERENCE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import runtime  # noqa: E402
import inference_modes  # noqa: E402
from traceable_action_buffer import ActionValueSourceItem  # noqa: E402
from traceable_action_buffer import TraceableAction  # noqa: E402


class _FakeIO:
    def __init__(self):
        self.actions = []
        self.observations = 0

    def get_observation(self):
        self.observations += 1
        return {
            "qpos": np.zeros(14, dtype=float),
            "state_timestamp": 1.0,
            "image_timestamp": 2.0,
        }

    def apply_action(self, action):
        self.actions.append(np.asarray(action, dtype=float).copy())


class _FakePolicy:
    def __init__(self):
        self.calls = 0

    def infer(self, _payload):
        self.calls += 1
        return {"actions": np.asarray([[1.0] * 14, [2.0] * 14], dtype=float)}


class _RequestIdPolicy:
    def infer(self, _payload):
        return {"actions": np.asarray([[1.0] * 14], dtype=float), "server_timing": {"request_id": 42}}


class _FakeRuntimeLogger:
    def __init__(self):
        self.rows = []

    def log_action_step(self, row):
        self.rows.append(dict(row))


def test_sleep_rate_sleeps_until_next_period(monkeypatch):
    monotonic_values = iter([10.02, 10.05])
    sleeps = []

    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(runtime.time, "sleep", sleeps.append)

    next_t = runtime._sleep_rate(last_t=10.0, rate_hz=20.0)

    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(0.03)
    assert next_t == pytest.approx(10.05)


def test_sleep_rate_does_not_sleep_when_period_already_elapsed(monkeypatch):
    monotonic_values = iter([10.08, 10.081])
    sleeps = []

    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(runtime.time, "sleep", sleeps.append)

    next_t = runtime._sleep_rate(last_t=10.0, rate_hz=20.0)

    assert sleeps == []
    assert next_t == pytest.approx(10.081)


def test_sleep_rate_clamps_non_positive_rate(monkeypatch):
    monotonic_values = iter([10.0, 10.1])
    sleeps = []

    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(runtime.time, "sleep", sleeps.append)

    next_t = runtime._sleep_rate(last_t=10.0, rate_hz=0.0)

    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(1_000_000.0)
    assert next_t == pytest.approx(10.1)


def test_legacy_boolean_flag_no_longer_enables_temporal_smoothing():
    assert inference_modes.stream_smooth_method({"mode": "naive_async", "use_temporal_smoothing": True}) == "raw"


def test_vlash_can_disable_temporal_smoothing():
    assert inference_modes.stream_smooth_method({"mode": "vlash_async", "use_temporal_smoothing": False}) == "raw"


def test_stream_smooth_method_reads_only_smooth_method_key():
    cfg = {
        "async_mode": "naive",
        "temporal_method": "temporal_ensembling",
        "postprocess_method": "temporal_smoothing",
        "smooth_method": "raw",
    }

    assert inference_modes.stream_smooth_method(cfg) == "raw"


def test_stream_smooth_method_ignores_legacy_temporal_keys():
    cfg = {"async_mode": "naive", "temporal_method": "temporal_ensembling", "postprocess_method": "temporal_smoothing"}

    assert inference_modes.stream_smooth_method(cfg) == "raw"


def test_explicit_smooth_method_overrides_boolean_flag():
    assert (
        inference_modes.stream_smooth_method(
            {"mode": "legato_async", "use_temporal_smoothing": True, "smooth_method": "raw"}
        )
        == "raw"
    )


def test_async_temporal_modes_use_standard_handler_with_temporal_postprocess():
    cfg = {"execution_mode": "async", "async_mode": "temporal_ensembling"}

    assert inference_modes.stream_smooth_method(cfg) == "temporal_ensembling"


def test_ttrtc_mode_builds_model_space_prefix_payload_and_alignment_context():
    class _Buffer:
        def get_chunk_progress(self):
            return {"chunk_id": 7, "executed_steps": 2, "remaining_steps": 3, "action_horizon": 5}

        def get_prev_action_chunk_model(self):
            return np.arange(10, dtype=np.float32).reshape(5, 2)

    rt = type("Runtime", (), {})()
    rt.cfg = {
        "execution_mode": "async",
        "async_mode": "ttrtc",
        "chunk_size": 5,
        "sample_num_steps": 4,
    }
    rt.stream_buffer = _Buffer()
    rt.request_progress_before = rt.stream_buffer.get_chunk_progress()
    rt.pending_actions_model = None
    rt.get_delay_steps = lambda: 2
    rt.update_delay_steps = lambda _rtt: None
    rt.log_mode_payload = lambda _payload: None

    mode = inference_modes.build_inference_mode(rt)
    payload = mode.build_payload({}, np.zeros(2, dtype=float))

    assert isinstance(mode, inference_modes.TtrtcMode)
    assert payload["enable_ttrtc"] is True
    assert payload["execute_horizon"] == 2
    assert payload["inference_delay"] == 2
    assert payload["num_steps"] == 4
    assert payload["prev_action_chunk_model"].shape == (5, 2)
    assert mode.integration_kwargs(rt.request_progress_before) == {"drop_reference_remaining": 3}

    actions = mode.handle_result(
        {
            "actions": np.ones((5, 2), dtype=np.float32),
            "actions_model": np.full((5, 2), 2.0, dtype=np.float32),
        },
        0.1,
    )
    assert actions == pytest.approx(np.ones((5, 2)))
    assert rt.pending_actions_model == pytest.approx(np.full((5, 2), 2.0))


def test_split_mode_fields_are_normalized():
    cfg = {"execution_mode": "async", "method": "rtc", "smooth_method": "temporal_ensembling"}

    assert inference_modes.execution_mode(cfg) == "async"
    assert inference_modes.stream_smooth_method(cfg) == "temporal_ensembling"


def test_sync_loop_infers_then_executes_returned_chunk(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_sleep_rate", lambda last_t, _rate_hz: last_t)
    rt = runtime.InferenceRuntime(
        io=_FakeIO(),
        cfg={
            "execution_mode": "sync",
            "ctrl_type": "joint",
            "max_publish_step": 2,
            "publish_rate": 30,
            "log_every_steps": 1000,
            "action_buffer": "traceable",
            "smooth_method": "raw",
        },
        recording_cfg={
            "root_dir": str(tmp_path),
            "record_runtime_events": False,
            "record_action_steps": False,
            "record_model_io": False,
        },
    )
    rt._policy = _FakePolicy()
    rt._build_payload = lambda obs: ({"state": obs["qpos"]}, {})
    rt.log_event = lambda _event: None

    rt._sync_loop()

    assert rt.policy.calls == 1
    assert rt.io.observations == 1
    assert len(rt.io.actions) == 2
    assert rt.io.actions[0] == pytest.approx([1.0] * 14)
    assert rt.io.actions[1] == pytest.approx([2.0] * 14)
    assert rt.shutdown.is_set()


def test_log_action_step_records_action_value_sources():
    rt = runtime.InferenceRuntime.__new__(runtime.InferenceRuntime)
    rt.runtime_logger = _FakeRuntimeLogger()
    rt.recording_cfg = {"record_action_steps": True}

    action_step = {
        "chunk_id": 3,
        "chunk_step_index": 4,
        "action": TraceableAction(
            np.arange(14, dtype=float),
            tuple(
                ((ActionValueSourceItem(1, 4, float(dim), 0.25), ActionValueSourceItem(3, 4, 100.0 + dim, 0.75)))
                for dim in range(14)
            ),
        ),
    }
    rt._log_action_step(action_step["action"].value, action_step)  # noqa: SLF001

    assert len(rt.runtime_logger.rows) == 1
    row = rt.runtime_logger.rows[0]
    assert "action_source" not in row
    assert json.loads(row["action_value_source"])[0] == [[1, 4, 0.0, 0.25], [3, 4, 100.0, 0.75]]


def test_infer_and_integrate_uses_request_id_as_chunk_id(tmp_path):
    rt = runtime.InferenceRuntime(
        io=_FakeIO(),
        cfg={
            "execution_mode": "async",
            "async_mode": "naive",
            "latency_k": 0,
            "min_smooth_steps": 1,
            "action_buffer": "traceable",
            "smooth_method": "raw",
        },
        recording_cfg={
            "root_dir": str(tmp_path),
            "record_runtime_events": False,
            "record_action_steps": False,
            "record_model_io": False,
        },
    )
    rt._policy = _RequestIdPolicy()
    rt._build_payload = lambda obs: ({"state": obs["qpos"]}, {})  # noqa: SLF001
    rt.get_delay_steps = lambda: 0
    rt.get_delay_ref_latency_ms = lambda: None
    rt.log_event = lambda _event: None

    rt._run_inference_once(  # noqa: SLF001
        {"qpos": np.zeros(14, dtype=float)},
    )
    action_step = rt.stream_buffer.pop_next_action()

    assert action_step is not None
    assert action_step["chunk_id"] == 42
    assert "action_value_source" not in action_step
    assert isinstance(action_step["action_trace"], TraceableAction)
    assert action_step["action_trace"].value_source[0][0].chunk_id == 42
    assert action_step["action_trace"].value_source[0][0].value == 1.0
    assert action_step["action_trace"].value_source[0][0].weight == 1.0
