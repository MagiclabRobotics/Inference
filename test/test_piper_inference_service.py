from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (
    REPO_ROOT / "client",
    REPO_ROOT / "client" / "inference",
    REPO_ROOT / "packages" / "openpi-client" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import InferenceConfig, apply_mode_override, load_config  # noqa: E402
from integration import inference_service  # noqa: E402
from integration.runtime_builder import build_runtime_profile  # noqa: E402
import robot_io_factory  # noqa: E402
from runtime import InferenceRuntime  # noqa: E402


PIPER_CONFIG = REPO_ROOT / "client" / "config_agilex.yaml"


class FakeCollectorIO:
    def __init__(self):
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def apply_action(self, action):
        raise AssertionError("Collector service must return actions to the caller for execution")

    def close(self):
        self.closed = True


@pytest.mark.parametrize("mode", [None, *load_config(PIPER_CONFIG).available_modes()])
def test_piper_service_session_preserves_actions_and_overrides(mode, tmp_path, monkeypatch):
    cfg = load_config(PIPER_CONFIG)
    cfg.raw["recording"] = {
        "root_dir": str(tmp_path / "records"),
        "record_model_io": False,
        "record_runtime_events": False,
        "record_action_steps": False,
    }
    cfg.raw["collector"]["keyboard_lcm"]["enabled"] = False
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg.raw), encoding="utf-8")
    io = FakeCollectorIO()
    monkeypatch.setattr(robot_io_factory, "create_robot_io", lambda config: io)

    def idle_runtime(runtime, *, execute_actions):
        assert execute_actions is False
        runtime.shutdown.wait(timeout=5.0)

    monkeypatch.setattr(inference_service, "run_embedded", idle_runtime)
    service = inference_service.InferenceService(str(config_path), startup_mode=mode)
    assert service.pop_action() == {"status": True, "action": None}
    try:
        reply = service.handle_request(
            {"cmd": "start", "params": {"policy_host": "127.0.0.1", "policy_port": 8123, "prompt": "fold"}}
        )
        assert reply["status"] is True
        assert io.started
        assert service.session_active
        assert isinstance(service._runtime, InferenceRuntime)
        runtime = service._runtime
        assert runtime.cfg["host"] == "127.0.0.1"
        assert runtime.cfg["port"] == 8123
        assert runtime.cfg["prompt"] == "fold"
        assert runtime.cfg["state_dim"] == 14
        assert runtime.cfg["publish_rate"] == 30
        if mode is not None:
            assert runtime.cfg["mode"] == mode
        runtime._policy = SimpleNamespace()
        assert service.status()["policy_ready"] is True

        actions = np.linspace(-0.5, 0.5, 28).reshape(2, 14)
        runtime.stream_buffer.integrate_new_chunk(actions, max_k=0, min_m=1, chunk_id=7)
        for index, action in enumerate(actions):
            reply = service.handle_request({"cmd": "pop_action"})
            assert reply == {
                "status": True,
                "action": action.tolist(),
                "chunk_id": 7,
                "chunk_step_index": index,
            }
        assert service.status()["pending_actions"] == 0
        reply = service.handle_request({"cmd": "stop"})
        assert reply["status"] is True
        assert reply["session_summary"]["action_pop_count"] == 2
        assert io.closed
        assert not service.session_active
    finally:
        service.stop_session()


@pytest.mark.parametrize("action_dim", [13, 16])
def test_piper_service_rejects_wrong_action_dimension(action_dim, tmp_path, monkeypatch):
    monkeypatch.setattr(inference_service.InferenceService, "_start_keyboard_lcm_listener", lambda self: None)
    service = inference_service.InferenceService(str(tmp_path / "missing.yaml"))
    service._runtime = SimpleNamespace(
        pop_action_step=lambda: {"action": np.zeros(action_dim)},
    )
    reply = service.pop_action()
    assert reply["status"] is False
    assert "expected 14" in reply["message"]


def test_unknown_mode_does_not_mutate_piper_config():
    cfg = load_config(PIPER_CONFIG)
    before = copy.deepcopy(cfg.raw)
    with pytest.raises(ValueError, match="Unsupported inference mode"):
        apply_mode_override(cfg, "unsupported_mode")
    assert cfg.raw == before


@pytest.mark.parametrize(
    "inference",
    [{"mode": "custom"}, {"execution_mode": "async", "async_mode": "unknown"}],
)
def test_service_profile_propagates_invalid_runtime_config(inference):
    cfg = InferenceConfig(raw={"inference": inference}, path=Path("config.yaml"))
    with pytest.raises(ValueError):
        build_runtime_profile(cfg)
