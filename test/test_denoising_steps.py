from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = REPO_ROOT / "packages" / "openpi-client" / "src"
CLIENT_INFERENCE_SRC = REPO_ROOT / "client" / "inference"
SERVER_SRC = REPO_ROOT / "server"

for path in (OPENPI_CLIENT_SRC, CLIENT_INFERENCE_SRC, SERVER_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import runtime  # noqa: E402
from openpi.policies import policy as server_policy  # noqa: E402
from openpi.models import pi0_config  # noqa: E402


class _FakeSnapflowTorchModel:
    def __init__(self):
        self.calls = []

    def to(self, _device):
        return self

    def eval(self):
        return self

    def sample_actions(self, device, observation, **kwargs):
        self.calls.append(("sample_actions", device, observation, kwargs))
        import torch

        return torch.zeros((observation.state.shape[0], 2, 3), device=observation.state.device)

    def sample_actions_one_step(self, device, observation, **kwargs):
        self.calls.append(("sample_actions_one_step", device, observation, kwargs))
        import torch

        return torch.ones((observation.state.shape[0], 2, 3), device=observation.state.device)


class _FakeTtrtcJaxModel:
    action_horizon = 2
    action_dim = 3

    def __init__(self):
        self.calls = []

    def sample_actions(self, rng, observation, **kwargs):
        self.calls.append(("sample_actions", rng, observation, kwargs))
        import jax.numpy as jnp

        return jnp.zeros((observation.state.shape[0], self.action_horizon, self.action_dim))

    def ttrtc_sample_actions(self, rng, observation, **kwargs):
        self.calls.append(("ttrtc_sample_actions", rng, observation, kwargs))
        import jax.numpy as jnp

        return jnp.ones((observation.state.shape[0], self.action_horizon, self.action_dim))


def _minimal_model_payload(*, num_steps: int | None = None) -> dict:
    import numpy as np

    payload = {
        "image": {
            "base_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.array(True),
            "left_wrist_0_rgb": np.array(True),
            "right_wrist_0_rgb": np.array(True),
        },
        "state": np.zeros((3,), dtype=np.float32),
        "prev_action_chunk": np.zeros((2, 3), dtype=np.float32),
    }
    if num_steps is not None:
        payload["num_steps"] = num_steps
    return payload


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        ({"num_denoising_steps": "7"}, 7),
        ({"denoising_steps": 6}, 6),
        ({"num_steps": 5}, 5),
        ({}, None),
    ],
)
def test_configured_num_denoising_steps_aliases(cfg, expected):
    assert runtime._configured_num_denoising_steps(cfg) == expected


def test_request_num_denoising_steps_prefers_canonical_num_steps():
    obs = {"num_steps": "4", "num_denoising_steps": 8}

    assert server_policy._request_num_denoising_steps(obs) == 4


def test_request_num_denoising_steps_rejects_non_positive_values():
    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        server_policy._request_num_denoising_steps({"num_steps": 0})


def test_snapflow_policy_uses_one_step_without_client_payload_changes():
    model = _FakeSnapflowTorchModel()
    policy = server_policy.Policy(
        model,
        sample_kwargs={"use_snapflow_inference": True},
        is_pytorch=True,
        pytorch_device="cpu",
    )

    result = policy.infer(_minimal_model_payload())

    assert result["actions"].shape == (2, 3)
    assert model.calls[0][0] == "sample_actions_one_step"
    assert model.calls[0][3] == {}


def test_snapflow_policy_uses_multistep_when_requested():
    model = _FakeSnapflowTorchModel()
    policy = server_policy.Policy(
        model,
        sample_kwargs={"use_snapflow_inference": True},
        is_pytorch=True,
        pytorch_device="cpu",
    )

    result = policy.infer(_minimal_model_payload(num_steps=4))

    assert result["actions"].shape == (2, 3)
    assert model.calls[0][0] == "sample_actions"
    assert model.calls[0][3] == {"num_steps": 4}


def test_policy_rejects_snapflow_and_legato_together():
    with pytest.raises(ValueError, match="cannot both be enabled"):
        server_policy.Policy(
            _FakeSnapflowTorchModel(),
            sample_kwargs={"use_snapflow_inference": True, "use_legato_inference": True},
            is_pytorch=True,
            pytorch_device="cpu",
        )


def test_pi0_snapflow_config_is_opt_in():
    assert pi0_config.Pi0Config(pi05=True).enable_snapflow is False
    assert pi0_config.Pi0Config(pi05=True, enable_snapflow=True).enable_snapflow is True


def test_ttrtc_policy_rejects_pytorch_backend():
    with pytest.raises(ValueError, match="only by the JAX"):
        server_policy.Policy(
            _FakeSnapflowTorchModel(),
            sample_kwargs={"use_ttrtc_inference": True},
            is_pytorch=True,
            pytorch_device="cpu",
        )


def test_ttrtc_policy_routes_model_space_prefix(monkeypatch):
    monkeypatch.setattr(server_policy.nnx_utils, "module_jit", lambda fn, **_kwargs: fn)
    model = _FakeTtrtcJaxModel()
    policy = server_policy.Policy(
        model,
        sample_kwargs={"use_ttrtc_inference": True},
        is_pytorch=False,
    )
    payload = _minimal_model_payload(num_steps=4)
    payload.pop("prev_action_chunk")
    payload["prev_action_chunk_model"] = np.zeros((2, 3), dtype=np.float32)
    payload["inference_delay"] = 1
    payload["execute_horizon"] = 1

    result = policy.infer(payload)

    assert result["actions"].shape == (2, 3)
    assert model.calls[0][0] == "ttrtc_sample_actions"
    kwargs = model.calls[0][3]
    assert kwargs["num_steps"] == 4
    assert kwargs["inference_delay"] == 1
    assert kwargs["execute_horizon"] == 1
    assert kwargs["prev_actions"].shape == (1, 2, 3)


def test_ttrtc_model_module_contains_no_training_loss():
    source_path = REPO_ROOT / "server" / "openpi" / "models" / "pi0_ttrtc.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    ttrtc_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Pi0TTRTC"
    )

    assert all(
        not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "compute_loss"
        for node in ttrtc_class.body
    )
    assert pi0_config.Pi0TTRTCConfig(pi05=False).model_type == pi0_config.Pi0Config(pi05=False).model_type
    assert pi0_config.Pi0TTRTCConfig(pi05=True).model_type == pi0_config.Pi0Config(pi05=True).model_type
