from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_SRC = REPO_ROOT / "server"

if str(SERVER_SRC) not in sys.path:
    sys.path.insert(0, str(SERVER_SRC))

from openpi.policies import policy as server_policy  # noqa: E402


def test_batch_tree_to_torch_reuses_cpu_numpy_storage():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)

    out = server_policy._batch_tree_to_torch({"x": arr}, "cpu")

    assert out["x"].shape == (1, 2, 3)
    arr[0, 0] = 99.0
    assert out["x"][0, 0, 0].item() == 99.0


def test_batch_tree_to_torch_reuses_cpu_tensor_storage():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    out = server_policy._batch_tree_to_torch({"x": tensor}, "cpu")

    assert out["x"].shape == (1, 2, 3)
    assert out["x"].data_ptr() == tensor.data_ptr()


def test_batch_tree_to_torch_makes_non_contiguous_numpy_contiguous():
    arr = np.arange(12, dtype=np.uint8).reshape(3, 2, 2).transpose(1, 2, 0)

    out = server_policy._batch_tree_to_torch({"x": arr}, "cpu")

    assert out["x"].shape == (1, 2, 2, 3)
    assert out["x"].is_contiguous()


def test_batch_tree_to_jax_adds_batch_before_device_conversion(monkeypatch):
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    seen_shapes = []

    def fake_asarray(value):
        seen_shapes.append(np.asarray(value).shape)
        return np.asarray(value)

    monkeypatch.setattr(server_policy.jnp, "asarray", fake_asarray)

    out = server_policy._batch_tree_to_jax({"x": arr, "flag": np.True_})

    assert seen_shapes == [(1, 2, 3), (1,)]
    assert out["x"].shape == (1, 2, 3)
    assert out["flag"].shape == (1,)


def test_unbatch_policy_outputs_keeps_raw_actions_model_separate_from_transformed_actions():
    state = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    actions = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)

    outputs, actions_model_np = server_policy._unbatch_policy_outputs(
        state=state,
        actions_model=actions,
        is_pytorch=True,
    )

    np.testing.assert_array_equal(outputs["state"], np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(actions_model_np, np.arange(6, dtype=np.float32).reshape(2, 3))
    np.testing.assert_array_equal(outputs["actions"], actions_model_np)

    outputs["actions"][0, 0] = -1.0
    assert actions_model_np[0, 0] == 0.0
