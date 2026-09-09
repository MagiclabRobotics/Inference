from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "client/inference", ROOT / "server", ROOT / "packages/openpi-client/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from action_buffers import create_action_buffer
from inference_modes import RtcMode
from runtime import InferenceRuntime
import runtime as runtime_module
from openpi.models import model as model_module
from openpi.models.pi0_rtc import Pi0RTC
from openpi.policies.policy import Policy


HORIZON = 6
ACTION_DIM = 14


class _TinyLLM(nnx.Module):
    def __call__(self, tokens, **kwargs):
        del kwargs
        prefix, suffix = tokens
        return (prefix, None if suffix is None else 0.25 * suffix), None


class _TinyRTC(Pi0RTC):
    """Run the real RTC sampler and JIT with a tiny differentiable velocity field."""

    def __init__(self, action_dim=ACTION_DIM):
        model_module.BaseModel.__init__(self, action_dim, HORIZON, 8)
        self.PaliGemma = nnx.Dict(llm=_TinyLLM())

    def embed_prefix(self, obs):
        batch = obs.state.shape[0]
        return jnp.zeros((batch, 1, self.action_dim)), jnp.ones((batch, 1), dtype=bool), jnp.array([False])

    def embed_suffix(self, obs, actions, timestep):
        del obs, timestep
        return actions, jnp.ones(actions.shape[:2], dtype=bool), jnp.array([True] + [False] * (HORIZON - 1)), None

    def action_out_proj(self, values):
        return values


class _StandardModel(nnx.Module):
    def sample_actions(self, rng, obs, *, num_steps=10):
        del rng, num_steps
        return jnp.zeros((obs.state.shape[0], HORIZON, ACTION_DIM))


def _model_payload(**kwargs):
    return {
        "image": {key: np.zeros((224, 224, 3), dtype=np.uint8) for key in model_module.IMAGE_KEYS},
        "image_mask": {key: np.True_ for key in model_module.IMAGE_KEYS},
        "state": np.zeros(ACTION_DIM, dtype=np.float32),
        "num_steps": 2,
        **kwargs,
    }


def _client_runtime(buffer_type="stream", smooth_method="raw"):
    cfg = {
        "chunk_size": HORIZON,
        "action_buffer": buffer_type,
        "execute_horizon": 4,
        "mask_prefix_delay": True,
        "prefix_attention_schedule": "linear",
        "max_guidance_weight": 0.07,
    }
    return SimpleNamespace(
        cfg=cfg,
        stream_buffer=create_action_buffer(cfg, max_chunks=4, state_dim=ACTION_DIM, smooth_method=smooth_method),
        pending_actions_model=None,
        request_progress_before=None,
        get_delay_steps=lambda: 1,
        update_delay_steps=lambda _: None,
        log_mode_payload=lambda _: None,
    )


@pytest.mark.parametrize("buffer_type,smoothing", [("stream", "raw"), ("traceable", "raw"), ("stream", "temporal_smoothing")])
def test_rtc_aligns_accepted_model_actions_and_skips_rpc_elapsed_steps(buffer_type, smoothing):
    rt = _client_runtime(buffer_type, smoothing)
    mode = RtcMode(rt)
    cold = mode.build_payload({}, np.zeros(ACTION_DIM))
    assert "prev_action_chunk_model" not in cold
    assert cold["enable_rtc"] is True
    old_model = np.arange(HORIZON * ACTION_DIM, dtype=np.float32).reshape(HORIZON, ACTION_DIM) / 100
    actions = mode.handle_result({"actions": old_model * 10, "actions_model": old_model}, 0.1)
    rt.stream_buffer.integrate_new_chunk(actions, max_k=0, actions_model_chunk=rt.pending_actions_model, drop_n=1)
    rt.stream_buffer.pop_next_action()
    rt.stream_buffer.pop_next_action()
    rt.request_progress_before = rt.stream_buffer.get_chunk_progress()
    payload = mode.build_payload({}, np.zeros(ACTION_DIM))
    assert "prev_action_chunk" not in payload
    np.testing.assert_array_equal(payload["prev_action_chunk_model"], old_model[[3, 4, 5, 5, 5, 5]])
    np.testing.assert_array_equal(rt.stream_buffer.get_prev_action_chunk_model(), old_model)

    # One action is consumed after submission and before the response is integrated.
    rt.stream_buffer.pop_next_action()
    next_model = old_model + 1
    actions = mode.handle_result({"actions": next_model * 10, "actions_model": next_model}, 0.1)
    # Only an accepted buffer update may replace the previous reference.
    np.testing.assert_array_equal(rt.stream_buffer.get_prev_action_chunk_model(), old_model)
    switch = rt.stream_buffer.integrate_new_chunk(
        actions, max_k=0, actions_model_chunk=rt.pending_actions_model,
        **mode.integration_kwargs(rt.request_progress_before),
    )
    assert switch["dropped_new_chunk_steps"] == 1
    if smoothing == "raw":
        published = rt.stream_buffer.pop_next_action()["action"]
        np.testing.assert_allclose(np.asarray(published), (next_model * 10)[1])


@pytest.mark.parametrize("model_actions", [None, np.zeros((2, ACTION_DIM)), np.full((HORIZON, ACTION_DIM), np.nan)])
def test_rtc_rejects_missing_or_invalid_model_space_output(model_actions):
    rt = _client_runtime()
    rt.pending_actions_model = np.ones((HORIZON, ACTION_DIM))
    with pytest.raises(ValueError):
        RtcMode(rt).handle_result({"actions": np.ones((HORIZON, ACTION_DIM)), "actions_model": model_actions}, 0.1)
    assert rt.pending_actions_model is None
    assert rt.stream_buffer.get_prev_action_chunk_model() is None


def test_rtc_warmup_uses_model_space_and_preserves_sampling_options():
    rt = object.__new__(InferenceRuntime)
    rt.cfg = {**_client_runtime().cfg, "async_mode": "rtc"}
    rt._base_payload = lambda _: ({"num_steps": 2}, {}, np.zeros(ACTION_DIM))
    model_actions = np.full((HORIZON, ACTION_DIM), 0.25, dtype=np.float32)
    payload = rt._build_warmup_continuation_payload({}, {"actions": model_actions * 10, "actions_model": model_actions})
    np.testing.assert_array_equal(payload["prev_action_chunk_model"], model_actions)
    assert "prev_action_chunk" not in payload
    assert payload["prefix_attention_schedule"] == "linear"
    assert payload["num_steps"] == 2
    assert payload["inference_delay"] == 0
    assert rt._build_warmup_continuation_payload({}, {"actions": model_actions * 10}) is None


def test_rtc_client_server_roundtrip_preserves_normalized_reference_and_options():
    def to_robot(data):
        return {**data, "actions": data["actions"][:, :ACTION_DIM] * 10}

    policy = Policy(_TinyRTC(action_dim=32), output_transforms=[to_robot])
    noise = np.ones((HORIZON, 32), dtype=np.float32)
    first = policy.infer(_model_payload(enable_rtc=True), noise=noise)
    rt = _client_runtime()
    mode = RtcMode(rt)
    actions = mode.handle_result(first, 0.1)
    rt.stream_buffer.integrate_new_chunk(actions, max_k=0, actions_model_chunk=rt.pending_actions_model)
    payload = mode.build_payload(_model_payload(), np.zeros(ACTION_DIM))
    calls = []
    sampler = policy._sample_actions

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return sampler(*args, **kwargs)

    policy._sample_actions = capture
    second = policy.infer(payload, noise=noise)
    kwargs = calls[-1]
    np.testing.assert_array_equal(np.asarray(kwargs["prev_action_chunk"])[0], first["actions_model"])
    assert np.asarray(kwargs["prev_action_chunk"]).shape == (1, HORIZON, 32)
    assert not np.allclose(first["actions"], first["actions_model"][:, :ACTION_DIM])
    assert kwargs["enable_rtc"] is True
    assert kwargs["mask_prefix_delay"] is True
    assert kwargs["prefix_attention_schedule"] == "linear"
    assert kwargs["max_guidance_weight"] == 0.07
    assert kwargs["execute_horizon"] == 4
    assert kwargs["inference_delay"] == 1
    assert np.isfinite(second["actions"]).all()


def test_runtime_rtc_continuation_uses_published_progress(monkeypatch, tmp_path):
    old_model = np.arange(HORIZON * 32, dtype=np.float32).reshape(HORIZON, 32) / 100
    next_model = old_model + 1
    requests = []

    def infer(payload):
        requests.append(payload)
        model_actions = old_model if len(requests) == 1 else next_model
        if len(requests) == 2:
            rt.stream_buffer.pop_next_action()  # Publication continues during RPC.
        return {"actions": model_actions[:, :ACTION_DIM] * 10, "actions_model": model_actions}

    monkeypatch.setattr(runtime_module, "create_policy_client", lambda _: SimpleNamespace(infer=infer))
    rt = InferenceRuntime(
        io=SimpleNamespace(),
        cfg={"execution_mode": "async", "async_mode": "rtc", "chunk_size": HORIZON, "smooth_method": "raw"},
        recording_cfg={"root_dir": str(tmp_path), "record_model_io": False,
                       "record_runtime_events": False, "record_action_steps": False},
    )
    obs = {"qpos": np.zeros(ACTION_DIM), "images": {
        key: np.zeros((8, 8, 3), dtype=np.uint8) for key in ("top_head", "hand_left", "hand_right")
    }}
    try:
        rt._run_inference_once(obs)
        rt.stream_buffer.pop_next_action()
        rt.stream_buffer.pop_next_action()
        rt._run_inference_once(obs)
        np.testing.assert_array_equal(requests[1]["prev_action_chunk_model"], old_model[[2, 3, 4, 5, 5, 5]])
        np.testing.assert_allclose(rt.stream_buffer.pop_next_action()["action"], next_model[1, :ACTION_DIM] * 10)
    finally:
        rt.close()


@pytest.mark.parametrize("schedule", ["exp", "linear", "ones", "zeros"])
@pytest.mark.parametrize("mask_prefix", [False, True])
def test_rtc_real_jit_accepts_static_options_and_disable_switch(schedule, mask_prefix):
    policy = Policy(_TinyRTC())
    reference = np.full((HORIZON, ACTION_DIM), 0.4, dtype=np.float32)
    noise = np.zeros_like(reference)
    payload = _model_payload(
        enable_rtc=True, prev_action_chunk_model=reference, inference_delay=1, execute_horizon=4,
        prefix_attention_schedule=schedule, mask_prefix_delay=mask_prefix, max_guidance_weight=0.7,
    )
    guided = policy.infer(payload, noise=noise)
    plain = policy.infer({**payload, "enable_rtc": False}, noise=noise)
    assert np.isfinite(guided["actions"]).all()
    assert np.any(np.abs(guided["actions"]) > 0)
    np.testing.assert_allclose(plain["actions"], 0, atol=1e-6)


def test_rtc_zero_guidance_weight_disables_correction():
    policy = Policy(_TinyRTC())
    reference = np.full((HORIZON, ACTION_DIM), 0.4, dtype=np.float32)
    result = policy.infer(_model_payload(
        enable_rtc=True, prev_action_chunk_model=reference, inference_delay=1, execute_horizon=4,
        max_guidance_weight=0.0, mask_prefix_delay=False,
    ), noise=np.zeros_like(reference))
    np.testing.assert_allclose(result["actions"], 0, atol=1e-6)


@pytest.mark.parametrize("options", [
    {"prev_action_chunk": np.zeros((HORIZON, ACTION_DIM))},
    {"prev_action_chunk_model": np.zeros((HORIZON, ACTION_DIM - 1))},
    {"prev_action_chunk_model": np.full((HORIZON, ACTION_DIM), np.nan)},
    {"mask_prefix_delay": "false"},
    {"max_guidance_weight": -0.1},
    {"max_guidance_weight": np.inf},
    {"prefix_attention_schedule": "typo"},
])
def test_rtc_rejects_ambiguous_or_invalid_requests(options):
    policy = Policy(_TinyRTC())
    with pytest.raises(ValueError, match="RTC"):
        policy.infer(_model_payload(enable_rtc=True, **options))


def test_rtc_request_requires_matching_server_but_standard_inference_still_works():
    policy = Policy(_StandardModel())
    with pytest.raises(ValueError, match="Pi0RTC"):
        policy.infer(_model_payload(enable_rtc=True))
    result = policy.infer(_model_payload())
    np.testing.assert_array_equal(result["actions"], np.zeros((HORIZON, ACTION_DIM)))


def test_rtc_cannot_select_another_special_sampler():
    with pytest.raises(ValueError, match="RTC requires"):
        Policy(_TinyRTC(), sample_kwargs={"use_legato_inference": True})
