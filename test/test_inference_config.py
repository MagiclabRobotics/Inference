from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENT_INFERENCE_SRC = REPO_ROOT / "client" / "inference"

if str(CLIENT_INFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(CLIENT_INFERENCE_SRC))

from config import InferenceConfig  # noqa: E402
from config import apply_mode_override  # noqa: E402
from config import load_config  # noqa: E402


def test_runtime_options_loads_agilex_config_profile():
    cfg = load_config(REPO_ROOT / "client" / "config_agilex.yaml")

    profile = cfg.runtime_options()

    assert profile["execution_mode"] == "async"
    assert profile["async_mode"] == "temporal_smoothing"
    assert profile["smooth_method"] == "temporal_smoothing"
    assert profile["action_buffer"] == "stream"
    assert profile["chunk_size"] == 50
    assert profile["publish_rate"] == 30
    assert profile["observation_rate"] == 30
    assert profile["inference_rate"] == 3
    assert profile["num_denoising_steps"] == 10
    assert profile["latency_k"] == 8
    assert profile["min_smooth_steps"] == 8
    assert "modes" not in profile


def test_runtime_options_defaults_async_mode_to_temporal_smoothing():
    cfg = InferenceConfig(
        raw={
            "inference": {
                "execution_mode": "async",
                "modes": {
                    "async": {
                        "temporal_smoothing": {"latency_k": 8, "min_smooth_steps": 8},
                    },
                },
            },
        },
        path=Path("config.yaml"),
    )

    profile = cfg.runtime_options()

    assert profile["execution_mode"] == "async"
    assert profile["async_mode"] == "temporal_smoothing"
    assert profile["smooth_method"] == "temporal_smoothing"
    assert profile["latency_k"] == 8
    assert profile["min_smooth_steps"] == 8


def test_runtime_options_merges_async_mode_options():
    cfg = InferenceConfig(
        raw={
            "inference": {
                "execution_mode": "async",
                "async_mode": "rtc",
                "smooth_method": "raw",
                "modes": {
                    "async": {
                        "rtc": {"execute_horizon": 8, "delay_clip_max": 4},
                    },
                },
            },
        },
        path=Path("config.yaml"),
    )

    profile = cfg.runtime_options()

    assert profile["execution_mode"] == "async"
    assert profile["async_mode"] == "rtc"
    assert profile["smooth_method"] == "raw"
    assert profile["execute_horizon"] == 8
    assert profile["delay_clip_max"] == 4


def test_runtime_options_merges_smooth_method_options():
    cfg = InferenceConfig(
        raw={
            "inference": {
                "execution_mode": "async",
                "async_mode": "rtc",
                "smooth_method": "temporal_ensembling",
                "modes": {
                    "async": {
                        "rtc": {"execute_horizon": 8},
                        "temporal_ensembling": {"ensemble_new_weight": 0.75},
                    },
                },
            },
        },
        path=Path("config.yaml"),
    )

    profile = cfg.runtime_options()

    assert profile["execution_mode"] == "async"
    assert profile["async_mode"] == "rtc"
    assert profile["smooth_method"] == "temporal_ensembling"
    assert profile["execute_horizon"] == 8
    assert profile["ensemble_new_weight"] == 0.75


def test_runtime_options_sync_ignores_async_submodes():
    cfg = InferenceConfig(
        raw={
            "inference": {
                "execution_mode": "sync",
                "async_mode": "rtc",
                "smooth_method": "temporal_smoothing",
                "modes": {
                    "sync": {"sync_only": True},
                    "async": {
                        "rtc": {"execute_horizon": 8},
                        "temporal_smoothing": {"min_smooth_steps": 6},
                    },
                },
            },
        },
        path=Path("config.yaml"),
    )

    profile = cfg.runtime_options()

    assert profile["execution_mode"] == "sync"
    assert profile["sync_only"] is True
    assert "async_mode" not in profile
    assert "execute_horizon" not in profile
    assert "min_smooth_steps" not in profile


def test_runtime_options_rejects_unsupported_async_mode():
    cfg = InferenceConfig(
        raw={
            "inference": {
                "execution_mode": "async",
                "async_mode": "unknown",
            },
        },
        path=Path("config.yaml"),
    )

    with pytest.raises(ValueError, match="Unsupported async_mode"):
        cfg.runtime_options()


def test_ttrtc_mode_override_loads_inference_only_profile():
    cfg = load_config(REPO_ROOT / "client" / "config_agilex.yaml")

    apply_mode_override(cfg, "ttrtc_async")
    profile = cfg.runtime_options()

    assert "ttrtc_async" in cfg.available_modes()
    assert profile["execution_mode"] == "async"
    assert profile["async_mode"] == "ttrtc"
    assert profile["smooth_method"] == "raw"
    assert profile["sample_num_steps"] == 5
