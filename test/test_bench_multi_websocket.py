from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_SRC = REPO_ROOT / "scripts" / "benchmark"
if str(SCRIPTS_SRC) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_SRC))

import bench_multi_websocket as bench  # noqa: E402


def _synthetic_args(**overrides):
    values = {
        "image_size": 8,
        "state_dim": 4,
        "prompt": "test prompt",
        "action_horizon": 3,
        "action_dim": 2,
        "num_steps": None,
        "sleep_ms": None,
        "seed": 123,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_synthetic_payload_uses_seeded_random_data():
    first = bench.build_synthetic_payload(_synthetic_args(seed=123))
    second = bench.build_synthetic_payload(_synthetic_args(seed=123))
    third = bench.build_synthetic_payload(_synthetic_args(seed=456))

    np.testing.assert_array_equal(first["state"], second["state"])
    np.testing.assert_array_equal(first["images"]["top_head"], second["images"]["top_head"])
    assert not np.all(first["state"] == 0)
    assert not np.array_equal(first["state"], third["state"])


@pytest.mark.parametrize("flag", ["--dataset-root", "--payload-npz"])
def test_cli_rejects_external_payload_sources(monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["bench_multi_websocket.py", flag, "/tmp/source"])

    with pytest.raises(SystemExit):
        bench.parse_args()


def test_latency_summary_reports_percentiles():
    summary = bench.latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary["count"] == 5
    assert summary["median"] == pytest.approx(3.0)
    assert summary["p90"] == pytest.approx(4.6)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(5.0)


def test_summarize_detects_non_unit_request_id_gaps():
    samples = [
        {"request_id": 1, "roundtrip_ms": 10.0},
        {"request_id": 3, "roundtrip_ms": 12.0},
        {"request_id": 4, "roundtrip_ms": 11.0},
    ]

    summary = bench.summarize(samples, calls=5, started=10.0, finished=12.0)

    assert summary["responses"] == 3
    assert summary["empty_polls"] == 2
    assert summary["request_id"]["monotonic"] is True
    assert summary["request_id"]["non_unit_gaps"] == 1
    assert summary["request_id"]["gap_examples"] == [2]
    assert summary["response_rate_hz"] == pytest.approx(1.5)
