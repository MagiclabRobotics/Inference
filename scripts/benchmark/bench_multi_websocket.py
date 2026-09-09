#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = REPO_ROOT / "packages" / "openpi-client" / "src"

if str(OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))

from openpi_client import multi_websocket_client_policy  # noqa: E402


def build_synthetic_payload(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    payload: dict[str, Any] = {
        "images": {
            "top_head": rng.integers(0, 256, size=(3, args.image_size, args.image_size), dtype=np.uint8),
            "hand_right": rng.integers(0, 256, size=(3, args.image_size, args.image_size), dtype=np.uint8),
            "hand_left": rng.integers(0, 256, size=(3, args.image_size, args.image_size), dtype=np.uint8),
        },
        "state": rng.normal(size=args.state_dim).astype(np.float32),
        "prompt": args.prompt,
        "action_horizon": args.action_horizon,
        "action_dim": args.action_dim,
    }
    if args.num_steps is not None:
        payload["num_steps"] = int(args.num_steps)
    if args.sleep_ms is not None:
        payload["sleep_ms"] = float(args.sleep_ms)
    return payload


def payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return _summarize_value(payload)


def _summarize_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _summarize_value(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return {"type": type(value).__name__, "len": len(value)}
    return value


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _response_request_id(out: dict[str, Any]) -> int | None:
    timing = out.get("server_timing")
    if isinstance(timing, dict):
        return _as_int(timing.get("request_id", timing.get("request_index")))
    client_timing = out.get("client_timing")
    if isinstance(client_timing, dict):
        return _as_int(client_timing.get("request_id", client_timing.get("request_index")))
    return None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def compact_sample(index: int, out: dict[str, Any], roundtrip_ms: float) -> dict[str, Any]:
    server_timing = out.get("server_timing") if isinstance(out.get("server_timing"), dict) else {}
    policy_timing = out.get("policy_timing") if isinstance(out.get("policy_timing"), dict) else {}
    client_timing = out.get("client_timing") if isinstance(out.get("client_timing"), dict) else {}
    request_id = _response_request_id(out)
    return {
        "call_index": index,
        "request_id": request_id,
        "roundtrip_ms": roundtrip_ms,
        "server_infer_ms": _as_float(server_timing.get("infer_ms")),
        "policy_infer_ms": _as_float(policy_timing.get("infer_ms")),
        "queue_wait_ms": _as_float(server_timing.get("queue_wait_ms")),
        "submit_to_receive_ms": _as_float(client_timing.get("submit_to_receive_ms")),
        "endpoint": client_timing.get("endpoint"),
        "worker_id": client_timing.get("worker_id"),
        "server_worker_id": server_timing.get("worker_id"),
    }


def summarize(samples: list[dict[str, Any]], calls: int, started: float, finished: float) -> dict[str, Any]:
    responses = [sample for sample in samples if sample.get("request_id") is not None]
    request_ids = [int(sample["request_id"]) for sample in responses]
    pairs = list(itertools.pairwise(request_ids))
    gaps = [b - a for a, b in pairs if b - a != 1]
    monotonic = all(b > a for a, b in pairs)
    duration_s = max(0.0, finished - started)
    endpoint_counts: dict[str, int] = {}
    for sample in responses:
        endpoint = sample.get("endpoint")
        if endpoint is not None:
            endpoint_counts[str(endpoint)] = endpoint_counts.get(str(endpoint), 0) + 1
    return {
        "calls": calls,
        "responses": len(responses),
        "empty_polls": calls - len(responses),
        "duration_s": duration_s,
        "call_rate_hz": calls / duration_s if duration_s > 0 else None,
        "response_rate_hz": len(responses) / duration_s if duration_s > 0 else None,
        "request_id": {
            "first": request_ids[0] if request_ids else None,
            "last": request_ids[-1] if request_ids else None,
            "monotonic": monotonic,
            "non_unit_gaps": len(gaps),
            "gap_examples": gaps[:10],
        },
        "roundtrip_ms": latency_summary([float(sample["roundtrip_ms"]) for sample in responses]),
        "server_infer_ms": latency_summary(
            [float(sample["server_infer_ms"]) for sample in responses if sample.get("server_infer_ms") is not None]
        ),
        "policy_infer_ms": latency_summary(
            [float(sample["policy_infer_ms"]) for sample in responses if sample.get("policy_infer_ms") is not None]
        ),
        "queue_wait_ms": latency_summary(
            [float(sample["queue_wait_ms"]) for sample in responses if sample.get("queue_wait_ms") is not None]
        ),
        "submit_to_receive_ms": latency_summary(
            [float(sample["submit_to_receive_ms"]) for sample in responses if sample.get("submit_to_receive_ms") is not None]
        ),
        "endpoint_counts": endpoint_counts,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    payload = make_payload(args)
    endpoints = [item.strip() for item in args.endpoints.split(",") if item.strip()] if args.endpoints else None
    policy = multi_websocket_client_policy.MultiWebsocketClientPolicy(
        host=args.host,
        port=args.port,
        endpoints=endpoints,
        connections_per_endpoint=args.connections_per_endpoint,
        max_in_flight=args.max_in_flight,
        result_timeout_s=args.result_timeout_s,
        first_result_timeout_s=args.first_result_timeout_s,
        connect_retry_s=args.connect_retry_s,
    )
    output_file = args.output_jsonl.open("w", encoding="utf-8") if args.output_jsonl else None
    samples: list[dict[str, Any]] = []
    started = time.monotonic()
    last_call = started
    period_s = 1.0 / args.rate_hz if args.rate_hz > 0 else 0.0
    calls = 0
    try:
        while True:
            now = time.monotonic()
            if args.requests is not None and calls >= args.requests:
                break
            if args.duration_s is not None and now - started >= args.duration_s:
                break
            if calls > 0 and period_s > 0:
                sleep_s = period_s - (now - last_call)
                if sleep_s > 0:
                    time.sleep(sleep_s)
            last_call = time.monotonic()
            call_payload = copy.deepcopy(payload)
            infer_start = time.monotonic()
            out = policy.infer(call_payload)
            roundtrip_ms = (time.monotonic() - infer_start) * 1000.0
            calls += 1
            if out:
                sample = compact_sample(calls, out, roundtrip_ms)
                samples.append(sample)
                if output_file is not None:
                    output_file.write(json.dumps(sample, sort_keys=True) + "\n")
    finally:
        finished = time.monotonic()
        if output_file is not None:
            output_file.close()
        policy.close()

    return {
        "config": {
            "host": args.host,
            "port": args.port,
            "endpoints": endpoints,
            "connections_per_endpoint": args.connections_per_endpoint,
            "max_in_flight": args.max_in_flight,
            "rate_hz": args.rate_hz,
            "requests": args.requests,
            "duration_s": args.duration_s,
            "result_timeout_s": args.result_timeout_s,
            "first_result_timeout_s": args.first_result_timeout_s,
        },
        "payload": payload_summary(payload),
        "server_metadata": policy.get_server_metadata(),
        "summary": summarize(samples, calls, started, finished),
    }


def make_payload(args: argparse.Namespace) -> dict[str, Any]:
    return build_synthetic_payload(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pressure-test the multi_websocket client path and report response latency, "
            "throughput, queue wait, and returned request-id monotonicity."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host, unless --endpoints is provided.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoints", help="Comma-separated ws://host:port endpoints. Overrides --host/--port target list.")
    parser.add_argument("--connections-per-endpoint", type=int, default=2)
    parser.add_argument("--max-in-flight", type=int, default=8)
    parser.add_argument("--result-timeout-s", type=float, default=0.0)
    parser.add_argument("--first-result-timeout-s", type=float, default=30.0)
    parser.add_argument("--connect-retry-s", type=float, default=1.0)
    parser.add_argument("--rate-hz", type=float, default=30.0, help="How often to call policy.infer. <=0 means as fast as possible.")
    parser.add_argument("--requests", type=int, default=200, help="Number of policy.infer calls to issue.")
    parser.add_argument("--duration-s", type=float, help="Run duration. If set, stops when either duration or --requests is reached.")
    parser.add_argument("--output-jsonl", type=Path, help="Optional per-response sample output.")

    parser.add_argument("--prompt", default="Flatten and fold the cloth.")
    parser.add_argument("--num-steps", type=int, help="Optional denoising step override sent in the request payload.")
    parser.add_argument("--sleep-ms", type=float, help="Optional mock-server delay field for synthetic stress tests.")

    parser.add_argument("--seed", type=int, default=0, help="Seed used to prebuild the random synthetic payload.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=14)
    args = parser.parse_args()
    if args.requests is None and args.duration_s is None:
        parser.error("at least one of --requests or --duration-s must be set")
    return args


def main() -> None:
    result = run_benchmark(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
