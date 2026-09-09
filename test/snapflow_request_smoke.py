#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from openpi_client import websocket_client_policy  # noqa: E402


def _make_image(seed: int, size: int = 224) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    image = np.stack(
        [
            (xx + seed * 17) % 256,
            (yy * 2 + seed * 29) % 256,
            ((xx // 2 + yy // 3) + seed * 43) % 256,
        ],
        axis=0,
    ).astype(np.uint8)
    noise = rng.integers(0, 8, size=image.shape, dtype=np.uint8)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def build_payload(*, num_steps: int | None = None, prompt: str = "Flatten and fold the cloth.") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "images": {
            "top_head": _make_image(1),
            "hand_left": _make_image(2),
            "hand_right": _make_image(3),
        },
        "state": np.linspace(-0.25, 0.25, 14, dtype=np.float32),
        "prompt": prompt,
    }
    if num_steps is not None:
        payload["num_steps"] = int(num_steps)
    return payload


def _summarize_array(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(np.nanmin(value)) if value.size else None,
        "max": float(np.nanmax(value)) if value.size else None,
        "mean": float(np.nanmean(value)) if value.size else None,
    }


def _summarize_output(output: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in output.items():
        summary[key] = _summarize_array(value) if isinstance(value, np.ndarray) else value
    return summary


def _stats(values: list[float]) -> dict[str, float | int]:
    clean = [v for v in values if v == v]
    if not clean:
        return {"n": 0}
    quantiles = statistics.quantiles(clean, n=100, method="inclusive") if len(clean) > 1 else [clean[0]] * 99
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
        "p90": quantiles[89],
        "p95": quantiles[94],
        "p99": quantiles[98],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Send constructed SnapFlow smoke-test requests to an OpenPI server.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host. Use 127.0.0.1 when running on 63.")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--num-steps", type=int, help="Optional SnapFlow multistep count. Omit for one-step.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--ignore-first", type=int, default=0)
    parser.add_argument("--prompt", default="Flatten and fold the cloth.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.ignore_first < 0 or args.ignore_first >= args.repeat:
        raise ValueError("--ignore-first must be >= 0 and smaller than --repeat")

    payload = build_payload(num_steps=args.num_steps, prompt=args.prompt)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    records: list[dict[str, Any]] = []
    last_response: dict[str, Any] | None = None

    for index in range(1, args.repeat + 1):
        start = time.monotonic()
        response = policy.infer(payload)
        elapsed_ms = (time.monotonic() - start) * 1000
        last_response = response
        client_timing = response.get("client_timing", {})
        server_timing = response.get("server_timing", {})
        policy_timing = response.get("policy_timing", {})
        record = {
            "index": index,
            "elapsed_ms": elapsed_ms,
            "roundtrip_ms": float(client_timing.get("roundtrip_ms", float("nan"))),
            "server_infer_ms": float(server_timing.get("infer_ms", float("nan"))),
            "policy_infer_ms": float(policy_timing.get("infer_ms", float("nan"))),
            "actions_shape": list(response["actions"].shape),
            "actions_model_shape": list(response["actions_model"].shape),
        }
        records.append(record)
        print(
            f"{index}/{args.repeat}: roundtrip={record['roundtrip_ms']:.2f}ms "
            f"policy={record['policy_infer_ms']:.2f}ms",
            flush=True,
        )

    measured = records[args.ignore_first :]
    result = {
        "request": {
            "host": args.host,
            "port": args.port,
            "num_steps": args.num_steps,
            "repeat": args.repeat,
            "ignore_first": args.ignore_first,
            "image_shapes": {key: list(value.shape) for key, value in payload["images"].items()},
            "state_shape": list(payload["state"].shape),
            "prompt": args.prompt,
        },
        "summaries": {
            "roundtrip_ms": _stats([r["roundtrip_ms"] for r in measured]),
            "server_infer_ms": _stats([r["server_infer_ms"] for r in measured]),
            "policy_infer_ms": _stats([r["policy_infer_ms"] for r in measured]),
        },
        "last_response": _summarize_output(last_response or {}),
        "records": records,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.output}")

    print(json.dumps({k: result[k] for k in ("request", "summaries", "last_response")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
