#!/usr/bin/env python3
"""Monitor inference TCP pop_action continuity at target Hz."""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def request(host: str, port: int, payload: dict, *, timeout: float = 5.0) -> dict:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((host, port))
    try:
        sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        buf = ""
        while "\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("connection closed")
            buf += chunk.decode("utf-8")
        return json.loads(buf.splitlines()[0])
    finally:
        sock.close()


def request_keepalive(sock: socket.socket, payload: dict) -> dict:
    sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    buf = ""
    while "\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk.decode("utf-8")
    return json.loads(buf.splitlines()[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor pop_action continuity")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=8.0, help="wait before sampling")
    parser.add_argument("--start-session", action="store_true")
    parser.add_argument(
        "--no-stop-on-exit",
        action="store_true",
        help="keep session alive after monitor (use secondary connection for start)",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config_agilex.yaml"),
    )
    args = parser.parse_args()

    interval = 1.0 / max(args.rate, 1e-6)
    samples: list[dict] = []
    t_end = time.perf_counter() + args.duration
    last_t = time.perf_counter()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(60.0)
    sock.connect((args.host, args.port))

    try:
        if args.start_session:
            resp = request_keepalive(
                sock,
                {"cmd": "start", "params": {"openpi_runtime_config": args.config}},
            )
            if not resp.get("status"):
                print(f"start failed: {resp}", file=sys.stderr)
                return 1
            print(f"session started, warming up {args.warmup}s (connection kept alive)...")
            time.sleep(args.warmup)

        while time.perf_counter() < t_end:
            loop_t0 = time.perf_counter()
            try:
                resp = request_keepalive(sock, {"cmd": "pop_action"})
            except Exception as exc:
                samples.append({"t": loop_t0, "ok": False, "error": str(exc)})
                break

            action = resp.get("action")
            pending = None
            if int(len(samples)) % 15 == 0:
                try:
                    st = request_keepalive(sock, {"cmd": "status"})
                    pending = st.get("pending_actions")
                except Exception:
                    pending = None

            dt = loop_t0 - last_t if samples else None
            last_t = loop_t0
            entry = {
                "t": loop_t0,
                "dt": dt,
                "ok": bool(resp.get("status")),
                "has_action": action is not None,
                "chunk_id": resp.get("chunk_id"),
                "chunk_step": resp.get("chunk_step_index"),
                "pending": pending,
            }
            if action is not None:
                arr = np.asarray(action, dtype=np.float64)
                entry["action_norm"] = float(np.linalg.norm(arr))
                if samples and samples[-1].get("has_action") and "prev_action" not in samples[-1]:
                    pass
            samples.append(entry)
            if action is not None and len(samples) >= 2:
                prev = None
                for s in reversed(samples[:-1]):
                    if s.get("has_action") and s.get("_action") is not None:
                        prev = s["_action"]
                        break
                if prev is None and samples[-2].get("_action") is not None:
                    prev = samples[-2]["_action"]
                arr = np.asarray(action, dtype=np.float64)
                samples[-1]["_action"] = arr
                if prev is not None:
                    samples[-1]["delta_norm"] = float(np.linalg.norm(arr - prev))
            elif action is not None:
                samples[-1]["_action"] = np.asarray(action, dtype=np.float64)

            sleep_left = interval - (time.perf_counter() - loop_t0)
            if sleep_left > 0:
                time.sleep(sleep_left)
    finally:
        sock.close()

    n = len(samples)
    if n == 0:
        print("no samples collected")
        return 1

    dts = [s["dt"] for s in samples[1:] if s.get("dt") is not None]
    has = sum(1 for s in samples if s.get("has_action"))
    null = n - has
    elapsed = samples[-1]["t"] - samples[0]["t"]
    actual_hz = (n - 1) / elapsed if elapsed > 0 else 0.0
    deliver_hz = has / elapsed if elapsed > 0 else 0.0

    gaps: list[tuple[int, float]] = []
    gap_start = None
    gap_t0 = None
    for i, s in enumerate(samples):
        if not s.get("has_action"):
            if gap_start is None:
                gap_start = i
                gap_t0 = s["t"]
        elif gap_start is not None:
            gaps.append((gap_start, s["t"] - gap_t0))
            gap_start = None
    if gap_start is not None:
        gaps.append((gap_start, samples[-1]["t"] - gap_t0))

    step_jumps = 0
    prev_step = None
    prev_chunk = None
    for s in samples:
        if not s.get("has_action"):
            prev_step = None
            continue
        cid = s.get("chunk_id")
        step = s.get("chunk_step")
        if prev_step is not None and cid == prev_chunk:
            if step != prev_step + 1:
                step_jumps += 1
        prev_step = step
        prev_chunk = cid

    deltas = [s["delta_norm"] for s in samples if s.get("delta_norm") is not None]
    pendings = [s["pending"] for s in samples if s.get("pending") is not None]

    print("=== pop_action continuity report ===")
    print(f"target_hz={args.rate:.1f}  duration={args.duration:.1f}s  samples={n}")
    print(f"actual_loop_hz={actual_hz:.2f}  delivered_action_hz={deliver_hz:.2f}")
    print(f"actions={has}  null={null}  null_ratio={null / n * 100:.1f}%")
    if dts:
        print(
            f"interval_ms: min={min(dts)*1000:.2f}  p50={statistics.median(dts)*1000:.2f}  "
            f"p95={sorted(dts)[int(len(dts)*0.95)]*1000:.2f}  max={max(dts)*1000:.2f}"
        )
    if gaps:
        print(f"gaps(null runs): count={len(gaps)}")
        for idx, (start, dur) in enumerate(gaps[:8]):
            print(f"  gap#{idx+1} start_sample={start} duration={dur*1000:.1f}ms (~{dur*args.rate:.0f} missed ticks)")
        if len(gaps) > 8:
            print(f"  ... {len(gaps)-8} more gaps")
    else:
        print("gaps(null runs): none")
    print(f"chunk_step_discontinuities={step_jumps}")
    if deltas:
        print(
            f"action_delta_norm: p50={statistics.median(deltas):.5f}  "
            f"p95={sorted(deltas)[int(len(deltas)*0.95)]:.5f}  max={max(deltas):.5f}"
        )
    if pendings:
        print(
            f"pending_actions: min={min(pendings)}  p50={statistics.median(pendings):.0f}  max={max(pendings)}"
        )

    # chunk timeline snippet
    timeline = []
    for s in samples:
        if s.get("has_action"):
            timeline.append((s.get("chunk_id"), s.get("chunk_step")))
    if timeline:
        print("chunk timeline (first 12 delivered):", timeline[:12])
        print("chunk timeline (last 12 delivered):", timeline[-12:])

    return 0 if null == 0 and step_jumps == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
