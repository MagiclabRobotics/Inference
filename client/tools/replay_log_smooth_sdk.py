#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = str(REPO_ROOT / "client/inference/log_smooth.txt")
DEFAULT_CONFIG = str(REPO_ROOT / "client/config_agilex.yaml")
INFERENCE_DIR = REPO_ROOT / "client/inference"


def _parse_array(text: str) -> np.ndarray:
    arr = np.fromstring(text.replace("\n", " "), sep=" ", dtype=float)
    if arr.size != 7:
        raise ValueError(f"expected 7 values, got {arr.size}: {text!r}")
    return arr


def load_actions(path: Path) -> list[tuple[float, np.ndarray]]:
    raw = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(?P<t>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
        r"\[(?P<left>.*?)\]\s*,\s*"
        r"\[(?P<right>.*?)\]",
        re.DOTALL,
    )
    actions: list[tuple[float, np.ndarray]] = []
    for match in pattern.finditer(raw):
        ts = float(match.group("t"))
        left = _parse_array(match.group("left"))
        right = _parse_array(match.group("right"))
        actions.append((ts, np.concatenate([left, right], axis=0)))
    if not actions:
        raise RuntimeError(f"no actions parsed from {path}")
    return actions


def sleep_until(last_t: float, rate_hz: float) -> float:
    period = 1.0 / float(rate_hz)
    now = time.monotonic()
    sleep_s = period - (now - last_t)
    if sleep_s > 0:
        time.sleep(sleep_s)
    return time.monotonic()


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay ROS log_smooth.txt actions through kai0-inference SDK control.")
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means replay all actions")
    parser.add_argument("--dry-run", action="store_true", help="parse and print actions without sending SDK commands")
    parser.add_argument("--move-to-first", action="store_true", help="interpolate to the first replay action before streaming")
    parser.add_argument("--move-steps", type=int, default=100)
    parser.add_argument("--move-sleep", type=float, default=0.01)
    parser.add_argument("--yes", action="store_true", help="required for live SDK replay")
    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("--rate must be positive")

    actions = load_actions(Path(args.log))
    start = max(0, int(args.start_index))
    end = None if int(args.limit) <= 0 else start + int(args.limit)
    selected = actions[start:end]
    if not selected:
        raise RuntimeError(f"empty replay range: start={start}, limit={args.limit}, parsed={len(actions)}")

    print(f"[replay] parsed={len(actions)} selected={len(selected)} rate={args.rate:.3f}Hz dry_run={args.dry_run}")
    print(f"[replay] first_ts={selected[0][0]:.9f} first_action={np.round(selected[0][1], 6).tolist()}")
    print(f"[replay] last_ts={selected[-1][0]:.9f} last_action={np.round(selected[-1][1], 6).tolist()}")

    if args.dry_run:
        return 0
    if not args.yes:
        print("[replay] refusing live SDK control without --yes", file=sys.stderr)
        return 2

    if str(INFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(INFERENCE_DIR))
    from config import load_config, section
    from robot_io import PiperDualArm

    cfg = load_config(args.config)
    arms = PiperDualArm.from_config(section(cfg, "arm"), section(cfg, "can"))
    try:
        print("[replay] connecting arms")
        arms.connect()
        if args.move_to_first:
            first = selected[0][1]
            print(f"[replay] moving to first action steps={args.move_steps} sleep={args.move_sleep}")
            arms.move_to(first[:7].tolist(), first[7:14].tolist(), steps=args.move_steps, sleep_s=args.move_sleep)
        print("[replay] streaming actions")
        last_t = time.monotonic()
        sent = 0
        start_t = time.monotonic()
        for i, (_, action) in enumerate(selected):
            arms.apply_action(action)
            sent += 1
            if sent % max(1, int(round(args.rate))) == 0:
                elapsed = time.monotonic() - start_t
                hz = sent / elapsed if elapsed > 0 else 0.0
                print(f"[replay] sent={sent}/{len(selected)} actual_hz={hz:.2f}")
            if i + 1 < len(selected):
                last_t = sleep_until(last_t, args.rate)
        elapsed = time.monotonic() - start_t
        hz = sent / elapsed if elapsed > 0 else 0.0
        print(f"[replay] done sent={sent} elapsed={elapsed:.3f}s actual_hz={hz:.2f}")
    finally:
        arms.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
