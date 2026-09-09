#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${REPO_ROOT}/client/config_agilex_mock.yaml"
HOST="127.0.0.1"
PORT=18123
DATASET_ROOT="${REPO_ROOT}/test/assets/task_a_base_episode001_subset"
MAX_PUBLISH_STEP=12
ACTION_HORIZON=50
ACTION_DIM=14
LOG_LEVEL="INFO"
DRY_RUN=0
KEEP_CONFIG=0
EXTRA_ARGS=()

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--config <path>] [--host <host>] [--port <port>] [--dataset-root <path>] [--max-publish-step <n>] [--dry-run] [--keep-config] [--log-level <level>] [-- <extra run_client args>]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --max-publish-step) MAX_PUBLISH_STEP="$2"; shift 2 ;;
    --action-horizon) ACTION_HORIZON="$2"; shift 2 ;;
    --action-dim) ACTION_DIM="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --keep-config) KEEP_CONFIG=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERR] config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[ERR] dataset root not found: ${DATASET_ROOT}" >&2
  exit 1
fi

TMP_CONFIG="$(mktemp /tmp/run_client_mock.XXXXXX.yaml)"
SERVER_PID=""
SERVER_LOG="/tmp/inference_mock_action_server_${PORT}.log"

cleanup() {
  set +e
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ "${KEEP_CONFIG}" != "1" ]]; then
    rm -f "${TMP_CONFIG}"
  fi
}
trap cleanup EXIT INT TERM

"${PYTHON_BIN:-python3}" - "${CONFIG}" "${TMP_CONFIG}" "${HOST}" "${PORT}" "${DATASET_ROOT}" "${MAX_PUBLISH_STEP}" <<'PY'
from pathlib import Path
import sys

import yaml

src, dst, host, port, dataset_root, max_publish_step = sys.argv[1:7]
with Path(src).open("r", encoding="utf-8") as fp:
    cfg = yaml.safe_load(fp) or {}

server = cfg.setdefault("server", {})
server["transport"] = "websocket"
server["host"] = host
server["port"] = int(port)

cfg["python_bin"] = ".venv/bin/python"

robot_io = cfg.setdefault("robot_io", {})
robot_io["type"] = "mock_lerobot"
robot_io["dataset_root"] = dataset_root
robot_io["episode_index"] = int(robot_io.get("episode_index", 0))
robot_io["loop"] = True

cfg.setdefault("can", {})["activate"] = False
cfg.setdefault("camera", {})["enable"] = False

inference = cfg.setdefault("inference", {})
inference["max_publish_step"] = int(max_publish_step)
inference["log_every_steps"] = 1

recording = cfg.setdefault("recording", {})
recording["record_model_io"] = False
recording["record_runtime_events"] = True
recording["record_action_steps"] = True
recording["root_dir"] = "/tmp/inference_mock_client_records"

Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

MOCK_CMD=(
  "${REPO_ROOT}/.venv/bin/python"
  "${REPO_ROOT}/client/tools/mock_action_server.py"
  --host "${HOST}"
  --port "${PORT}"
  --action-horizon "${ACTION_HORIZON}"
  --action-dim "${ACTION_DIM}"
)
CLIENT_CMD=(
  "${REPO_ROOT}/scripts/run_client.sh"
  --config "${TMP_CONFIG}"
  --log-level "${LOG_LEVEL}"
  "${EXTRA_ARGS[@]}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] generated config: ${TMP_CONFIG}"
  sed -n '1,180p' "${TMP_CONFIG}"
  echo "[INFO] mock server: ${MOCK_CMD[*]}"
  echo "[INFO] client: ${CLIENT_CMD[*]}"
  exit 0
fi

rm -f "${SERVER_LOG}"
echo "[INFO] starting mock action server: ${MOCK_CMD[*]}"
"${MOCK_CMD[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

"${REPO_ROOT}/.venv/bin/python" - "${HOST}" "${PORT}" <<'PY'
import socket
import sys
import time

host, port = sys.argv[1], int(sys.argv[2])
for _ in range(100):
    sock = socket.socket()
    try:
        sock.settimeout(0.1)
        sock.connect((host, port))
        sock.close()
        break
    except OSError:
        time.sleep(0.05)
else:
    raise SystemExit(f"mock action server did not open {host}:{port}")
PY

echo "[INFO] mock action server log: ${SERVER_LOG}"
echo "[INFO] running client: ${CLIENT_CMD[*]}"
"${CLIENT_CMD[@]}"
