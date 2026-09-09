#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${REPO_ROOT}/client/config_agilex.yaml"
INFER_DIR="${REPO_ROOT}/client/inference"
CAN_ACTIVATE_SCRIPT="${REPO_ROOT}/scripts/can_activate.sh"
DRY_RUN=0
CHECK_HARDWARE=0
LOG_LEVEL="INFO"
EXTRA_ARGS=()

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--config <path>] [--dry-run] [--check-hardware] [--log-level <level>] [-- <extra inference args>]

Environment variables:
  PYTHON_BIN_PATH   Override the python interpreter path (takes precedence over config's python_bin).
                    Accepts an absolute path, a path relative to the repo root, or a plain executable name.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --check-hardware) CHECK_HARDWARE=1; shift ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERR] config not found: ${CONFIG}" >&2
  exit 1
fi

read_cfg() {
  local expr="$1"
  local default_value="${2:-}"
  "${PYTHON_BIN:-python3}" - "$CONFIG" "$expr" "$default_value" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to read config files. Run: uv pip install -r client/requirements_inference.txt") from exc

path, expr, default = sys.argv[1:4]
with Path(path).open("r", encoding="utf-8") as fp:
    data = yaml.safe_load(fp) or {}
cur = data
for part in expr.split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        cur = default
        break
if isinstance(cur, bool):
    print("true" if cur else "false")
elif cur is None:
    print("")
else:
    print(cur)
PY
}

resolve_python_bin() {
  local python_bin="$1"
  if [[ "${python_bin}" == /* ]]; then
    printf '%s\n' "${python_bin}"
  elif [[ "${python_bin}" == */* ]]; then
    printf '%s\n' "${REPO_ROOT}/${python_bin}"
  else
    printf '%s\n' "${python_bin}"
  fi
}

if [[ -n "${PYTHON_BIN_PATH:-}" ]]; then
  PYTHON_BIN_RESOLVED="$(resolve_python_bin "${PYTHON_BIN_PATH}")"
  echo "[INFO] using python from PYTHON_BIN_PATH: ${PYTHON_BIN_RESOLVED}"
else
  PYTHON_BIN_FROM_CONFIG="$(read_cfg python_bin .venv/bin/python)"
  PYTHON_BIN_RESOLVED="$(resolve_python_bin "${PYTHON_BIN_FROM_CONFIG}")"
fi
ACTIVATE_CAN="$(read_cfg can.activate true)"
CAN_BITRATE="$(read_cfg can.bitrate 1000000)"
CAN_LEFT_USB="$(read_cfg can.left.usb "")"
CAN_RIGHT_USB="$(read_cfg can.right.usb "")"

export PYTHONPATH="${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
TASKSET_PREFIX=(taskset -c "9,10,11,12,13")
cmd=("${TASKSET_PREFIX[@]}" "${PYTHON_BIN_RESOLVED}" "${INFER_DIR}/agilex_inference_openpi.py" "--config" "${CONFIG}" "--log-level" "${LOG_LEVEL}")
if [[ ${CHECK_HARDWARE} -eq 1 ]]; then
  cmd+=("--check-hardware")
fi
cmd+=("${EXTRA_ARGS[@]}")

echo "[INFO] ${cmd[*]}"
if [[ ${DRY_RUN} -eq 1 ]]; then
  exit 0
fi

if [[ "${ACTIVATE_CAN}" == "true" ]]; then
  if [[ ! -f "${CAN_ACTIVATE_SCRIPT}" ]]; then
    echo "[ERR] CAN activation script not found: ${CAN_ACTIVATE_SCRIPT}" >&2
    exit 1
  fi
  echo "[INFO] activating CAN interfaces (can_left_slave / can_right_slave)..."
  CAN_ACTIVATE_ARGS=("--bitrate=${CAN_BITRATE}")
  [[ -n "${CAN_LEFT_USB}" ]] && CAN_ACTIVATE_ARGS+=("--left-usb=${CAN_LEFT_USB}")
  [[ -n "${CAN_RIGHT_USB}" ]] && CAN_ACTIVATE_ARGS+=("--right-usb=${CAN_RIGHT_USB}")
  sudo bash "${CAN_ACTIVATE_SCRIPT}" "${CAN_ACTIVATE_ARGS[@]}"
fi

# Keep stdin attached to the terminal; the Python client handles keys and signals.
exec "${cmd[@]}"
