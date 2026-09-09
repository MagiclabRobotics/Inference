#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${REPO_ROOT}/server/config.yaml"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--config <path>] [--json-config <json>] [--policy-config <config>] [--policy-dir <dir>] [--dry-run] [-- <extra serve_policy args>]
USAGE
}

DRY_RUN=0
EXTRA_ARGS=()
OVERRIDE_CONFIG=""
OVERRIDE_DIR=""
JSON_CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --json-config) JSON_CONFIG="$2"; shift 2 ;;
    --policy-config) OVERRIDE_CONFIG="$2"; shift 2 ;;
    --policy-dir) OVERRIDE_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERR] config not found: ${CONFIG}" >&2
  exit 1
fi

# Export overrides for the Python script
export POLICY_CONFIG_OVERRIDE="${OVERRIDE_CONFIG}"
export POLICY_DIR_OVERRIDE="${OVERRIDE_DIR}"
export JSON_CONFIG_OVERRIDE="${JSON_CONFIG}"

mapfile -t CMD_ARGS < <("${PYTHON_BIN:-python3}" - "${CONFIG}" <<'PY'
import os
import sys
try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to read config files: pip install pyyaml") from exc

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

# Override / replace from JSON environment variable
json_override = os.environ.get("JSON_CONFIG_OVERRIDE", "")
if json_override:
    import json as _json
    json_cfg = _json.loads(json_override)
    # Deep-merge JSON into cfg (JSON takes priority)
    def deep_merge(base, overlay):
        for k, v in overlay.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                deep_merge(base[k], v)
            else:
                base[k] = v
    deep_merge(cfg, json_cfg)

# Override from environment variables
policy_config_override = os.environ.get("POLICY_CONFIG_OVERRIDE", "")
policy_dir_override = os.environ.get("POLICY_DIR_OVERRIDE", "")

args = []
python_bin = str(cfg.get("python_bin") or "python3")
args.append(python_bin)
args.append(str(cfg.get("port", 8000)))
if cfg.get("transport") is not None:
    transport = str(cfg["transport"]).replace("-", "_").upper()
    args.extend(["--transport", transport])
if cfg.get("shared_memory_socket_path") is not None:
    args.extend(["--shared-memory-socket-path", str(cfg["shared_memory_socket_path"])])
if cfg.get("websocket_workers") is not None:
    args.extend(["--websocket-workers", str(cfg["websocket_workers"])])
if cfg.get("websocket_queue_size") is not None:
    args.extend(["--websocket-queue-size", str(cfg["websocket_queue_size"])])
if cfg.get("default_prompt") is not None:
    args.extend(["--default-prompt", str(cfg["default_prompt"])])
if cfg.get("use_legato_inference", False):
    args.append("--use-legato-inference")
if cfg.get("use_ttrtc_inference", False):
    args.append("--use-ttrtc-inference")
if cfg.get("use_snapflow_inference", False):
    args.append("--use-snapflow-inference")
if cfg.get("record", False):
    args.append("--record")
if cfg.get("log_level") is not None:
    args.extend(["--log-level", str(cfg["log_level"])])
if cfg.get("log_file") is not None:
    args.extend(["--log-file", str(cfg["log_file"])])
if cfg.get("event_log_file") is not None:
    args.extend(["--event-log-file", str(cfg["event_log_file"])])
if cfg.get("request_log_every_n") is not None:
    args.extend(["--request-log-every-n", str(cfg["request_log_every_n"])])
policy = cfg.get("policy") or {"type": "default"}
ptype = str(policy.get("type", "default"))
if ptype == "checkpoint":
    policy_config = policy_config_override if policy_config_override else str(policy.get("config", ""))
    policy_dir = policy_dir_override if policy_dir_override else str(policy.get("dir", ""))
    args.extend(["policy:checkpoint", "--policy.config", policy_config, "--policy.dir", policy_dir])
    if policy.get("asset_id") is not None:
        args.extend(["--policy.asset-id", str(policy["asset_id"])])
elif ptype in {"tensorrt", "trt", "tensorrt_fp16"}:
    args.extend(["policy:tensorrt", "--policy.config", str(policy["config"]), "--policy.engine", str(policy["engine"])])
    optional_fields = {
        "assets_dir": "--policy.assets-dir",
        "asset_id": "--policy.asset-id",
        "device": "--policy.device",
        "seed": "--policy.seed",
        "output_name": "--policy.output-name",
        "precision": "--policy.precision",
    }
    for key, flag in optional_fields.items():
        if policy.get(key) is not None:
            args.extend([flag, str(policy[key])])
elif ptype == "default":
    args.append("policy:default")
else:
    raise SystemExit(f"Unsupported policy.type: {ptype}")
for arg in args:
    print(arg)
PY
)

PYTHON_BIN_FROM_CONFIG="${CMD_ARGS[0]}"
if [[ "${PYTHON_BIN_FROM_CONFIG}" != /* && "${PYTHON_BIN_FROM_CONFIG}" == */* ]]; then
  PYTHON_BIN_FROM_CONFIG="${REPO_ROOT}/${PYTHON_BIN_FROM_CONFIG}"
fi
SERVE_ARGS=("--port" "${CMD_ARGS[1]}" "${CMD_ARGS[@]:2}" "${EXTRA_ARGS[@]}")
export PYTHONPATH="${REPO_ROOT}/server:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

TASKSET_PREFIX=(taskset -c "15,16,17,18,19,20")
cmd=("${TASKSET_PREFIX[@]}" "${PYTHON_BIN_FROM_CONFIG}" "${REPO_ROOT}/server/serve_policy.py" "${SERVE_ARGS[@]}")
echo "[INFO] ${cmd[*]}"
if [[ ${DRY_RUN} -eq 1 ]]; then
  exit 0
fi
exec "${cmd[@]}"
