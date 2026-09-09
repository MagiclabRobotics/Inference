#!/usr/bin/env bash
set -euo pipefail

LEFT_NAME="can_left_slave"
RIGHT_NAME="can_right_slave"
BITRATE="1000000"
LEFT_USB=""
RIGHT_USB=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--bitrate <bps>] [--left-usb <bus-info>] [--right-usb <bus-info>]

Brings up exactly two CAN adapters, sets bitrate, and renames them to ${LEFT_NAME} and ${RIGHT_NAME}.

  --bitrate     CAN bitrate (default: ${BITRATE})
  --left-usb    ethtool bus-info for the left arm adapter (e.g. 3-1.4:1.0)
  --right-usb   ethtool bus-info for the right arm adapter

If both --left-usb and --right-usb are set, those ports are used directly.
Otherwise, existing CAN interfaces are listed; if fewer than two are found, the script exits with a warning.
If at least two are present, you are prompted to pick which port is left and which is right (by index).

Examples:
  sudo $(basename "$0") --bitrate 1000000 --left-usb 3-1.4:1.0 --right-usb 3-5.2:1.0
  sudo $(basename "$0") --bitrate 1000000
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --bitrate=*)
      BITRATE="${1#*=}"
      shift
      ;;
    --bitrate)
      BITRATE="${2:?--bitrate requires a value}"
      shift 2
      ;;
    --left-usb=*)
      LEFT_USB="${1#*=}"
      shift
      ;;
    --left-usb)
      LEFT_USB="${2:?--left-usb requires a value}"
      shift 2
      ;;
    --right-usb=*)
      RIGHT_USB="${1#*=}"
      shift
      ;;
    --right-usb)
      RIGHT_USB="${2:?--right-usb requires a value}"
      shift 2
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

echo "-------------------START-----------------------"

require_cmd() {
  local cmd="$1"
  local package="$2"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Error: ${cmd} not found."
    echo "Install it with: sudo apt update && sudo apt install -y ${package}"
    exit 1
  fi
}

require_cmd ip iproute2
require_cmd ethtool ethtool

if ! command -v candump >/dev/null 2>&1 && ! dpkg -s can-utils >/dev/null 2>&1; then
  echo "Error: can-utils not detected in the system."
  echo "Install it with: sudo apt update && sudo apt install -y can-utils"
  exit 1
fi

echo "CAN dependencies are installed."

refresh_can_list() {
  mapfile -t CAN_INTERFACES < <(ip -br link show type can | awk '{print $1}')
}

bus_info_for() {
  local iface="$1"
  sudo ethtool -i "${iface}" 2>/dev/null | awk '/bus-info/ {print $2; exit}'
}

iface_for_usb() {
  local want="$1"
  local iface bus
  refresh_can_list
  for iface in "${CAN_INTERFACES[@]}"; do
    bus="$(bus_info_for "${iface}")"
    if [[ -n "${bus}" && "${bus}" == "${want}" ]]; then
      printf '%s\n' "${iface}"
      return 0
    fi
  done
  return 1
}

is_link_up() {
  ip link show "$1" 2>/dev/null | grep -q "UP"
}

current_bitrate() {
  ip -details link show "$1" 2>/dev/null | grep -oP 'bitrate \K\d+' || true
}

activate_iface() {
  local iface="$1"
  local br="$2"
  local up cur
  up="no"
  if is_link_up "${iface}"; then
    up="yes"
  fi
  cur="$(current_bitrate "${iface}")"
  if [[ "${up}" == "yes" && "${cur}" == "${br}" ]]; then
    echo "Interface ${iface} is already up at bitrate ${br}."
    return 0
  fi
  if [[ "${up}" == "yes" ]]; then
    echo "Interface ${iface} is up at bitrate ${cur:-unknown}; resetting to ${br}."
  else
    echo "Interface ${iface} is down or bitrate unset; configuring bitrate ${br}."
  fi
  sudo ip link set "${iface}" down
  sudo ip link set "${iface}" type can bitrate "${br}"
  sudo ip link set "${iface}" up
  echo "Interface ${iface} is up at bitrate ${br}."
}

unique_tmp_name() {
  local p="$1"
  local c
  while true; do
    c="${p}${RANDOM}"
    if ! ip link show "${c}" >/dev/null 2>&1; then
      printf '%s\n' "${c}"
      return 0
    fi
  done
}

rename_iface() {
  local old="$1"
  local new="$2"
  if [[ "${old}" == "${new}" ]]; then
    return 0
  fi
  echo "Rename interface ${old} -> ${new}."
  sudo ip link set "${old}" down
  sudo ip link set "${old}" name "${new}"
  sudo ip link set "${new}" up
}

evict_name_if_not() {
  local name="$1"
  local keep="$2"
  local bus
  if ! ip link show "${name}" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "${name}" == "${keep}" ]]; then
    return 0
  fi
  bus="$(bus_info_for "${name}")"
  echo "Evict unrelated interface ${name} (bus-info ${bus:-unknown}) from name collision."
  rename_iface "${name}" "$(unique_tmp_name ev_)"
}

assign_final_names() {
  local iface_left="$1"
  local iface_right="$2"
  local tl tr

  evict_name_if_not "${LEFT_NAME}" "${iface_left}"
  evict_name_if_not "${RIGHT_NAME}" "${iface_right}"

  tl="$(unique_tmp_name tl)"
  tr="$(unique_tmp_name tr)"

  if [[ "${iface_left}" != "${LEFT_NAME}" ]]; then
    rename_iface "${iface_left}" "${tl}"
    iface_left="${tl}"
  fi
  if [[ "${iface_right}" != "${RIGHT_NAME}" ]]; then
    rename_iface "${iface_right}" "${tr}"
    iface_right="${tr}"
  fi

  if [[ "${iface_left}" != "${LEFT_NAME}" ]]; then
    rename_iface "${iface_left}" "${LEFT_NAME}"
  fi
  if [[ "${iface_right}" != "${RIGHT_NAME}" ]]; then
    rename_iface "${iface_right}" "${RIGHT_NAME}"
  fi
}

interactive_pick_usb() {
  local n i bus idx_l idx_r
  refresh_can_list
  n="${#CAN_INTERFACES[@]}"
  if [[ "${n}" -lt 2 ]]; then
    echo "Warning: need at least two CAN interfaces for dual-arm setup; found ${n}." >&2
    echo "-------------------ERROR-----------------------" >&2
    exit 1
  fi
  if [[ ! -r /dev/tty ]]; then
    echo "Error: cannot access /dev/tty for interactive left/right CAN selection." >&2
    echo "Set both can.left.usb and can.right.usb in the client config, or run this script from a terminal." >&2
    exit 1
  fi

  echo "Detected ${n} CAN interface(s):" >&2
  for i in "${!CAN_INTERFACES[@]}"; do
    bus="$(bus_info_for "${CAN_INTERFACES[$i]}")"
    echo "  [$((i + 1))] ${CAN_INTERFACES[$i]}  bus-info: ${bus:-<unknown>}" >&2
  done
  while true; do
    read -r -p "Enter index for LEFT arm (${LEFT_NAME}) [1-${n}]: " idx_l </dev/tty || true
    if [[ "${idx_l}" =~ ^[0-9]+$ ]] && [[ "${idx_l}" -ge 1 && "${idx_l}" -le "${n}" ]]; then
      break
    fi
    echo "Invalid choice; enter a number between 1 and ${n}." >&2
  done
  while true; do
    read -r -p "Enter index for RIGHT arm (${RIGHT_NAME}) [1-${n}]: " idx_r </dev/tty || true
    if [[ "${idx_r}" =~ ^[0-9]+$ ]] && [[ "${idx_r}" -ge 1 && "${idx_r}" -le "${n}" ]] && [[ "${idx_r}" != "${idx_l}" ]]; then
      break
    fi
    echo "Invalid choice; enter a different number between 1 and ${n} (must differ from left)." >&2
  done
  LEFT_USB="$(bus_info_for "${CAN_INTERFACES[$((idx_l - 1))]}")"
  RIGHT_USB="$(bus_info_for "${CAN_INTERFACES[$((idx_r - 1))]}")"
  if [[ -z "${LEFT_USB}" || -z "${RIGHT_USB}" ]]; then
    echo "Error: could not read bus-info for the selected interface(s)." >&2
    exit 1
  fi
  echo "Selected LEFT  (${LEFT_NAME}): USB ${LEFT_USB}" >&2
  echo "Selected RIGHT (${RIGHT_NAME}): USB ${RIGHT_USB}" >&2
}

resolve_pair_into() {
  local il ir
  if [[ -n "${LEFT_USB}" && -n "${RIGHT_USB}" ]]; then
    if [[ "${LEFT_USB}" == "${RIGHT_USB}" ]]; then
      echo "Error: --left-usb and --right-usb must differ." >&2
      exit 1
    fi
    echo "Using configured USB ports: left=${LEFT_USB} right=${RIGHT_USB}" >&2
  else
    if [[ -n "${LEFT_USB}" || -n "${RIGHT_USB}" ]]; then
      echo "Warning: only one of left/right USB was set; ignoring partial USB settings and entering interactive selection." >&2
      LEFT_USB=""
      RIGHT_USB=""
    fi
    interactive_pick_usb
  fi

  il="$(iface_for_usb "${LEFT_USB}")" || true
  ir="$(iface_for_usb "${RIGHT_USB}")" || true
  if [[ -z "${il}" ]]; then
    echo "Error: no CAN interface with bus-info ${LEFT_USB}." >&2
    exit 1
  fi
  if [[ -z "${ir}" ]]; then
    echo "Error: no CAN interface with bus-info ${RIGHT_USB}." >&2
    exit 1
  fi
  if [[ "${il}" == "${ir}" ]]; then
    echo "Error: left and right resolved to the same interface ${il}." >&2
    exit 1
  fi
  IFACE_LEFT="${il}"
  IFACE_RIGHT="${ir}"
}

resolve_pair_into


activate_iface "${IFACE_LEFT}" "${BITRATE}"
activate_iface "${IFACE_RIGHT}" "${BITRATE}"
echo "Left arm interface:  ${IFACE_LEFT}  -> ${LEFT_NAME}"
echo "Right arm interface: ${IFACE_RIGHT} -> ${RIGHT_NAME}"
assign_final_names "${IFACE_LEFT}" "${IFACE_RIGHT}"

echo "CAN setup complete: ${LEFT_NAME} and ${RIGHT_NAME} at bitrate ${BITRATE}."
echo "-------------------OVER------------------------"
