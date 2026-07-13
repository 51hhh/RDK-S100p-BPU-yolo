#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-rdk}"
INTERFACE="${2:-eth0}"
if [[ "${ROLE}" == "nx" ]]; then
  LOCAL_IP="10.42.0.10"
  PEER_IP="10.42.0.20"
else
  LOCAL_IP="10.42.0.20"
  PEER_IP="10.42.0.10"
fi

ip -br address show "${INTERFACE}"
ip address show dev "${INTERFACE}" | grep -F "${LOCAL_IP}/24"
ethtool "${INTERFACE}" | grep -E 'Speed:|Duplex:|Link detected:'
ping -I "${INTERFACE}" -c 5 -W 1 "${PEER_IP}"

if command -v ros2 >/dev/null 2>&1; then
  ros2 topic info /nx/ball/observation --verbose
  ros2 topic info /d435/ball/observation --verbose
  ros2 topic info /diagnostics/time_sync --verbose
  ros2 topic info /diagnostics/transport --verbose
fi
