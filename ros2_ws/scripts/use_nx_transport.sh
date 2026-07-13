#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://${SCRIPT_DIR}/../config/cyclonedds_nx.xml"
printf 'NX ROS transport configured: domain=%s, config=%s\n' \
  "${ROS_DOMAIN_ID}" "${CYCLONEDDS_URI}"
