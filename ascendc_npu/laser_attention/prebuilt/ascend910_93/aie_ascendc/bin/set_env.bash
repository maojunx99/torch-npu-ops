#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
# Source this vendor environment before starting the serving process.
_joyo_laser_vendor="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export ASCEND_CUSTOM_OPP_PATH="${_joyo_laser_vendor}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="${_joyo_laser_vendor}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset _joyo_laser_vendor
