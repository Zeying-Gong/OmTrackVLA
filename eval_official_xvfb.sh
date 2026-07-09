#!/usr/bin/env bash
set -euo pipefail

export RUN_WRAPPER="${RUN_WRAPPER:-./run_xvfb.sh}"
exec bash eval_official_glx.sh "$@"
