#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export RUN_WRAPPER="${RUN_WRAPPER:-$ROOT/scripts/runtime/run_xvfb.sh}"
exec bash "$ROOT/scripts/official/eval_official_glx.sh" "$@"
