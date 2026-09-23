#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT"
WORKERS_PER_TASK="${WORKERS_PER_TASK:-4}" bash "$ROOT/scripts/official/prepare_official_0420.sh"
GPU_LIST="${GPU_LIST:-0}" NNODES=1 NODE_RANK=0 BATCH_SIZE="${BATCH_SIZE:-8}" \
  bash "$ROOT/scripts/official/precache_official_0420.sh"
