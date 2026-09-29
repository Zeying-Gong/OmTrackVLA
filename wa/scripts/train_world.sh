#!/usr/bin/env bash
set -euo pipefail
: "${WA_ROOT:?persistent project asset root required}"
: "${WA_WLA_SOURCE:?pinned WLA dependency required}"
: "${WA_WLA_CHECKPOINT:?trained WLA checkpoint required}"
: "${WA_ENCODER_WEIGHTS:?DINOv2 checkpoint required}"
: "${WA_CACHE:?audited EVT cache required}"
: "${WA_INDEX_ROOT:?robot transition audit directory required}"
: "${WA_OUTPUT:?unique run output required}"
: "${WA_WORLD_KIND:?jepa or dino required}"
WA_PYTHON="${WA_PYTHON:-python}"
exec "$WA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${WA_GPUS:-8}" \
  -m wa.wm.train --root "$WA_ROOT" --wla-source "$WA_WLA_SOURCE" \
  --wla-checkpoint "$WA_WLA_CHECKPOINT" --encoder-weight "$WA_ENCODER_WEIGHTS" \
  --cache "$WA_CACHE" --index-root "$WA_INDEX_ROOT" --output "$WA_OUTPUT" \
  --kind "$WA_WORLD_KIND" "$@"
