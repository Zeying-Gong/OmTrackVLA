#!/usr/bin/env bash
set -euo pipefail
: "${WA_CACHE:?Set copied cache directory}"
: "${WA_OUTPUT:?Set a NEW persistent output directory}"
: "${WA_ENCODER_WEIGHTS:?Set local torchvision ResNet18 state_dict}"
cd "$(dirname "$0")/../.."
exec "${WA_PYTHON:-python3}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m wa.train --cache "$WA_CACHE" --output "$WA_OUTPUT" --encoder-weights "$WA_ENCODER_WEIGHTS" \
  --lane external-h100 "$@" 2>&1 | tee "${WA_OUTPUT}.log"
