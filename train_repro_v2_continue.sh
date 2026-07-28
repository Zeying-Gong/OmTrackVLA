#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SOURCE_CKPT="${SOURCE_CKPT:-/robot/robot-research-raw-data-0/user/gzy/omtrackvla_ckpts/repro_v2_grouped_cosine/model_epoch01_step459720.pt}"
OUT_DIR="${OUT_DIR:-/robot/robot-research-raw-data-0/user/gzy/omtrackvla_ckpts/repro_v2_continue_480k}"
TARGET_STEP="${TARGET_STEP:-480000}"

if [ ! -f "$SOURCE_CKPT" ]; then
  echo "Missing continuation checkpoint: $SOURCE_CKPT" >&2
  exit 2
fi
if [ "$TARGET_STEP" -le 459720 ]; then
  echo "TARGET_STEP must be greater than 459720" >&2
  exit 2
fi

export OUT_DIR
export RUN_NAME="${RUN_NAME:-repro_v2_continue_480k}"
export RESUME=1
export RESUME_CKPT="$SOURCE_CKPT"
export MAX_TRAIN_STEPS="$TARGET_STEP"
export EPOCHS=1
export SAVE_EVERY="${SAVE_EVERY:-10000}"
export MAX_CKPTS="${MAX_CKPTS:-4}"
export LOG_EVERY="${LOG_EVERY:-100}"
export VIS_EVERY=0

exec bash "$ROOT/train_repro_v2.sh"
