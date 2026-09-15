#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
CONFIG="$ROOT/configs/phases/v2_017_phase3_safety_head_calibration.yaml"
OUTPUT="$ROOT/outputs/training/v2_017_phase3_safety_head_calibration"
LOG="$ROOT/outputs/training/v2_017_phase3_safety_head_calibration.launch.log"
EXIT_FILE="$ROOT/outputs/training/v2_017_phase3_safety_head_calibration.EXIT_CODE"
SESSION=v2_017_phase3_safety

test ! -e "$OUTPUT"
test ! -e "$LOG"
test ! -e "$EXIT_FILE"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 2
fi

printf -v command \
  "cd %q && set -o pipefail && CUDA_VISIBLE_DEVICES=3 PYTHONPATH=%q OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 %q -u scripts/train_v2_phase3_safety_heads.py --config %q --output-dir %q --device cuda:0 > %q 2>&1; code=\$?; printf '%%s\\n' \"\$code\" > %q; exit \"\$code\"" \
  "$ROOT" "$ROOT" "$PYTHON" "$CONFIG" "$OUTPUT" "$LOG" "$EXIT_FILE"
tmux new-session -d -s "$SESSION" "$command"
echo "launched session=$SESSION gpu=3 output=$OUTPUT"
