#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
SESSION=evt_corrected_at_v2
CONTINUATION="$ROOT/outputs/takeover/clean48_teacher_v5_semantic_fixed_at_v2/_runner"
LOG="$CONTINUATION/tmux.log"

cd "$ROOT"
mkdir -p "$CONTINUATION"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && env CUDA_VISIBLE_DEVICES= '$PYTHON' -B -u scripts/resume_clean48_semantic_fixed_at.py > '$LOG' 2>&1"

tmux list-sessions | grep -F "$SESSION"
