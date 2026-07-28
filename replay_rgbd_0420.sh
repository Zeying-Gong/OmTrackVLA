#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:?Set TASK to stt, dt, or at}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
STRIDE="${STRIDE:-4}"
REPLAY_SEED="${REPLAY_SEED:-1000}"
DEPTH_MAX="${DEPTH_MAX:-10.0}"
DEPTH_SCALE="${DEPTH_SCALE:-1000}"
SOURCE_ROOT="${SOURCE_ROOT:-sim_data/${TASK}_seed1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/robot/robot-research-raw-data-0/user/gzy/omtrackvla_flux_rgbd_0420}"
PYTHON_BIN="${PYTHON_BIN:-/robot/robot-research-raw-data-0/user/gzy/temp/miniconda3/envs/tracevln_V2/bin/python}"
LOG_DIR="${LOG_DIR:-logs/replay_rgbd}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${TASK}_shard${SHARD_ID}of${NUM_SHARDS}_${RUN_ID}.log}"

args=(
  replay_rgbd.py
  --task "$TASK"
  --source-root "$SOURCE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --shard-id "$SHARD_ID"
  --num-shards "$NUM_SHARDS"
  --stride "$STRIDE"
  --depth-max "$DEPTH_MAX"
  --depth-scale "$DEPTH_SCALE"
)
if [[ -n "${MAX_EPISODES:-}" ]]; then args+=(--max-episodes "$MAX_EPISODES"); fi
if [[ -n "${EPISODE_ID:-}" ]]; then args+=(--episode-id "$EPISODE_ID"); fi
if [[ "${ONE_SCENE_PER_PROCESS:-1}" == "1" ]]; then args+=(--one-scene-per-process); fi
args+=(habitat.simulator.seed="$REPLAY_SEED")

export PYTHONPATH="habitat-lab:${PYTHONPATH:-}"
echo "[rgbd-replay] log=$LOG_FILE"
"$PYTHON_BIN" -u "${args[@]}" 2>&1 | tee "$LOG_FILE"
