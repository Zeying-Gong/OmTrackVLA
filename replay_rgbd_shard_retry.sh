#!/usr/bin/env bash
set -uo pipefail

RUN_WRAPPER="${RUN_WRAPPER:-./run_glx.sh}"
MAX_RESTARTS="${MAX_RESTARTS:-1000}"
RESTART_DELAY="${RESTART_DELAY:-2}"
MAX_NO_PROGRESS_RESTARTS="${MAX_NO_PROGRESS_RESTARTS:-3}"
attempt=0
no_progress=0

completed_count() {
  find "${OUTPUT_ROOT:?}/${TASK:?}" -name complete.json -type f 2>/dev/null | wc -l
}

while true; do
  before="$(completed_count)"
  echo "[rgbd-retry] shard=${SHARD_ID:-?}/${NUM_SHARDS:-?} attempt=$attempt"
  "$RUN_WRAPPER" bash replay_rgbd_0420.sh
  rc=$?
  if [[ "$rc" -eq 0 ]]; then
    exit 0
  fi
  if [[ "$rc" -eq 75 ]]; then
    echo "[rgbd-retry] scene batch complete; starting next pending scene in ${RESTART_DELAY}s"
    sleep "$RESTART_DELAY"
    continue
  fi
  after="$(completed_count)"
  if [[ "$after" -gt "$before" ]]; then
    no_progress=0
  else
    no_progress=$((no_progress + 1))
  fi
  attempt=$((attempt + 1))
  if [[ "$attempt" -gt "$MAX_RESTARTS" ]]; then
    echo "[rgbd-retry] ERROR exhausted restarts=$MAX_RESTARTS last_rc=$rc" >&2
    exit "$rc"
  fi
  if [[ "$no_progress" -gt "$MAX_NO_PROGRESS_RESTARTS" ]]; then
    echo "[rgbd-retry] ERROR no completed-episode progress across $no_progress failures; last_rc=$rc" >&2
    exit "$rc"
  fi
  echo "[rgbd-retry] native worker failed rc=$rc; completed episodes are durable, restarting in ${RESTART_DELAY}s" >&2
  sleep "$RESTART_DELAY"
done
