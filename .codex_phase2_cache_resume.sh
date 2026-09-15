#!/usr/bin/env bash
set -euo pipefail

cd /data/nfs/share/wam_tracking/OmTrackVLA
export NUM_SHARDS=7
export GPU_IDS="1 2 3 4 5 6 7"
export OUTPUT_ROOT=/data/nfs/share/wam_tracking/OmTrackVLA/outputs/perception/sage3d_phase2_frozen_frontend_v1
export PYTHON_BIN=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
mkdir -p "$OUTPUT_ROOT"
exec >"$OUTPUT_ROOT/cache_resume_evt.log" 2>&1
exec bash scripts/run_sage3d_perception_cache_8gpu.sh all
