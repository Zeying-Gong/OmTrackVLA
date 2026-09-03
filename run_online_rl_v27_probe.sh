#!/usr/bin/env bash
set -euo pipefail

# V27 one-GPU protocol probe.  This is deliberately short and uses an
# 8-step horizon only to validate imports, observations, reward components,
# PPO anchor, checkpoint writing, and validation plumbing.  It is not a
# benchmark result and must not be used as the formal training command.
PROJECT_ROOT="${PROJECT_ROOT:-/data/nfs/share/OmTrackVLA}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/hybrid_flux}"
PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
CHECKPOINT="${V27_IL_CHECKPOINT:-${PROJECT_ROOT}/experiments/flux_ilrl_evt_train_0901_clean/il/il_warmstart_best_val.pt}"
OUTPUT_DIR="${V27_PROBE_OUTPUT_DIR:-${PROJECT_ROOT}/experiments/flux_ilrl_evt_train_0903_v27_probe}"
MASTER_PORT="${MASTER_PORT:-29647}"

cd "${PROJECT_ROOT}"
if [[ -e "${OUTPUT_DIR}/online_rl_status.json" ]]; then
  echo "refusing to reuse completed output: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}/obs_diag"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/tools/data_loader:/data/nfs/share/FLUX/baselines/flux"
export HYBRID_EXTRA_SITE_PACKAGES="/data/nfs/share/gzy/miniconda3/envs/internnav/lib/python3.10/site-packages"
export FLUX_ROOT="${FLUX_ROOT:-/data/nfs/share/FLUX/baselines/flux}"
export OMTRACKVLA_NVIDIA_GL_LIBS="system"
export V27_RL_ONLY="1"
export V27_FOLLOWING_PIXEL_THRESHOLD="3000"
export V27_ANCHOR_COEF="0.05"
export V27_OBS_DIAG_DIR="${OUTPUT_DIR}/obs_diag"

exec "${PROJECT_ROOT}/run_egl.sh" "${PYTHON_BIN}" \
  "${CODE_ROOT}/train_online_rl_v27.py" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --tasks stt \
  --max-episode-steps 8 \
  --rollout-steps 2 \
  --ppo-epochs 1 \
  --minibatch-size 2 \
  --max-updates 1 \
  --min-updates 1 \
  --eval-every 1 \
  --eval-episodes 1 \
  --max-train-episodes 1 \
  --max-val-episodes 1 \
  --save-every 1 \
  --patience 2 \
  --unfreeze-decoder-last-n 0 \
  --amp-dtype bf16 \
  --dist-timeout-seconds 300
