#!/usr/bin/env bash
set -euo pipefail

# V27 formal candidate.  Run only after v27_probe completes and its reward /
# observation diagnostics are inspected.  The default is the conservative
# RL-only warm phase; it keeps the pretrained FLUX Linear-RF path fixed while
# learning the stochastic recovery residual and critic.
PROJECT_ROOT="${PROJECT_ROOT:-/data/nfs/share/OmTrackVLA}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/hybrid_flux}"
PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
CHECKPOINT="${V27_IL_CHECKPOINT:-${PROJECT_ROOT}/experiments/flux_ilrl_evt_train_0901_clean/il/il_warmstart_best_val.pt}"
OUTPUT_DIR="${V27_FORMAL_OUTPUT_DIR:-${PROJECT_ROOT}/experiments/flux_ilrl_evt_train_0903_v27_formal}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29648}"

cd "${PROJECT_ROOT}"
if [[ -e "${OUTPUT_DIR}/online_rl_status.json" ]]; then
  echo "refusing to reuse completed output: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}/obs_diag"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/tools/data_loader:/data/nfs/share/FLUX/baselines/flux"
export HYBRID_EXTRA_SITE_PACKAGES="/data/nfs/share/gzy/miniconda3/envs/internnav/lib/python3.10/site-packages"
export FLUX_ROOT="${FLUX_ROOT:-/data/nfs/share/FLUX/baselines/flux}"
export OMTRACKVLA_NVIDIA_GL_LIBS="system"
export V27_RL_ONLY="${V27_RL_ONLY:-1}"
export V27_FOLLOWING_PIXEL_THRESHOLD="${V27_FOLLOWING_PIXEL_THRESHOLD:-3000}"
export V27_ANCHOR_COEF="${V27_ANCHOR_COEF:-0.05}"
export V27_OBS_DIAG_DIR="${OUTPUT_DIR}/obs_diag"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

exec "${PROJECT_ROOT}/run_egl.sh" "${PYTHON_BIN}" \
  -m torch.distributed.run --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  "${CODE_ROOT}/train_online_rl_v27.py" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --tasks stt,at,dt \
  --max-episode-steps "${MAX_EPISODE_STEPS:-300}" \
  --rollout-steps "${ROLLOUT_STEPS:-64}" \
  --ppo-epochs "${PPO_EPOCHS:-4}" \
  --minibatch-size "${MINIBATCH_SIZE:-64}" \
  --max-updates "${MAX_UPDATES:-100}" \
  --min-updates "${MIN_UPDATES:-10}" \
  --eval-every "${EVAL_EVERY:-5}" \
  --eval-episodes "${EVAL_EPISODES:-4}" \
  --save-every "${SAVE_EVERY:-5}" \
  --patience "${PATIENCE:-3}" \
  --unfreeze-decoder-last-n "${UNFREEZE_DECODER_LAST_N:-0}" \
  --amp-dtype "${AMP_DTYPE:-bf16}" \
  --dist-timeout-seconds "${DIST_TIMEOUT_SECONDS:-1800}"
