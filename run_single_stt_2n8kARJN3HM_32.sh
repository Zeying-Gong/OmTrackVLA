#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export OMTRACKVLA_XVFB_DISPLAY_NUM="${OMTRACKVLA_XVFB_DISPLAY_NUM:-150}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_MODEL_DIR="${HF_MODEL_DIR:-$PWD/official_ckpt/ckpt_0401_text_hf}"
export HF_MODEL_ID="${HF_MODEL_ID:-}"
export TRACKVLA_SAVE_VIDEO=1
export SAVE_VIDEO=1
export TRACKVLA_VERBOSE_STEPS="${TRACKVLA_VERBOSE_STEPS:-0}"
export HF_HOME="${HF_HOME:-/robot/robot-research-raw-data-0/user/gzy/temp/tmp/omtrack_hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export OMTRACKVLA_DISABLE_HF_SSL_VERIFY="${OMTRACKVLA_DISABLE_HF_SSL_VERIFY:-1}"
export OMTRACKVLA_LLM_NAME="${OMTRACKVLA_LLM_NAME:-/robot/robot-research-raw-data-0/user/gzy/temp/tmp/omtrack_hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
export DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-/robot/robot-research-raw-data-0/user/gzy/temp/tmp/modelscope_dinov3_vits16}"
export PYTHONPATH="habitat-lab:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/robot/robot-research-raw-data-0/user/gzy/temp/miniconda3/envs/tracevln_V2/bin/python}"
SAVE_PATH="${SAVE_PATH:-sim_data/eval/single_2n8kARJN3HM_32_avatar_fix_20260714}"

./run_xvfb.sh "$PYTHON_BIN" -u -c '
import random

import habitat
import numpy as np
import evt_bench
from habitat.datasets import make_dataset

from trained_agent import evaluate_agent

config = habitat.get_config(
    "habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml",
    [
        "habitat.simulator.habitat_sim_v0.gpu_device_id=0",
        "habitat.simulator.scene_dataset=data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
    ],
)
random.seed(config.habitat.simulator.seed)
np.random.seed(config.habitat.simulator.seed)

dataset = make_dataset(
    id_dataset=config.habitat.dataset.type,
    config=config.habitat.dataset,
)
dataset.episodes = [
    episode
    for episode in dataset.episodes
    if episode.scene_id.split("/")[-2] == "2n8kARJN3HM"
    and str(episode.episode_id) == "32"
]
assert len(dataset.episodes) == 1, len(dataset.episodes)

episode = dataset.episodes[0]
print(
    "[single-episode]",
    episode.scene_id,
    episode.episode_id,
    episode.info["main_humanoid_name"],
)
evaluate_agent(config, dataset, "'"$SAVE_PATH"'")
'

echo "Video: $SAVE_PATH/2n8kARJN3HM/32.mp4"
