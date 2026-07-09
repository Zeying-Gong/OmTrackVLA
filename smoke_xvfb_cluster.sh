#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${PYTHON_BIN:=/robot/robot-research-raw-data-0/user/gzy/temp/miniconda3/envs/tracevln_V2/bin/python}"
: "${HF_MODEL_DIR:=/robot/robot-research-raw-data-0/user/gzy/temp/tmp/omtrackvla_0_6b_hf}"
: "${OMTRACKVLA_LLM_NAME:=/robot/robot-research-raw-data-0/user/gzy/temp/tmp/omtrack_hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
: "${HF_HOME:=/robot/robot-research-raw-data-0/user/gzy/temp/tmp/omtrack_hf_home}"
: "${DINO_MODEL:=/robot/robot-research-raw-data-0/user/gzy/temp/tmp/modelscope_dinov3_vits16}"
: "${HAB_SIM_TEST_SCENE:=/robot/robot-research-exp-0/user/gzy/habitat-sim-src/data/test_assets/scenes/simple_room.glb}"
: "${SAVE_PATH:=sim_data/eval_smoke/cluster_xvfb_stt_5steps_$(date +%Y%m%d_%H%M%S)}"
: "${LOG_DIR:=logs/xvfb_smoke}"
: "${LOG_FILE:=${LOG_DIR}/smoke_xvfb_cluster_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

export CUDA_VISIBLE_DEVICES
export HF_MODEL_DIR OMTRACKVLA_LLM_NAME HF_HOME DINO_MODEL
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export SAVE_VIDEO="${SAVE_VIDEO:-0}"
export TRACKVLA_SAVE_VIDEO="${TRACKVLA_SAVE_VIDEO:-0}"
export TRACKVLA_VERBOSE_STEPS="${TRACKVLA_VERBOSE_STEPS:-1}"
export HAB_SIM_TEST_SCENE

echo "================================================"
echo "Xvfb/GLX smoke settings"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  HF_MODEL_DIR=${HF_MODEL_DIR}"
echo "  SAVE_PATH=${SAVE_PATH}"
echo "  LOG_FILE=${LOG_FILE}"
echo "================================================"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -f "${HAB_SIM_TEST_SCENE}" ]]; then
  echo "ERROR: HAB_SIM_TEST_SCENE does not exist: ${HAB_SIM_TEST_SCENE}" >&2
  exit 2
fi

echo
echo "================================================"
echo "Cluster diagnostics"
echo "================================================"
date
hostname || true
pwd
echo "DISPLAY=${DISPLAY:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}"
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-}"
echo "__GLX_VENDOR_LIBRARY_NAME=${__GLX_VENDOR_LIBRARY_NAME:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "LD_PRELOAD=${LD_PRELOAD:-}"
echo
echo "[nvidia-smi]"
nvidia-smi || true
echo
echo "[NVIDIA device files]"
ls -l /dev/nvidia* 2>/dev/null || true
echo
echo "[Xvfb]"
command -v Xvfb || true
echo
echo "[run_xvfb.sh]"
ls -l ./run_xvfb.sh

echo
echo "================================================"
echo "Step 1/2: minimal Habitat-Sim Xvfb/GLX context test"
echo "================================================"

./run_xvfb.sh "${PYTHON_BIN}" - <<'PY'
import os
import habitat_sim
import magnum

print("DISPLAY =", os.environ.get("DISPLAY"))
print("habitat_sim =", habitat_sim.__version__)
print("cuda_enabled =", habitat_sim.cuda_enabled)
print("magnum.TARGET_EGL =", getattr(magnum, "TARGET_EGL", None))

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.gpu_device_id = 0
sim_cfg.scene_id = os.environ["HAB_SIM_TEST_SCENE"]

sensor = habitat_sim.CameraSensorSpec()
sensor.uuid = "color_sensor"
sensor.sensor_type = habitat_sim.SensorType.COLOR
sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
sensor.resolution = [64, 64]

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [sensor]

sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
obs = sim.get_sensor_observations()
print("obs =", obs["color_sensor"].shape, obs["color_sensor"].dtype)
sim.close()
PY

echo
echo "================================================"
echo "Step 2/2: OmTrack STT 1-episode / 5-step smoke"
echo "================================================"

./run_xvfb.sh "${PYTHON_BIN}" run_eval.py \
  --split-num 1 --split-id 0 --max-episodes 1 \
  --exp-config habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml \
  --run-type eval \
  --save-path "${SAVE_PATH}" \
  habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
  habitat.environment.max_episode_steps=5

echo
echo "================================================"
echo "Smoke finished"
echo "  SAVE_PATH=${SAVE_PATH}"
echo "  LOG_FILE=${LOG_FILE}"
echo "================================================"

if [[ -f summarize_eval.py ]]; then
  "${PYTHON_BIN}" summarize_eval.py "${SAVE_PATH}" || true
fi
