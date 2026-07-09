CHUNKS=${CHUNKS:-30}
NUM_PARALLEL=${NUM_PARALLEL:-1}
SAVE_PATH=${SAVE_PATH:-"sim_data/eval/stt"}
SAVE_VIDEO=${SAVE_VIDEO:-1}
GPU_BASE=${GPU_BASE:-1}
HABITAT_GPU_ID=${HABITAT_GPU_ID:-0}
USE_GLX=${USE_GLX:-1}
PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG_OVERRIDES=${CONFIG_OVERRIDES:-}
export DINO_MODEL=${DINO_MODEL:-facebook/dinov3-vits16-pretrain-lvd1689m}

if [ -n "${HF_MODEL_DIR:-}" ]; then
    export HF_MODEL_DIR
    echo "[eval] Using HuggingFace planner weights from ${HF_MODEL_DIR}"
fi

IDX=0
while [ $IDX -lt $CHUNKS ]; do
    for ((i = 0; i < NUM_PARALLEL && IDX < CHUNKS; i++)); do
        GPU_ID=$((GPU_BASE + i))
        RUN_PREFIX=()
        if [ "$USE_GLX" = "1" ]; then
            RUN_PREFIX=(./run_glx.sh)
        fi
        echo "Launching job IDX=$IDX on GPU=$GPU_ID; habitat_gpu=$HABITAT_GPU_ID; glx=$USE_GLX"
        CUDA_VISIBLE_DEVICES=$GPU_ID SAVE_VIDEO=$SAVE_VIDEO PYTHONPATH="habitat-lab:${PYTHONPATH:-}" "${RUN_PREFIX[@]}" "$PYTHON_BIN" run_eval.py \
            --split-num $CHUNKS \
            --split-id $IDX \
            --exp-config 'habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml' \
            --run-type 'eval' \
            --save-path $SAVE_PATH \
            habitat.simulator.habitat_sim_v0.gpu_device_id=$HABITAT_GPU_ID \
            $CONFIG_OVERRIDES &
        ((IDX++))
    done
    wait
done
