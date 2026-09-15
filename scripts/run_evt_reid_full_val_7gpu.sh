#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
KPR_ROOT="${KPR_ROOT:-/data/nfs/share/gzy/omtrackvla_external/kpr}"
KPR_CALIBRATION="${KPR_CALIBRATION:-$ROOT/outputs/evaluation/kpr_phase2b_memory_v1_preview/train_calibration.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/evaluation/evt_reid_full_val_v1}"
GPU_LIST="${GPU_LIST:-1,2,3,4,5,6,7}"
NUM_WORKERS="${NUM_WORKERS:-4}"
METHODS="${METHODS:-osnet,kpr}"

DETECTOR_WEIGHTS="$ROOT/models/torchvision/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
OSNET_WEIGHTS="$ROOT/models/reid/osnet_x0_25_msmt17.pt"
OSNET_FUSION="$ROOT/configs/models/candidate_fusion_resnet50_tpt_train37_dualop_v3.json"
KPR_WEIGHTS="$KPR_ROOT/downloads/kpr_occ_duke_SOLIDER_75.12_84.25_41443413.pth.tar"
KPR_SOURCE="$KPR_ROOT/source/keypoint_promptable_reidentification-main"

for required in "$PYTHON_BIN" "$KPR_CALIBRATION" "$DETECTOR_WEIGHTS" "$OSNET_WEIGHTS" "$OSNET_FUSION" "$KPR_WEIGHTS"; do
  [[ -e "$required" ]] || { echo "[evt-full-val] missing required path: $required" >&2; exit 2; }
done
[[ -d "$KPR_SOURCE" ]] || { echo "[evt-full-val] missing KPR source: $KPR_SOURCE" >&2; exit 2; }

jget() {
  "$PYTHON_BIN" - "$KPR_CALIBRATION" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split('.'):
    value = value[key]
print(value)
PY
}

run_method() {
  local method="$1"
  local -a gpu_array
  IFS=',' read -r -a gpu_array <<< "$GPU_LIST"
  echo "[$(date -Is)] [evt-full-val] starting method=$method"
  (
    export PYTHON_BIN GPU_LIST NUM_WORKERS
    # Keep the original shard topology when resuming on a smaller healthy GPU
    # set.  This lets finished episode JSONs remain valid and processes the
    # remaining shards in waves instead of silently repartitioning the run.
    export NUM_SHARDS="${NUM_SHARDS:-$((${#gpu_array[@]} * NUM_WORKERS))}"
    export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
    export TASKS="stt,dt,at" SPLITS="val"
    export OUTPUT_ROOT="$OUTPUT_ROOT/$method"
    export LOG_ROOT="$OUTPUT_ROOT/logs/$method"
    export RUN_ID="evt_reid_full_val_v1_${method}"
    export DISPLAY_BASE=600
    export CONTINUE_ON_ERROR=1 SAVE_VIDEO=0 SAVE_STEPS=1
    export REQUIRE_100_SUCCESS=0 GPU_BURN_DUTY=0
    export SCENES_PER_PROCESS=1 MAX_CONSECUTIVE_NATIVE_RESTARTS=10
    export PROGRESS_INTERVAL=60
    export PERCEPTION="rgb-person" CONTROLLER="reactive" TARGET_MODE="point"
    export TARGET_INITIALIZATION="goal-crop" LOST_TARGET_POLICY="coordinate"
    export PERSON_DETECTOR_WEIGHTS="$DETECTOR_WEIGHTS"
    export PERSON_DETECTOR_ARCHITECTURE="fasterrcnn_resnet50_fpn_v2"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    unset MAX_EPISODES_PER_SHARD EXCLUDE_DATASET_INDICES

    if [[ "$method" == "osnet" ]]; then
      export PERSON_REID_BACKEND="osnet"
      export PERSON_REID_WEIGHTS="$OSNET_WEIGHTS"
      export PERSON_FUSION_WEIGHTS="$OSNET_FUSION"
      unset PERSON_KPR_SOURCE PERSON_TRACKLET_IDENTITY_FLOOR
    elif [[ "$method" == "kpr" ]]; then
      export PERSON_REID_BACKEND="kpr"
      export PERSON_REID_WEIGHTS="$KPR_WEIGHTS"
      export PERSON_KPR_SOURCE="$KPR_SOURCE"
      export PERSON_FUSION_WEIGHTS=""
      export PERSON_REID_THRESHOLD="$(jget tracking.short_reacquisition_identity_threshold)"
      export PERSON_TRACKLET_IDENTITY_FLOOR="$(jget tracking.identity_floor)"
      export PERSON_SHORT_REACQUISITION_IDENTITY_THRESHOLD="$(jget tracking.short_reacquisition_identity_threshold)"
      export PERSON_GLOBAL_IDENTITY_THRESHOLD="$(jget global_multiple.score_threshold)"
      export PERSON_GLOBAL_SINGLE_IDENTITY_THRESHOLD="$(jget global_single.score_threshold)"
      export PERSON_GLOBAL_IDENTITY_MARGIN="$(jget global_multiple.margin_threshold)"
      export PERSON_REACQUISITION_CONFIRM_FRAMES=3
      export PERSON_REACQUISITION_CONSISTENCY_REID="$(jget reacquisition.consistency_reid)"
      export PERSON_MEMORY_UPDATE_DETECTOR_THRESHOLD="$(jget memory_updates.update_detector_threshold)"
      export PERSON_MEMORY_UPDATE_IDENTITY_THRESHOLD="$(jget memory_updates.update_identity_threshold)"
      export PERSON_MEMORY_UPDATE_ANCHOR_THRESHOLD="$(jget memory_updates.update_anchor_threshold)"
      export PERSON_MEMORY_UPDATE_ASSOCIATION_THRESHOLD="$(jget memory_updates.update_association_threshold)"
      export PERSON_MEMORY_UPDATE_MARGIN="$(jget memory_updates.update_margin)"
      export PERSON_MEMORY_UPDATE_MIN_CONFIRMED_STEPS="$(jget memory_updates.update_min_confirmed_steps)"
    else
      echo "[evt-full-val] unsupported method: $method" >&2
      exit 2
    fi
    scripts/eval_oracle_modular_8gpu.sh
  )
  echo "[$(date -Is)] [evt-full-val] completed method=$method"
}

mkdir -p "$OUTPUT_ROOT/logs"
IFS=',' read -r -a method_array <<< "$METHODS"
for method in "${method_array[@]}"; do
  run_method "$method"
done

"$PYTHON_BIN" scripts/summarize_evt_reid_full_val.py \
  --root "$OUTPUT_ROOT" \
  --output-json "$OUTPUT_ROOT/REPORT.json" \
  --output-csv "$OUTPUT_ROOT/REPORT.csv" \
  --require-complete
touch "$OUTPUT_ROOT/COMPLETE"
echo "[$(date -Is)] [evt-full-val] complete root=$OUTPUT_ROOT"
