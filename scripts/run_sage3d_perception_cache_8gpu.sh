#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/perception/sage3d_phase2_frozen_frontend_v1}"
DATA_ROOT="${SAGE3D_DATA_ROOT:-/data/nfs/share/OmTrackVLA/data/sage3d_extracted}"
MANIFEST="${PHASE_MANIFEST:-$REPO_ROOT/configs/manifests/phase1_v1.json}"
SIDECAR_ROOT="${SAGE3D_SIDECAR_ROOT:-$REPO_ROOT/results/sage3d_bbox_sidecar_v1}"
POLICY_ADMISSION="${SAGE3D_POLICY_ADMISSION:-$REPO_ROOT/results/sage3d_policy_v1_audit/admission.json}"
DETECTOR_WEIGHTS="${DETECTOR_WEIGHTS:-$REPO_ROOT/models/torchvision/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth}"
REID_WEIGHTS="${REID_WEIGHTS:-$REPO_ROOT/models/reid/osnet_x0_25_msmt17.pt}"
REID_CODE="${REID_CODE:-$REPO_ROOT/third_party/torchreid/torchreid/reid/models/osnet.py}"
FUSION_WEIGHTS="${FUSION_WEIGHTS:-$REPO_ROOT/configs/models/candidate_fusion_resnet50_tpt_train37_dualop_v3.json}"
RECORD_STRIDE="${RECORD_STRIDE:-3}"
SPLIT="${1:-all}"

if [[ "$SPLIT" == "all" ]]; then
  SPLITS=(train val viz_val)
elif [[ "$SPLIT" == "train" || "$SPLIT" == "val" || "$SPLIT" == "viz_val" ]]; then
  SPLITS=("$SPLIT")
else
  printf 'split must be all, train, val, or viz_val\n' >&2
  exit 2
fi

if [[ ! "$NUM_SHARDS" =~ ^[2-8]$ ]]; then
  printf 'NUM_SHARDS must be an integer in [2, 8]\n' >&2
  exit 2
fi
read -r -a GPU_ARRAY <<<"$GPU_IDS"
if ((${#GPU_ARRAY[@]} != NUM_SHARDS)); then
  printf 'GPU_IDS must contain exactly NUM_SHARDS device indices\n' >&2
  exit 2
fi
declare -A seen_gpu=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "$gpu" =~ ^[0-7]$ ]] || [[ -n "${seen_gpu[$gpu]:-}" ]]; then
    printf 'GPU_IDS must contain unique indices in [0, 7]\n' >&2
    exit 2
  fi
  seen_gpu[$gpu]=1
done
for path in "$PYTHON_BIN" "$MANIFEST" "$POLICY_ADMISSION" "$DETECTOR_WEIGHTS" "$REID_WEIGHTS" "$REID_CODE" "$FUSION_WEIGHTS"; do
  [[ -f "$path" ]] || { printf 'missing required file: %s\n' "$path" >&2; exit 2; }
done
[[ -d "$DATA_ROOT" && -d "$SIDECAR_ROOT" ]] || { printf 'SAGE3D data/sidecar root missing\n' >&2; exit 2; }

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for split in "${SPLITS[@]}"; do
  split_root="$OUTPUT_ROOT/$split"
  log_root="$split_root/logs"
  shard_root="$split_root/shards"
  mkdir -p "$log_root" "$shard_root"
  pids=()
  for ((index=0; index<NUM_SHARDS; index++)); do
    shard_name="$(printf 'shard-%03d-of-%03d' "$index" "$NUM_SHARDS")"
    shard_output="$shard_root/$shard_name"
    log_path="$log_root/$shard_name.log"
    gpu="${GPU_ARRAY[$index]}"
    printf '[%s] launch %s on physical cuda:%s\n' "$split" "$shard_name" "$gpu"
    PYTHONPATH="$REPO_ROOT" CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
      scripts/build_sage3d_perception_cache.py \
      --data-root "$DATA_ROOT" \
      --manifest "$MANIFEST" \
      --split "$split" \
      --sidecar-root "$SIDECAR_ROOT" \
      --policy-admission "$POLICY_ADMISSION" \
      --output-dir "$shard_output" \
      --detector-weights "$DETECTOR_WEIGHTS" \
      --reid-weights "$REID_WEIGHTS" \
      --reid-code "$REID_CODE" \
      --fusion-weights "$FUSION_WEIGHTS" \
      --device cuda \
      --record-stride "$RECORD_STRIDE" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$index" \
      --resume >"$log_path" 2>&1 &
    pids+=("$!")
  done
  failures=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done
  pids=()
  if ((failures > 0)); then
    printf '[%s] %d shard(s) failed; inspect %s\n' "$split" "$failures" "$log_root" >&2
    exit 1
  fi
  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" scripts/merge_sage3d_perception_cache.py \
    --shard-root "$shard_root" \
    --output-dir "$split_root" \
    --num-shards "$NUM_SHARDS" | tee "$log_root/merge.log"
done

"$PYTHON_BIN" - "$OUTPUT_ROOT" "${SPLITS[@]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
splits = sys.argv[2:]
payload = {"status": "complete", "splits": splits, "test_locked_used": False}
(root / "CACHE_COMPLETE.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
