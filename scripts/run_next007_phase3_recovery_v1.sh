#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/training/next007_phase3_recovery_pilot_v1}"
CONFIG="${CONFIG:-$ROOT/configs/phases/next007_phase3_recovery_v1.json}"
BASE_CONFIG="$ROOT/configs/phases/phase2_end_to_end_v1_phase1_init_formal.yaml"
BASELINE_CHECKPOINT="$ROOT/outputs/ablations/next026_abl08_converged_v1/teacher_on/gru_curriculum_4k/checkpoints/best.ckpt"
MANIFEST="$OUTPUT_DIR/recovery_manifest_frozen.json"
MASTER_PORT="${MASTER_PORT:-29707}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

# Keep this descriptor open for the full orchestration (and its children).
# Retain the lock file: unlinking an active flock can create two lock inodes.
command -v flock >/dev/null 2>&1 || {
  echo "flock is required for exclusive Phase 3 orchestration" >&2
  exit 69
}
exec 9>>"$OUTPUT_DIR/.orchestrator.lock"
if ! flock -n 9; then
  echo "another Phase 3 orchestrator owns output directory: $OUTPUT_DIR" >&2
  exit 75
fi

# The completed v1 pilot is an immutable artifact. Evaluation continuation must
# use its own explicit runner/output; do not restart training from step zero.
if [[ -f "$OUTPUT_DIR/TRAINING_COMPLETE.json" ]]; then
  echo "existing Phase 3 completion marker; preserving completed run: $OUTPUT_DIR"
  exit 0
fi

exec > >(tee -a "$OUTPUT_DIR/orchestrator.log") 2>&1

while tmux list-sessions 2>/dev/null | grep -q 'next007_collect_v1_'; do
  echo "[$(date --iso-8601=seconds)] waiting for Phase 3 collection workers"
  sleep 20
done

pipeline_args=()
while IFS= read -r path; do
  pipeline_args+=(--pipeline-complete "$path")
done < <(find outputs/evaluation -path '*next007_train_candidate_formal_*_20260914_v1/PIPELINE_COMPLETE.json' -type f | sort)

if ((${#pipeline_args[@]} == 0)); then
  echo "no successful formal Phase 3 collection candidates" >&2
  exit 2
fi

if [[ ! -f "$MANIFEST" ]]; then
  "$PYTHON_BIN" scripts/build_next007_phase3_manifest.py \
    --output "$MANIFEST" \
    --minimum-samples 6 \
    --sample outputs/evaluation/next007_train_relabel_stt_000_fresh_v1/phase3_sample/sample.json \
    --sample outputs/evaluation/next007_train_relabel_dt_200_fresh_v1/phase3_sample/sample.json \
    --sample outputs/evaluation/next007_train_relabel_at_401_v1/phase3_sample/sample.json \
    "${pipeline_args[@]}"
fi

PREFLIGHT_DIR="$ROOT/outputs/evaluation/next007_phase3_recovery_preflight_v1"
if [[ ! -f "$PREFLIGHT_DIR/TRAINING_COMPLETE.json" ]]; then
  CUDA_VISIBLE_DEVICES=7 "$PYTHON_BIN" scripts/train_next007_phase3.py \
    --config "$CONFIG" \
    --output-dir "$PREFLIGHT_DIR" \
    --max-steps 1
fi

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

resume_args=()
if [[ -f "$OUTPUT_DIR/checkpoints/last.ckpt" && ! -f "$OUTPUT_DIR/TRAINING_COMPLETE.json" ]]; then
  resume_args+=(--resume-from "$OUTPUT_DIR/checkpoints/last.ckpt")
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=7 \
  --master_port="$MASTER_PORT" \
  scripts/train_next007_phase3.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  "${resume_args[@]}"

mkdir -p "$OUTPUT_DIR/eval_baseline_224" "$OUTPUT_DIR/eval_candidate_224"
"$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=7 --master_port="$((MASTER_PORT + 1))" \
  --module omtrackvla.evaluation.end_to_end_evaluate \
  --config "$BASE_CONFIG" \
  --checkpoint "$BASELINE_CHECKPOINT" \
  --output "$OUTPUT_DIR/eval_baseline_224/metrics.json" \
  --split val --samples-per-mode 224 --batch-size-per-device 4 --num-workers 3

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=7 --master_port="$((MASTER_PORT + 2))" \
  --module omtrackvla.evaluation.end_to_end_evaluate \
  --config "$BASE_CONFIG" \
  --checkpoint "$OUTPUT_DIR/checkpoints/best.ckpt" \
  --output "$OUTPUT_DIR/eval_candidate_224/metrics.json" \
  --split val --samples-per-mode 224 --batch-size-per-device 4 --num-workers 3

set +e
"$PYTHON_BIN" scripts/gate_next007_phase3_pilot.py \
  --training-complete "$OUTPUT_DIR/TRAINING_COMPLETE.json" \
  --baseline-metrics "$OUTPUT_DIR/eval_baseline_224/metrics.json" \
  --candidate-metrics "$OUTPUT_DIR/eval_candidate_224/metrics.json" \
  --output "$OUTPUT_DIR/PILOT_GATE.json"
gate_code=$?
set -e

CUDA_VISIBLE_DEVICES=7 "$PYTHON_BIN" scripts/render_end_to_end_v1_checkpoint.py \
  --config "$BASE_CONFIG" \
  --checkpoint "$OUTPUT_DIR/checkpoints/best.ckpt" \
  --output "$OUTPUT_DIR/fixed_sample.png" \
  --report "$OUTPUT_DIR/fixed_sample.json" \
  --device cuda:0

CUDA_VISIBLE_DEVICES=7 "$PYTHON_BIN" scripts/render_next007_phase3_recovery.py \
  --config "$BASE_CONFIG" \
  --manifest "$MANIFEST" \
  --before-checkpoint "$BASELINE_CHECKPOINT" \
  --after-checkpoint "$OUTPUT_DIR/checkpoints/best.ckpt" \
  --output-dir "$OUTPUT_DIR/recovery_visualization" \
  --samples-per-task 2 \
  --device cuda:0

if ((gate_code != 0)); then
  echo "Phase 3 pilot gate failed; closed-loop replay intentionally not started"
  exit "$gate_code"
fi

env \
  PHYSICAL_GPU=7 \
  TASK=stt \
  SPLIT=val \
  DATASET_INDEX=514 \
  INITIALIZATION_SCAN_EPISODES=1 \
  INITIALIZATION_WAIT_STEPS=0 \
  MAX_STEPS=50 \
  MODEL_CONFIG="$BASE_CONFIG" \
  CHECKPOINT="$OUTPUT_DIR/checkpoints/best.ckpt" \
  OUTPUT_ROOT="$OUTPUT_DIR/closed_loop_stt_val_514" \
  bash scripts/run_next027_closed_loop_smoke.sh

echo "[$(date --iso-8601=seconds)] Phase 3 pilot, open-loop gate, and closed-loop replay complete"
