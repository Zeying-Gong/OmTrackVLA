#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

CONFIG_FILE="$REPO_ROOT/configs/pipeline/h100_8gpu.env"
PHASE_SELECTION="all"
RUN_ID=""
CLI_RUNS_ROOT=""
CLI_NPROC=""
CLI_MASTER_PORT=""
FROM_CHECKPOINT=""
RESUME_MODE="auto"
DRY_RUN=0
PREFLIGHT_ONLY=0
SKIP_GATE=0
ALLOW_NON_H100=0
ALLOW_DIRTY=0
CURRENT_PHASE=""
RUN_DIR=""
LOCK_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_pipeline_8xh100.sh [options]

Run the three OmTrackVLA training phases on one already-allocated 8xH100 node.

Options:
  --config PATH              Pipeline environment config.
  --phase all|1|2|3          Run all phases or one phase (default: all).
  --run-id ID                Stable run id. Reuse it with --resume auto.
  --runs-root PATH           Override the output root.
  --nproc-per-node N         Distributed workers (default from config: 8).
  --master-port PORT         torchrun rendezvous port.
  --from-checkpoint PATH     Initialize the first selected phase from PATH.
  --resume auto|none         Resume/skip completed work (default: auto).
  --preflight-only           Validate environment, configs and Python modules.
  --dry-run                  Print commands without checking GPUs or running jobs.
  --skip-gate                Run eval/render but do not enforce the metric gate.
  --allow-non-h100           Permit non-H100 GPUs for development.
  --allow-dirty              Permit a dirty Git worktree.
  -h, --help                 Show this help.

Examples:
  bash scripts/run_pipeline_8xh100.sh --phase all --run-id baseline_v1
  bash scripts/run_pipeline_8xh100.sh --phase 2 --run-id baseline_v1 --resume auto
  bash scripts/run_pipeline_8xh100.sh --dry-run --allow-dirty --allow-non-h100
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

while (($#)); do
  case "$1" in
    --config)
      (($# >= 2)) || die "--config requires a path"
      CONFIG_FILE="$2"
      shift 2
      ;;
    --phase)
      (($# >= 2)) || die "--phase requires all, 1, 2, or 3"
      PHASE_SELECTION="$2"
      shift 2
      ;;
    --run-id)
      (($# >= 2)) || die "--run-id requires a value"
      RUN_ID="$2"
      shift 2
      ;;
    --runs-root)
      (($# >= 2)) || die "--runs-root requires a path"
      CLI_RUNS_ROOT="$2"
      shift 2
      ;;
    --nproc-per-node)
      (($# >= 2)) || die "--nproc-per-node requires a value"
      CLI_NPROC="$2"
      shift 2
      ;;
    --master-port)
      (($# >= 2)) || die "--master-port requires a value"
      CLI_MASTER_PORT="$2"
      shift 2
      ;;
    --from-checkpoint)
      (($# >= 2)) || die "--from-checkpoint requires a path"
      FROM_CHECKPOINT="$2"
      shift 2
      ;;
    --resume)
      (($# >= 2)) || die "--resume requires auto or none"
      RESUME_MODE="$2"
      shift 2
      ;;
    --preflight-only)
      PREFLIGHT_ONLY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-gate)
      SKIP_GATE=1
      shift
      ;;
    --allow-non-h100)
      ALLOW_NON_H100=1
      shift
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "$PHASE_SELECTION" in
  all|1|2|3) ;;
  *) die "--phase must be all, 1, 2, or 3" ;;
esac
case "$RESUME_MODE" in
  auto|none) ;;
  *) die "--resume must be auto or none" ;;
esac

if [[ "$CONFIG_FILE" != /* ]]; then
  CONFIG_FILE="$REPO_ROOT/$CONFIG_FILE"
fi
[[ -f "$CONFIG_FILE" ]] || die "pipeline config not found: $CONFIG_FILE"

# This file is version-controlled shell configuration. Review changes before sourcing.
# shellcheck source=/dev/null
source "$CONFIG_FILE"

PIPELINE_PYTHON="${PIPELINE_PYTHON:-}"
if [[ -z "$PIPELINE_PYTHON" ]]; then
  if command -v python >/dev/null 2>&1; then
    PIPELINE_PYTHON="python"
  elif command -v python3 >/dev/null 2>&1; then
    PIPELINE_PYTHON="python3"
  else
    die "neither python nor python3 was found; activate the training environment"
  fi
fi
PIPELINE_NPROC_PER_NODE="${CLI_NPROC:-${PIPELINE_NPROC_PER_NODE:-8}}"
PIPELINE_MASTER_ADDR="${PIPELINE_MASTER_ADDR:-127.0.0.1}"
PIPELINE_MASTER_PORT="${CLI_MASTER_PORT:-${PIPELINE_MASTER_PORT:-29500}}"
PIPELINE_RUNS_ROOT="${CLI_RUNS_ROOT:-${PIPELINE_RUNS_ROOT:-$REPO_ROOT/outputs/training}}"
PIPELINE_DATA_ROOT="${PIPELINE_DATA_ROOT:-$REPO_ROOT/data}"
PIPELINE_REQUIRE_CLEAN_GIT="${PIPELINE_REQUIRE_CLEAN_GIT:-1}"
PIPELINE_EXPECT_H100="${PIPELINE_EXPECT_H100:-1}"
PIPELINE_LAST_CHECKPOINT_REL="${PIPELINE_LAST_CHECKPOINT_REL:-checkpoints/last.ckpt}"
PIPELINE_BEST_CHECKPOINT_REL="${PIPELINE_BEST_CHECKPOINT_REL:-checkpoints/best.ckpt}"

[[ "$PIPELINE_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "invalid nproc: $PIPELINE_NPROC_PER_NODE"
[[ "$PIPELINE_MASTER_PORT" =~ ^[0-9]+$ ]] || die "invalid master port: $PIPELINE_MASTER_PORT"
((PIPELINE_MASTER_PORT >= 1024 && PIPELINE_MASTER_PORT <= 65535)) || die "master port must be 1024..65535"

if [[ "$PIPELINE_RUNS_ROOT" != /* ]]; then
  PIPELINE_RUNS_ROOT="$REPO_ROOT/$PIPELINE_RUNS_ROOT"
fi
if [[ "$PIPELINE_DATA_ROOT" != /* ]]; then
  PIPELINE_DATA_ROOT="$REPO_ROOT/$PIPELINE_DATA_ROOT"
fi

phase_value() {
  local phase="$1"
  local suffix="$2"
  local variable="PHASE${phase}_${suffix}"
  printf '%s' "${!variable:-}"
}

repo_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s' "$path"
  else
    printf '%s' "$REPO_ROOT/$path"
  fi
}

selected_phases() {
  if [[ "$PHASE_SELECTION" == all ]]; then
    printf '%s\n' 1 2 3
  else
    printf '%s\n' "$PHASE_SELECTION"
  fi
}

check_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    if ((DRY_RUN)); then
      warn "$label is not implemented yet: $path"
    else
      die "$label not found: $path"
    fi
  fi
}

check_module() {
  local module="$1"
  local label="$2"
  if ((DRY_RUN)); then
    return
  fi
  if ! "$PIPELINE_PYTHON" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)' "$module"; then
    die "$label Python module is not implemented/importable: $module"
  fi
}

print_command() {
  printf 'DRY-RUN:'
  printf ' %q' "$@"
  printf '\n'
}

run_logged() {
  local label="$1"
  local log_file="$2"
  shift 2
  if ((DRY_RUN)); then
    printf '[%s] ' "$label"
    print_command "$@"
    return
  fi
  printf '\n[%s] starting at %s\n' "$label" "$(date --iso-8601=seconds)" | tee -a "$log_file"
  "$@" 2>&1 | tee -a "$log_file"
  printf '[%s] completed at %s\n' "$label" "$(date --iso-8601=seconds)" | tee -a "$log_file"
}

append_phase_args() {
  local destination_name="$1"
  local source_name="$2"
  if declare -p "$source_name" >/dev/null 2>&1; then
    local -n destination_ref="$destination_name"
    local -n source_ref="$source_name"
    destination_ref+=("${source_ref[@]}")
  fi
}

git_commit="unknown"
git_dirty=0
if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    git_dirty=1
  fi
fi

preflight() {
  command -v "$PIPELINE_PYTHON" >/dev/null 2>&1 || die "Python not found: $PIPELINE_PYTHON"
  [[ -d "$PIPELINE_DATA_ROOT" ]] || die "data root not found: $PIPELINE_DATA_ROOT"

  if ((git_dirty)) && [[ "$PIPELINE_REQUIRE_CLEAN_GIT" == 1 ]] && ((ALLOW_DIRTY == 0)); then
    die "Git worktree is dirty; commit/stash changes or pass --allow-dirty for development"
  fi

  if ((DRY_RUN == 0)); then
    "$PIPELINE_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
    local gpu_count
    gpu_count="$("$PIPELINE_PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
    [[ "$gpu_count" == "$PIPELINE_NPROC_PER_NODE" ]] || die "visible GPU count is $gpu_count, expected $PIPELINE_NPROC_PER_NODE"

    if [[ "$PIPELINE_EXPECT_H100" == 1 ]] && ((ALLOW_NON_H100 == 0)); then
      local unexpected_gpu
      unexpected_gpu="$("$PIPELINE_PYTHON" -c 'import torch; print("\n".join(n for n in (torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) if "H100" not in n))')"
      [[ -z "$unexpected_gpu" ]] || die "non-H100 GPU detected: $unexpected_gpu; pass --allow-non-h100 only for development"
    fi
  fi

  local phase
  while IFS= read -r phase; do
    local train_config benchmark_config gate_config
    train_config="$(repo_path "$(phase_value "$phase" TRAIN_CONFIG)")"
    benchmark_config="$(repo_path "$(phase_value "$phase" BENCHMARK_CONFIG)")"
    gate_config="$(repo_path "$(phase_value "$phase" GATE_CONFIG)")"
    check_file "$train_config" "Phase $phase train config"
    check_file "$benchmark_config" "Phase $phase benchmark config"
    if ((SKIP_GATE == 0)); then
      check_file "$gate_config" "Phase $phase gate config"
    fi
  done < <(selected_phases)

  check_module "$PIPELINE_TRAIN_MODULE" "training"
  check_module "$PIPELINE_EVAL_MODULE" "evaluation"
  check_module "$PIPELINE_RENDER_MODULE" "render"
  if ((SKIP_GATE == 0)); then
    check_module "$PIPELINE_GATE_MODULE" "gate"
  fi

  if ((DRY_RUN)); then
    printf 'Dry-run validation completed (GPU/module/file enforcement skipped): repo=%s data=%s nproc=%s git=%s dirty=%s\n' \
      "$REPO_ROOT" "$PIPELINE_DATA_ROOT" "$PIPELINE_NPROC_PER_NODE" "$git_commit" "$git_dirty"
  else
    printf 'Preflight passed: repo=%s data=%s nproc=%s git=%s dirty=%s\n' \
      "$REPO_ROOT" "$PIPELINE_DATA_ROOT" "$PIPELINE_NPROC_PER_NODE" "$git_commit" "$git_dirty"
  fi
}

write_run_manifest() {
  local manifest_path="$RUN_DIR/run_manifest.json"
  local gpu_names
  gpu_names="$("$PIPELINE_PYTHON" -c 'import json, torch; print(json.dumps([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))')"
  "$PIPELINE_PYTHON" -c '
import json, pathlib, socket, sys
path, run_id, git_commit, git_dirty, config, data_root, nproc, gpu_names = sys.argv[1:]
payload = {
    "run_id": run_id,
    "host": socket.gethostname(),
    "git_commit": git_commit,
    "git_dirty": bool(int(git_dirty)),
    "pipeline_config": config,
    "data_root": data_root,
    "nproc_per_node": int(nproc),
    "gpu_names": json.loads(gpu_names),
}
pathlib.Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
' "$manifest_path" "$RUN_ID" "$git_commit" "$git_dirty" "$CONFIG_FILE" "$PIPELINE_DATA_ROOT" "$PIPELINE_NPROC_PER_NODE" "$gpu_names"
  cp -- "$CONFIG_FILE" "$RUN_DIR/pipeline.env"
}

resolve_initial_checkpoint() {
  local phase="$1"
  local is_first="$2"
  local checkpoint=""
  local use_previous
  use_previous="$(phase_value "$phase" USE_PREVIOUS_CHECKPOINT)"

  if [[ "$is_first" == 1 && -n "$FROM_CHECKPOINT" ]]; then
    checkpoint="$FROM_CHECKPOINT"
  elif ((phase > 1)) && [[ "$use_previous" == 1 ]]; then
    checkpoint="$RUN_DIR/phase_$((phase - 1))/$PIPELINE_BEST_CHECKPOINT_REL"
  else
    checkpoint="$(phase_value "$phase" INIT_CHECKPOINT)"
  fi

  if [[ -n "$checkpoint" && "$checkpoint" != /* ]]; then
    checkpoint="$REPO_ROOT/$checkpoint"
  fi
  printf '%s' "$checkpoint"
}

run_phase() {
  local phase="$1"
  local is_first="$2"
  CURRENT_PHASE="$phase"

  local phase_dir="$RUN_DIR/phase_$phase"
  local log_dir="$phase_dir/logs"
  local best_checkpoint="$phase_dir/$PIPELINE_BEST_CHECKPOINT_REL"
  local last_checkpoint="$phase_dir/$PIPELINE_LAST_CHECKPOINT_REL"
  local metrics_path="$phase_dir/metrics.json"
  local report_path="$phase_dir/report.md"
  local gate_path="$phase_dir/gate.json"
  local train_config benchmark_config gate_config init_checkpoint
  train_config="$(repo_path "$(phase_value "$phase" TRAIN_CONFIG)")"
  benchmark_config="$(repo_path "$(phase_value "$phase" BENCHMARK_CONFIG)")"
  gate_config="$(repo_path "$(phase_value "$phase" GATE_CONFIG)")"
  init_checkpoint="$(resolve_initial_checkpoint "$phase" "$is_first")"

  if ((DRY_RUN == 0)); then
    mkdir -p "$log_dir" "$phase_dir/checkpoints" "$phase_dir/eval" "$phase_dir/visualizations" "$phase_dir/failures"
    if [[ "$RESUME_MODE" == auto && -f "$phase_dir/GATE_PASSED" ]]; then
      printf 'Phase %s already passed; skipping (%s)\n' "$phase" "$phase_dir/GATE_PASSED"
      return
    fi
  fi

  local train_cmd=(
    "$PIPELINE_PYTHON" -m torch.distributed.run
    --nnodes=1
    --nproc-per-node="$PIPELINE_NPROC_PER_NODE"
    --master-addr="$PIPELINE_MASTER_ADDR"
    --master-port="$PIPELINE_MASTER_PORT"
    --module "$PIPELINE_TRAIN_MODULE"
    --phase "$phase"
    --config "$train_config"
    --data-root "$PIPELINE_DATA_ROOT"
    --output-dir "$phase_dir"
    --run-manifest "$RUN_DIR/run_manifest.json"
  )
  if [[ "$RESUME_MODE" == auto && -f "$last_checkpoint" ]]; then
    train_cmd+=(--resume-from "$last_checkpoint")
  elif [[ -n "$init_checkpoint" ]]; then
    if ((DRY_RUN == 0)) && [[ ! -f "$init_checkpoint" ]]; then
      die "Phase $phase initialization checkpoint not found: $init_checkpoint"
    fi
    train_cmd+=(--init-checkpoint "$init_checkpoint")
  fi
  append_phase_args train_cmd "PHASE${phase}_TRAIN_ARGS"
  run_logged "phase_${phase}/train" "$log_dir/train.log" "${train_cmd[@]}"

  if ((DRY_RUN == 0)) && [[ ! -f "$best_checkpoint" ]]; then
    die "Phase $phase trainer completed without required best checkpoint: $best_checkpoint"
  fi

  local eval_cmd=(
    "$PIPELINE_PYTHON" -m torch.distributed.run
    --nnodes=1
    --nproc-per-node="$PIPELINE_NPROC_PER_NODE"
    --master-addr="$PIPELINE_MASTER_ADDR"
    --master-port="$PIPELINE_MASTER_PORT"
    --module "$PIPELINE_EVAL_MODULE"
    --phase "$phase"
    --config "$benchmark_config"
    --data-root "$PIPELINE_DATA_ROOT"
    --checkpoint "$best_checkpoint"
    --output-dir "$phase_dir/eval"
    --metrics-out "$metrics_path"
    --report-out "$report_path"
  )
  append_phase_args eval_cmd "PHASE${phase}_EVAL_ARGS"
  run_logged "phase_${phase}/eval" "$log_dir/eval.log" "${eval_cmd[@]}"

  if ((DRY_RUN == 0)) && [[ ! -f "$metrics_path" ]]; then
    die "Phase $phase evaluator completed without metrics: $metrics_path"
  fi

  local render_cmd=(
    "$PIPELINE_PYTHON" -m torch.distributed.run
    --nnodes=1
    --nproc-per-node="$PIPELINE_NPROC_PER_NODE"
    --master-addr="$PIPELINE_MASTER_ADDR"
    --master-port="$PIPELINE_MASTER_PORT"
    --module "$PIPELINE_RENDER_MODULE"
    --phase "$phase"
    --config "$benchmark_config"
    --data-root "$PIPELINE_DATA_ROOT"
    --checkpoint "$best_checkpoint"
    --split viz_val
    --output-dir "$phase_dir/visualizations"
  )
  append_phase_args render_cmd "PHASE${phase}_RENDER_ARGS"
  run_logged "phase_${phase}/render" "$log_dir/render.log" "${render_cmd[@]}"

  if ((SKIP_GATE)); then
    if ((DRY_RUN == 0)); then
      touch "$phase_dir/PHASE_COMPLETE_GATE_SKIPPED"
    fi
    warn "Phase $phase gate skipped"
    return
  fi

  local gate_cmd=(
    "$PIPELINE_PYTHON" -m "$PIPELINE_GATE_MODULE"
    --phase "$phase"
    --config "$gate_config"
    --metrics "$metrics_path"
    --output "$gate_path"
  )
  append_phase_args gate_cmd "PHASE${phase}_GATE_ARGS"
  run_logged "phase_${phase}/gate" "$log_dir/gate.log" "${gate_cmd[@]}"
  if ((DRY_RUN == 0)); then
    touch "$phase_dir/GATE_PASSED"
  fi
}

on_error() {
  local exit_code=$?
  trap - ERR
  if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
    printf 'phase=%s\nexit_code=%s\ntime=%s\n' "$CURRENT_PHASE" "$exit_code" "$(date --iso-8601=seconds)" >"$RUN_DIR/FAILED"
  fi
  exit "$exit_code"
}

cleanup() {
  if [[ -n "$LOCK_DIR" && -d "$LOCK_DIR" ]]; then
    rm -f -- "$LOCK_DIR/pid"
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
}

trap on_error ERR
trap cleanup EXIT

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/habitat-lab:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

preflight
if ((PREFLIGHT_ONLY)); then
  exit 0
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${git_commit:0:8}"
fi
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "run id may contain only letters, digits, dot, underscore and dash"
RUN_DIR="$PIPELINE_RUNS_ROOT/$RUN_ID"

if ((DRY_RUN == 0)); then
  mkdir -p "$RUN_DIR"
  LOCK_DIR="$RUN_DIR/.pipeline_lock"
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    die "run is already active or has a stale lock: $LOCK_DIR"
  fi
  printf '%s\n' "$$" >"$LOCK_DIR/pid"
  write_run_manifest
fi

first_phase=1
while IFS= read -r phase; do
  run_phase "$phase" "$first_phase"
  first_phase=0
done < <(selected_phases)

if ((DRY_RUN == 0)); then
  rm -f "$RUN_DIR/FAILED"
  touch "$RUN_DIR/PIPELINE_COMPLETE"
fi
printf 'Pipeline completed: %s\n' "$RUN_DIR"
