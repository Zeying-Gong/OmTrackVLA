#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT=/data/nfs/share/wam_tracking/OmTrackVLA
PYTHON_BIN=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
OUTPUT_ROOT="$REPOSITORY_ROOT/outputs/evaluation/next026_overnight_verify_v1"
VIS_ROOT="$REPOSITORY_ROOT/outputs/visualization/next026_overnight_verify_v1"
LOG_ROOT="$REPOSITORY_ROOT/outputs/logs/next026_overnight_verify_v1"

mkdir -p "$OUTPUT_ROOT/direct" "$OUTPUT_ROOT/phase1_init" \
  "$VIS_ROOT/direct" "$VIS_ROOT/phase1_init" "$LOG_ROOT"
cd "$REPOSITORY_ROOT"

exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

status=running
current_stage=bootstrap
write_status() {
  local exit_code="$1"
  "$PYTHON_BIN" - "$LOG_ROOT/status.json" "$status" "$current_stage" "$exit_code" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, status, stage, exit_code = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "schema_version": 1,
            "task": "NEXT-026 overnight reproducibility verification",
            "status": status,
            "current_stage": stage,
            "exit_code": int(exit_code),
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "test_locked_used": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}
on_exit() {
  local exit_code="$?"
  if [[ "$exit_code" -eq 0 ]]; then
    status=complete
    current_stage=complete
  else
    status=failed
  fi
  write_status "$exit_code"
}
trap on_exit EXIT
write_status 0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

current_stage=direct_formal_val
write_status 0
"$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=8 --master_port=29641 \
  --module omtrackvla.evaluation.end_to_end_evaluate \
  --config configs/phases/phase2_end_to_end_v1_formal.yaml \
  --checkpoint outputs/training/next026_direct_phase2_core_v1/checkpoints/best.ckpt \
  --output "$OUTPUT_ROOT/direct/metrics.json" \
  --split val --samples-per-mode 1024 --batch-size-per-device 4 --num-workers 4

current_stage=phase1_init_formal_val
write_status 0
"$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=8 --master_port=29642 \
  --module omtrackvla.evaluation.end_to_end_evaluate \
  --config configs/phases/phase2_end_to_end_v1_phase1_init_formal.yaml \
  --checkpoint outputs/training/next026_phase1_init_phase2_core_v1/checkpoints/best.ckpt \
  --output "$OUTPUT_ROOT/phase1_init/metrics.json" \
  --split val --samples-per-mode 1024 --batch-size-per-device 4 --num-workers 4

current_stage=trained_dashboards
write_status 0
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/render_end_to_end_v1_checkpoint.py \
  --config configs/phases/phase2_end_to_end_v1_formal.yaml \
  --checkpoint outputs/training/next026_direct_phase2_core_v1/checkpoints/best.ckpt \
  --output "$VIS_ROOT/direct/dashboard.png" \
  --report "$VIS_ROOT/direct/dashboard.json"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/render_end_to_end_v1_checkpoint.py \
  --config configs/phases/phase2_end_to_end_v1_phase1_init_formal.yaml \
  --checkpoint outputs/training/next026_phase1_init_phase2_core_v1/checkpoints/best.ckpt \
  --output "$VIS_ROOT/phase1_init/dashboard.png" \
  --report "$VIS_ROOT/phase1_init/dashboard.json"

current_stage=abl_v1_08_comparison
write_status 0
"$PYTHON_BIN" scripts/compare_next026_abl_v1_08.py \
  --direct-run outputs/training/next026_direct_phase2_core_v1 \
  --phase1-init-run outputs/training/next026_phase1_init_phase2_core_v1 \
  --output "$OUTPUT_ROOT/abl_v1_08_comparison.json"

current_stage=unit_tests
write_status 0
"$PYTHON_BIN" -m unittest discover -s tests -v 2>&1 | tee "$LOG_ROOT/unittest.log"

current_stage=artifact_hashes
write_status 0
sha256sum \
  "$OUTPUT_ROOT/direct/metrics.json" \
  "$OUTPUT_ROOT/phase1_init/metrics.json" \
  "$OUTPUT_ROOT/abl_v1_08_comparison.json" \
  "$VIS_ROOT/direct/dashboard.png" \
  "$VIS_ROOT/direct/dashboard.json" \
  "$VIS_ROOT/phase1_init/dashboard.png" \
  "$VIS_ROOT/phase1_init/dashboard.json" \
  outputs/training/next026_direct_phase2_core_v1/checkpoints/best.ckpt \
  outputs/training/next026_phase1_init_phase2_core_v1/checkpoints/best.ckpt \
  > "$OUTPUT_ROOT/SHA256SUMS"

