#!/usr/bin/env bash
set -euo pipefail
: "${WA_WORLD_KIND:?jepa or dino required}"
export WA_ROOT=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928
export WA_WLA_SOURCE="$WA_ROOT/dependencies/wla_v1"
export WA_WLA_CHECKPOINT=/data/nas_ray/project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt
export WA_ENCODER_WEIGHTS=/data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth
export WA_CACHE=/data/nas_ray/home/zeying.gong/datasets/wla_evt_se2_cache_20260925_v2
export WA_INDEX_ROOT="$WA_ROOT/artifacts/robot_transition_audit_v2"
export WA_OUTPUT="/data/nas_ray/project/md-ak/users/zeying.gong/job_${MD_AK_JOB_ID:?}/task_${MD_AK_TASK_ID:?}/wm_${WA_WORLD_KIND}_robot_v1"
export WA_PYTHON="$WA_ROOT/probe_env/bin/python"
export WA_GPUS=8 OMP_NUM_THREADS=2 NCCL_DEBUG=WARN
for required in "$WA_PYTHON" "$WA_ENCODER_WEIGHTS" "$WA_WLA_CHECKPOINT" "$WA_INDEX_ROOT/audit.json" "$WA_WLA_SOURCE/src/md_wla/models/queries.py"; do
  test -f "$required"
done
hostname
findmnt -T /data/nas_ray
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$WA_PYTHON" -c 'import torch; names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; print(names); assert len(names)==8 and all("4090" in n for n in names)'
bash wa/scripts/train_world.sh --epochs 1 --batch-size 2 --accumulation 2 --workers 2 --seed 42 --world-weight 0.1
"$WA_PYTHON" wa/tools/exchange.py pack --run "$WA_OUTPUT" --output "${WA_OUTPUT}.zip"
"$WA_PYTHON" wa/tools/exchange.py verify --bundle "${WA_OUTPUT}.zip"
