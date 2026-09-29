#!/usr/bin/env bash
set -euo pipefail
export WA_ROOT=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928
export WA_WLA_SOURCE="$WA_ROOT/dependencies/wla_v1"
export WA_WLA_CHECKPOINT=/data/nas_ray/project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt
export WA_ENCODER_WEIGHTS=/data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth
export WA_CACHE=/data/nas_ray/home/zeying.gong/datasets/wla_evt_se2_cache_20260925_v2
export WA_INDEX_ROOT="$WA_ROOT/artifacts/robot_transition_audit_v2"
export WA_PYTHON="$WA_ROOT/probe_env/bin/python" WA_WORLD_KIND=jepa OMP_NUM_THREADS=2 NCCL_DEBUG=WARN
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
EXTRA=()
if [[ "${WA_DEVELOPER_DIAGNOSTIC:-0}" == 1 ]]; then
  test -z "${MD_AK_JOB_ID:-}"
  export WA_GPUS=2 WA_OUTPUT="$WA_ROOT/artifacts/epoch2_resume_developer_v1"
  EXTRA=(--diagnostic)
else
  export WA_GPUS=8 WA_OUTPUT="/data/nas_ray/project/md-ak/users/zeying.gong/job_${MD_AK_JOB_ID:?}/task_${MD_AK_TASK_ID:?}/wm_jepa_epoch2_v1"
  "$WA_PYTHON" -c 'import torch; names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; print(names); assert len(names)==8 and all("4090" in n for n in names)'
fi
bash wa/scripts/train_world.sh --epochs 1 --completed-epochs 1 --batch-size 2 --accumulation 2 --workers 2 --seed 42 --world-weight 0.1 \
  --resume /data/nas_ray/project/md-ak/users/zeying.gong/job_59566/task_70423/wm_jepa_robot_v1/checkpoint.pt \
  --resume-sha256 2cb78751977cff0a18eee887ae58157fa3f78d6ebf1208534c291a95c5035c52 "${EXTRA[@]}"
