#!/usr/bin/env bash
set -euo pipefail
OUT="$1"
SOURCE="$(pwd)"
ROOT=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928
WLA=/data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925
BENCH=/data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA-da3-polar-20260924
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TRACKVLA_SAVE_VIDEO=0 SAVE_VIDEO=0 TRACKVLA_VERBOSE_STEPS=0 TRACKVLA_LIVE_FRAME_INTERVAL=4
export HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet WLA_GEOMETRY_CONTROL=1
unset DA3_EVT_CHECKPOINT
sha256sum -c wa/wm/closed_loop_dependencies.sha256 > "$OUT/dependencies_check.log"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu.csv"
PYTHONPATH="$SOURCE" "$ROOT/probe_env/bin/python" -u -m wa.wm.eval_server \
 --root "$ROOT" --encoder-weight /data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth \
 --wla-source "$ROOT/dependencies/wla_v1" \
 --wla-checkpoint /data/nas_ray/project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt \
 --checkpoint /data/nas_ray/project/md-ak/users/zeying.gong/job_59566/task_70423/wm_jepa_robot_v1/checkpoint.pt \
 --ready "$OUT/server_ready.json" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
for ((i=0;i<300;i++)); do
 if test -f "$OUT/server_ready.json"; then break; fi
 if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 40 "$OUT/server.log"; exit 1; fi
 sleep 3
done
test -f "$OUT/server_ready.json"
source "$BENCH/scripts/runtime/install_xvfb_bundle.sh" "$BENCH/artifacts/xvfb_bundle"
if [ ! -e /usr/bin/xkbcomp ]; then ln -s "$XVFB_ROOT/usr/bin/xkbcomp" /usr/bin/xkbcomp; fi
if [ ! -e /usr/share/X11/xkb ]; then ln -s "$XVFB_ROOT/usr/share/X11/xkb" /usr/share/X11/xkb; fi
export OMTRACKVLA_XVFB_DISPLAY_NUM=157 OMTRACKVLA_HAB_SIM_GLX_ROOT="$BENCH"
export OMTRACKVLA_XVFB_LOG="$OUT/xvfb.log"
export PYTHONPATH="$BENCH/artifacts/official_runtime:$BENCH/torch_overlay:$BENCH:$BENCH/habitat-lab:$SOURCE:$WLA"
cd "$BENCH"
"$BENCH/scripts/runtime/run_xvfb.sh" /data/nas_ray/home/zeying.gong/algorithm/envs/habitat/bin/python -u -m wa.wm.eval_ten \
 --benchmark "$BENCH" --ready "$OUT/server_ready.json" --output "$OUT/wa" \
 --baseline /data/nas_ray/project/md-ak/users/zeying.gong/job_57564/task_68176/wla_evt_text_eval/official \
 > "$OUT/worker.log" 2>&1
