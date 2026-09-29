#!/usr/bin/env bash
set -euo pipefail
OUT="$1"
SOURCE="$(pwd)"
BENCH=/data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA-da3-polar-20260924
ROOT=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TRACKVLA_SAVE_VIDEO=0 SAVE_VIDEO=0 TRACKVLA_VERBOSE_STEPS=0 TRACKVLA_LIVE_FRAME_INTERVAL=10
export HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet WLA_GEOMETRY_CONTROL=1
unset DA3_EVT_CHECKPOINT
sha256sum -c wa/wm/closed_loop_dependencies.sha256
sha256sum -c wa/wm/full_eval_dependencies.sha256
source "$BENCH/scripts/runtime/install_xvfb_bundle.sh" "$BENCH/artifacts/xvfb_bundle"
if [ ! -e /usr/bin/xkbcomp ]; then ln -s "$XVFB_ROOT/usr/bin/xkbcomp" /usr/bin/xkbcomp; fi
if [ ! -e /usr/share/X11/xkb ]; then ln -s "$XVFB_ROOT/usr/share/X11/xkb" /usr/share/X11/xkb; fi
PYTHONPATH="$SOURCE" "$ROOT/probe_env/bin/python" -u -m wa.wm.diagnostic_launch "$OUT"
