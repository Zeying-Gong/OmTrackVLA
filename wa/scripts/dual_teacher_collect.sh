#!/usr/bin/env bash
set -euo pipefail
OUT="$1"
ROOT=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928
BENCH=/data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA-da3-polar-20260924
test -z "$(git status --porcelain)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
sha256sum -c wa/wm/dual_teacher_dependencies.sha256
nvidia-smi --query-gpu=index,name,memory.total --format=csv
git rev-parse HEAD
export OMTRACKVLA_XVFB_ROOT="/tmp/wa_dual_teacher_xvfb_${MD_AK_JOB_ID}"
source "$BENCH/scripts/runtime/install_xvfb_bundle.sh" "$BENCH/artifacts/xvfb_bundle"
if [ ! -e /usr/bin/xkbcomp ]; then ln -s "$XVFB_ROOT/usr/bin/xkbcomp" /usr/bin/xkbcomp; fi
if [ ! -e /usr/share/X11/xkb ]; then ln -s "$XVFB_ROOT/usr/share/X11/xkb" /usr/share/X11/xkb; fi
"$ROOT/probe_env/bin/python" -u wa/tools/dual_teacher_launch.py "$OUT" --formal
PYTHONPATH="$(pwd)" "$ROOT/probe_env/bin/python" -u wa/tools/audit_dual_teacher.py "$OUT" --output "$OUT/paired_audit.json"
sha256sum -c wa/wm/dual_teacher_dependencies.sha256
