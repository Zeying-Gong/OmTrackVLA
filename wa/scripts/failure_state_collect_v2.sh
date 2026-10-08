#!/usr/bin/env bash
# Exact125 new +1 frozen continuation; completion is NOT training admission/SR.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: failure_state_collect_v2.sh ABSOLUTE_NEW_JOB_OUTPUT" >&2
  exit 2
fi
if ! [[ "${MD_AK_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${MD_AK_TASK_ID:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo "requires a real scheduler Job/Task identity; no developer/cluster smoke" >&2
  exit 2
fi
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$SOURCE"
test "$(git rev-parse --show-toplevel)" = "$SOURCE"
test -z "$(git status --porcelain)"
ROOT=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928
BENCH=/data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA-da3-polar-20260924
PYTHON="$ROOT/probe_env/bin/python"
PLAN="$ROOT/artifacts/failure_state_plan_61609_v1.json"
PLAN_SHA=2eff9e83e89008ce1ee73632def406a27131fece6e77b01f6d4ca8c02624292b
CONTINUATION="$ROOT/artifacts/failure_state_continuation_61833_v2.json"
CONTINUATION_SHA=06d098f30ec52d988dc707249e73b17d11642f44846a75015455315891182822
TASK_ROOT="/data/nas_ray/project/md-ak/users/zeying.gong/job_${MD_AK_JOB_ID}/task_${MD_AK_TASK_ID}"
OUT="$1"
if [[ "$OUT" != "$TASK_ROOT/"* ]] ||
   ! [[ "${OUT#"$TASK_ROOT/"}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
   [ -e "$OUT" ] || [ -L "$OUT" ]; then
  echo "output must be a new direct child of this scheduler task NAS directory" >&2
  exit 2
fi
# Do not mkdir OUT: the launcher exclusively creates it after its own checks.
test -x "$PYTHON"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SOURCE"

printf 'FAILURE_STATE_V2_WRAPPER_START job=%s task=%s source=%s utc=%s\n' \
  "$MD_AK_JOB_ID" "$MD_AK_TASK_ID" "$SOURCE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git rev-parse HEAD
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader
"$PYTHON" -B - <<'PY'
import csv, io, json, subprocess
import torch
rows = list(csv.reader(io.StringIO(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"],
    text=True))))
if len(rows) != 8 or len({r[1].strip() for r in rows}) != 8:
    raise ValueError("formal allocation requires eight distinct visible GPU UUIDs")
if any("A800" not in r[2] for r in rows):
    raise ValueError("this wrapper is only for the actual A800 worker")
if torch.cuda.device_count() != 8:
    raise ValueError("formal collection requires eight actual CUDA-visible devices")
names = [torch.cuda.get_device_name(i) for i in range(8)]
if any("A800" not in name for name in names):
    raise ValueError("CUDA model names do not match the A800 allocation")
print("GPU_ALLOCATION_VERIFIED", json.dumps(dict(
    count=8, gpu_uuids=[r[1].strip() for r in rows], cuda_names=names)), flush=True)
PY

check_dependencies() {
  local manifest
  for manifest in wa/wm/dual_teacher_dependencies.sha256 \
                  wa/wm/closed_loop_dependencies.sha256 \
                  wa/wm/full_eval_dependencies.sha256 \
                  wa/wm/failure_state_dependencies.sha256; do
    test -f "$manifest" || return 1
    sha256sum -c "$manifest" || return $?
  done
  printf '%s  %s\n' "$PLAN_SHA" "$PLAN" | sha256sum -c - || return $?
  printf '%s  %s\n' "$CONTINUATION_SHA" "$CONTINUATION" | sha256sum -c - || return $?
}
check_dependencies

# Extract only into a new private worker-local directory. Never delete/reuse it,
# never alter dpkg's package database, and never copy models/data out of NAS.
XVFB_ROOT="$(mktemp -d /tmp/wa_failure_xvfb.XXXXXX)"
export XVFB_ROOT
printf 'XVFB_PRIVATE_PREFIX %s\n' "$XVFB_ROOT"
shopt -s nullglob
DEBS=("$BENCH/artifacts/xvfb_bundle/"*.deb)
if [ "${#DEBS[@]}" -eq 0 ]; then
  echo "missing existing offline Xvfb bundle" >&2
  exit 1
fi
for deb in "${DEBS[@]}"; do
  dpkg-deb -x "$deb" "$XVFB_ROOT"
done
test -x "$XVFB_ROOT/usr/bin/Xvfb"
test -x "$XVFB_ROOT/usr/bin/xkbcomp"
test -d "$XVFB_ROOT/usr/share/X11/xkb"

ensure_worker_link() {
  local target="$1" link="$2" kind="$3"
  # Called only after scheduler identity and actual8A800 checks, inside its
  # isolated worker. Never override an existing or dangling system entry.
  if [ -e "$link" ] || [ -L "$link" ]; then
    if [ "$kind" = executable ]; then test -x "$link"; else test -d "$link"; fi
    printf 'XKB_EXISTING_VALID %s -> %s\n' "$link" "$(readlink -f -- "$link")"
  else
    test -d "$(dirname "$link")"
    ln -s -- "$target" "$link"
    if [ "$kind" = executable ]; then test -x "$link"; else test -d "$link"; fi
    printf 'XKB_WORKER_LINK_CREATED %s -> %s\n' "$link" "$target"
  fi
}
ensure_worker_link "$XVFB_ROOT/usr/bin/xkbcomp" /usr/bin/xkbcomp executable
ensure_worker_link "$XVFB_ROOT/usr/share/X11/xkb" /usr/share/X11/xkb directory
export PATH="$XVFB_ROOT/usr/bin:$XVFB_ROOT/usr/sbin:${PATH:-}"
export XKB_CONFIG_ROOT="$XVFB_ROOT/usr/share/X11/xkb"
export OMTRACKVLA_GLX_LIB_DIRS="/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/local/nvidia/lib64:$XVFB_ROOT/usr/lib/x86_64-linux-gnu:$XVFB_ROOT/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$OMTRACKVLA_GLX_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_rc=0
"$PYTHON" -B -u -m wa.tools.failure_state_launch_v2 "$OUT" \
  --plan "$PLAN" --plan-sha "$PLAN_SHA" --formal \
  --continuation "$CONTINUATION" --continuation-sha "$CONTINUATION_SHA" \
  --port-base 19180 --display-base 580 || run_rc=$?
post_rc=0
check_dependencies || post_rc=$?
printf 'FAILURE_STATE_V2_WRAPPER_END launcher_exit=%s dependency_exit=%s utc=%s\n' \
  "$run_rc" "$post_rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$run_rc" -ne 0 ]; then exit "$run_rc"; fi
exit "$post_rc"
