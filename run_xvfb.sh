#!/usr/bin/env bash
set -euo pipefail

RANK_FOR_DISPLAY="${RANK:-${MLP_ROLE_INDEX:-0}}"
DISPLAY_NUM="${OMTRACKVLA_XVFB_DISPLAY_NUM:-$((99 + RANK_FOR_DISPLAY % 50))}"
HAB_SIM_GLX_ROOT="${OMTRACKVLA_HAB_SIM_GLX_ROOT:-/robot/robot-research-exp-0/user/gzy/habitat-sim-src}"
XVFB_LOG="${OMTRACKVLA_XVFB_LOG:-/tmp/omtrackvla_xvfb_${DISPLAY_NUM}_${UID:-0}.log}"
GLX_LIB_DIRS="${OMTRACKVLA_GLX_LIB_DIRS:-/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/local/nvidia/lib64}"
BASE_LD_LIBRARY_PATH="${OMTRACKVLA_XVFB_KEEP_LD_LIBRARY_PATH:-0}"

export DISPLAY=":${DISPLAY_NUM}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export __VK_LAYER_NV_optimus="${__VK_LAYER_NV_optimus:-NVIDIA_only}"
if [ "$BASE_LD_LIBRARY_PATH" = "1" ]; then
  export LD_LIBRARY_PATH="${GLX_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${GLX_LIB_DIRS}"
fi
export PYTHONPATH="${HAB_SIM_GLX_ROOT}/build/deps/magnum-bindings/src/python:${HAB_SIM_GLX_ROOT}/build/lib.linux-x86_64-cpython-39:${HAB_SIM_GLX_ROOT}/src_python:habitat-lab:${PYTHONPATH:-}"

# Do not inherit the old llvmpipe/software flags from run_glx.sh or parent jobs.
unset LIBGL_ALWAYS_SOFTWARE
unset GALLIUM_DRIVER
unset MESA_GL_VERSION_OVERRIDE

echo "[run_xvfb] DISPLAY=${DISPLAY}"
echo "[run_xvfb] HAB_SIM_GLX_ROOT=${HAB_SIM_GLX_ROOT}"
echo "[run_xvfb] __GLX_VENDOR_LIBRARY_NAME=${__GLX_VENDOR_LIBRARY_NAME}"
echo "[run_xvfb] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "[run_xvfb] XVFB_LOG=${XVFB_LOG}"

if command -v xdpyinfo >/dev/null 2>&1 && xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
  echo "[run_xvfb] using existing X display ${DISPLAY}"
else
  if ! pgrep -f "Xvfb :${DISPLAY_NUM} " >/dev/null 2>&1; then
    Xvfb "$DISPLAY" -screen 0 1920x1080x24 -nolisten tcp >"$XVFB_LOG" 2>&1 &
    XVFB_PID=$!
    sleep 2
    if ! kill -0 "$XVFB_PID" >/dev/null 2>&1; then
      echo "[run_xvfb] ERROR: failed to start Xvfb on ${DISPLAY}. Log: ${XVFB_LOG}" >&2
      cat "$XVFB_LOG" >&2 || true
      exit 1
    fi
  fi
fi

exec "$@"
