#!/usr/bin/env bash
set -euo pipefail

DISPLAY_NUM="${OMTRACKVLA_GLX_DISPLAY_NUM:-109}"
HAB_SIM_GLX_ROOT="${OMTRACKVLA_HAB_SIM_GLX_ROOT:-/robot/robot-research-exp-0/user/gzy/habitat-sim-src}"
GLX_LIB_DIRS="${OMTRACKVLA_GLX_LIB_DIRS:-/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/local/nvidia/lib64}"

export DISPLAY=":${DISPLAY_NUM}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export MESA_GL_VERSION_OVERRIDE="${MESA_GL_VERSION_OVERRIDE:-4.1}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
export LD_LIBRARY_PATH="${GLX_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${HAB_SIM_GLX_ROOT}/build/deps/magnum-bindings/src/python:${HAB_SIM_GLX_ROOT}/build/lib.linux-x86_64-cpython-39:${HAB_SIM_GLX_ROOT}/src_python:habitat-lab:${PYTHONPATH:-}"

find_opengl_lib() {
  local path dir
  IFS=':' read -ra _glx_dirs <<< "$LD_LIBRARY_PATH"
  for dir in "${_glx_dirs[@]}"; do
    path="${dir}/libOpenGL.so.0"
    if [ -e "$path" ]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | awk '/libOpenGL\.so\.0 / {print $NF; exit}'
  fi
}

OPENGL_LIB="$(find_opengl_lib || true)"
echo "[run_glx] DISPLAY=${DISPLAY} HAB_SIM_GLX_ROOT=${HAB_SIM_GLX_ROOT}"
echo "[run_glx] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "[run_glx] libOpenGL.so.0=${OPENGL_LIB:-NOT_FOUND}"

if [ -z "$OPENGL_LIB" ] && [ "${OMTRACKVLA_GLX_SKIP_OPENGL_CHECK:-0}" != "1" ]; then
  echo "[run_glx] ERROR: libOpenGL.so.0 was not found. Set OMTRACKVLA_GLX_LIB_DIRS to the directory containing libOpenGL.so.0." >&2
  exit 127
fi

if ! pgrep -f "Xvfb :${DISPLAY_NUM} " >/dev/null 2>&1; then
  Xvfb ":${DISPLAY_NUM}" -screen 0 1024x768x24 +extension GLX +render -noreset >/tmp/omtrackvla_xvfb_${DISPLAY_NUM}.log 2>&1 &
  sleep 2
fi

exec "$@"
