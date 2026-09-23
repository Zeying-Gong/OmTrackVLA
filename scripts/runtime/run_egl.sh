#!/usr/bin/env bash
set -euo pipefail

HAB_SIM_EGL_ROOT="${OMTRACKVLA_HAB_SIM_EGL_ROOT:-/robot/robot-research-exp-0/user/gzy/habitat-sim-src-egl}"
NVIDIA_GL_LIBS="${OMTRACKVLA_NVIDIA_GL_LIBS:-/robot/robot-research-exp-0/user/zlf/nvidia_gl_libs}"
EGL_VENDOR_JSON="${OMTRACKVLA_EGL_VENDOR_JSON:-/tmp/omtrackvla_10_nvidia_${UID:-0}.json}"

if [ "$NVIDIA_GL_LIBS" = "system" ] || [ "$NVIDIA_GL_LIBS" = "none" ]; then
  EGL_VENDOR_LIB="${OMTRACKVLA_EGL_VENDOR_LIB:-libEGL_nvidia.so.0}"
elif [ -d "$NVIDIA_GL_LIBS" ]; then
  EGL_VENDOR_LIB="${NVIDIA_GL_LIBS}/libEGL_nvidia.so.0"
  export LD_LIBRARY_PATH="${NVIDIA_GL_LIBS}:${LD_LIBRARY_PATH:-}"

  if [ "${OMTRACKVLA_EGL_PRELOAD:-1}" = "1" ]; then
    _egl_preload="${NVIDIA_GL_LIBS}/libEGL_nvidia.so.570.133.20"
    _glx_preload="${NVIDIA_GL_LIBS}/libGLX_nvidia.so.570.133.20"
    [ -e "$_egl_preload" ] || _egl_preload="${NVIDIA_GL_LIBS}/libEGL_nvidia.so.0"
    [ -e "$_glx_preload" ] || _glx_preload="${NVIDIA_GL_LIBS}/libGLX_nvidia.so.0"
    export LD_PRELOAD="${_egl_preload}:${_glx_preload}${LD_PRELOAD:+:${LD_PRELOAD}}"
  fi
else
  EGL_VENDOR_LIB="libEGL_nvidia.so.0"
fi

cat > "$EGL_VENDOR_JSON" <<JSON
{
  "file_format_version": "1.0.0",
  "ICD": { "library_path": "${EGL_VENDOR_LIB}" }
}
JSON

export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export MAGNUM_CUDA_DEVICE="${MAGNUM_CUDA_DEVICE:-0}"
export MAGNUM_WINDOWLESS_EGL_PLATFORM="${MAGNUM_WINDOWLESS_EGL_PLATFORM:-surfaceless}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
unset DISPLAY

export PYTHONPATH="${HAB_SIM_EGL_ROOT}/build/deps/magnum-bindings/src/python:${HAB_SIM_EGL_ROOT}/src_python:habitat-lab:${PYTHONPATH:-}"

echo "[run_egl] HAB_SIM_EGL_ROOT=${HAB_SIM_EGL_ROOT}"
echo "[run_egl] EGL_VENDOR_JSON=${EGL_VENDOR_JSON}"
echo "[run_egl] EGL_VENDOR_LIB=${EGL_VENDOR_LIB}"
echo "[run_egl] NVIDIA_GL_LIBS=${NVIDIA_GL_LIBS}"
echo "[run_egl] MAGNUM_CUDA_DEVICE=${MAGNUM_CUDA_DEVICE}"

exec "$@"
