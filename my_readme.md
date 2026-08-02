# OmTrackVLA 开发机迁移与测试记录

最后验证时间：2026-08-01，使用单张 NVIDIA H20 GPU。

## 本机路径

```bash
ROOT=/high_perf_store4/evad-tech-vla/gzy
REPO=$ROOT/OmTrackVLA
PYTHON_BIN=$ROOT/envs/tracevln_V2/bin/python
```

原集群命令包含 `/robot/...` 路径。这些路径在当前开发机上不存在，执行时必须替换为上面的本机路径。

## EGL 运行环境

开发机没有安装 Xvfb，因此评测使用 NVIDIA EGL 验证。已安装以下 Ubuntu 24.04 软件包：

```text
libegl1      1.7.0-1build1
libopengl0   1.7.0-1build1
libegl-mesa0 25.2.8-0ubuntu0.24.04.2
```

火山 apt 镜像无法访问，因此软件包改从小米 Ubuntu 镜像下载并在本机安装。下载的 `.deb` 文件位于：

```text
/high_perf_store4/evad-tech-vla/gzy/tmp/apt-downloads
```

`data/scene_datasets/navmeshes` 原来是指向旧机器 `/robot/...` 路径的失效软链接。现已替换为本机真实目录，使 Habitat 能够创建 episode 所需的 navmesh。

EGL 所需环境变量：

```bash
export PYTHON_BIN=$ROOT/envs/tracevln_V2/bin/python
export OMTRACKVLA_HAB_SIM_EGL_ROOT=$ROOT/habitat-sim-src-egl
export OMTRACKVLA_NVIDIA_GL_LIBS=system
export LD_LIBRARY_PATH=$ROOT/envs/tracevln_V2/lib/python3.9/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMTRACKVLA_LLM_NAME=$REPO/models/Qwen3-0.6B
export DINOV3_MODEL_PATH=$REPO/models/dinov3-vits16-pretrain-lvd1689m
export SIGLIP_MODEL_PATH=$REPO/models/siglip-so400m-patch14-384
```

## 原模型评测冒烟测试

官方 checkpoint 已使用 `RUN_WRAPPER=./run_egl.sh` 完成测试。测试配置为单张 GPU、一个 chunk、一个 episode、不保存视频并关闭 GPU burn。

测试结果：

- STT：通过，运行 1 个 episode，每个 episode 5 步。
- DT：通过，运行 1 个 episode，每个 episode 5 步。
- AT：通过，运行 1 个 episode、1 步。5 步测试已进入模型推理，但因冒烟测试超时而停止。

测试产物：

```text
$ROOT/tmp/omtrack_eval_smoke/stt_egl_siglip
$ROOT/tmp/omtrack_eval_smoke/dt_egl
$ROOT/tmp/omtrack_eval_smoke/at_egl_1step
```

单 GPU 命令模板：

```bash
cd "$REPO"

CUDA_VISIBLE_DEVICES=0 \
TASK=stt GPU_LIST=0 JOBS_PER_GPU=1 CHUNKS=1 \
MAX_EPISODES=1 MAX_STEPS=5 RESUME=0 \
SAVE_VIDEO=0 TRACKVLA_SAVE_VIDEO=0 GPU_BURN_DUTY=0 \
RUN_WRAPPER=./run_egl.sh \
PYTHON_BIN="$PYTHON_BIN" \
HF_MODEL_DIR="$REPO/official_ckpt/ckpt_0401_text_hf" \
OMTRACKVLA_HAB_SIM_EGL_ROOT="$ROOT/habitat-sim-src-egl" \
OMTRACKVLA_NVIDIA_GL_LIBS=system \
OMTRACKVLA_LLM_NAME="$REPO/models/Qwen3-0.6B" \
DINOV3_MODEL_PATH="$REPO/models/dinov3-vits16-pretrain-lvd1689m" \
SIGLIP_MODEL_PATH="$REPO/models/siglip-so400m-patch14-384" \
LD_LIBRARY_PATH="$ROOT/envs/tracevln_V2/lib/python3.9/site-packages/nvidia/cuda_runtime/lib" \
SAVE_PATH="$ROOT/tmp/omtrack_eval_smoke/manual_stt/results" \
LOG_DIR="$ROOT/tmp/omtrack_eval_smoke/manual_stt/logs" \
bash eval_official_glx.sh
```

测试其他任务时，将 `TASK` 改为 `dt` 或 `at`。

## 训练冒烟测试

已使用样例 JSONL 和视觉缓存完成离线推理，以及一步全参数训练。两项测试均已通过，生成的 checkpoint 位于：

```text
$ROOT/tmp/omtrack_migration_train_20260731/model_epoch00_step000001.pt
```

记录本文档时，完整正式训练数据的缓存仍在传输。该情况不影响 Habitat 评测和 oracle 测试。

## Oracle 单 GPU 冒烟测试模板

测试配置为单张 GPU、一个 shard、一个 split、一个 episode、不保存视频并关闭 GPU burn：

```bash
cd "$REPO"

RUN_ID=oracle_smoke_stt \
PYTHON_BIN="$PYTHON_BIN" \
OUTPUT_ROOT="$ROOT/tmp/omtrack_oracle_smoke/stt/output" \
LOG_ROOT="$ROOT/tmp/omtrack_oracle_smoke/stt/logs" \
EGL_VENDOR_DIR="$ROOT/tmp/omtrack_oracle_smoke/stt/egl" \
GPU_LIST=0 NUM_SHARDS=1 TASKS=stt SPLITS=val \
MAX_EPISODES_PER_SHARD=1 \
MAX_STEPS_PER_EPISODE=1 \
SAVE_VIDEO=0 RENDER_BACKEND=egl \
REQUIRE_100_SUCCESS=0 CONTINUE_ON_ERROR=0 GPU_BURN_DUTY=0 \
OMTRACKVLA_HAB_SIM_EGL_ROOT="$ROOT/habitat-sim-src-egl" \
OMTRACKVLA_NVIDIA_GL_LIBS=system \
LD_LIBRARY_PATH="$ROOT/envs/tracevln_V2/lib/python3.9/site-packages/nvidia/cuda_runtime/lib" \
bash eval_oracle_modular_8gpu.sh
```

测试 DT 和 AT 时，需要修改 `TASKS`，并相应修改输出和日志子目录。

Oracle 冒烟测试结果：

- STT val：通过，运行 1 个 episode、1 步；`completed=1`，`errors=0`。
- DT val：通过，排除数据集 index 186 后运行 1 个 episode、1 步；`completed=1`，`errors=0`。
- AT val：通过，运行 1 个 episode、1 步；`completed=1`，`errors=0`。

测试产物：

```text
$ROOT/tmp/omtrack_oracle_smoke/stt_1step
$ROOT/tmp/omtrack_oracle_smoke/dt_1step
$ROOT/tmp/omtrack_oracle_smoke/at_1step
```

冒烟测试会警告第一个 HM3D val 场景缺少可选的语义场景描述文件（`.basis.scn` / `info_semantic.json`）。由于跟踪任务会提供目标人物的 panoptic observation，oracle episode 仍可正常完成。如果需要使完整 benchmark 的运行环境与旧机器严格一致，仍应补齐 HM3D 语义标注文件。


## STT EGL 下 RGB 与 depth 同时使用的修复

`spot_agent_simplified_flux` 原来按照 RGB、depth、panoptic、third RGB 的顺序注册传感器。在当前 Habitat-Sim 0.3.1 和 NVIDIA EGL 环境中，这会导致 jaw RGB 持续接近黑色，但 depth 和 oracle 控制流程仍能运行。

修复方法是保留所有传感器和原有参数，仅将 jaw depth 的注册顺序及配置块移动到最后：RGB、panoptic、third RGB、depth。修复后正式 STT flux 配置的检查结果如下：

- jaw RGB：`384x384x3 uint8`，首帧均值 `110.91`。
- jaw depth：`384x384x1 float32`，有限值比例 100%，非零比例 99.96%。
- jaw panoptic：`384x384x1 int32`，输出正常。
- 50 步视频：50 帧、6.25 秒，平均亮度 `100.27`，`success=1`，跟踪率 96%。

修复后的正式 flux 视频位于：

```text
$ROOT/tmp/omtrack_depth_debug_20260802/formal_flux_50step/output/stt/val/videos/000000_bzCsHPLDztK_ep_1_610fde814f.mp4
```

可复用的诊断入口为 `debug_stt_depth_observation.py`，默认加载 `track_infer_stt_depth_debug.yaml` 并打印 RGB、depth、panoptic 的 shape、dtype 和数值统计。
