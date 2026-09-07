# OmTrackVLA Modular

一个面向仿真行人跟随的精简实验仓库。当前仓库只保留：

- EVT-Bench / Habitat 跟随环境；
- Oracle perception 和 Oracle NavMesh 上限；
- RGB-D detector/ReID 感知；
- reactive、obstacle-map、A* 和动态安全控制；
- 分片评测、进度监控、结果汇总和相关回归测试。

原始 OmTrackVLA 训练/推理、端到端 waypoint policy、FLUX/RL、数据转换试验和历史输出已移出当前主线，便于从模块化 baseline 上开始新方法。

## 目录结构

```text
OmTrackVLA/
├── omtrackvla/                     模块化方法和评测核心
│   ├── oracle_modular_follow.py   感知/控制接口与控制器
│   ├── modular_obstacle_map.py    深度局部地图与路径规划
│   ├── rgb_person_perception.py   RGB-D detector/ReID 感知
│   ├── oracle_modular_batch.py    episode/分片评测入口
│   └── oracle_modular_follow_v6.py 历史 V6 NavMesh 参数对照
├── scripts/                        启动、渲染、监控、汇总与诊断
├── evt_bench/                      EVT-Bench 动作、sensor、metric 和环境注册
├── habitat-lab/                    项目所需的 Habitat 定制版
├── data/                           task config、episode 和机器人资产
├── third_party/torchreid/           RGB 感知所需的 OSNet 实现
├── tests/                          模块化/Oracle 回归测试
├── results/                        紧凑的可追溯结果表
├── README.md                       当前使用方式
└── PROJECT_NOTES.md                 设计约定与开发注意事项
```

## 当前 baseline

| 编号 | 感知 | 控制 | 用途 |
|---|---|---|---|
| A | Oracle panoptic + GT point | Oracle NavMesh | 环境上限 |
| B | RGB-D detector + ReID | Oracle NavMesh | 单测感知 |
| C0 | Oracle point | Direct reactive | 最小控制 baseline |
| C1--C3 | Oracle point + depth | 局部地图 + A* + 历史轨迹 | 模块递进消融 |
| C4 | Oracle point + depth | C3 + 动态行人安全层 | 当前主 baseline |
| D | RGB/RGB-D + ReID | C4 | 非 Oracle 模块化系统 |

C4 在每个任务 1,405 个 val episode 上的历史结果：

| Task | Success | Tracking | Collision | Finish |
|---|---:|---:|---:|---:|
| STT | 66.55% | 77.35% | 3.35% | 96.65% |
| DT | 60.07% | 68.87% | 3.35% | 96.65% |
| AT | 64.27% | 74.91% | 3.20% | 96.80% |

详细历史表见 `results/oracle_v5_summary.csv`。

## 环境

已在 4090 机器上创建独立环境，包含 Python 3.9、Habitat-Sim 0.3.1 headless/Bullet、PyTorch 2.5.1 + CUDA 12.4 和基础依赖：

```bash
export PYTHON_BIN=/data/nas_ray/home/zeying.gong/venvs/omtrackvla-modular/bin/python
export PYTHONPATH="$PWD/habitat-lab:$PWD"
```

可以执行 `mamba activate /data/nas_ray/home/zeying.gong/venvs/omtrackvla-modular` 激活，也可直接使用上述 `PYTHON_BIN`。环境可通过根目录的 `environment.yml` 重建。

模型权重已安装在 `models/`（不入 Git）。重新拉取或校验可执行：

```bash
bash scripts/download_weights.sh
```

脚本从 TorchVision/OSNet 的公开模型源下载到临时文件，SHA-256 校验通过后才替换目标文件；重复执行时会跳过已校验权重。

## 运行

所有命令均从仓库根目录执行。

单 episode 调试：

```bash
"$PYTHON_BIN" -m omtrackvla.oracle_modular_batch \
  --task at \
  --split val \
  --shard-id 0 \
  --num-shards 1 \
  --dataset-index 49 \
  --output-root outputs/debug_at_49 \
  --perception oracle \
  --controller map-reactive-c4 \
  --save-steps \
  --save-video \
  --no-resume
```

多 GPU 全量评测：

```bash
GPU_LIST=0,1,2,3,4,5,6,7 \
NUM_WORKERS=2 \
PERCEPTION=oracle \
CONTROLLER=map-reactive-c4 \
TASKS=stt,dt,at \
SPLITS=val \
RENDER_BACKEND=egl \
SAVE_VIDEO=1 \
OUTPUT_ROOT="$PWD/outputs/c4_oracle_val" \
bash scripts/eval_oracle_modular_8gpu.sh
```

指定 episode 并行回归：

```bash
DATASET_INDICES=0,1,2,49 \
TASK=at \
SPLIT=val \
PERCEPTION=oracle \
CONTROLLER=map-reactive-c4 \
bash scripts/eval_oracle_indices_8gpu.sh
```

将 `PERCEPTION=oracle` 改为 `PERCEPTION=rgb-person` 可测 RGB-D detector/ReID 前端。不同感知或 controller 必须使用不同 `OUTPUT_ROOT`。

## 诊断和测试

```bash
"$PYTHON_BIN" -m scripts.diagnostics.avatar_switch_smoke
"$PYTHON_BIN" -m scripts.diagnostics.debug_stt_depth_observation
"$PYTHON_BIN" -m scripts.diagnostics.default_pointnav_map_debug
python -m pytest -q
```

脚本语法检查：

```bash
bash -n scripts/*.sh
```

## 数据与本地产物

- EVT episode：`data/datasets/track/{STT,DT,AT}/{train,val}`；
- avatar：`data/humanoids/humanoid_data`；
- scene：`data/scene_datasets` 软链接；
- Spot 资产：`data/robots/hab_spot_arm`；
- detector/ReID 权重：`models/`；
- 评测输出：`outputs/`，默认不入 Git。

数据、权重和视频都不应直接提交。只将固定配置、紧凑 summary 和必要的小型回归样例纳入 Git。

## 开发新方法

新方法优先实现为已有接口的替换件：

- 新感知：输出 `TargetObservation`；
- 新控制器：输入目标观测和可选 depth/map，输出 `ContinuousAction`；
- 新建图/规划：替换或扩展 `LocalObstacleMap`；
- 新评测组合：在 `oracle_modular_batch.py` 中注册，保持相同 metric 和输出 schema。

约定、已知坑和扩展点见 `PROJECT_NOTES.md`。

## 归档

2026-09-04 清理前的未跟踪第三方源码、RL 诊断脚本和约 9.3 GB 历史输出已移到：

```text
/data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA_archive_20260904
```

已跟踪的旧方法仍可从 Git 历史恢复。
