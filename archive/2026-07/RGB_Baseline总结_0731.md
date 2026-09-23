# RGB 感知 + V5 控制 Baseline 总结

更新时间：2026-07-31

## 1. 目标与输入定义

建立一套可运行、可复现、后续可由 learned control/RL 改进的真实感知 baseline。算法不使用文本 grounding。

- STT：场景中只有目标人物，绑定 first-visible track。
- DT：输入目标 RGB crop，在外观不同的干扰者中匹配目标。
- AT：目标与干扰者外观相同，绑定 first-visible track，依靠时序保持身份。

本阶段对应：

```text
真实 RGB 感知 + v5 coordinate control
```

## 2. 冻结 Baseline

```text
Detector: Faster R-CNN MobileNetV3
Detector threshold: 0.30
ReID: OSNet-x0.25 MSMT17
Association: ReID + HSV + bbox motion + 遮挡/lost 状态机
Controller: v5 coordinate
Renderer: Xvfb
STT/AT initialization: first-visible
DT initialization: goal-crop
```

关键实现：

- `rgb_person_perception.py`：检测、RGB-D 距离、ReID 和时序关联；
- `oracle_modular_follow.py`：v5 control 及可选 safety policy；
- `oracle_modular_batch.py`：episode 评测和逐步指标；
- `eval_oracle_indices_8gpu.sh`：指定 indices 的小批量评测。

`LOST_TARGET_POLICY=coordinate` 是主 baseline；`stop-search` 是低碰撞但较保守的 safety ablation。

## 3. Val-20 结果

每个任务运行 val dataset indices 0--19，共 60 个 episode。

| Task | SR | TR | Collision | Detection | ID Precision | ID Recall |
|---|---:|---:|---:|---:|---:|---:|
| STT | 45.00% | 68.69% | 35.00% | 47.64% | 99.29% | 52.97% |
| DT | 45.00% | 63.68% | 30.00% | 55.44% | 96.67% | 62.61% |
| AT | 30.00% | 65.31% | 45.00% | 42.79% | 100.00% | 50.58% |

同 indices 的 v5 Oracle Reference：

| Task | Baseline SR | Oracle SR | Baseline TR | Oracle TR | Baseline CR | Oracle CR |
|---|---:|---:|---:|---:|---:|---:|
| STT | 45.00% | 85.00% | 68.69% | 81.27% | 35.00% | 10.00% |
| DT | 45.00% | 75.00% | 63.68% | 79.48% | 30.00% | 10.00% |
| AT | 30.00% | 85.00% | 65.31% | 80.85% | 45.00% | 10.00% |

结论：身份 precision 已较高；主要瓶颈是 detector/track recall 和近距离控制。AT 的 SR 差距与碰撞率最大，应优先改进。

## 4. 运行命令

以下以 DT/val 0--19 为例。当前机器只暴露 GPU 0，建议最多两路并发以减少 Xvfb/native 偶发失败。

```bash
cd /robot/robot-research-exp-0/user/gzy/OmTrackVLA

PERCEPTION=rgb-person \
PERSON_SCORE_THRESHOLD=0.30 \
TARGET_INITIALIZATION=goal-crop \
LOST_TARGET_POLICY=coordinate \
TASK=dt SPLIT=val \
DATASET_INDICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
GPU_LIST=0,0 \
RUN_ID=baseline_val20_dt \
OUTPUT_ROOT=outputs/rgb_person_v5_baseline_val20_20260730 \
LOG_ROOT=outputs/rgb_person_v5_baseline_val20_20260730/logs/dt \
RENDER_BACKEND=xvfb SAVE_STEPS=1 \
bash eval_oracle_indices_8gpu.sh
```

STT/AT 将 `TASK` 改为 `stt`/`at`，并设置：

```bash
TARGET_INITIALIZATION=first-visible
```

生成汇总：

```bash
python summarize_rgb_baseline.py \
  outputs/rgb_person_v5_baseline_val20_20260730
```

## 5. 结果位置

```text
outputs/rgb_person_v5_baseline_val20_20260730/
  summary.csv                 三任务汇总
  episodes.csv                60 个 episode 指标
  comparison_oracle.csv       同 indices Oracle 对照
  {stt,dt,at}/val/episodes/   逐 step JSON
  {stt,dt,at}/val/videos/     视频
  logs/                       worker 日志
```

## 6. 下一步

1. 冻结当前感知作为第一版 baseline，避免继续围绕单 episode 调规则。
2. 用逐 step 数据进行 BC，先模仿 v5 控制。
3. 训练 residual RL：`action = v5_action + learned_residual`。
4. RL 优先优化近距离避碰、速度调节和目标丢失恢复。
5. 同时保留 `GT 感知 + learned control`，用于测量纯控制差距。

建议 RL 状态包含 range、bearing、bbox、检测/ReID 置信度、lost steps、历史动作和局部地图；奖励包含跟随距离、持续跟随、碰撞惩罚、目标丢失惩罚和动作平滑。
