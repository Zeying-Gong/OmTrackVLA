# OmTrackVLA 评测中目标人物模型错配问题报告

## 问题概述

在 STT 评测过程中，部分 episode 实际渲染的人物模型与该 episode 在
`val.json.gz` 中指定的 `main_humanoid_name` 不一致。

目前任务代码只在每个评测 worker 处理第一个 episode 时调用一次
`_switch_avatar()`。当同一个 worker 连续评测多个 episode 时，后续 episode
会继续使用第一个 episode 加载的人物模型。

此时，语言指令、起始位置、路径点、目标语义 ID、评测指标和输出文件名均来自
当前 episode，但画面中的人物外观可能来自该 worker 的第一个 episode。

## 具体复现案例

使用的数据集为：

```text
data/datasets/track/STT/val/val.json.gz
```

场景 `2n8kARJN3HM` 的 episode `32` 在数据集中明确指定：

```text
main_humanoid_name: male_58
main_human_semantic_id: 1089
instruction: Track the person wearing a green tunic with brown leather straps and gloves.
```

在 NumPy 随机种子为 `100`、`CHUNKS=256`，并使用 Habitat 默认的
`Dataset.get_splits(..., collate_scene_ids=True)` 时，该 episode 被分配到
chunk 13。

chunk 13 日志记录的实际 episode 执行顺序为：

```text
246, 245, 177, 32, 22, 168
```

其中第一个执行的 episode `246` 指定的人物是 `male_47`。由于人物切换逻辑
只执行一次，因此运行 episode `32` 时仍然沿用 `male_47`，而不是数据集指定的
`male_58`。

## 根本原因

问题代码位于：

```text
habitat-lab/habitat/tasks/track/track_task.py
```

原始 reset 逻辑如下：

```python
if self._sim_reset:
    if self._first_init:
        self._switch_avatar(self._sim, episode)
        self._first_init = False
    self._sim.reset()
```

`_first_init` 只在 `MultiHumanTrackingTask` 初始化时设置一次，在 episode 之间
不会重新设置。因此，每个 worker 进程中的 `_switch_avatar()` 只会为第一个
episode 执行一次。

该问题发生在模型开始根据观测执行动作之前，与具体使用的模型 checkpoint
无关。

## 为什么并行评测会暴露该问题

评测脚本以 split/chunk 为单位启动 worker。STT 验证集共有 1,405 个 episode。
当设置 `CHUNKS=256` 时，每个 worker 大约会连续处理 5 至 6 个 episode。

在当前实现下，每个 worker 只有第一个 episode 能保证加载正确的人物模型。
后续 episode 都会沿用第一个 episode 的人物，除非两者恰好指定了相同的人物
身份。

修改 GPU 数量或 `JOBS_PER_GPU` 不能解决这个问题。将每个 worker 限制为只处理
一个 episode 可以临时绕过问题，但会产生较大的模型和模拟器重复启动开销。

## 建议修复

应在每个 episode reset 时根据当前 episode 重新切换人物：

```python
if self._sim_reset:
    self._switch_avatar(self._sim, episode)
    self._first_init = False
    self._sim.reset()
```

为了方便后续排查，建议在 reset 时记录以下信息：

```text
scene_id, episode_id, main_humanoid_name, main_human_semantic_id
```

还可以额外记录实际加载的主目标人物 URDF 路径，并检查其是否与
`episode.info["main_humanoid_name"]` 一致。

## 对评测结果的影响

该问题会导致视觉目标人物与当前 episode 的语言指令不匹配，从而直接改变模型
的跟踪动作，并影响 STT 的成功率和跟踪率。

使用“每个 worker 运行多个 episode”方式生成的历史结果可能受到该问题影响，
建议在修复后重新运行评测。
