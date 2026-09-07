# Modular / Oracle 设计笔记

本文只记录当前保留主线的稳定约定。运行命令和当前结果以 `README.md` 为准。

## 1. 模块边界

### 感知

感知层统一输出 `TargetObservation`，包含：

- `visible`、`confidence` 和 mask/bbox；
- 像素脚点；
- robot frame 下的 `(forward, left)`；
- range 和 bearing。

Oracle perception 使用 panoptic 与 GT pose，RGB 前端使用 detector/ReID/深度。两者必须输出同一接口，不让 controller 暗中依赖 GT。

### 控制

控制层输出 `ContinuousAction`，并明确限制 forward/lateral/yaw。主要实现：

- `OracleNavmeshFollower`：使用 NavMesh 的上限参考；
- `ModularReactiveFollower`：基于目标相对位置的反应式控制；
- `MapReactiveFollower`：局部深度地图、A*、历史轨迹和动态安全层。

### 建图

`LocalObstacleMap` 的核心约定：

- depth 点先做高度过滤，再转到 episode 固定坐标系；
- 静态障碍持久累积，动态人物层每帧重建；
- 目标人物像素不得写入持久静态层；
- 障碍按机器人半径膨胀，A* 输出短程 carrot；
- 历史目标点按时间优先，避免退回过时路径分支。

## 2. 评测口径

EVT-Bench 包含 STT、DT 和 AT。每个 val split 历史上有 1,405 episodes。最低报告项为 Success、Tracking、Collision 和 Finish。

必须分开报告：

- Oracle perception 与 RGB perception；
- Oracle NavMesh 与模块化局部控制；
- GT coordinate takeover 与纯视觉目标位置；
- 单 episode/smoke 与全量 benchmark。

对比前固定 commit、config、task/split、dataset indices、seed、avatar 资产、controller version 和输出目录。

## 3. 已知风险

### Avatar 错配

episode 可能指定 `male_46`/`male_56` 等 TrackVLA 自定义 avatar。并行 worker 中不能假定 semantic ID、列表索引和模型文件始终一一对应。回归时同时验证配置路径、runtime object ID 和 panoptic 值。

### Metric 口径

- 可见性/主行人像素阈值要在实验间一致；
- human collision 应按事件计数，不应将 sticky 状态每帧重复惩罚；
- `TrackEnv.step()` 后不要再重复读 raw sensor，否则会重复渲染并改变并行速度。

### 坐标与动作

在新 controller 或 learned policy 接入前，用人工 waypoint 逐步检查：

```text
target/waypoint
→ robot-frame forward/left
→ yaw 和速度限制
→ Habitat action
→ simulator pose
→ benchmark metric
```

重点是 forward/left 符号、yaw 方向、米/秒尺度、control timestep 和 action repeat。

## 4. RGB 感知约定

- target reference 来自 episode 首个可见帧，不能每帧用当前 bbox 重裁；
- 当前 bbox 只作为位置/可见性观测；
- 目标丢失时进入 brake/coast/search/recovery，不伪造有效 target；
- ReID、HSV、bbox motion 和时序证据只用于身份关联，不改变 controller 接口；
- detector/ReID 权重是本地依赖，不入 Git。

## 5. 并行与输出

- 每个 worker 只处理自己的 shard；
- episode JSON 先原子写入，再视为可 resume；
- 被截断或 schema 错误的结果在启动 worker 前清理；
- Oracle、RGB、不同 controller/version 绝不共用 `OUTPUT_ROOT`；
- summary 可入 Git，视频、raw episode JSON、log 和 checkpoint 不入 Git。

## 6. 新方法的接入顺序

1. 以现有 `TargetObservation` 或 `ContinuousAction` 作为边界，不同时改感知、控制和 metric。
2. 添加小型单元测试，覆盖坐标、mask、动作限制和丢失恢复。
3. 跑单 episode 并保存 step visualization。
4. 跑固定的 8-episode 回归集。
5. 确认口径不变后再跑 STT/DT/AT 全量。

新实验不再创建按日期命名的 Markdown。当前命令/结果更新 `README.md`，稳定设计约定更新本文。
