# V3 真实闭环独立验收

结论：**原始数据完整性与对齐、实际网格冻结、动作预测和 guard 选择均通过。本例中的离开可导航面、滑入不连通分量和 approach 缺路径停追问题已修复。教师整体质量仍未通过，不开放动作模仿训练准入。**

远端只读 CPU 审计核对全部 262 个封存文件，共 119,735,031 字节；精确文件集合、SHA256、大小均与 supervisor 一致。124 张 RGB 和 124 份原始 panoptic 的数组、目标/干扰人标签、初始化框和模型输入因果边界独立重算通过；123 个意图、执行、状态、时钟与终止帧一致。没有重新采集、创建 Simulator/renderer 或使用 GPU。

| 项目 | V2 | V3 |
|---|---:|---:|
| 目标可见帧（至少一个目标语义像素） | 35/124 | 84/124 |
| 上游 `human_following` 动作后计数 | 10/123（8.13%） | 12/123（9.76%） |
| 最终 `human_following_success` | 0 | 0 |
| 最小目标距离 | 1.1799 m | 0.8883 m |
| 最终目标距离 | 7.1933 m | 1.7260 m |
| `human_collision` 距离代理 | 全 0 | 全 0 |
| 保存的 target-robot 实际接触指标 | 全 0 | 全 0 |
| 当前网格上到目标可达的观测 | 23/124，采集后复查 | 124/124，live 网格已封存 |

两版全部 124 个实际目标位置和世界时刻完全相同，初始机器人/RGB/panoptic 一致；V3 机器人位置从帧 16 才与 V2 分歧。没有观察到目标轨迹混淆。终止仍为目标完成两个导航 goal，等待 8 步超过阈值 7，最后 `task_should_end=true`、`task_is_stop_called=true`，不是跟随成功或 300 步上限。

## Guard 的独立复算

运行前冻结的实际缓存和 reset 后序列化的 live PathFinder 字节 SHA 均为 `32c18267379578ee878c82477d2eac80ad987364dd78ebdf72f96bd9a2bf7708`，21,356 字节、5 个岛屿。审计加载的是封存的 live 文件，不依赖采集后的外部缓存状态。

独立脚本没有导入生产 guard、控制器或标签辅助函数。它按固定 `BaseVelNonCylinderAction` 源码重写运动学和三个偏移点的碰撞/sliding 公式，使用新的 native VelocityControl 与 CPU PathFinder，复算全部 **3,218 个候选**。预测 transform/basepos、偏移点过滤输入输出、候选可导航性、路径、拒绝理由和最近命令选择全部吻合。3,128 个候选接受，90 个因预测中心离开可导航面被拒绝。

guard 实际修改动作 15–25，共 11 步，包括停止 yaw 的候选；没有只靠缩放平移。所有 123 次实际执行后的 ground basepos 与预测逐项完全一致，最大 yaw 残差约 8.28e-7 rad。全部 124 帧机器人可导航，机器人与当前目标都在 island 0，且到目标路径均存在；81 个 approach 动作没有缺失 waypoint。

运行时记录了预测前后完整状态的相同哈希，并通过机器人/目标/所有角色 transform、时钟、步数及真实动作控制器速度/标志不变断言。完整的所有角色快照没有保存，审计不能从哈希独立重建这些内部状态；这是证据边界。NavMesh 约束与保存的接触指标也不等于所有动态/静态物理接触均已独立认证。

## “facing” 是面积门槛，不能直接解释为朝向

固定 `evt_bench/additional_sensor.py`（SHA256 `ab9c266bf03a7ba44968b60df0bcdb5f624e63fc025fe2cf6cdb5de566cf0747`）的 `MainHumanoidDetectorSensor.get_observation()` 每次从 `episode.info["main_human_semantic_id"]` 读取目标 ID 1060，覆盖配置初始值 100。`_assign_unique_humanoid_semantic_ids()` 保持目标 ID 1060，只为干扰人分配 2002–2008。没有发现主目标 ID 不一致。

该传感器的真实规则是 **`10000 < 目标语义像素数 < 384×384×0.3 = 44236.8`**。`human_following` 再要求距离不大于 3 m。它不是直接的机器人 yaw、相机朝向或目标朝向测试。

审计用固定真实传感器方法，在全部 124 份封存 panoptic 上重新执行，逐帧评分 **124/124 完全匹配**。84 个可见帧中，72 帧面积不超过 10,000；只有 12 帧过面积门槛，且这 12 帧距离均满足要求。最终帧目标虽然可见且距离 1.726 m，面积只有 8,884，因此 `facing=0`。没有发现当前帧评分错位；没有保存传感器内部缓存快照，不能把这个结果扩写为所有缓存机制都已被穷尽验证。

因此下一版教师应显式使用当前可见目标面积反馈，并兼顾距离、遮挡和原 guard；不能仅凭指标名称推断“朝向错误”，也不能修改 metric 来让结果通过。该项依然只是 GT 教师改进，不等于主模型完成任意目标、三模式导航或部署主动搜索。

另有独立的 formal runner 问题：`OtherHumanoidDetectorSensor` 固定遍历 1000–1100，无法覆盖已重新分配为 2002–2008 的干扰人 ID。它不进入本次 `HumanFollowing` 计算，原始 panoptic 和独立干扰人标签也不受该传感器影响。今后若使用它的避障或评分输出，应单独修复与测试，不能冒充本次 raw 数据问题。

证据文件：`raw_verification.json`、`guard_verification.json`、`facing_verification.json`、对应执行回执和源码收据。Supervisor SHA256：`772cc41265bd4468ff8c96133247a9bbffff27f868d92f00ae5657670b70637b`；冻结计划文件 SHA256：`46ea73201d6874f6a9f50839ede48d56e8196a7a38edcb1893de9d3ce96f6001`。
