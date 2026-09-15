# FSM → Habitat 仿真 yaw 单位与时间合同

**FSM 的 rad/s 不能直接填进 `agent_1_base_vel[2]`，也不能简单除以 6.28。当前可靠关系是“每次动作的转角”；换成每个仿真世界秒的角速度必须明确动作时间区间。**

## 已核实的执行关系

冻结 AT/DT/STT 配置均为 `BaseVelNonCylinderAction`、`ang_speed=6.28`、`ctrl_freq=40`、`ac_freq_ratio=4`、`physics_target_sps=60`。输入顺序是 `[forward, lateral, yaw]`；虽然声明的 action space 较宽，实际三个分量分别裁剪到 `[-1,1]`。

真实实现先令 `angular_velocity=(0, clip(u,-1,1)*6.28, 0)`，然后**每次环境动作仅执行一次** `integrate_transform(1/40, ...)`。因此当前冻结实现的名义关系为：

```text
β = ang_speed / ctrl_freq = 0.157 rad / action / normalized_yaw
Δθ_k = β × clip(u_k, -1, 1)
ω_observed,k = Δθ_k / (world_time_k - world_time_(k-1))
```

动作积分的 `0.025 s` 不等于环境世界时间增量。`EmbodiedTask.step` 还调用 `step_physics(1/60)`，随后 `RearrangeSim.step` 执行 4 次 `internal_step(-1)`；其后端物理子步不能用 `40/4` 注释推算。配置引用的 `./data/default.physics_config.json` 在本次远端只读检查中不存在，因此更不能仅凭该配置路径宣称已固定真实物理时间步长。

证据代码：`yaw_source_snapshot/habitat-lab/habitat/tasks/rearrange/actions/actions.py:654,715`、`core/embodied_task.py:329`、`tasks/rearrange/rearrange_sim.py:911,987`。这三份下载文件的 SHA 已与本次 sidecar 冻结计划中的代码 pin 比较一致。

## 已用现成 sidecar 完成 CPU 测量

对 370 个连续观测、366 个原动作，读取相邻 camera/world time 与已经回放的动作；没有创建环境或使用 GPU。

| 来源 | 动作数 | Δt=0.048 s | Δt=0.056 s | 拟合 β，rad/action/unit |
|---|---:|---:|---:|---:|
| AT 401 | 85 | 78 | 7 | 0.15699985 |
| DT 200 | 104 | 96 | 8 | 0.15700073 |
| STT 0 | 84 | 77 | 7 | 0.15699967 |
| STT 3100 | 93 | 86 | 7 | 0.15700009 |

全部动作对 `Δθ=0.157u` 的最大绝对误差为 **1.16×10⁻⁶ rad**。但世界时间增量确有两种：337 次约 0.048 s、29 次约 0.056 s。因此每单位 normalized yaw 对应约 **3.270833 或 2.803571 rad/world-s**，不能写死 `0.048`，也不能使用一个全局拟合速率系数替代时间合同。

详见 `SIMULATOR_YAW_MEASUREMENT.json`；复核脚本为 `measure_sidecar_yaw_cpu.py`。脚本核对代码 pin、每个 labels 文件与完成状态中记录的 SHA、连续索引、正时间增量和每帧通过的来源审计。本测量未代替独立标签准入；这批动作均含平移，**没有单独原地搜索动作的验收证据**。

## 后续接入的最小合同

1. FSM 保留 `ω_requested` 的 rad/s 单位；单独的仿真适配边界计算 `u = clip(ω_requested × Δt_command / β, -1, 1)`，并记录裁剪前后值。`Δt_command` 必须是明确约定的本次命令作用区间；不能把刚测到的上一段 Δt 当作下一段必然相同，更不能提前读取下一帧时间来控制当前动作。当前只由 resolved config 可可靠得到 β，不能保证下一步世界 Δt。
2. 接入前需由仿真调度边界明确其下一段时间合同及容许速率误差，再核对实际 `Δt_world` 与 `Δθ/Δt_world`。在此之前，rad/s → normalized yaw 的转换尚未闭合，不应直接接通 FSM 输出。不能把本次观测到的最小 Δt 当作未经保证的通用下界。
3. 仿真中 FSM 的 capture 时间、控制时间、许可时间和预算统一采用同一仿真时钟域；真实 elapsed 从世界时间差获得，不能使用 `step×0.048` 或 GPU/渲染耗时。相机姿态只供离线/验收测量，不能变成搜索控制的 GT 输入。

## 必需字段与 camera 约定

保留 `run_id`、冻结 plan/source/code/config SHA、`environment_step`、`source_audit.action_just_replayed=[forward,lateral,yaw]`、相邻 `world_time_s`、`source_audit.capture_evidence.camera_before_render/camera_after_render`、对应 render 前后 world time、`source_audit.passed` 和 terminal 标记。后续另记 `ω_requested`、`Δt_command`、normalized yaw、裁剪结果、实际 Δt。

当前 `_matrix4_list` 按 Magnum `(column,row)` 索引写 JSON；该 **wire field** 的平移在最后一行。仅对这个已核实字段先 `T_world_camera = np.asarray(raw).T`，验证最后一行为 `[0,0,0,1]`、旋转正交且行列式为 +1；不要对已经规范化的 NumPy transform 再转置。

冻结固定相机可取 `f=-T_world_camera[:3,2]`，`θ=atan2(f_x,f_z)`，相邻角差用 `atan2(sin(θ_k-θ_(k-1)), cos(...))` 避免跨 ±π。该符号对应 Habitat +Y，与此次动作实测同号。相机近竖直或安装姿态变化时不能沿用此平面航向测量。角度与时间只用于独立审计及转换验证，不用目标 GT 决定动作。

本任务未修改运行代码、FSM、训练或配置，也未启动任何新的仿真/GPU 任务。
