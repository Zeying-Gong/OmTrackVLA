# 端到端目标人物跟随模型：项目进展

> 2026-09-14 19:18 更新：V8实例修复已通过真实渲染与独立核验，修正后的48例采集已启动；旧数据/权重仍待实例复核，不作SOTA声明。LightNav安装与CPU导入通过，主权重传输中。详见 docs/SIMULATION_SOTA_STATUS_20260914.md。

> 2026-09-14 18:46 更新：已确认同外观人物共享原编号，公开节点赋值候选无效。加载阶段修复正在处理原生绑定兼容问题，尚未放行采集或训练。LightNav 181 依赖安装核验完成，权重传输保留断点恢复中。详见 docs/SIMULATION_SOTA_STATUS_20260914.md。

> 2026-09-14 17:46 纠正：真实 Habitat 探针已确认，人物的 visual_scene_nodes 为空，旧编号分配循环没有执行任何写入。旧感知框、可见性和依赖它们的跟随指标暂不能认定对应唯一物理目标。已暂停新增训练、数据准入和权重晋升，正在验证根/骨骼节点修复并准备重新采集。原结果和失败记录均保留；没有 SOTA 成绩。详见 docs/SIMULATION_SOTA_STATUS_20260914.md。

> 2026-09-14 17:14：34场景三组感知训练全部完成（总6144步），固定4案例/2场景独立复算通过；B1开发集框IoU为0.4132。V5教师同例最终成功1，但全程跟随仅43/123；新V5原48候选串行评估运行中。尚无新导航闭环或SOTA成绩。详见 docs/SIMULATION_SOTA_STATUS_20260914.md。

> 2026-09-14 16:17：固定48场景采集完成，共3003帧；34场景通过感知限定准入，2921个训练观测，其他14场景保留台账。V3修复固定例中的网格越界，整体跟随仍未成功；面积反馈V4已真实试跑，等待独立效果核验。下一轮三组各2048步感知训练准备中，尚无新闭环/SOTA成绩。详见 docs/SIMULATION_SOTA_STATUS_20260914.md。

> 2026-09-14 15:20：首轮清洁感知训练完成（3组各256步）；共同四场景raw诊断，新空间读出IoU 0.341、IoU≥0.5为15/48，同预算pooled约0.143–0.145。教师V2仍不合格，已发现实际缓存网格的分量与底盘滑移问题，修复推进中。暂无新闭环或SOTA成绩。详见 docs/SIMULATION_SOTA_STATUS_20260914.md。

> 2026-09-14 14:08：共同 raw 验证基线已完成（225 帧、48 帧可见，平均框 IoU 0.253，IoU≥0.5 为4/48）；旧 Phase2 的祖先训练清单存在官方 val 场景重合，正式参评路线改用来源可核验的新训练。clean48 跨场景候选与完整1405基准清单已固定，LightNav 独立环境已建、依赖中转安装推进中。暂无 SOTA 成绩。详见 docs/SIMULATION_SOTA_STATUS_20260914.md。

> 当前优先级：用户明确暂无实机硬件，只关注仿真并争取 SOTA。实机适配暂停，近期以 EVT-Bench 为主核对公开基线、完整 episode 清单、视角与任务条件，以及开发数据和官方评测集是否重叠；小开发集成绩不能代替可比基准结果。

> 2026-09-14 统一接管更新：完整产品继续按原接入说明验收，包括任意目标跟随、图像/坐标/混合三模式 Tracking 与 Navigation、丢失后主动搜索和重获、动态避障及 Thor 离线部署。后文“人物跟随”“不主动搜索”等是历史阶段范围，不能再据此缩小最终目标。最终模型主链仍保持端到端；外部逐帧 bbox、GT point/depth 和冻结感知 cache 不进入部署主链。
>
> 最新事实：真实时钟和连续状态训练已修复；v2b、v3、v4 开发训练均已完成，但没有新模型晋升。v4 离线选中 best16，固定四例完整闭环仍为 0/4 成功，仅比 Phase 2 多 1 个跟随帧；诊断 last64 也为 0/4。保留原 Phase 2 为工作基线。搜索状态机基础模块已安装并通过远端 Python 3.9 的 50 项 CPU 测试，尚未接入真实身份/运动许可与执行端 watchdog，不能称为主动搜索完成。固定四条训练源的 370 个同次渲染观测已完成采集和独立核验；新增感知监督接口已通过 19 条真实前缀检查、65 项不同 CPU 测试和 GPU7 的 BF16 双头反向预检；0 optimizer step，身份确认与运动许可仍未补齐。43 个低置信停车且目标仍可见的决策中有 32 个预测框零重叠，不能简单放宽停车阈值。
>
> 接手请先读 [接管记录](docs/phase3_takeover.md)、[LightNav-0 参考审查](docs/lightnav_reference_review.md)。详细闭环证据在 `outputs/takeover/permanent_val4_closed_loop_v1` 及 `permanent_val4_independent_audit_v1`；场景资产盘点在 `outputs/takeover/scene_asset_inventory_v1`，990 条资产记录尚不等于通过原要求的 1000+ 环境验收。以下旧实验、失败和约定作为历史保留。

> 本文件用于持续记录已确认决策、数据状态、训练与实验尝试、失败原因、项目修改和下一步计划。
> 原则：事实、提案和待确认事项必须分开；实验与失败记录尽量追加，不覆盖历史。
> 最近更新：2026-09-15

## 0. 当前快照

- 任务：端到端目标人物跟随。
- 目标条件：视觉侧只输入一次“初始化图像 + 目标 bbox”，后续不接收外部逐帧 bbox；UWB 是持续但可能缺失的空间条件。理想启动时，用户佩戴 UWB 并在机器人面前露面，系统据此建立视觉身份与 UWB 标签的绑定。
- 冷启动：支持没有视觉初始化的 UWB-only 启动。机器人先依据 UWB 保守接近；目标进入视野后，模型利用 UWB 投影位置、RGB 中的人物候选与时间连续性完成自动视觉绑定，并在内部保存目标身份参考。
- 缺失情形：目标不在画面时可依靠 UWB 粗定位；UWB 不可用时可在已有视觉绑定的前提下纯视觉跟随；当目标无法被视觉定位且 UWB 同时失效时，默认安全减速至停止，等待任一信号恢复，不主动盲目搜索。
- 主要观测：后续 egocentric RGB 历史。
- 输出：机器人未来的局部 waypoint trajectory。
- 明确不需要：自然语言描述、VQA、语义 CoT。
- 当前状态：WP-0数据审计、WP-1统一数据契约、WP-2 Phase 1闭环和WP-3 Phase 2A人物识别对照均已完成。EVT-Bench全量`val`对照于2026-09-11完成：OSNet与KPR各4,215/4,215 episode、共490,824步/方法，0错误、0缺失，4,215组逐episode轨迹全部一致。OSNet总体precision/recall/Micro-F1为88.5708%/63.9865%/74.2978%，KPR为83.9559%/61.7710%/71.1748%；OSNet总体领先3.1230个百分点，作为身份teacher/模块化baseline优先选项，KPR仅在AT切片Micro-F1上领先。最初GPU 1～7的28-worker任务因GPU 1、2、4～7的Habitat/EGL原生失败改为GPU 3的4 worker、保持28分片断点恢复；另将故障GPU遗留的1个黑帧OSNet结果归档并在GPU 3按原协议真实重跑，最终严格配对gate通过。此前6个episode预实验继续只作工程留档。2026-09-11代码审计进一步确认：现有Phase 2只训练读取冻结感知cache的MLP waypoint decoder，明确拒绝Phase 1 checkpoint，既不是原始RGB历史到waypoint的联合训练，也没有让Phase 1表示参与轨迹预测。用户据此重申最终研究主线必须是严格端到端人物跟随；现有1,303/5,584个SAGE3D OSNet cache完整保留，但降级为可复现模块化baseline，不再作为下一P0。始终未访问`test_locked`。
- 评测优先级：按2026-09-10项目确认，Phase 2A行人识别主评测是EVT-Bench；TpT是重要的真实域/遮挡/长序列压力测试，不再单独决定快速迭代去留。EVT快速对照须固定episode和控制轨迹，正式阈值后续只从EVT `train`冻结。
- 集群入口：`scripts/run_pipeline_8xh100.sh`已接通Phase 1和旧Phase 2模块化baseline；Architecture v1统一模型、正式Phase 2训练/评测/渲染、ABL-08/09及独立closed-loop runner均已实现并验证。Phase 3目前只有模型访问状态采集、expert relabel和小批量恢复smoke，正式3个配置、可恢复训练与退出gate仍缺，因此`--phase all`仍会明确失败。OmTrackVLA使用`wam`分支并通过GitHub `origin/wam`协作。
- 当前方案：采用3个正式阶段——Phase 1共享视觉身份与World-Action预训练、Phase 2从“一次初始化RGB+bbox + 后续RGB历史 + 可缺失UWB”到未来局部waypoint的联合监督微调、Phase 3噪声与闭环恢复训练。阶段可以分开训练，但最终Phase 2推理链不得依赖预计算cache、外部逐帧bbox、GT目标位置或GT depth，waypoint主损失必须至少回传到目标条件时序融合与共享视觉表示。
- Architecture v1已于2026-09-11冻结：DA3-SMALL L11经Target Cross-Attention得到`z_target`，Scene Attention与camera-token pair经显式SE(2)瓶颈形成`w_t`；UWB同时生成20×36 camera-geometry Gaussian patch bias和独立`z_uwb`，再由Fusion MLP、GRU输出8×2 waypoint与stop。部署删除DualDPT/Camera/GS heads。完整定义、tensor shape、loss和强制消融见第3.1节。
- NEXT-025最小闭环已于2026-09-11通过：官方DA3-SMALL加载437/437 tensor、34,299,463/34,299,463参数（100%）；冻结DA3 22,059,008参数、adapter 100,736个可训练参数、新policy 2,117,907个可训练参数。真实、已准入且不读cache的SAGE3D `train` batch仅用waypoint SmoothL1反向时，Fusion/GRU/DA3后层adapter梯度范数分别为`5.740385e-3`/`6.065018e-3`/`5.033064e-5`；单卡与一步8×H100 DDP均通过，未使用辅助loss、未正式训练、未访问`test_locked`。
- NEXT-026已在用户确认训练前可视化后启动正式Architecture v1 Phase 2训练。当前最低ADE/FDE模型为`next026_phase2_world_action_long_v1`：在既有4,096步后追加32,768步，有效累计36,864步；SAGE3D `val`每模式1,024样本的正常模式ADE/FDE为`0.087408/0.151858 m`，路径长度比`0.766329`，finite与stop accuracy均100%；checkpoint SHA-256为`a1dc37ff...`。该阶段只用已准入SAGE3D训练waypoint与辅助标签，不用EVT-Bench训练，也未访问`test_locked`。
- 用户指出固定图中绿色预测轨迹明显短于专家后，已进行受控精修而非盲目续训。直接path-length loss虽可把路径比推到约1.0，但固定图出现锯齿/回退，已否决；terminal+delta、普通低学习率续训和terminal-only均存在误差/长度权衡。当前“更长且自然”候选为`next026_phase2_waypoint_radial_progress_v1`的1,000步checkpoint：正式正常模式ADE/FDE `0.088484/0.153559 m`、路径长度比`0.800828`，相对最低误差模型路径长度提高4.50%、ADE/FDE代价1.23%/1.12%；固定样例所有相邻点向前、无回退。两种checkpoint均保留，不互相覆盖。
- NEXT-026最初13臂各4,096步结果现统一降级为`4k diagnostic`：除single-step外，direct随机policy各臂均停留在约`0.176～0.177 m` ADE的短轨迹平台，不能据此做模块因果归因。等预算36,864步复核中，随机GRU主对照为`0.119734/0.205298 m`、路径比`0.701199`，single-step为`0.090825/0.156395 m`、路径比`0.734556`。进一步采用“36,864步single-step表示学习→近恒等重置GRU→GRU单独高学习率、其余低学习率4,096步”的课程后，GRU模型达到`0.089646/0.154162 m`、路径比`0.746039`；强制每步清零hidden后退化为`0.090231/0.155013 m`、路径比`0.741733`，证明小幅收益中确有真实递归贡献。该课程只改变优化顺序，不改变冻结Deployment主链。
- ABL-V1-08四个收敛臂已于2026-09-13全部完成。teacher-off/on的single-step ADE/FDE分别为`0.086013/0.148800`与`0.084867/0.147118 m`；对应GRU课程为`0.085573/0.148125`与当前最优`0.084341/0.146654 m`。teacher-on对matched single-step改善ADE/FDE 1.33%/1.13%，GRU课程再改善0.62%/0.32%；四臂finite coverage与safe-stop均100%，固定图仍显示共同的终点偏短。全部只用SAGE3D非locked split，DA3加载100%，waypoint-only梯度对Fusion/GRU/adapter非零。
- ABL-V1-09 Frozen Frontend Baseline已完成。cache显式记账为train/val/viz_val `5578/733/389`，另有train中5个canonical rejected被结构化跳过，`cached + skipped = requested`且`CACHE_COMPLETE.json`确认未使用`test_locked`。19项专项测试、2-step smoke、36,864步正式训练、每模式1,024个val评测和64帧固定render均完成。冻结前端基线ADE/FDE为`0.168628/0.282747 m`，Architecture v1最优为`0.084341/0.146654 m`，相对误差改善49.98%/48.13%；说明端到端视觉适配有明确价值。首次评测启动因环境`torchrun`脚本残留旧Python shebang退出126，失败记录已归档，改用当前`PYTHON_BIN -m torch.distributed.run`后从评测阶段续跑成功，没有重训或覆盖既有checkpoint。GPU0未使用；不用EVT-Bench waypoint训练，不访问`test_locked`。
- NEXT-027已启动并完成首个固定Habitat episode的端到端闭环诊断，但未通过最终gate。决策输入审计确认首帧只用RGB+bbox，后续只用新RGB和可选UWB；本次UWB缺失，不用GT target point、后续bbox、depth、pose或perception cache。修复`env.step()`返回旧RGB后，50步v6完整运行、0碰撞且视频50/50帧可解码；但人物在墙角遮挡后第28步丢失，模型未重获，following rate仅52%，因此只能称“安全子门通过，跟随与重获失败”，不能称NEXT-027完成。
- NEXT-007的train-split最小恢复闭环已通过。STT/DT/AT三个独立train episode各完成12步真实模型闭环，逐步强制刷新RGB均为新帧；在模型访问的post-step-9状态，用独立Habitat实例预演未来目标并生成8点expert标签。三样本replay/未来目标误差为0，坐标误差不超过`2.25e-7 m`，21步内相对静止anchor的距离改善为`0.488/0.499/0.495 m`，方向余弦`0.959/0.999/0.987`，碰撞均为0。3样本、8 optimizer-step、仅waypoint loss的minibatch smoke使loss从`0.014094`降至`0.002434`，Fusion/GRU/DA3 adapter梯度为`0.076001/0.011950/0.012074`；没有保存checkpoint或开始正式Phase 3，现等待用户检查训练前/后叠加图。
- Architecture v2后续开发固定为8帧历史、相邻帧0.1秒，并保留Architecture v1的DA3/Target/World/UWB双路/Fusion主链；当前Phase 3只是小预算DAgger与可观测性pilot，不是正式大规模训练。v2_014/v2_015共采集8条多场景完整rollout并成功重标18/20次；index1300在此阶段完全未见，留作闭环gate。
- v2_016使用严格场景与状态不重叠的20个train/13个recovery-val样本训练128步：recovery ADE `0.23934→0.22053 m`、expert ADE `0.37993→0.36507 m`、safe-stop path `0.14875→0.10620 m`，SAGE3D ADE小幅回退`0.05763→0.06172 m`；唯一未过项为false-visible仍为5。该step128只作闭环候选，SHA-256 `329b589f...`。
- v2_017从v2_016 step128只训练visibility/stop heads 32步，其他参数逐tensor哈希确认不变；离线recovery-val上false-visible `5→4`、false-invisible `3→1`、visibility accuracy `38.46%→61.54%`，waypoint指标漂移严格为0，best SHA-256 `ab8d8063...`。但stop accuracy仍为`0.53846`且概率偏低，因此离线通过不能直接晋升。
- v2_018已在此前完全未见的Habitat `train/index1300`完成v2_006b/v2_016/v2_017 × STT/DT/AT × `polar_reactive/se2_waypoint`共18条闭环，0运行失败、0碰撞且未访问`test_locked`。三个模型成功率全为0；v2_017虽把polar平均跟随率由v2_016的`0.1275`提高到`0.1531`，却把se2-waypoint跟随率从`0.0916`降到`0.0350`，三条se2 rollout机器人总路程由`2.5997 m`降为`0 m`。根因是三条se2中visibility没有一帧超过冻结阈值0.005，安全门全程停车；v2_017闭环gate失败并拒绝晋升，index1300不得再用于调阈值或训练。
- World-Action 路线：v1采用action-conditioned future latent + future target `(x,y,visibility)`的选项B，并保留inverse dynamics作训练期辅助；DA3完整pose/depth只允许离线伪标或审计，不进入部署主链。普通无任务 ego 视频只训练几何与状态转移辅助能力，不直接提供 policy trajectory 监督。
- 当前主要风险：UWB 冷启动后的首次视觉绑定；首帧/首次绑定目标身份如何长期保留；UWB 时间延迟、坐标转换与误差标定；pseudo ego-motion 的尺度及可执行性；专家状态与模型实际访问状态存在分布偏移；TpT 视频时钟与 GT/ODOM 时钟语义尚未统一；SAGE3D侧车是pose+depth几何包络而非像素级分割，冻结样本仍有2.45% detector conflict，后续模型结果必须继续按场景和遮挡切片检查。

## 0.1 协作者接手入口：当前该做什么

本节是协作者开始工作的入口。先核对事实和数据，再实现训练；不要把流水线dry-run误认为模型已经可以训练。

### 2026-09-11端到端主线纠偏（协作者必须先读）

- **最终研究交付是严格端到端人物跟随，不是“冻结OSNet坐标 + MLP”。** 推理输入固定为一次初始化`RGB+bbox`、后续RGB历史、可缺失UWB及其mask/质量/age；输出为当前底盘系未来8点`[B,8,2]` waypoint及安全停车判断。
- **现有Phase 2是模块化baseline。** `Phase2WaypointPolicy`只接收`visual_xy/confidence/valid`、UWB和mode，`omtrackvla/training/phase2.py`还明确要求`--init-checkpoint`必须是Phase 2 checkpoint，因此正式Phase 1 v3当前没有进入Phase 2前向或梯度图。
- **Phase 1保留，但只有Architecture-v1同构预训练才能主张迁移。** 现有4,096-step ResNet18+GRU Phase 1 v3与DA3-v1主干不同，只保留为历史baseline/teacher候选，不能强行加载并宣称覆盖率。新同构Phase 1须共享DA3 projector、Target/World表示和SE(2)/World-Action heads；Phase 2加载时报告逐键覆盖率，并验证waypoint loss对时序融合及DA3后层adapter/LoRA产生非零梯度。收益必须用`Architecture-v1 Phase1 init vs direct Phase2`消融验证，不能预设有效。
- **OSNet/KPR结果仍有效但角色改变。** OSNet是训练期身份teacher、工程baseline和消融优先选项；不得把冻结OSNet/cache当最终论文主模型的必需推理链。SAGE3D depth/逐帧bbox/target pose只允许作训练辅助标签或审计，不得进入最终模型推理输入。
- **立即执行顺序：** NEXT-025、NEXT-026及ABL-08/09已完成；Architecture-v2 Phase 3多场景小预算训练、纯安全头校准和index1300闭环gate也已完成。v2_017因se2-waypoint全程被visibility安全门停车而拒绝晋升。下一步不得在已消费的index1300上调阈值；须扩大跨场景、连续时序的模型访问状态数据，重新冻结train/calibration与全新闭环holdout，并在保持干净Phase 2混合和waypoint主链梯度的条件下做下一轮有界pilot。无真实UWB时当前安全策略仍禁止平移盲搜；冻结OSNet cache继续只作ABL-09模块化baseline。

### 工作目录与安全边界

- GitHub开发分支是`wam`。在H100上使用干净共享目录`/data/nfs/share/wam_tracking/OmTrackVLA`；旧目录`/data/nfs/share/OmTrackVLA`含有大量未提交实验代码和数据，暂时只读，不执行`git clean`、`git reset --hard`或`git add -A`。
- 开始前运行`git status --short`并记录`git rev-parse HEAD`。正式流水线默认拒绝dirty worktree。
- Git只保存代码、配置、小型manifest、文档，以及已审计的`example_datasets/samples`微型真实样例。除此之外的完整数据、checkpoint、视频及运行日志不得提交；它们通过`OMTRACKVLA_DATA_ROOT`和`outputs/training/`管理。
- 本文件中的“已确认”是任务约束；“提案”和“待开始”可以通过实验修改，但修改时必须记录证据、失败原因和关联commit。

### 已核对的仓库与数据事实（2026-09-07，g0014/NFS复核）

| 路径/对象 | 实际状态 | 可直接用于什么 | 不能假设什么 |
|---|---|---|---|
| `scripts/run_pipeline_8xh100.sh` | 已实现编排、preflight、8进程启动、断点续跑、eval/render/gate和产物检查；Phase 1/2已接入真实模块 | Phase 1正式入口、Phase 2 open-loop入口及全Phase dry-run | Phase 2尚无closed-loop退出gate；不代表Phase 3已经实现 |
| `configs/pipeline/h100_8gpu.env` | 已实现，实际配置格式是Bash env，不是YAML | 集中定义模块、Phase配置、GPU和产物契约 | 不包含具体模型或数据schema |
| `omtrackvla/` | 已包含原模块化oracle/感知代码、Phase 1闭环和Phase 2冻结感知cache/四模式waypoint模块 | Phase 1双流baseline；Phase 2 open-loop train/eval/render/gate | Phase 2 smoke不代表正式规模或closed-loop效果，Phase 3仍未实现 |
| `configs/phases/`、`configs/benchmarks/`、`configs/gates/` | Phase 1/2各有3个配置并已接通；Phase 3的3个配置尚缺 | 当前可运行`--phase 1`及完成cache后的`--phase 2` | Phase 2 gate明确为`open_loop_development`，不得包装为完整阶段退出gate |
| `data/datasets/track/{STT,DT,AT}/{train,val}/*.json.gz` | Git跟踪的6个episode文件；每类train 7,257、val 1,405，共25,986 episodes | 启动Habitat仿真、生成观测/rollout、建立固定episode划分 | 文件本身不含已录制RGB历史、逐帧bbox、UWB日志或expert future-waypoint张量 |
| episode字段 | 已看到`scene_id`、机器人初始pose、`main_human_semantic_id`、humanoid名称与人物waypoint字段；还含`instruction` | 识别目标人物、构造仿真真值和生成样本 | `instruction`不得进入本项目模型输入；人物waypoint不自动等于机器人expert trajectory |
| `data/scene_datasets` | 本机软链接到`/data/nas_ray/home/zeying.gong/datasets/scene_datasets`，目标存在；该链接不进Git | Habitat场景资产 | 新clone/H100会自动拥有同一路径 |
| `data/humanoids`、`data/versioned_data` | 本机存在且被Git忽略，约573 MB和57 MB；主humanoid目录有100个人物资产目录 | Habitat人物资产 | 它们是训练样本或会随Git下载 |
| `/h100-2/vln_n1/traj_data` | 完整的 InternData-N1 展开目录；12个group、196,536 episodes；3,730个scene目录中3,725个含正式episode，另5个仅含未索引depth残留 | Phase 1导航几何、pose/action与future-state预训练 | 不是人物跟随数据；自然语言task不得进入模型；5个无metadata/parquet的残留目录不得入manifest；旧报告的85,124 episodes是漏扫结果 |
| `/data/nfs/share/OmTrackVLA/data/sage3d_extracted` | 约160 GiB（`du -s -B1`为171,095,801,856字节）；912 runs、7,110个索引episode、其中7,105个canonical accepted；2,132,276 steps；另有准入的2,131,500步只读修复侧车 | RGB/depth、robot/target pose和8点ego waypoint；Phase 1身份标签只能从`results/sage3d_bbox_sidecar_v1`读取 | 无真实UWB；156,420个尾部step无未来waypoint；源`derived.json`的bbox/visible仍因水平符号、画面相交和遮挡错误而禁止直接训练；见`docs/sage3d_bbox_audit.md`和`docs/sage3d_bbox_sidecar.md` |
| `/data/nfs/share/OmTrackVLA/data/tpt_bench_clean_v2` | 约17 GiB（`du -s -B1`为17,604,104,192字节）；47序列、141,326帧；parquet与RGB逐帧对应 | Phase 1B身份保持、遮挡/干扰人与可见性监督 | 无expert waypoint/UWB；`vid_pts_ms`与GT/ODOM时钟的整段时长约差4.48倍，冻结horizon前必须核实 |
| `example_datasets/samples` | 三套正式数据各16个连续真实帧；5,854,999字节、99文件；含48 RGB、32个16-bit depth、2个16行Parquet及3张预览 | 协作者在GitHub检查真实外观、目录结构、深度编码和标签形状 | 不是训练/评测划分；预览黄框和逐帧标签不得作为模型输入；不授予上游数据额外权利 |
| 仓库Git跟踪的`data/` | 31个文件，约19.7 MB；WAM提交没有新增大数据 | 小型episode元数据和Spot机器人资产 | `??`本地数据已经上传GitHub |

### 协作者按顺序执行的工作包

1. **WP-0：数据审计，已完成。** 正式外部数据范围固定为`/h100-2/vln_n1/traj_data`（InternData-N1）、`/data/nfs/share/OmTrackVLA/data/sage3d_extracted`和`/data/nfs/share/OmTrackVLA/data/tpt_bench_clean_v2`。产物为`docs/data_inventory.md`、`configs/data_inventory.json`和`scripts/audit_data_inventory.py`；审计只读、检查全量元数据/路径并按group/mode-camera/sequence分层解码媒体，不生成target crop。另有`example_datasets/samples`保存每套16帧的可公开浏览微型真实样例，仅用于理解数据。
2. **WP-1：冻结数据契约，已完成。** `docs/data_contract.md`和`configs/data_contract.json`已冻结一次性初始化事件、RGB历史、UWB与路由元数据、8点底盘系expert轨迹、辅助GT及`null + valid/mask`缺失值；`scripts/validate_data_contract.py`只读强制执行输入/标签隔离、四种条件模式、坐标/时钟不变量及源准入gate。逐帧bbox只能在label域，不能出现在model input域。
3. **WP-2：打通Phase 1最小闭环，已完成。** InternData-N1提供pose ego几何流，TpT及通过完整性/质量准入的SAGE3D修复侧车提供identity流；源SAGE3D bbox/visible继续禁用。正式8×H100 baseline经过v1、v2两次保留的失败迭代后，`phase1_baseline_v3_5103f88`完成4,096步训练并通过B1-ID、B1-GEO、B1-PROBE全部12项gate，固定`viz_val`视频包含可见与absent样本。DA3-SMALL分相机尺度和置信度准入保持冻结，唯一一次`test_locked`不得重跑或查看。后续转入WP-3 Phase 2。
4. **WP-3：打通Phase 2端到端基本跟随，进行中。** Phase 2A已完成EVT全量人物识别受控对照，为身份teacher/baseline选出OSNet，但该结论不等于最终端到端模型已经具备目标感知。Phase 2B主线改为共享模型的联合监督微调：一次初始化`RGB+bbox`建立identity token，后续RGB历史经共享视觉与时序身份模块建模，可缺失UWB经独立encoder融合，统一decoder输出未来8点waypoint与安全停车判断；waypoint、identity/bbox/visibility、World-Action和stop辅助损失共同训练，主轨迹损失须进入视觉/时序梯度图。现有冻结感知cache/train/eval/render smoke完整保留并改称`Frozen Frontend Baseline`，不得再包装成最终方法。最后必须补无GT point的Habitat closed-loop episode与退出gate。没有真实UWB日志时，只能使用明确标记的`simulated_uwb`，不得声称已覆盖真实UWB误差。
5. **WP-4：打通Phase 3恢复。** 先完成3A离线受控扰动，再验证仿真可从模型访问状态查询expert后实现3B DAgger。始终混入Phase 2干净专家数据，并用相同scene/seed对比Phase 2与Phase 3。
6. **WP-5：每个工作包都回填。** 在第9～11节追加实验、失败和修改记录；写明命令、commit、数据manifest、checkpoint、指标和产物路径。失败也要记录，不覆盖历史，不只汇报总loss。

### 当前明确阻塞项

- Phase 1与旧Phase 2冻结前端baseline均已接通4个Python入口和3个Phase/benchmark/gate配置；严格端到端Phase 2的统一模型、训练期辅助监督、正式训练/eval/render及ABL-08已完成，ABL-09恢复中。NEXT-027的执行接口和首个50步固定episode已完成，但闭环跟随/重获gate失败。旧ResNet18权重不再要求迁移。Phase 3仍缺arbitrary-state expert relabel验证、训练配置及实现，因此当前`--phase all`失败是预期行为。
- 四种目标条件模式、canonical坐标/时间接口和最小样本schema已由WP-1冻结，但这不等于所有源字段均可训练：SAGE3D源bbox/visible仍永久阻断，身份loss和评测只能读取完整、hash匹配且已准入的修复侧车；TpT没有expert waypoint且物理时钟未解决；InternData-N1不是人物跟随数据。严格数据划分仍由NEXT-015冻结。
- 当前没有真实UWB日志、设备标定或误差统计被纳入仓库；在获得这些数据前，只能验证接口和仿真UWB，不能完成产品级UWB鲁棒性结论。

### 当前可视化与判读边界

| 可视化 | 位置 | 现在可以检查什么 | 不能据此判断什么 |
|---|---|---|---|
| InternData-N1预览 | `example_datasets/samples/previews/intern_data_n1.jpg` | egocentric RGB连续性、视角与场景外观 | 无人物目标，不能检查身份跟踪或跟随效果 |
| SAGE3D预览 | `example_datasets/samples/previews/sage3d_extracted.jpg` | 仿真人物外观、连续帧和GT投影bbox是否合理 | 黄框是离线GT，不是模型预测，也不是后续模型输入 |
| SAGE3D bbox审计 | `docs/sage3d_bbox_audit.md`及H100的`results/wp2_sage_bbox_audit_v2/` | 黄框投影错误、水平镜像反事实、独立person detector框及mode/camera/时间分层统计 | detector是审计证据，不自动等于目标GT；当前源bbox/visible未准入训练 |
| SAGE3D修复侧车审计 | `docs/sage3d_bbox_sidecar.md`及H100的`results/sage3d_bbox_sidecar_v1_audit/` | 黄框修复投影、绿框最佳独立person detection、青框其他人物检测，以及深度support/near证据 | 这是数据标签准入检查，不是模型预测；2.45% detector conflict需保留为标签噪声风险，detector不能自动替换目标身份 |
| TpT clean v2预览 | `example_datasets/samples/previews/tpt_bench_clean_v2.jpg` | 真实机器人鱼眼画面、人物尺度变化和GT bbox | 黄框是离线GT；不能证明时钟已对齐或模型已学会重识别 |
| 冻结检测/ReID双运行点跟踪 | H100 `outputs/evaluation/pretrained_identity_v4_resnet50_fusion_dualop_viz/visual_review/`；本地`Desktop/OmTrackVLA_visual_review/pretrained_identity_v4_resnet50_fusion_dualop_viz/visual_review/` | `correct`、`wrong_target`、`visible_miss`、`absent_false_positive`、`absent_correct`各16张；绿框为PRED、黄框为GT | 仅为TpT两个固定`viz_val`序列的开发验收，不是locked test，也不代表完整Phase 2机器人waypoint闭环 |
| Phase 2冻结前端smoke | H100 `outputs/training/phase2_frozen_frontend_smoke/phase_2/visualizations/phase2_frames/`；本地`Desktop/OmTrackVLA_visual_review/phase2_frozen_frontend_smoke/visualizations/phase2_frames/` | 四模式各4张PNG；绿框/轨迹为启用的PRED，灰框为已屏蔽的冻结前端诊断，黄框/轨迹为GT，青圈为模拟UWB | 仅1个`viz_val` episode和20-step开发模型；无closed-loop，不能作为正式Phase 2效果 |
| Phase 1 smoke视频 | H100共享工作副本的`results/wp2_phase1_memory_smoke/phase_1/visualizations/phase1_viz.mp4` | render链路、`PRED/GT`标识、身份/几何面板和视频编码是否工作 | 仅2-step开发smoke，正式gate预期失败，不是可报告的模型效果 |
| Phase 1正式视频 | H100的`outputs/training/phase1_baseline_v3_5103f88/phase_1/visualizations/phase1_viz.mp4`；本地复核副本`Desktop/OmTrackVLA_visual_review/phase1_baseline_v3_5103f88_phase1_viz.mp4` | 前32帧身份预测、后32帧几何预测；身份段固定包含8个absent案例，并明确显示置信度、0.999阈值、`PRED visible`与`GT visible` | 这是通过gate的`viz_val`人工复核产物，不是locked test；绿框是模型PRED，黄框是GT，预测absent时不画无意义绿框 |
| Phase 1 smoke指标 | 同目录的`metrics.json`、`report.md`、`gate.json` | 指标字段、报告和失败原因是否完整 | 不能作为baseline数值或Phase 1验收结果 |
| DA3单clip probe | `docs/da3_geometry_probe.md`及H100的`results/wp2_da3_probe/report.json` | 坐标转换、输出shape和单clip误差 | 同clip校准与评测，尚不能证明跨场景尺度/置信度准入 |
| DA3多场景开发复核 | `docs/da3_multiscene_admission.md`及H100的`results/wp2_da3_multiscene_v2_viz/` | `viz_val`的depth/confidence、预测/GT鸟瞰轨迹、置信度—误差关系和时域误差曲线 | 只用于开发复核；不能查看或用来调参`test_locked`，也不能把DA3 pose称为专家控制动作 |

DA3的`test_locked`已唯一运行一次且未生成/查看图片，不得重跑或人工检查。正式8卡baseline必须另行产出固定`viz_val`视频，不能复用上述smoke或DA3审计图片充当验收。

## 1. 已确认决策

| ID | 日期 | 决策 | 理由/依据 | 状态 |
|---|---|---|---|---|
| DEC-001 | 2026-09-04 | 任务输出为机器人未来局部 waypoint trajectory | 当前任务定义 | 已确认 |
| DEC-002 | 2026-09-04 | 首帧人物 bbox 用于指定目标身份 | 当前任务定义 | 已确认 |
| DEC-003 | 2026-09-04 | 目标不可见时允许使用有噪声、漂移或延迟的相对目标点 | 当前任务定义 | 已确认 |
| DEC-004 | 2026-09-04 | 后续主要感知输入为 egocentric RGB 历史 | 当前任务定义 | 已确认 |
| DEC-005 | 2026-09-04 | 不依赖自然语言描述、VQA 或语义 CoT | 当前任务定义 | 已确认 |
| DEC-006 | 2026-09-04 | 当前阶段不限定 backbone | 避免过早绑定模型结构 | 已确认 |
| DEC-007 | 2026-09-04 | 当前阶段不预先确定 diffusion、RVQ、直接回归、概率策略或 RL 等动作头方案 | 应先确认训练阶段和数据闭环 | 已确认 |
| DEC-008 | 2026-09-04 | 视觉 bbox 与 UWB 相对坐标是可同时存在、可独立缺失的互补模态 | 视觉主要提供目标身份，UWB主要提供带噪空间位置 | 已确认 |
| DEC-009 | 2026-09-04 | UWB输入必须保留有效性、置信度、时间戳/age和坐标系 | 实际UWB存在噪声、漂移、延迟与丢失 | 已确认 |
| DEC-010 | 2026-09-04 | 不采用 WALA 式不可解释 latent action 作为主路线 | DA3类几何模型可从视频恢复显式相机自运动轨迹 | 已确认 |
| DEC-011 | 2026-09-04 | 将几何模型恢复的轨迹称为 pseudo ego-motion/realized motion，而不直接等同于底层控制指令 | 实际相机运动还受到动力学、打滑、碰撞和控制器影响 | 已确认 |
| DEC-012 | 2026-09-04 | World-Action Modeling优先采用训练期未来状态与状态转移辅助监督 | FutureNav为该路线提供实验证据；迁移到连续pseudo motion仍需本项目验证 | 已确认 |
| DEC-013 | 2026-09-07 | 视觉bbox只随初始化图像输入一次，后续不接收外部逐帧bbox；若离线训练集带后续帧bbox，可选作GT辅助监督和评测，不要求部署时生成 | 保持端到端RGB历史建模边界，同时允许直接检查身份漂移 | 已确认；细化DEC-002 |
| DEC-014 | 2026-09-07 | 支持无视觉初始化的UWB-only冷启动，并在目标首次进入视野后自动建立视觉身份绑定 | 覆盖真实启动与远距离接近场景 | 已确认 |
| DEC-015 | 2026-09-07 | UWB主路径按“目标佩戴指定tag，定位系统输出anchor/world坐标中的米制位置”理解；适配层对齐到RGB时刻并转换为机器人底盘系相对点 `(x_forward, y_left)`，保留tag ID、原始时间戳、age、质量/协方差和LOS/NLOS | 符合主流定位产品接口，同时把设备差异隔离在模型外 | 已确认 |
| DEC-016 | 2026-09-07 | 当目标无法被视觉定位且UWB同时失效时，策略安全减速至停止并等待恢复 | 此时继续主动跟踪缺少可靠目标条件，产品级默认应安全停车 | 已确认 |
| DEC-017 | 2026-09-07 | 普通无任务ego视频不用于直接监督policy future trajectory | pseudo ego-motion描述“实际发生了什么”，无目标/意图时不能唯一表示“应该做什么” | 已确认 |
| DEC-018 | 2026-09-07 | DA3输出先按w2c保存并显式转换为c2w，再结合camera-to-base外参计算base-frame相对运动，最后投影为SE(2) | 避免矩阵约定、坐标轴和相机/底盘运动混淆 | 已确认 |
| DEC-019 | 2026-09-07 | FutureNav式辅助目标第一版采用单步next-state target，辅助head默认仅在训练时使用 | 最接近已有论文证据，便于先验证收益与成本 | 已确认 |
| DEC-020 | 2026-09-07 | Stage 0采用几何/动态与目标身份两类数据流；Stage 2内部拆为2A离线扰动和2B闭环DAgger | 保留三阶段总框架，同时保证数据来源和收益可独立消融 | 已确认 |
| DEC-021 | 2026-09-07 | 当前阶段暂不以许可证筛选候选模型，但在任何发布或商用决策前重新审查 | 当前优先验证技术可行性 | 已确认；暂缓项 |
| DEC-022 | 2026-09-07 | 整个RGB传感器流失效时默认安全停车，即便UWB仍有效；除非另有独立且已验证的避障安全栈 | UWB只提供目标位置，不能替代环境感知与避障 | 已确认 |
| DEC-023 | 2026-09-07 | 对外统一使用Phase 1/2/3；分别对应原Stage 0预训练、Stage 1基本跟随、Stage 2鲁棒恢复 | 避免“三阶段”却从Stage 0开始造成沟通歧义 | 已确认 |
| DEC-024 | 2026-09-07 | 每个Phase结束必须运行固定benchmark、生成指标报告和可视化，并通过冻结的gate后才能进入下一Phase | 训练loss不能证明身份保持、闭环跟随或恢复能力 | 已确认 |
| DEC-025 | 2026-09-07 | 数据划分至少区分train、val、viz_val和locked test；频繁人工查看只使用viz_val | 防止反复查看测试案例导致人工过拟合 | 已确认 |
| DEC-026 | 2026-09-07 | 首帧后的可视化bbox/heatmap只能是模型诊断输出或GT叠加，不得作为模型输入 | 保持“一次视觉初始化”的推理边界，同时让身份漂移可检查 | 已确认 |
| DEC-027 | 2026-09-07 | GitHub版本提供8×H100一键流水线，逐Phase保存配置、数据manifest、代码版本、checkpoint、指标、失败案例和视频 | 让协作者可复现训练并快速定位阶段性问题 | 启动器及Phase 1入口已实现；Phase 2/3待实现 |
| DEC-028 | 2026-09-07 | 以当前模块化精简后的OmTrackVLA状态为WAM开发基线，并使用独立`wam`分支 | 保留旧分支历史，同时让新方法从已验证的干净快照开始 | 已确认；已推送`origin/wam` |
| DEC-029 | 2026-09-07 | 在GitHub发布三套正式数据各16个连续真实帧的微型样例，并保留目录结构、必要元数据/Parquet切片、RGB/depth和可直接浏览的预览 | 让协作者无需访问完整数据即可认识真实结构与外观；用户明确确认该用途 | 已确认；禁止将样例视为训练/评测划分，禁止上传视频、点云、crop/cache或绝对symlink |
| DEC-030 | 2026-09-07 | WP-1统一使用anchor时刻底盘系（x前、y左、z上，米/弧度）、归一化xyxy bbox、含anchor的8点绝对局部XY轨迹和显式时间偏移；序列只在history index 0消费一次外部bbox，UWB tag ID仅作路由；所有缺失值用`null + valid/mask` | 消除不同源的坐标、时间、缺失值和输入/标签边界歧义，同时阻止未确认source semantics静默进入训练 | 已确认；SAGE3D只允许准入侧车，TpT physical clock与real UWB继续由机器可读gate阻断 |
| DEC-031 | 2026-09-08 | Phase 1固定按Intern `group/scene`、SAGE3D `run`、TpT `sequence`隔离划分，种子为20260907；训练adapter只能内存裁剪初始化目标，不得向源数据写crop/cache | 防止scene/人物/episode泄漏，并修复旧外部loader会在源目录生成`_target_crops/_target_refs`的问题 | 已实现；manifest只枚举split unit和必要元数据，不扫描全量媒体 |
| DEC-032 | 2026-09-08 | Phase 1先以无下载权重的共享ResNet-18建立可执行双流baseline；DA3-SMALL作为Apache-2.0几何teacher候选，实际输出验证前不得生成pseudo label | 先验证数据、分布式训练和评测闭环，同时不把尚未安装/验证的DA3接口伪装成已完成 | baseline smoke、DA3接口与多场景pseudo-motion准入及正式8×H100 v3均已完成；teacher消融仍单独保留，不影响当前baseline准入 |
| DEC-033 | 2026-09-08 | DA3使用官方commit `3d835ec1...`和DA3-SMALL revision `e08cab65...`的独立源码、Python target目录和校验权重；pose必须经过OpenCV/Habitat相机轴桥接、Intern底盘外参、canonical基变换和显式metric scale | 不污染现有训练环境；真实probe证明直接混用OpenCV与Habitat外参会产生近90度平移和yaw符号错误 | 已确认；单clip坐标修正及后续多场景分相机尺度/置信度准入均已通过，具体冻结规则见DEC-036 |
| DEC-034 | 2026-09-08 | SAGE3D源`bbox/visible`在修复侧车通过独立person detector、深度和人工拼图审计前，不得进入Phase 1身份loss或指标；修复与伪标注不得覆盖只读源，detector/ReID仅用于离线监督或审计，不构成后续推理bbox输入 | 384帧分层审计确认抽取器水平轴符号错误、画面外零面积框和未验证遮挡可见性；直接训练会把背景当目标并污染身份记忆 | 已确认；TpT身份流和Intern几何流不受影响，SAGE3D修复转NEXT-021 |
| DEC-035 | 2026-09-08 | SAGE3D身份监督仅接受`selection=full`、7,105个episode完整、无生成失败、源metadata和episode SHA-256匹配且独立审计通过的`v1`侧车；person detector/DINO/ReID只验证而不替换目标身份 | 防止旧源框回退、部分侧车误用或多人场景generic detection串换身份，同时允许经证据验证的修复标签进入Phase 1 | 已确认；`sage3d-bbox-depth-v1`已准入并加入Phase 1 `identity_datasets` |
| DEC-036 | 2026-09-08 | DA3 pseudo ego-motion仅接受policy `da3-small-intern-multiscene-v1`：在`val`冻结D435i/ZED分层metric scale和median depth confidence阈值，仅用`viz_val`开发复核，并只运行一次无可视化的`test_locked`正式准入；任何参数漂移须fail closed | 单一全局尺度在跨相机场景上以24% admitted bad rate超过20%上限；分相机尺度在开发集通过且唯一locked结果以56.25% coverage、16.67% bad rate通过14项gate | 已确认；原始DA3平移不得直接作metric label，confidence不是校准概率，pseudo motion不等于专家控制动作 |
| DEC-037 | 2026-09-08 | Phase 1 visibility使用仅在train split校准并写入checkpoint模型配置的固定运行阈值0.999；val只执行一次固定评测，不搜索阈值 | 加权BCE使输出概率高度饱和，固定0.5并不对应部署运行点；train上0.999得到FPR 7.41%、FNR 10.61%，且未消费locked test | 已确认；正式val FPR 16.42%、accuracy 86.40%，通过冻结gate |
| DEC-038 | 2026-09-08 | B1-PROBE对预训练与随机编码器分别使用各自train feature的均值/标准差标准化，再以相同ridge配置评测val | 未标准化的固定绝对正则会把特征幅值/条件数误当表示质量；v2原始probe为-127.20%，标准化后为提升69.46% | 已确认；只使用train统计，不读取val统计，不降低probe gate |
| DEC-039 | 2026-09-09 | 目标人物感知复用冻结COCO person detector与冻结MSMT17 OSNet，不从零训练检测/ReID；只训练候选融合，并以不可覆盖首帧身份anchor、谨慎正样本图库和干扰负样本图库维持身份 | 现成模型已提供通用人体定位和重识别能力；从零训练会重复造轮子且增加身份漂移风险 | 已确认；权重大文件不进Git，融合头及特征契约可版本化 |
| DEC-040 | 2026-09-09 | 候选融合使用train-calibration冻结的双运行点：连续跟踪偏召回，丢失后的全局重获偏精度；旧单阈值权重继续向后兼容 | 单一严格阈值虽降低误跟，却会因一次拒绝进入更难的全局搜索并持续漏检；全局误重获的代价又高于连续跟踪短时误差 | 已确认；tracking为score 0.92/margin 0.02，reacquisition为0.95/0.06，`viz_val`只评测不搜索阈值，禁止使用`test_locked` |
| DEC-041 | 2026-09-09 | SAGE3D Phase 2 expert waypoint不得直接读取源`waypoints_ego`，而须从robot pose按当前base帧重算控制步`0,+3,…,+21`；控制频率冻结为30 Hz，target pose只能生成显式无噪声`simulated_uwb` | 源extractor实际保存`+1,+4,…,+22`，不满足WP-1第0点为`[0,0]`；真实记录目标移动速度与30 Hz命令速度比例提供独立时钟证据 | 已确认；spec `sage3d-policy-se2-30hz-v1`，256个非locked episode的11项gate全通过，真实UWB继续阻断 |
| DEC-042 | 2026-09-09 | Phase 2将冻结Faster R-CNN/OSNet/融合跟踪器作为端到端系统内的固定感知前端，离线cache其身份关联与depth相对位置，只训练waypoint decoder；cache必须绑定前端、sidecar、policy admission、source index和split manifest哈希，非视觉模式须同时屏蔽视觉XY/confidence/valid | 复用现成通用检测/ReID避免从零训练；严格哈希和partial标记防止旧cache、开发子集或标签漂移静默进入正式训练；smoke发现只屏蔽XY/valid仍会泄漏confidence | 已确认；safe-stop由模型接口硬归零，当前B2 gate仅为open-loop开发gate，closed-loop退出条件仍未完成 |
| DEC-043 | 2026-09-09 | Phase 2 waypoint训练前新增2A感知适配gate：冻结Faster R-CNN与OSNet基础权重，训练候选融合/时序身份保持/重获；训练和阈值只用TpT `train`，模型选择用`val`，既有`viz_val`仅作确定性开发对照，禁止访问`test_locked`。2A须相对冻结基线E2E至少提升2个百分点，同时output precision≥88%、absent FPR≤5%、wrong-target帧≤60且reappearance≥33.33%，否则不得生成2B正式cache | 同口径5,928帧对照证明冻结前端显著优于Phase 1身份头，但现有冻结前端仍只有46.30% E2E、58.47%候选选择率和33.33%重获率，身份关联仍是进入waypoint前的主要瓶颈 | 已确认；旧`v1` cache绑定融合SHA-256 `03a78884...`，只保留作负基线/消融，不得用于正式2B |
| DEC-044 | 2026-09-09 | Phase 2A时序融合适配必须从既有双运行点fusion精确展开初始化，新特征初始权重为零；只训练当前冻结前端产生的on-policy候选。epoch 0也参与`val`候选排序选模，故离线排序若无改善则保留原模型，不允许随机新头覆盖强基线 | `phase2a_temporal_fusion_v1`虽降低误跟，但从随机头训练且混入缺少新增时序字段的旧记录，`val`候选排序precision仅49.35%，最终E2E从42.80%降到24.99% | 已确认；阈值仍只由train calibration拟合，`val`只选epoch，`viz_val`只在val safeguard通过后运行 |
| DEC-045 | 2026-09-09 | Phase 2A v3冻结旧fusion的legacy列、hidden bias和输出层，只允许新增时序列产生梯度；运行点继承旧模型已由train-only冻结的tracking 0.92/0.02与reacquisition 0.95/0.06，不在新on-policy rollout上做反事实阈值重拟合 | v2正确选择epoch 0、离线排序与旧fusion一致，但固定策略记录上的反事实校准选出tracking margin 0.30并把在线val E2E从42.80%降至13.47%，证明权重防退化不足以约束阈值防退化 | 已确认；val仍只用于模型版本准入，未搜索阈值；若新增列无排序改善则epoch 0保持全系统等价并应在val以“不提升”被拒绝 |
| DEC-046 | 2026-09-09 | 所有包含时序状态特征的Phase 2A正式train rollout必须使用与val/viz/部署相同的frame stride 1；stride 4记录保留为失败消融，不得与正式时序适配混用 | v3在stride 4 train上获得val提升但viz退化；`missed_steps`、`confirmed_track_steps`和最近相似度窗口均按调用次数演化，stride变化会改变其物理时间语义 | 已确认；v4重新生成独立on-policy root，禁止复用v1～v3的stride 4 train记录，仍不得访问`test_locked` |
| DEC-047 | 2026-09-09 | Phase 2A离线候选排序改善但在线val退化时，下一迭代必须做train-only on-policy dataset aggregation：用失败候选策略重新rollout train并与冻结基线状态合并；不得通过反复查看viz或放宽gate解决 | v4的val离线top候选增加108帧，但在线成功帧减少300，说明候选动作改变tracker状态后出现covariate shift，固定基线记录上的监督无法覆盖 | 已确认；v5只增加一次v4-policy train rollout，模型选择仍用val，viz只在val通过后运行，`test_locked`禁止访问 |
| DEC-048 | 2026-09-09 | v5若仍不能通过固定viz gate，则停止同一逐帧BCE/ranking融合家族，不得继续用viz反馈调阈值或追加同类迭代；Phase 2B继续阻塞，下一方案必须引入显式序列级身份保持/重获训练目标，或以独立val协议比较更强冻结tracker | v3与v5均在val改善而viz退化，v5加入train-only on-policy状态后仍出现E2E -6.98pp、wrong-target 137和重获21.84%，已证明当前离线目标缺乏跨序列稳定性 | 已确认；冻结基线继续作为可用前端和比较下界，但在DEC-043的+2pp gate未满足前不得声称2A完成 |
| DEC-049 | 2026-09-10 | KPR/SOLIDER只使用官方共同可见部位归一化欧氏距离；所有动态图库写入、连续tracklet floor、短时/全局重获及漂移终止运行点必须仅由37条train序列校准。动态图库先满足高精度写入约束，再用多帧一致性扩展外观；不得直接复用OSNet的0.82/0.72 memory阈值或用5条val调参 | KPR完整val虽较严格静态替换提高约10～12倍，但两个tracklet方案均未通过precision safeguard，且`memory_updates=0`；train KPR正样本anchor中位数仅0.4643，证明OSNet尺度阈值使“adaptive”配置实际退化为静态anchor | 已确认；先以现有train records拟合KPR专属更新/漂移策略并记录覆盖率和精度，再运行完整val；`viz_val/test_locked`继续封锁 |
| DEC-050 | 2026-09-10 | Phase 2A行人识别的实际主评测切换为EVT-Bench；TpT保留为真实域、遮挡和长序列压力测试，但不得再作为快速迭代的唯一准入结论。快速EVT对照固定相同episode，以GT point/轻量控制隔离导航影响，并报告目标precision、recall、IoU及图库写入污染 | 用户确认实际部署评测在EVT-Bench；首个EVT AT val episode已显示KPR precision 97.30%、recall 59.02%、IoU 0.918，与TpT低precision结论显著不同 | 已确认；先完成固定EVT小样本KPR/OSNet同协议比较，再从EVT train冻结正式运行点；TpT结果作为跨域风险保留，仍不访问`test_locked` |
| DEC-051 | 2026-09-10 | Phase 2B冻结感知前端选用Faster R-CNN ResNet50 + OSNet + TpT train-only双运行点fusion；KPR保留为高精度/遮挡消融，不再阻塞waypoint主线。EVT快速对照用于工程选型而非宣称统计充分的完整benchmark | OSNet在完整6个固定EVT episode上以95.31% precision、85.92% recall和90.4% micro-F1通过快速验证；KPR已完成4个episode的recall为52.69%，即使其余两条满分，micro-F1上界仍约88.2%，且长episode两次出现Habitat原生abort | 已确认；NEXT-023快速选型关闭并解除NEXT-022阻塞，后续只用EVT `train`校准新运行点；扩大EVT评测与TpT压力测试作为审计保留，不访问`test_locked` |
| DEC-052 | 2026-09-10 | Phase 2A正式前端选择必须覆盖EVT-Bench全量`val` 4,215 episode/方法：STT/DT/AT各1,405条，使用相同GT point + reactive controller、官方300步上限且无人工截断；OSNet与KPR须各完成4,215条、错误/缺失均为0且逐episode轨迹一致后，才按总体micro-F1排名 | 用户复核指出6条仅占全量val的0.1423%，episode内帧又高度相关，不能支撑benchmark泛化或论文结论；此前KPR理论上界也只约束那6条子集 | 已确认并执行；DEC-051仅保留为快速工程预选记录，不再构成最终准入；Phase 2B cache/训练暂停等待EXP-022正式结果，仍不访问`test_locked` |
| DEC-053 | 2026-09-11 | 最终主方法必须是严格端到端目标人物跟随：推理只接收一次初始化`RGB+bbox`、后续RGB历史、可缺失UWB及有效性/质量/age，直接输出未来8点底盘系waypoint与安全停车判断；不得依赖预计算OSNet cache、后续外部bbox、GT target point、GT target pose或GT depth。训练允许identity/bbox/visibility、几何与World-Action辅助监督，但waypoint主损失必须至少回传至目标条件时序融合和共享视觉表示 | 用户明确要求端到端方法；代码审计确认当前冻结OSNet/depth位置cache + MLP只证明模块闭环，不能代表RGB历史到轨迹的联合学习 | 已确认；旧Phase 2改称`Frozen Frontend Baseline`，NEXT-025～027成为WP-3主线 |
| DEC-054 | 2026-09-11 | Phase 1继续作为端到端共享backbone/identity memory/World-Action预训练概念，但不得仅凭阶段名称声称已经迁移；只有与最终主模型同构的checkpoint才能加载、报告逐键覆盖率并进行非零梯度测试。OSNet优先作为训练期identity teacher和模块化对照，不作为最终模型永久冻结的外部黑盒 | 当前ResNet18 `Phase1WorldIdentityModel`与冻结后的DA3 Architecture v1并不同构，而`Phase2WaypointPolicy`又是独立MLP；旧Phase 1→旧Phase 2实际迁移为0，旧v3只能保留为baseline，不能强行映射为DA3-v1初始化 | 已确认；由DEC-055～057细化。Architecture-v1同构Phase 1 init vs direct Phase 2必须消融，先完成模型/梯度smoke再扩算 |
| DEC-055 | 2026-09-11 | 冻结论文主模型`Architecture v1`：部署视觉backbone为官方commit `3d835ec1...`/revision `e08cab65...`的DA3-SMALL encoder，只使用L11的local/global拼接token；`T=4`、输入`280×504`、patch grid `20×36=720`、policy维度`C=256`。一次初始化RGB+bbox经L11投影与RoIAlign建立单个Target Memory；每步只保留Target Token、World Token、UWB Token和单向量GRU状态`h_t`，直接回归8×2 waypoint与stop | 该结构是“Spatial Foundation Model → Target-conditioned World-Action Policy”的最小充分实现；不再堆叠第二个patch级Temporal Transformer、part tokens、fixed/adaptive多套身份状态或rectified flow | 已确认并冻结为v1；L5+L11、多身份token及更复杂decoder只能作为后续消融/增强，不能静默进入v1 |
| DEC-056 | 2026-09-11 | UWB在v1采用显式几何Gaussian spatial bias：同步/age补偿后的base-frame位置与协方差，经已知base-to-camera外参、处理后intrinsics、tag高度不确定性和标定误差传播到图像，再积分为20×36 patch prior并以log-prior加入target attention；UWB连续特征另编码为`z_uwb`作late fusion。FoV外、相机后方、过大协方差或过高age必须退化为弱/零spatial bias，禁止把投影clamp到边缘 | 已知几何时直接学习`(x,y,...)→720`数据效率和外推性更差；UWB tag只提供空间共位线索，不提供人物外观身份 | 已确认；learned projection为必做消融。UWB-only仅在候选唯一且视觉/personness/时序一致时绑定，歧义时必须保持unbound并等待，不能强制选人 |
| DEC-057 | 2026-09-11 | DA3 camera token不是显式pose；v1禁止将`c_t-c_(t-1)`直接解释为ego-motion。最后两个L11 camera tokens经pair MLP预测显式`[dx,dy,sin(dyaw),cos(dyaw)]`瓶颈，再编码为`z_ego`，并用realized SE(2)监督。World-Action采用选项B：由当前World/Target表示和`a_real`预测stop-gradient future latent，同时预测下一底盘系target `(x,y,visibility)`；Forward/Inverse heads仅训练期存在 | 官方CameraDec对每帧camera token作非线性绝对pose解码，隐空间差分没有SE(3)/SE(2)群运算保证；仅future latent容易退化为普通特征平滑，增加未来目标状态是最小可解释动作后果 | 已确认；部署保留SE(2)预测瓶颈但不保留CameraDec、DualDPT或World-Action辅助heads；必须以no/shuffled-action验证动态分支确实使用`a_real` |
| DEC-058 | 2026-09-12 | NEXT-026 checkpoint按双轨保留：`world_action_long_v1/best.ckpt`作为最低ADE/FDE基线，`waypoint_radial_progress_v1/step_0001000.ckpt`作为“更长且自然”的Pareto候选；不得用单一总路径长度指标选模，必须同时检查ADE/FDE、逐horizon半径和固定图的回退/锯齿 | 用户人工检查发现绿色预测偏短；直接path-length loss将路径比提高到0.875，但固定图通过左右摆动和局部回退增加长度，证明aggregate长度可被错误轨迹利用。沿专家方向的逐horizon进度约束避免该漏洞，正式路径比提高4.50%，误差代价约1.2% | 已确认；当前不继续在同一`val`上无边界扫权重。NEXT-026仍需完成冻结消融矩阵，最终方法结论必须再由NEXT-027无GT point closed-loop决定 |
| DEC-059 | 2026-09-12 | Architecture v1的GRU保留，但正式训练采用受控两段课程：先用single-step路径学好DA3 adapter/Fusion/waypoint表示，再加载该checkpoint、近恒等重置GRU，并为GRU与其余policy/adapter设置独立学习率；所有阶段仍须用waypoint-only probe证明Fusion、GRU、DA3后层adapter梯度非零 | 等预算direct训练显示GRU从第0步起就比single-step差，第二步没有额外累积恶化；单改近恒等初始化的direct 4k仍停在`0.176821/0.298233 m`，说明问题是联合随机优化顺序而非单一初值。两段课程正式1,024×4验证相对single-step将ADE/FDE降低1.30%/1.43%，清零hidden反事实又确认递归本身贡献约0.65%/0.55% | 已确认；这只改变训练初始化和参数组，不给Deployment增加旁路或新模块。4k矩阵一律标为diagnostic，剩余消融必须等预算收敛或明确报告训练日程差异，不能把未收敛平台当模块结论 |
| DEC-060 | 2026-09-13 | NEXT-027闭环决策API必须机器检查为首帧`RGB+bbox`、后续`RGB + optional UWB`；GT target point、后续GT bbox、depth、pose、panoptic和perception cache只能在动作后用于评测，不得进入决策。无UWB且视觉置信度低于0.90时三轴动作严格归零；高置信人物框高度达到画面85%时禁止继续前进，但允许横移/转向脱离近距风险 | open-loop ADE不能发现输入泄漏、过近碰撞或丢失后的危险动作；首轮闭环会追至约0.40 m并碰撞，而低置信误框又不能驱动盲搜 | 已确认并实现；v6低置信非零动作违规0、碰撞0。安全子门通过不等于跟随/重获gate通过，阈值是当前仿真开发策略而非真实产品标定 |
| DEC-061 | 2026-09-13 | 当前固定Habitat episode的NEXT-027开发rollout上限设为50次`env.step()`；帧在rollout中逐张保存为PNG，结束后由隔离FFmpeg子进程编码H.264 MP4。不得为追求60步调用已稳定复现原生abort的第51次，也不得把视频编码失败与仿真进程abort混为一谈 | v3/v4均在第51次`env.step()`发生Habitat原生abort；关闭imageio/OpenCV写视频后仍复现，且PNG spool证明第51帧已保存而只有前50个动作完成。50步v6退出码0、JSON完整、MP4 50帧可独立解码 | 已确认；结果新增`requested_max_steps`与`termination_reason`，但v6启动早于该最后字段上传，既有v6 JSON不追写伪造；后续运行自然生成 |
| DEC-062 | 2026-09-13 | SAGE3D根`index.json`只作目录catalog，冻结感知cache不能把7,110条索引等同于7,105条canonical accepted。缺sidecar时仅当episode同时“无`_ACCEPTED`且`quality.status=rejected`”才允许以结构化记录跳过；任一证据不满足仍fail-closed。正式分片合并必须满足`requested = cached + skipped`且路径无重复 | ABL-09首轮在4,910个cache处遇到前三个被源质量门拒绝的episode而退出；直接忽略所有missing sidecar会掩盖真正的数据损坏 | 已确认并实现；5个源拒绝episode均在train run中，原失败日志已归档，已完成cache保留并断点重挂 |
| DEC-063 | 2026-09-13 | Phase 3模型访问状态样本必须保存一次初始化`RGB+bbox`、与Architecture v1一致的`T=4`连续RGB history、当时真实存在或缺失的UWB/masks，以及标签侧expert 8点；GT pose/navmesh不得进入模型输入。expert除轨迹长度外还必须证明终点比“机器人留在采样anchor不动”更接近同一时刻目标且移动方向朝向目标。来自`val`的首个失败样本只可做数据加载、waypoint-only backward和可视化smoke，正式恢复训练只能扩展到隔离的`train` episode | v1 expert轨迹长度4.2205 m、末点3.5902 m，明显越出SAGE3D监督分布；仅看动作后的绝对人机距离又会因人物仍在快速前进而误判正确跟随为反向。直接训练单个`val`失败状态还会造成评测泄漏 | 已确认并实现最小smoke；v4保存step 25～28原始RGB及初始化图，方向/进度审计和waypoint-only反传通过，但`formal_training_eligible=false`，尚未开始Phase 3优化器训练 |
| DEC-064 | 2026-09-15 | Architecture v2 Phase 3 pilot固定8帧、0.1秒间隔；训练与recovery-val必须按scene和模型访问状态双重隔离，且保留一个完全未见的闭环index。离线选模同时约束recovery/expert waypoint、safe-stop path、false-visible、false-invisible和SAGE3D干净回退 | 单场景少样本可以降低离线loss却无法证明新场景闭环泛化；visibility/stop辅助头也不能代替waypoint主链有效性 | 已确认并用于v2_016～v2_018；train/val场景与状态重叠均为0，index1300在闭环gate前保持未见，未访问`test_locked` |
| DEC-065 | 2026-09-15 | 纯safety-head校准只允许更新`trajectory.visibility_head.*`与`trajectory.stop_head.*`；DA3、Fusion、trajectory decoder和waypoint heads须逐tensor哈希不变，waypoint指标漂移必须为0。即使离线gate通过，也必须经过全新场景闭环后才能晋升 | 若直接继续全模型训练以修复0.005运行阈值，会把安全概率问题与waypoint变化混在一起，无法判断收益来源 | v2_017按此执行，冻结哈希前后均为`edf90f95...`且waypoint漂移为0；离线通过但闭环失败，故未晋升 |
| DEC-066 | 2026-09-15 | v2_017在index1300闭环gate失败后封存为负结果；不得基于该index重新选择visibility阈值、改安全门或训练后再宣称同一index为未见泛化。下一候选必须使用更广的连续时序train/calibration数据，并预先冻结另一个全新闭环holdout | v2_017仅在13个离线点上改善，到了新场景三条se2-waypoint的visibility均全程低于0.005，导致机器人总路程为0；这是明确的校准过拟合而非仿真崩溃 | 已确认；v2_016继续只作未通过最终闭环的父候选，v2_017拒绝部署，正式Phase 3大训练仍未开始 |

## 2. 当前提案与待确认决策

以下内容是当前建议，并非已经锁定的项目决策。

| ID | 提案 | 主要理由 | 确认所需证据 | 状态 |
|---|---|---|---|---|
| PROP-001 | 采用“几何 World-Action 预训练 → 跟随监督 → 鲁棒闭环恢复”三阶段 | 同时利用无动作视频、专家数据和模型访问状态 | 数据盘点、最小闭环实验 | 已采纳；效果待验证 |
| PROP-002 | 跟随监督阶段覆盖视觉+UWB、视觉-only、UWB-only，以及双信号失效时的安全停止 | 对应真实产品中的模态可用性组合 | 同步视觉/UWB数据与缺失模式统计 | 已采纳；见DEC-014～016 |
| PROP-003 | 将目标身份记忆与目标位置条件概念上分开 | 初始化bbox回答“跟谁”，UWB主要回答“在哪里” | 遮挡及干扰人物实验 | 已采纳；效果待验证 |
| PROP-004 | Phase 3优先使用DAgger式expert relabel，三阶段完成后才考虑RL | 可直接监督模型走偏后的访问状态，调试更清晰 | 仿真环境是否支持任意状态查询 expert | 已采纳；可行性待验证 |
| PROP-005 | 原始专家数据持续混入后续阶段 | 防止鲁棒性/恢复训练破坏基本跟随能力 | 灾难性遗忘对照实验 | 已采纳；比例待验证 |

## 3. 统一目标条件接口

外部输入的数据层接口：

```text
target_condition = {
  visual_initialization: {
    initial_rgb,
    initial_bbox,
    valid,
    timestamp
  },
  uwb_target: {
    target_tag_id,
    relative_position_base_xy_m,
    valid,
    covariance_xy_or_quality,
    source_timestamp,
    receive_timestamp,
    age,
    los_nlos_optional
  }
}
```

模型还维护不由外部逐帧喂入的内部状态：

```text
internal_target_state = {
  visual_identity_memory,
  binding_valid,
  binding_confidence,
  binding_source: user_bbox | uwb_auto
}
```

训练约束：

- 初始化 bbox 必须与对应图像绑定；模型后续不接收外部逐帧 bbox。逐帧 target bbox、mask或track ID只能作为辅助监督和评测标签。
- UWB适配层默认接收指定tag在anchor/world坐标中的米制位置，结合时钟同步、robot pose历史和外参，输出当前RGB时刻底盘坐标系下的 `(x_forward, y_left)`；原始测量时间和age必须保留。若设备只输出range/bearing或其他形式，由适配层转换，模型接口不随产品变化。
- `target_tag_id`用于数据配对和选择指定tag流，默认不作为可泛化的数值身份特征输入网络。
- 初始化视觉参考与 UWB 位置分别归一化和编码，再进行置信度感知融合，不把二者伪装成同一种原始坐标。
- 有同步 pose、标定和可见性标签时，可构造同一时刻的 bbox/相对点配对，训练跨模态一致性。
- 训练应覆盖视觉绑定+UWB、已绑定视觉-only、UWB-only冷启动/恢复，以及双信号失效时的安全停止；完整模态样本用于建立“视觉人物—UWB标签”的身份位置绑定。
- 不可靠相对点不应直接覆盖稳定的目标身份记忆。
- 只有UWB且目标尚未入画时，`visual_initialization.valid=false`；机器人可先向UWB位置保守接近。目标进入多人画面后，模型通过UWB投影位置、RGB人物候选和历史连续性更新内部身份状态，不要求外部系统提供候选bbox作为模型输入。
- 若多人候选在UWB误差范围内无法可靠区分，策略应减速或停止等待消歧，而不是强制绑定某一人。
- 当目标无法被视觉定位且UWB无效时，输出安全减速/零运动轨迹；仅用短暂去抖避免信号抖动导致频繁启停，不执行主动盲搜。

## 3.1 Architecture v1（2026-09-11冻结）

本节是NEXT-025的唯一主模型定义。除下列“必须完成的ablation”外，实现不得在v1中加入第二个patch级Temporal Transformer、DA3 depth/camera heads、part tokens、多套身份anchor、自回归/flow waypoint decoder或其他未经重新确认的模块。

### 3.1.1 固定主链与数据流

```text
一次初始化：
initial RGB_0 ──> DA3-SMALL encoder L11 ──> 768→256 projector
initial bbox_0 ────────────────────────────> RoIAlign 3×3 → mean
                                                      │
                                                      └──> Target Memory m_t [B,256]

每个policy step：
Ego RGB window [B,4,3,280,504]
        │
        └──> DA3-SMALL encoder（只取L11）
                 ├──> spatial tokens [B,4,720,768]
                 │      └──> current X_t [B,720,256]
                 └──> camera tokens [B,4,768]
                        └──> last pair → supervised SE(2) bottleneck → z_ego [B,256]

UWB (xy,covariance,quality,age,valid) + camera calibration
        ├──> explicit geometric Gaussian B_uwb [B,720] ──┐
        └──> UWB encoder z_uwb [B,256]                   │
                                                        ▼
m_t single-query attention over X_t + B_uwb ──> z_target [B,256]
scene-query pooling over X_t ──────────────────> z_scene  [B,256]
MLP([z_scene,z_ego]) ──────────────────────────> w_t      [B,256]

FusionMLP([z_target,w_t,z_uwb,valid masks]) ──> r_t [B,256]
GRUCell(r_t,h_(t-1)) ─────────────────────────> h_t [B,256]
        ├──> Linear 14 → future 7×2 + hard-coded point 0=(0,0) → waypoint [B,8,2]
        └──> stop head → stop_logit [B,1]
```

固定规则：

- DA3采用方案B：复用DA3-SMALL pretrained encoder/intermediate representation作为部署backbone；主链不运行`DualDPT`、`CameraDec`、`CameraEnc`或GS heads。必须直接调用内部backbone，不能使用总会运行heads且带`inference_mode`的公开完整forward。
- 第一版只消费L11 token，不融合L5/7/9；DA3自身已在4帧window内完成patch级local/global interaction，GRU只接收压缩后的policy token。
- 初始化身份只保存一个`m_t [B,256]`。正常bbox初始化时以固定身份参考为主；UWB-only模式从`UNKNOWN_PERSON`开始，只有唯一、可见且空间/时间一致的候选达到绑定门槛时才写入memory。
- UWB不能覆盖已有视觉身份，只作为空间先验和late-fusion条件；多人同时落在UWB不确定区域时保持`unbound`。因此v1支持“条件式空间共位绑定”，不声称UWB tag自身提供视觉身份。
- Waypoint第一版固定直接回归；第0点严格为当前底盘原点，只对未来7点计算轨迹回归损失。

### 3.1.2 Ego-motion与UWB spatial bias

DA3 L11 camera token是可被非线性`CameraDec`解码的隐变量，不是显式pose。v1使用最后两个camera tokens的pair head预测物理瓶颈：

```text
[LN(c_(t-1)), LN(c_t)] [B,1536]
  → Pair Motion MLP
  → xi_hat_t=[dx,dy,sin(dyaw),cos(dyaw)] [B,4]
  → Motion Embedding MLP
  → z_ego [B,256]
```

`z_ego`只能由四维预测运动量生成，禁止pair-head隐藏特征绕过该瓶颈进入policy。realized SE(2)标签由`(T^world_base,t-1)^-1 T^world_base,t`在上一时刻底盘系提取；训练和部署的policy均使用预测运动，不用GT motion teacher forcing。

UWB bias使用显式几何而非主方案learned projection。若UWB源时间为`tau`，先把点和协方差补偿到当前RGB时刻底盘系，并以目标运动噪声按`age²`膨胀协方差。仅有二维UWB时使用已标定tag高度分布`z_tag~N(z_mean,sigma_z²)`；未可靠标定高度时增大`sigma_z`，形成垂直方向较宽的prior，禁止默认投影成地面点。之后执行：

```text
p_camera = T_camera<-base · [x_forward,y_left,z_tag,1]
(u,v) = project(K_after_resize_crop,p_camera)
Sigma_uv = projection Jacobian传播(UWB covariance + age motion noise
           + tag-height uncertainty + intrinsics/extrinsics calibration uncertainty)
G_ij = Gaussian在第(i,j)个14×14 patch内的概率质量
rho_fov = sum_ij G_ij
gamma = valid · calibrated_quality · exp(-age/tau_age) · rho_fov
P_ij = (1-gamma)/720 + gamma · G_ij/(rho_fov+eps)
B_uwb,ij = log(P_ij+eps) - log(1/720)
```

`B_uwb`按row-major展开后加到target-query attention logits。投影均值在FoV外但不确定椭圆与图像相交时，仅保留图像内概率质量且由`rho_fov`降低强度；完全FoV外或`Z_camera<=0`时令bias为0而不clamp到边缘；covariance很大时prior自然趋近均匀；age超过冻结上限时按invalid处理。上述情形仍可由late-fusion `z_uwb`表达“目标在画面外/后方”的导航条件。

### 3.1.3 Deployment Path

- DA3-SMALL encoder及L11 spatial/camera tokens；L11 `768→256` projector。
- 初始化bbox上的RoIAlign `3×3`、单Target Memory、`UNKNOWN_PERSON`和置信度门控更新。
- 单query target attention、scene query pooling。
- L11 camera-token pair的显式SE(2)预测瓶颈与motion embedding。
- UWB显式几何Gaussian patch bias、UWB连续特征encoder和valid/quality/age处理。
- Fusion MLP、`GRUCell(C=256)`、bbox/visibility/binding诊断heads。
- 7个未来点回归并前置固定原点的`8×2` waypoint head，以及stop head。
- 若训练采用DA3后层adapter/LoRA，它属于encoder本身并随模型部署；不存在仅在训练时开启、部署时删除的视觉捷径。

Deployment明确不包含：OSNet/KPR、冻结perception cache、后续外部bbox、GT target point/pose/depth、GT realized motion、DA3 `DualDPT/CameraDec/CameraEnc/GS`、Forward/Inverse Dynamics heads。

### 3.1.4 Training-only Path

- 后续帧target bbox、visibility、track identity和OSNet teacher输出只作为辅助label/loss；OSNet不得产生部署输入。
- realized `a_real=(dx,dy,dyaw)`监督部署所需的SE(2)物理瓶颈，并作为Forward Dynamics条件；它不是expert waypoint。
- World-Action选项B：为使scene-only `w_t`能够预测目标后果，训练期Forward Dynamics同时接收`w_t`、`z_target_t`和`a_real`，从同一future hidden state预测：
  - normalized future latent `w_hat_(t+1)`，target为`stop_gradient(w_(t+1))`；
  - 下一时刻底盘系target relative `(x,y)`；
  - 下一时刻target visibility。
- Inverse Dynamics由相邻压缩world state预测realized SE(2)。Forward/Inverse均不预测完整DA3 patch feature，也不进入部署。
- DA3 depth/pose heads如用于离线pseudo-label或审计，必须独立于v1主梯度图与部署输入；不得把完整DA3 online偷偷恢复到主链。

### 3.1.5 核心tensor shapes

| Tensor | Shape | 语义 |
|---|---:|---|
| `initial_rgb` | `[B,3,280,504]` | 一次视觉初始化图像 |
| `initial_bbox` | `[B,4]` | resize/crop后图像上的`xyxy`；另有`[B,1]` valid |
| `ego_rgb` | `[B,4,3,280,504]` | 当前policy step的4帧窗口 |
| `da3_l11_spatial` | `[B,4,720,768]` | L11 local/global concat；grid为20×36 |
| `da3_l11_camera` | `[B,4,768]` | L11 camera latent，不直接当pose |
| `X_t` | `[B,720,256]` | 当前帧projected patch tokens |
| `roi_feature` | `[B,256,3,3]` | 初始化bbox RoIAlign输出 |
| `m_t` | `[B,256]` | 唯一持久Target Memory |
| `B_uwb` | `[B,720]` | 当前帧patch上的additive log-prior |
| `xi_hat_t` | `[B,4]` | `[dx,dy,sin(dyaw),cos(dyaw)]`物理瓶颈 |
| `z_target,z_scene,z_ego,w_t,z_uwb` | 各`[B,256]` | 压缩目标、场景、运动、world和UWB表示 |
| `fusion_masks` | `[B,4]` | `visual_init/uwb/rgb/binding`有效性 |
| `r_t,h_t` | 各`[B,256]` | GRU输入与统一shared policy state |
| `future_waypoints_raw` | `[B,7,2]` | 真正回归的未来7点 |
| `waypoints` | `[B,8,2]` | 前置固定`(0,0)`后的部署输出 |
| `stop_logit` | `[B,1]` | 安全停止logit |
| `bbox_pred` | `[B,4]` | 诊断/辅助预测，不是外部输入 |
| `visibility_logit,binding_logit` | 各`[B,1]` | 可见性与绑定置信度 |
| `a_real` | `[B,3]` | 训练期realized `[dx,dy,dyaw]` |
| `w_hat_(t+1)` | `[B,256]` | 训练期future compressed latent |
| `target_xy_hat_(t+1)` | `[B,2]` | 训练期未来目标相对位置 |
| `target_vis_logit_(t+1)` | `[B,1]` | 训练期未来可见性 |

### 3.1.6 Architecture v1总loss

```text
L_v1 = L_waypoint
     + 0.5 L_stop
     + 0.5 L_box
     + 0.5 L_visibility
     + 0.1 L_identity_or_binding
     + 0.1 L_ego
     + 0.1 L_world_action
     + 0.1 L_inverse

L_world_action = L_future_latent + L_future_target_xy + L_future_visibility
```

- `L_waypoint`：未来7点的masked SmoothL1；固定第0点不计loss。
- `L_stop`：stop BCE；双信号失效/RGB传感器失效样本必须覆盖。
- `L_box/L_visibility/L_identity_or_binding`：只消费label域的后续bbox/visibility/identity或训练期teacher，不产生部署输入；歧义UWB冷启动样本监督拒绝强制绑定。
- `L_ego`：`dx/dy` SmoothL1加`1-cos(dyaw_hat-dyaw)`圆周角损失。
- `L_future_latent`：归一化latent MSE，target端stop-gradient；`L_future_target_xy`使用独立valid mask的SmoothL1；`L_future_visibility`使用BCE。
- `L_inverse`：realized SE(2) SmoothL1/圆周角损失。上述系数是v1首个smoke与正式起跑默认值；任何变更必须写入配置和实验记录，不能静默修改。

### 3.1.7 必须完成的ablation

| ID | 对照 | 要验证的核心问题 |
|---|---|---|
| `ABL-V1-01` | DA3 pretrained vs 同结构DINOv2初始化；DA3 frozen vs 后层adapter/LoRA | 收益是否来自DA3 spatial pretraining，及是否需要任务适配 |
| `ABL-V1-02` | L11 only vs L5+L11 | 单层最小设计是否已足够，多层是否真实增益 |
| `ABL-V1-03` | RoIAlign `3×3` vs bbox mask mean pooling | 初始化Target Token的局部对齐是否必要 |
| `ABL-V1-04` | GRU history vs 无GRU单步fusion | 轻量history memory是否必要；不得替换成重Temporal Transformer |
| `ABL-V1-05` | supervised SE(2) bottleneck vs raw camera-token difference；另含no-ego | 显式运动语义是否优于无约束latent差分 |
| `ABL-V1-06` | geometric early bias + late UWB token vs late-only；geometric projection vs learned `UWB→720` | 显式空间先验是否帮助正常模式和UWB-only冷启动 |
| `ABL-V1-07` | World-Action B vs latent-only A vs no dynamics；正确action vs zero/shuffled action | future target state是否提供可解释增益，模型是否真的使用realized action |
| `ABL-V1-08` | Architecture-v1同构Phase 1初始化 vs 从DA3主干直接开始Phase 2；OSNet identity teacher on/off | Phase 1和teacher的实际贡献；现有ResNet18 Phase 1 v3只作baseline，不能伪装成DA3-v1权重迁移 |
| `ABL-V1-09` | 完整端到端v1 vs Frozen Frontend Baseline | 最终主方法是否优于OSNet/cache+MLP模块化系统 |

所有ablation必须共享scene/person/episode split、采样预算、评测脚本与运行种子；优先在固定open-loop val筛选，再对必要对照执行无GT target point的Habitat closed-loop。EVT人物识别F1只评估身份前端，不能替代这些端到端对照。

## 4. 建议训练阶段

### Phase 1（原 Stage 0）：身份与几何 World-Action 预训练

- 状态：训练方案已采纳，具体收益待实验验证。优先评估 DA3，也保留其他几何基础模型作为对照。
- 目标：在不要求专家动作的情况下，先学习 egocentric 几何、相机/底盘状态转移和跨遮挡目标身份保持；本阶段不把普通视频中的相机运动当成“正确跟随动作”。
- 内部拆为两类可独立采样和消融的数据流：
  - **Phase 1A，几何/动态流**：使用连续 ego RGB；几何教师产生 depth、confidence、camera pose 和空间特征。pseudo ego-motion只作为 inverse dynamics 的已实现运动标签，以及 forward dynamics 的条件，不监督跟随 policy waypoint。
  - **Phase 1B，目标身份流**：使用普通人物跟踪视频；输入序列起点的一次目标图像+bbox，利用后续一致 track ID、bbox、mask、visibility 等训练从 RGB 历史保持身份。逐帧标注只产生损失，不作为后续模型输入。
- 主要辅助目标：inverse dynamics、pseudo-motion-conditioned forward dynamics、单步 action-free next-state prediction，以及目标匹配/可见性/身份一致性监督。辅助 head 默认只在训练时使用。
- DA3 pose 约定：记 `^A T_B` 为从坐标系 B 到 A 的变换；DA3输出 `E_t = ^C_t T_W`（w2c），先求 `^W T_C_t = E_t^{-1}`。结合固定外参 `^C T_B` 得到 `^W T_B_t = ^W T_C_t · ^C T_B`，再计算 `^B_t T_B_{t+k} = (^W T_B_t)^{-1} · ^W T_B_{t+k}`，最后取底盘平面的 x、y、yaw。
- 尺度规则：米制 waypoint 只能使用具有可信 metric scale 的片段；尺度来自 metric 几何模型或里程计、UWB、相机高度等外部标定。任意尺度的单目序列不能与米制轨迹标签直接混用。
- 过滤规则：剔除低几何置信度、运动主体占画面过大、明显手持/非底盘运动、时间不连续及严重非平面运动片段。
- 与 WALA 的区别：不学习额外 latent action encoder；使用几何模型恢复的显式 realized motion，但保留“从无动作视频学习状态转移”的思想。

### Phase 2（原 Stage 1）：目标人物跟随的 World-Action 联合监督

- 状态：训练方案已采纳；对应原方案的专家监督阶段。
- 目标：学习目标身份保持、跟随距离控制、转弯、停止/等待和局部运动规划，并建立视觉身份与指定 UWB tag 的对应关系。
- 必要输入：RGB历史；可选的一次初始化图像+bbox；统一格式的UWB位置、时间戳、age、有效性和质量信息。后续逐帧bbox不作为输入。
- 必要监督：每个访问状态在当前机器人坐标系下的 expert future waypoints。
- UWB模式的必要数据：同步RGB与UWB日志、相机与UWB/底盘坐标标定；或由robot pose与target pose生成并经过真实误差模型校验的UWB观测。
- UWB-only冷启动的必要辅助标签：指定tag对应的目标track ID、逐帧bbox/visibility，以及UWB位置在相机中的投影及不确定区域。这些只监督何时、与谁完成内部视觉绑定，不进入推理输入。
- 训练必须显式覆盖四类模式：
  - **正常初始化**：一次图像+bbox建立身份，UWB同时存在；
  - **视觉-only**：已有身份绑定，UWB暂时缺失，依靠RGB历史继续跟随；
  - **UWB-only冷启动/恢复**：没有视觉参考或视觉身份已丢失，先按UWB保守趋近，目标可见且关联置信度足够后自动绑定；
  - **无可靠目标条件**：目标无法视觉定位且UWB无效，监督安全减速至停止；若整个RGB传感器失效，即使UWB仍有效也默认停车，除非未来另有独立、已验证的避障安全栈。
- 可选辅助标注：目标mask、visibility、遮挡状态、目标速度、相对目标pose、UWB误差/LOS-NLOS状态、碰撞和free-space。
- 数据课程建议：单人且持续可见 → 短遮挡 → 多人干扰 → UWB冷启动绑定 → 较长遮挡、模态冲突和复杂转弯。
- World-Action辅助监督：继续训练 forward dynamics 与单步 next-state prediction；可加入未来目标相对位置、未来visibility和未来空间feature预测，但不改变expert waypoint主监督。
- 退出条件：正常初始化能够稳定跟随；视觉-only能保持身份；UWB-only能安全趋近并在入画后正确绑定；多人歧义时不误绑定；双信号失效时能够稳定停车。

### Phase 3（原 Stage 2）：噪声、模态冲突与闭环恢复

- 状态：训练方案已采纳；内部按 3A 后 3B 实施和消融。
- 目标：抵抗初始化误差、UWB噪声/漂移/延迟、身份混淆、模态冲突和机器人自身走偏，并学习回到可跟随状态。
- **Phase 3A，离线受控扰动**：在 Phase 2 专家样本上污染输入，动作监督仍使用对应的干净 expert waypoints。
  - 初始化bbox只在首次视觉绑定时施加抖动、尺度变化、截断、错框和背景污染，不在每个后续时刻伪造bbox输入；
  - UWB加入随机误差、持续偏置、时间相关漂移、延迟、丢包、离群值，以及confidence与真实误差失配；
  - 加入UWB缺失、视觉身份置信度下降、RGB帧丢失和模态冲突；
  - 随机化FoV、相机高度/外参小误差、机器人姿态、目标速度和多人布局；
  - 双信号失效样本只占小比例，用于学习安全停车，不将其包装成“无条件恢复跟踪”。
- **Phase 3B，闭环 rollout + expert relabel（DAgger式）**：让当前模型在仿真中执行，从模型实际访问的任意状态查询 expert，生成恢复 waypoint；持续混入 Phase 2 原始专家数据。
- 数据优先级：真实UWB产品日志、真实视觉关联失败与传感器时间戳高于纯人工高斯噪声；仿真噪声分布应由真实统计校准。
- 模态冲突：重点覆盖稳定视觉身份与陈旧/错误UWB矛盾、多个候选同时落入UWB不确定区域，以及自动绑定后发现身份错误的样本。
- 闭环采样：重点收集走偏、误跟、错误绑定、目标丢失、stuck、oscillation、collision-near及expert takeover前后状态。
- 退出条件：受控噪声下性能平稳退化；闭环恢复率和自动绑定准确率提高；歧义及双信号失效时能安全停车；干净状态性能没有明显回退。

### 三阶段之后：闭环目标优化（可选）

- 暂不决定是否采用RL；只有在BC + World-Action监督 + 条件扰动 + DAgger稳定后仍有明确长时瓶颈时，再评估outcome-based post-training。

## 5. 阶段 Benchmark、测试集与可视化

### 5.1 通用验收协议

- 每个Phase执行固定闭环：`train → eval → render → gate`。gate失败时保留产物并停止流水线，不默认进入下一Phase。
- 项目主gate使用与最终任务输入/输出一致的自建或仿真held-out数据；公开跟踪或pose数据只作为Phase 1能力检查，不能替代端到端跟随benchmark。
- 数据至少拆分为：
  - `train`：允许参与参数更新；
  - `val`：用于选checkpoint、调超参数和设定初始gate；
  - `viz_val`：固定的小型人工检查集，可频繁生成视频；
  - `test_locked`：按里程碑或正式对比运行，不作为日常调试视频来源。
- 所有划分按scene、人物和episode隔离；每个benchmark保存数据版本、episode列表、仿真seed、场景配置和checksum。
- 初次建立baseline后冻结通过阈值；正式实验不得根据`test_locked`结果反复修改阈值。
- 所有后续帧中的bbox/heatmap必须标记来源：`PRED`表示模型诊断输出，`GT`表示评测真值；二者均不是推理输入。画面从第1帧起固定显示`external bbox input: false`。
- 每个Phase至少输出：`best.ckpt`、完整配置、git commit/dirty状态、数据manifest、`metrics.json`、`report.md`、失败案例索引和固定episode视频。

### 5.2 Phase 1验收：身份与几何是否真的学会

Phase 1不要求机器人已经完成闭环人物跟随；理想结果是证明后续policy可复用的身份记忆和egocentric几何能力已经形成。

| Benchmark | 固定测试数据 | 主要指标 | 必要可视化 | 理想表现 |
|---|---|---|---|---|
| `B1-ID` 首帧指定目标保持 | held-out人物视频，覆盖单人、相似人物交叉、部分/完全遮挡、出画重入、尺度/视角变化和目标持续缺失 | tracking success/AUC、中心误差、wrong-person/ID-switch率、遮挡后重获取率与耗时、目标缺失时误报率 | 第0帧用户bbox；后续`PRED`目标热力图/诊断框与`GT`框；identity confidence和visibility时间线 | 后续无bbox输入时仍锁定原人物；交叉不换人；遮挡后恢复原身份；目标缺失时不强行匹配干扰者 |
| `B1-GEO` 几何与运动 | held-out带可信metric robot/camera pose的视频，场景和轨迹与训练集隔离 | SE(2) relative pose平移/转角误差、分段漂移、尺度误差、有效预测覆盖率；next-state辅助误差 | RGB、depth/confidence、预测与GT的鸟瞰ego轨迹、逐时刻误差曲线 | 预测运动方向和转弯与GT一致，米制尺度稳定；低置信度片段能被识别而不是静默产生大错 |
| `B1-PROBE` 低数据迁移检查 | 固定小比例expert数据，只训练统一的轻量probe或采用统一短程微调协议 | 相比无Phase 1预训练的sample efficiency与验证误差 | 同配置学习曲线和少量waypoint叠加 | 在相同数据量与训练预算下稳定优于from-scratch基线 |

Phase 1推荐视频版式：左侧为RGB与目标诊断输出，右侧为预测/GT ego-motion鸟瞰图，底部显示`bbox input`、目标可见性、身份置信度、平移误差和yaw误差。第一帧之后出现的框只能来自`PRED`或`GT`图层。

### 5.3 Phase 2验收：干净条件下能否完成基本跟随

固定benchmark同时包含open-loop expert片段与closed-loop episode，至少覆盖：

- 一次视觉bbox初始化 + UWB持续可用；
- 一次视觉bbox初始化后UWB缺失，依靠RGB历史继续跟随；
- 无视觉初始化的UWB-only冷启动、保守接近和首次自动视觉绑定；
- 目标出画但UWB仍有效，随后重新入画；
- 目标无法视觉定位且UWB失效时安全减速停车；
- 相似人物交叉和多人同时落入UWB不确定区域。

主要指标：open-loop waypoint ADE/FDE与转角误差；closed-loop跟随成功率/持续时间、目标距离与方位误差、碰撞/near-collision率、wrong-person率、UWB冷启动绑定准确率与绑定耗时、安全停车成功率及制动距离。

每个episode视频至少同时显示：RGB、仅作诊断的`PRED/GT`目标图层、UWB相对点及不确定区域、UWB age/valid、预测与expert waypoint、机器人和目标的鸟瞰轨迹、当前绑定来源与置信度。理想结果是完成基本稳定跟随、单模态退化运行、正确冷启动绑定和双信号失效停车，而不是只在open-loop上得到较低ADE。

### 5.4 Phase 3验收：噪声和走偏后能否恢复

- 固定离线扰动矩阵：初始化bbox误差、UWB随机误差/偏置/相关漂移、延迟、丢包、离群值、置信度失配、视觉遮挡、相似人物、RGB帧丢失和相机参数扰动。
- 固定闭环失败集：走偏、误绑定、目标丢失、stuck、oscillation、collision-near及expert takeover前后状态；相同scene/seed同时运行Phase 2与Phase 3 checkpoint。
- 主要指标：随噪声强度变化的性能曲线、恢复成功率、恢复耗时、wrong-person率、碰撞率、安全停车率，以及干净benchmark相对Phase 2的性能回退。
- 必要可视化：Phase 2与Phase 3同场景并排视频、恢复事件时间线、输入噪声真值与模型置信度、成功/失败案例画廊。
- 通过条件：Phase 3在固定恢复集上显著优于Phase 2，同时干净性能回退不超过预先冻结的容忍值；双信号失效只要求安全停车，不计为无条件跟踪恢复。

### 5.5 GitHub一键训练与产物约定

目标入口形式（当前实际配置是`.env`）：

```bash
bash scripts/run_pipeline_8xh100.sh --config configs/pipeline/h100_8gpu.env
```

脚本职责：检查数据与环境 → 以8卡分布式方式训练当前Phase → 运行固定benchmark → 生成报告和视频 → 判断gate → 通过后加载上一Phase的best checkpoint继续。应支持断点续跑和单独运行某个Phase，但具体训练框架暂不限定。

建议产物结构：

```text
runs/<run_id>/
  run_manifest.json
  phase_1/{checkpoints,best.ckpt,metrics.json,report.md,visualizations,failures}
  phase_2/{checkpoints,best.ckpt,metrics.json,report.md,visualizations,failures}
  phase_3/{checkpoints,best.ckpt,metrics.json,report.md,visualizations,failures}
```

GitHub只保存代码、配置、环境锁定文件、数据manifest和已审计的`example_datasets/samples`微型样例；完整训练数据与大checkpoint通过外部存储路径解析。每个报告必须能追溯到代码版本、数据版本、配置、随机种子和父checkpoint。

## 6. 数据清单

> 正式外部数据范围已固定为 InternData-N1、Sage3D extracted 和 TpT clean v2。2026-09-07 已完成`docs/data_inventory.md`、`configs/data_inventory.json`、只读审计脚本、全量元数据/路径检查和确定性分层媒体解码。严格数据划分由NEXT-015继续完成。`未知` 不应默认解释为不存在。

| DATA-ID | 数据集/位置 | RGB | 首次 bbox | 逐帧 ID/bbox | Robot pose | Target pose/UWB | Expert trajectory | Rollout | 可用于阶段 | 当前问题 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| DATA-001 | 普通人物跟踪视频 | 未知 | 未知 | 未知 | 通常无 | 通常无 | 无 | 无 | Phase 1B；Phase 2/3辅助 | 需盘点规模、场景和标注质量 |
| DATA-002 | 普通ego视频/带pose视频 | 未知 | 通常无 | 通常无 | 无、真值或可伪标 | 通常无 | 无 | 无 | Phase 1A | 必须筛掉非底盘运动并确认metric scale来源 |
| DATA-003 | 同步视觉+UWB跟踪数据 | 未知 | 可选 | 必需作辅助标签 | 必需或可标定 | 必需 | 可选 | 无 | Phase 2；Phase 3A | 需确认tag ID、LOS/NLOS、时间同步及冷启动入画样本 |
| DATA-004 | Expert trajectory 数据 | 未知 | 必需或明确UWB-only | 强烈建议 | 强烈建议 | 必需 | 必需 | 无 | Phase 2；Phase 3A | 需确认waypoint坐标系、horizon、时间间隔和四种输入模式覆盖率 |
| DATA-005 | 仿真 rollout 数据 | 可生成 | 可生成 | 可生成 | 可生成 | 可生成含UWB噪声 | 可从任意状态重标注 | 可生成 | Phase 3B；三阶段后可选RL | 需确认arbitrary-state expert、传感器失效注入和仿真吞吐量 |
| DATA-006 | Git内`data/datasets/track/{STT,DT,AT}` | 需运行仿真生成 | 当前文件无预计算bbox | 有`main_human_semantic_id`和人物配置，逐帧框需渲染/生成 | 只有episode初始robot pose及场景配置 | 有人物起点/waypoint元数据，无UWB日志 | 无预计算robot expert future waypoints | 无 | 仿真样本生成；Phase 2/3 benchmark基础 | 6个文件、共25,986 episodes；必须验证episode语义、生成器和scene隔离后再做manifest |
| DATA-007 | 4090本地Habitat资产：`data/scene_datasets`、`data/humanoids`、`data/versioned_data` | 资产可渲染 | 可由仿真GT生成 | 可由semantic ID生成 | 仿真可得 | 仿真可得 | 需oracle生成 | 可生成 | Phase 1～3仿真数据生产 | 被Git忽略；H100需独立挂载/软链接，路径不能写死为4090绝对路径 |
| DATA-008 | InternData-N1：`/h100-2/vln_n1/traj_data` | 有RGB+depth | 无人物bbox | 无人物ID/bbox | 有逐帧pose矩阵和离散action | 有导航goal，无人物target pose/UWB | 可由pose/action构造导航轨迹监督 | 已展开episode | Phase 1A几何/动态预训练 | 12 group、196,536 episodes、3,725个正式scene；5个仅含未索引depth的目录已排除；旧85,124统计漏扫；自然语言task不得输入；按group/scene隔离 |
| DATA-009 | Sage3D：`/data/nfs/share/OmTrackVLA/data/sage3d_extracted` | 有，2,132,276帧，另有等量depth | 从首个侧车visible bbox构造 | Phase 1/2只读准入侧车，源bbox/visible禁用 | 有逐帧robot pose/yaw | 有逐帧target pose/target_local；仅可派生无噪声`simulated_uwb` | 从robot pose重算8点`0,+3,…,+21`，源`waypoints_ego`禁用 | 已抽取仿真run | Phase 1B、Phase 2、Phase 3A | 7,105个canonical accepted；256个非locked episode验证30 Hz、x-forward/y-left和轨迹；按run隔离 |
| DATA-010 | TpT clean v2：`/data/nfs/share/OmTrackVLA/data/tpt_bench_clean_v2` | 有，141,326帧 | 可用每序列首个可见bbox构造 | 有逐帧bbox、is_exist、玻璃遮挡标签；无显式跨序列person ID | 有ODOM position/quaternion | 无target pose/UWB | 无 | 47条真实跟踪序列 | Phase 1B；Phase 2/3身份辅助 | 约17 GiB；RGB/parquet无缺帧；视频时钟与GT/ODOM时钟约差4.48倍；必须按sequence隔离 |

### 6.1 2026-09-07容量与规模基线

| 数据根 | 当前容量/规模 | 与此前记录比较 |
|---|---|---|
| `/h100-2/vln_n1/traj_data` | 按项目确认视为完整数据；12 group、196,536 episodes；不再对海量小文件执行全量`du` | 顶层目录自2026-06-12～15未变化；旧85,124 episodes是统计漏扫，不代表数据近期增长 |
| `/data/nfs/share/OmTrackVLA/data/sage3d_extracted` | 约160 GiB；171,095,801,856分配字节；2,132,276 steps | 与本次会话前次复测的160 GiB一致，未见变化 |
| `/data/nfs/share/OmTrackVLA/data/tpt_bench_clean_v2` | 约17 GiB；17,604,104,192分配字节；141,326帧 | 与本次会话前次复测的17 GiB一致，未见变化 |

### 6.2 WP-0审计结论

- `scripts/audit_data_inventory.py`只在显式`--output`位于数据根之外时写报告；数据根内没有创建或修改文件，也不会生成`_target_crops`。
- InternData-N1：全量metadata/Parquet episode ID与声明计数一致；48个按12 group分层的RGB/depth样本解码通过；5个无metadata/Parquet的depth-only残留目录以warning排除；不执行全量`du`。
- SAGE3D：7,110个索引episode的derived step、RGB/depth文件ID与路径全量一致；108个按mode-camera分层的RGB/depth样本解码通过；canonical accepted为“根索引成员 + `_ACCEPTED` + `quality.status=accepted”的7,105个episode。
- TpT clean v2：47个sequence的141,326个Parquet行与非空RGB逐一对应；47个按sequence分层的JPEG样本解码通过；GT/ODOM相对视频时钟倍率中位4.4752、范围4.2553～4.8668，继续作为WP-1阻塞项。
- 三套数据均无真实UWB；自然语言字段已列入禁止模型输入项。

### 数据进入训练前的必要检查

- 时间戳是否同步。
- 相机内外参是否可追溯。
- robot/target pose 坐标系定义是否唯一。
- waypoint 是增量、绝对局部坐标还是积分后的 SE(2) 轨迹。
- waypoint horizon、采样间隔和控制执行频率是否一致。
- bbox 对应的 target ID 是否在遮挡和重现后保持一致。
- UWB能否输出指定tag在anchor/world坐标中的米制2D/3D位置；若只有range/bearing或原始测距，适配层如何定位。仅有单一range时不得假设可唯一恢复二维目标点。
- UWB tag ID、anchor map、更新频率、端到端延迟、quality字段、LOS/NLOS状态、丢包方式和时间戳语义是否可追溯。
- RGB相机、底盘与UWB坐标系的外参是否标定，目标UWB位置投影到图像后的系统偏差和不确定区域是否有实测统计。
- “视觉缺失”必须区分目标出画/遮挡、模型关联失败与整个RGB传感器故障；三者对应不同训练标签和安全策略。
- 双信号失效时的安全减速轨迹、最大制动距离及等待/恢复条件是否由控制与安全接口明确定义。
- 数据集 train/validation/test 是否按 scene、人物或 episode 做隔离。

## 7. 数据类型与阶段映射

| 数据类型 | 主要价值 | 推荐阶段 | 限制 |
|---|---|---|---|
| 普通跟踪视频 | 身份保持、遮挡重识别、visibility、干扰人物区分 | Phase 1B；Phase 2/3辅助 | 无机器人动作时不能直接监督waypoint；逐帧框只作标签 |
| 普通 ego RGB 视频 | 经DA3类几何模型恢复realized motion，监督forward/inverse dynamics与单步未来空间状态 | Phase 1A | 不产生policy loss；需过滤非导航运动并处理metric scale |
| 同步 robot+target pose 数据 | 生成理想相对目标点、目标速度、延迟点和坐标变换监督 | Phase 1A/2/3A | pose不等同于安全expert轨迹，也不能替代真实UWB误差统计 |
| 同步视觉+UWB数据 | 建立tag—视觉人物绑定，训练UWB-only冷启动、多模态融合和单模态缺失 | Phase 2/3A | 必须保留tag ID、时间戳、age、坐标系、LOS/NLOS和真实误差 |
| Expert trajectory 数据 | 基本跟随动作学习及离线扰动训练 | Phase 2/3A | 只覆盖expert状态时不能充分训练恢复 |
| 当前模型仿真 rollout | 暴露模型诱导状态、错误绑定与真实失败分布 | Phase 3B；三阶段后可选RL | 需要arbitrary-state expert relabel；噪声需经真实日志校准 |

## 8. 论文设计取舍记录

| 论文 | 值得参考 | 当前不采用/不优先 |
|---|---|---|
| TrackVLA | expert waypoint imitation；遮挡/干扰人物数据；当前帧与历史帧差异化组织 | VQA/语言联合训练；暂不确定 anchor diffusion |
| TrackVLA++ | confidence-gated TIM；pose 生成相对角度/距离；不可见状态标注 | Polar-CoT、语言 token、多视角依赖 |
| LightNav-0 | DAgger 状态；相机随机化；轨迹模式平衡；关键失败事件采样；难度分桶 | ER/VQA、pointing token；暂不确定 RVQ/GRPO |
| ReferTrack | bbox条件注入；目标记忆；错误人物与缺失噪声；expert data curation | 语言Refer-CoT、候选框文本catalog、Refer-QA；不把外部历史bbox作为持续输入 |
| VLingNav | stuck/oscillation/failure 后 expert 接管；hybrid buffer；保留原专家数据 | adaptive CoT、语言摘要记忆；暂不确定 PPO/Gaussian head |
| FutureNav | action policy、action-conditioned forward dynamics、inverse dynamics、action-free future spatial-state prediction；世界分支可仅用于训练 | 语言指令与自回归action token不是本项目必需；迁移到连续realized motion仍是待验证假设 |
| WALA | 使用未来视觉/深度变化提供训练期world dynamics监督；无动作视频仍可利用 | 不采用其latent action encoder；改用DA3等恢复的显式pseudo ego-motion，且不把它当专家动作 |

参考导读：`/data/nas_ray/home/zeying.gong/algorithm/repos/Tracking_TRAINING_DETAILS_v2.md`

## 9. 实验记录

| EXP-ID | 日期 | 目标/假设 | 代码版本 | 配置 | 数据版本 | 随机种子 | 指标 | 结果 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| EXP-TEMPLATE | YYYY-MM-DD |  |  |  |  |  |  |  |  |
| EXP-001 | 2026-09-08 | 验证Phase 1从真实source adapter、RGB历史身份记忆到train/eval/render/gate的最小闭环 | 基于`f487a13`的未提交WP-2实现 | `phase1_pretrain.yaml`；单卡每源1 split unit/2 step；8卡每卡1 batch；8+8 eval、8+8 probe | `phase1_v1.json`，SHA-256 `254c7c6c...`; Intern/SAGE/TpT正式根 | 20260907 | B1-ID visibility=0.5、IoU=0.193、success@0.5=0；B1-GEO translation RMSE=0.155m、yaw MAE=0.116rad；B1-PROBE相对改善=0.156 | 带GRU history的checkpoint、三类指标、report和8帧960×380 MP4均生成；8×H100 DDP/NCCL及rank-0 checkpoint通过；正式gate按预期未通过 | 接口、历史记忆、单卡/8卡反向传播闭环成立；2-step结果不代表正式能力，必须保留gate失败并继续正式训练 |
| EXP-002 | 2026-09-08 | 实际验证DA3-SMALL的几何与feature接口，并检查DA3 pose到Intern canonical SE(2)的约定 | `4fa9143`加坐标修正工作树 | DA3官方commit `3d835ec1...`；模型revision `e08cab65...`；504px；4帧；feature layer 5/11 | Intern `3dfront_d435i/00154c06...` episode 0，indices 0/4/8/12 | 不适用 | depth/conf/pose/intrinsics及两层feature均有限；scale=2.07116；修正后translation mean=0.00496m、yaw mean=0.00519rad | GPU前向约0.68s；发现并修复OpenCV/Habitat轴混用和缺失canonical基变换；confidence中位4.3767 | NEXT-009接口验证完成；同clip用于尺度标定与误差计算，不能据此准入pseudo label，继续NEXT-010多场景held-out验证 |
| EXP-003 | 2026-09-08 | 审计SAGE3D投影bbox/visible能否作为身份监督 | `0bc9075`加只读审计工作树 | 384帧、128 episodes、每episode首/中/末3帧、覆盖18个mode/camera分层；Faster R-CNN MobileNet person阈值0.30；IoU 0.50/0.20 | SAGE3D正式根只读；detector权重SHA-256 `907ea3f9...` | 20260908 | 5个源框裁剪后零面积；379个有效框中331帧检测到人；强一致176、部分86、冲突69；中位IoU 0.533 | 初始/中间/末帧冲突率分别1.87%/32.43%/27.43%；水平镜像后中位IoU 0.638、强一致247、冲突14；报告和72例拼图在`results/wp2_sage_bbox_audit_v2/` | 当前SAGE bbox/visible未准入；根因是左轴误作右轴并缺少画面相交/遮挡检查，转NEXT-021修复侧车 |
| EXP-004 | 2026-09-08 | 不修改源数据地修复并重新准入SAGE3D bbox/visible身份监督 | 基于`e7ebaef`的NEXT-021工作树 | 正确image-right轴；完整相机平移；1.7m×0.5m人物包络；RGB对齐毫米depth；容差`max(0.30m,15%)`、support≥20%、near≤50%；32 workers；冻结384帧审计 | 7,105个canonical accepted episode、2,131,500 steps；manifest SHA-256 `f103f034...`；原检测记录SHA-256 `62fac918...` | 20260908 | 全量visible 2,111,385；out-of-view 6,919、depth-inconsistent 7,220、behind-camera 3,797、near-occluded 2,158、missing-depth 21；样本median IoU 0.6662、strong 82.21%、partial 15.34%、conflict 2.45%、rejected-strong 0 | 7,105/7,105完成、0失败；7项完整性和4项质量gate全通过，`admission.json`已生成；4张最差案例拼图位于`results/sage3d_bbox_sidecar_v1_audit/` | NEXT-021完成；Phase 1只从准入侧车读取SAGE3D身份标签，源bbox/visible继续禁用；残余冲突作为标签噪声风险保留 |
| EXP-005 | 2026-09-08 | 用互斥Intern场景冻结DA3尺度与置信度过滤，并正式准入pseudo ego-motion | 基于`f825845`的NEXT-010工作树 | policy `da3-small-intern-multiscene-v1`；4帧0/4/8/12；504px；每group 3 clips；`val`标定、`viz_val`开发、唯一一次`test_locked`；D435i/ZED尺度2.074511/1.806175；confidence阈值2.947961 | `phase1_v1.json`；locked selection/report/admission/records SHA-256分别为`88a14504...`/`fce834f9...`/`093892c6...`/`5f414031...` | 20260908 | `viz_val` coverage 61.29%、bad 5.26%、translation median/P90 0.03785/0.09934m；locked coverage 56.25%、bad 16.67%、translation 0.044991/0.119449m、yaw 0.002505/0.012585rad、scale error 0.128413/0.393865 | 31个val、32个locked clips、12 groups、0推理失败；locked 14项gate全通过且`visualizations=[]`；开发图片位于`results/wp2_da3_multiscene_v2_viz/` | NEXT-010完成；只准入冻结分相机尺度与confidence gate，禁止重跑/查看locked；正式Phase 1 baseline可继续 |
| EXP-006 | 2026-09-08 | 首次运行包含Intern几何流、SAGE3D/TpT身份流的正式Phase 1 baseline | `e774542` | 8×H100；8 epochs；65,536 samples/epoch；每卡batch 16；4,096 optimizer steps；run `phase1_baseline_v1_e774542` | `phase1_v1.json`及准入SAGE3D侧车；身份源按帧数比例采样 | 20260907 | B1-ID visibility 0.98364、absent FPR 1.0、IoU 0.65034、success 0.76620；B1-GEO translation RMSE 0.09222m、yaw MAE 0.06650rad；B1-PROBE +0.05134 | best checkpoint step 3,072；11项gate中10项通过，absent FPR失败；失败产物完整保留 | 正式训练链可复现，但4096个val身份样本仅67个absent且训练采样压低TpT负例，模型退化为恒预测visible；必须修复后重跑 |
| EXP-007 | 2026-09-08 | 修复v1不可见样本学习与当前帧身份记忆污染 | `ad9ab72` | 与v1相同正式规模；身份源等概率采样；absent visibility loss 10×；当前帧不写入自身判定前的身份记忆；run `phase1_baseline_v2_ad9ab72` | 与EXP-006相同；评测新增67个invisible样本计数gate | 20260907 | visibility 0.97437、absent FPR 0.77612、IoU 0.72060、success 0.86696；translation RMSE 0.08823m、yaw MAE 0.06571rad；未标准化B1-PROBE -1.27198 | best checkpoint step 4,096；12项gate中10项通过；FPR改善但仍失败，probe因特征尺度与固定绝对正则失配而失败；失败产物完整保留 | 数据/记忆修复有效提升bbox与FPR，但部署阈值和probe方法仍须按train-only原则校准，不能降低gate |
| EXP-008 | 2026-09-08 | 验证train-only visibility运行点与尺度公平的linear probe能否完成Phase 1准入 | `5103f88` | v2训练配置不变；checkpoint固定threshold 0.999；probe分别按train feature标准化；视频从4,096个标签候选中固定抽取8个absent；run `phase1_baseline_v3_5103f88` | 与EXP-006相同；未运行任何locked test | 20260907 | visibility 0.86401、absent FPR 0.16418、IoU 0.72060、success 0.86696；translation RMSE 0.08823m、yaw MAE 0.06571rad；标准化B1-PROBE +0.69457 | 4,096步训练及train/eval/render/gate全部完成；12/12 gate通过；`GATE_PASSED`、`PIPELINE_COMPLETE`存在，失败清理后无`.pipeline_lock`；64帧MP4可完整解码 | WP-2正式Phase 1 baseline完成；v1/v2负结果继续保留，下一工作包转Phase 2基本跟随 |
| EXP-009 | 2026-09-09 | 审计现成detector/ReID能否直接成为端到端感知组件，并定位检测与身份关联瓶颈 | 基于`d677ff7`的感知工作树 | 冻结Faster R-CNN MobileNet/ResNet50-FPN-v2、冻结OSNet x0.25 MSMT17；adaptive/anchor-only；320/640/默认分辨率 | TpT `viz_val`序列0005/0032，共5,928评测帧；未运行locked test | 20260908 | MobileNet320 candidate recall 51.22%、selection 53.08%、E2E 27.19%；MobileNet640为68.94%/37.42%/25.80%；ResNet50无融合为79.10%/32.14%/25.47% | 动态图库相对anchor-only将E2E从16.92%提升到27.19%；升分辨率/换强检测器提高候选召回，但更多干扰候选使手工关联退化 | 冻结现成组件方向成立；主要瓶颈已从检测召回转为候选身份融合，不微调OSNet |
| EXP-010 | 2026-09-09 | 用全部TpT train序列训练小型候选融合头，并验证单一高精度阈值 | 感知融合v2工作树 | 37个train序列、stride 4、8×H100；399,247候选；32维隐藏层；单阈值score 0.95/margin 0.06 | `phase1_v1.json`的37个TpT train序列；评测仅0005/0032 `viz_val` | 20260908 | candidate recall 79.10%、selection 41.94%、E2E 33.20%、visibility recall 36.30%、output precision 88.97%、absent FPR 1.74%、wrong-target 111 | 误跟显著下降，但单一严格阈值过度保守；相对8序列融合的E2E 51.27%明显回落 | 保留为负结果；必须拆分连续跟踪与全局重获运行点 |
| EXP-011 | 2026-09-09 | 验证train-only双运行点能否同时恢复召回并保持低误跟 | 感知融合v3工作树 | 与EXP-010相同权重训练；train-calibration按`global_search`分区，tracking precision floor 0.67/F1，reacquisition floor 0.70/F0.5 | 与EXP-010相同；未运行或查看`test_locked` | 20260908 | 阈值0.92/0.02与0.95/0.06；candidate recall 79.10%、selection 58.47%、E2E 46.30%、visibility recall 48.62%、output precision 90.68%、absent FPR 4.16%、wrong-target 60、reappearance 33.33% | 相对单严格阈值E2E +13.10pp、召回 +12.32pp、wrong-target 111→60且precision反升；80张五类图片全部解码 | 双运行点优于单严格阈值并作为当前开发基线；仍低于8序列高误报版本的51.27% E2E，下一步接入Phase 2并补连续tracklet/重获确认消融 |
| EXP-012 | 2026-09-09 | 准入SAGE3D Phase 2 waypoint坐标、时间与模拟UWB语义 | `aa32bd4`后的WP-3工作树 | 固定seed 20260909；`val`/`viz_val`各128 episode；13个mode/camera分层；30 Hz；8点stride 3；禁止`test_locked` | SAGE3D只读源；phase1 v1 split manifest；bbox sidecar既有准入 | 不适用 | 256/256有效且step连续；target-local/source重建最大误差均0.00005m；0.7s最大位移1.7370m；速度证据coverage 98.44%、ratio中位1.0124/P10 0.9044/P90 1.0701 | 11/11 gate通过；报告SHA-256 `ec90f757...` | SAGE3D policy标签准入；必须重算`0,+3,…,+21`，不得直读源`+1,+4,…,+22`；只声明无噪声模拟UWB |
| EXP-013 | 2026-09-09 | 验证冻结身份前端cache到Phase 2四模式waypoint训练、评测、渲染和gate的最小闭环 | `7d42bae`后的WP-3工作树 | train/val/viz_val各1个非locked episode；record stride 12；单卡20 steps、batch 16；四模式各24个val样本和4张图 | SAGE3D只读源；冻结ResNet50/OSNet/双运行点融合；partial cache显式标记并仅以`--allow-partial-cache`开发加载 | 20260909 | 正常三模式ADE 0.22561m、FDE 0.38741m；visual+UWB/visual-only/UWB-only ADE分别0.23059/0.22692/0.21933m；safe-stop 24/24精确全零；冻结前端诊断IoU 0.78950 | 20-step checkpoint、metrics/report、16张PNG和16帧MP4均生成并完整解码；smoke gate仅5个样本数项失败，8个数值/安全项通过；2-shard真实cache/merge另以2 episode验证 | 最小open-loop闭环成立；修复非视觉模式confidence泄漏；数值不可作为正式效果，仍需全量cache/训练和closed-loop benchmark |

| EXP-014 | 2026-09-09 | 在完全相同的TpT连续序列协议下比较Phase 1身份头与冻结detector/ReID/fusion，决定Phase 2前端 | `71547b9`；一次性只读评测 | `viz_val`序列0005/0032；5,928帧；首个可见框初始化；缺失输出按IoU 0计；未访问`test_locked` | Phase 1正式v3 checkpoint step 4,096；冻结ResNet50/OSNet/双运行点fusion v3 | 20260907/20260908 | Phase 1：E2E 2.59%、precision 9.33%、visible recall 19.19%、absent FPR 14.72%、reappearance 10.34%、wrong-target 470；冻结前端：E2E 46.30%、precision 90.68%、visible recall 48.62%、absent FPR 4.16%、reappearance 33.33%、wrong-target 60 | Phase 1同协议97个成功帧，冻结前端1,732个；评测帧及visible/absent计数严格一致为5,928/3,741/2,187；Phase 1结果写入`outputs/evaluation/phase1_v3_tpt_viz_same_protocol/metrics.json` | 冻结前端明确胜出，但46.30%仍不足以直接冻结进入waypoint；先执行DEC-043的2A训练与gate |
| EXP-015 | 2026-09-09 | Phase 2A首个时序hard-negative融合适配 | `2506da0`；`phase2a_temporal_fusion_v1` | TpT train 37序列stride 4；val 5序列全12,457帧；冻结Faster R-CNN/OSNet；48 epoch；train-only calibration；禁止`viz_val/test_locked`直到val通过 | 旧no-fusion与新on-policy train记录；随机初始化19维融合头；123,202 hard pairs | 20260909 | val基线/候选：E2E 42.80%→24.99%，precision 72.60%→98.06%，absent FPR 17.31%→0.72%，reappearance 30.77%→14.79%，wrong-target 1,377→26；候选排序precision 49.35%，选中epoch 2 | val safety的precision/FPR通过，E2E improvement失败；流水线退出码非零，未生成`PHASE2A_COMPLETE`，未运行`viz_val` | 失败保留；模型过度拒绝且排序弱于基线，按DEC-044改为基线精确初始化、on-policy-only及含epoch 0的排序选模 |
| EXP-016 | 2026-09-09 | Phase 2A基线精确初始化与epoch 0防退化验证 | `ae9c7a8`；`phase2a_temporal_fusion_v2` | 复用EXP-015 on-policy train/val记录；24 epoch；hidden 32；LR 1e-4；val候选F0.5选模；禁止`viz_val/test_locked`直到val通过 | 旧双运行点fusion精确展开到19维；新增列初始为零；on-policy-only | 20260909 | epoch 0以val F0.5 53.46%、TP 6,266胜过所有训练epoch并被正确选回；但train calibration重拟合为tracking 0.97/margin 0.30、reacquisition 0.98/0.06；在线val E2E 42.80%→13.47%，precision 72.60%→94.18%，absent FPR 17.31%→0.92%，reappearance 30.77%→14.79% | E2E safeguard失败并退出；未生成`PHASE2A_COMPLETE`，未运行`viz_val` | 失败保留；确认权重防退化有效、反事实阈值校准失效；按DEC-045继承旧运行点并只训练新增列 |
| EXP-017 | 2026-09-09 | Phase 2A仅新增时序列适配并继承旧运行点 | `a48026f`；`phase2a_temporal_fusion_v3` | 复用stride 4 train记录；24 epoch；仅7个新增列更新；val F0.5选模；旧双运行点不变 | 旧fusion精确初始化；on-policy-only；epoch 0全系统防退化 | 20260909 | val基线/候选：E2E 42.80%→51.13%、precision 72.60%→80.61%、FPR 17.31%→12.07%、reappearance 30.77%→35.50%、wrong 1,377→972；选中epoch 2。viz基线/候选：E2E 46.30%→44.88%、precision 90.68%→89.31%、FPR 4.16%→3.48%、reappearance 33.33%→34.48%、wrong 60→101 | val safeguard通过后首次运行固定viz；正式gate的precision/FPR/reappearance及计数通过，E2E提升与wrong-target失败；未生成`PHASE2A_COMPLETE` | 失败保留；不调viz阈值，按DEC-046修复train/部署stride 4→1分布偏移后重训 |
| EXP-018 | 2026-09-09 | Phase 2A同节拍stride 1时序适配 | `3075db4`；`phase2a_temporal_fusion_v4` | 37 train序列stride 1；5 val序列12,457帧；24 epoch；仅新增列更新；继承旧运行点 | 1,333,572个model-train候选；246,505 hard pairs；旧fusion精确初始化 | 20260909 | val离线top候选TP 6,266→6,374，选中epoch 8；在线val基线/候选：E2E 42.80%→40.06%、precision 72.60%→77.84%、FPR 17.31%→9.05%、reappearance 30.77%→24.26%、wrong 1,377→998 | E2E safeguard失败；未运行viz、未生成`PHASE2A_COMPLETE` | 失败保留；stride对齐必要但不足，离线排序与闭环状态分布不一致，按DEC-047做一次聚合 |
| EXP-019 | 2026-09-09 | Phase 2A一次train-only on-policy dataset aggregation | `b4b7dd3`；`phase2a_temporal_fusion_v5` | 复用v4冻结基线stride 1 records；新增v4-policy 37 train序列stride 1 rollout；12 epoch；只更新新增列；继承旧运行点 | 两个独立record root；冻结基线与v4-policy各37条train序列；禁止test_locked | 20260909 | 在线val基线/候选：E2E 42.80%→46.35%、precision 72.60%→80.35%、FPR 17.31%→10.56%、reappearance 30.77%→31.36%、wrong 1,377→888。viz基线/候选：E2E 46.30%→39.31%、precision 90.68%→85.92%、FPR 4.16%→4.07%、reappearance 33.33%→21.84%、wrong 60→137 | val safeguard通过；固定viz正式gate仅计数/FPR通过，E2E/precision/wrong/reappearance失败；未生成`PHASE2A_COMPLETE` | 失败保留；按DEC-048停止同类逐帧融合迭代，Phase 2B继续阻塞 |
| EXP-020 | 2026-09-10 | 评估冻结KPR/SOLIDER及显式tracklet/多帧重获能否突破逐帧OSNet fusion | 2026-09-09 KPR工作树；`kpr_phase2a_overnight` | 官方KPR Occ-Duke SOLIDER checkpoint；Faster R-CNN ResNet50；37 train序列stride 4校准；5 val序列全12,457帧；v2确认3帧，v3较宽阈值并确认4帧；禁止`viz_val/test_locked` | train 28,888帧、19,381正候选、379,866负候选；KPR 9个前景/部位分支、513维含可见性；5条val detector candidate recall 80.27% | 20260909 | v2/v3：E2E 12.60%/14.51%，precision 79.05%/74.67%，absent FPR 1.18%/2.56%，reappearance 4.14%/5.92%，wrong-target 325/469；`memory_updates=0`；`0013`静态anchor top-1 73.96% | 两个`COMPLETE`及报告正常生成，但二者均未达到88% precision safeguard；报告的v3 selection只是无合格方案时按E2E兜底 | 失败保留；KPR排序有潜力但当前未形成动态序列记忆，按DEC-049校准专属图库写入和漂移策略后重跑val |
| EXP-021 | 2026-09-10 | 在实际主评测EVT-Bench上快速选择Phase 2B人物识别前端 | 2026-09-10工作树；`evt_reid_compare_v1` | EVT `val` STT/DT/AT；各固定index 0/1；最多60步；reactive controller + GT point；相同episode；ResNet50 detector；KPR与OSNet+fusion；不访问`test_locked` | OSNet完整6条、213步；KPR完成4条、93步后提前停止；逐步GT IoU、候选存在率和memory update审计；OSNet轨迹协议6/6一致 | 20260910 | OSNet：183/192输出正确，precision 95.31%、recall 85.92%、IoU 0.889、detector target recall 93.90%、memory 174/180正确；STT/DT/AT precision为98.57%/86.27%/98.59%，recall为97.18%/61.97%/98.59%。KPR已完成4条为49/50、precision 98.00%、recall 52.69%；缺失两条全满分时F1上界仍低于OSNet | OSNet 6/6完整；KPR长STT/0初跑及自动重试均原生abort，partial保留并按不可改变选型结论的上界提前停止；3个index-1 episode均因相同控制轨迹在11步碰撞，故本实验只作感知快速选型、不作导航结论 | 通过快速选型；按DEC-051冻结OSNet前端并恢复Phase 2B cache；DT仍是后续EVT train校准与扩大审计的重点 |
| EXP-022 | 2026-09-10～11 | 在EVT-Bench全量`val`上正式比较OSNet与KPR并重做Phase 2A最终选择 | 2026-09-10工作树；`evt_reid_full_val_v1` | 每方法4,215 episode；STT/DT/AT各1,405；GT point + reactive controller；官方300步上限；无人工step截断；目标初始化goal crop；ResNet50 detector；train-only冻结阈值；保持28个可恢复分片；不访问`test_locked` | 8,430个方法-episode；每方法490,824步；逐帧GT可见性、precision/recall/F1/IoU、detector target recall、memory污染；episode macro指标、结束状态/错误率及逐episode配对轨迹哈希 | 20260910 | OSNet总体P/R/Micro-F1/Macro-F1为88.5708%/63.9865%/74.2978%/73.2168%，KPR为83.9559%/61.7710%/71.1748%/71.7425%；STT/DT/AT Micro-F1分别为OSNet 82.6470/75.2399/64.9409%，KPR 80.6759/66.2458/67.0831% | 两方法各4,215/4,215、0错误、0缺失；4,215组轨迹全一致；最初GPU 1、2、4～7原生失败后仅用健康GPU 3的4 worker断点恢复；故障GPU留下的STT index 1黑帧结果已归档并真实重跑，未手改指标；`COMPLETE`及正式JSON/CSV均生成 | 通过；总体胜者OSNet（Micro-F1高3.1230个百分点），冻结为Phase 2B前端；KPR在AT切片高2.1422个百分点，保留为切片分析与后续消融 |
| EXP-023 | 2026-09-11 | 审计现有Phase 1是否实际帮助Phase 2轨迹模型，并核对当前方法能否称为严格端到端 | 2026-09-11本地/远端同步前工作树；只读代码审计 | 检查`omtrackvla/models/{phase1,phase2}.py`、`omtrackvla/training/phase2.py`及`configs/phases/phase2_following_sft.yaml`的数据流、checkpoint约束和梯度边界 | 不涉及数据或locked split | 不适用 | Phase 1含共享ResNet-18、identity GRU、bbox/visibility与World-Action辅助头；Phase 2 forward只接收7类冻结低维感知/UWB特征并由MLP输出8点轨迹；Phase 2初始化明确拒绝`phase!=2` checkpoint；配置依赖冻结perception cache | 当前Phase 1到Phase 2权重迁移覆盖率为0，waypoint loss不可能回传至RGB encoder/identity memory；当前可运行链是模块化baseline而非严格端到端 | 审计不通过端到端定义；按DEC-053～054停止将cache+MLP作为主方法，转NEXT-025统一模型实现；不覆盖EXP-013等既有baseline证据 |

| EXP-024 | 2026-09-11 | 验证Architecture v1真正的`RGB→DA3→target/world/UWB fusion→GRU→waypoint`主链，并证明waypoint主损失自身进入DA3后层adapter | 基于`b420517`的未提交NEXT-025工作树 | DA3-SMALL L11；280×504；4帧history；20×36 patch；最后2个block residual adapter；单卡1 batch及8×H100一步DDP；只用masked waypoint SmoothL1，无辅助loss/优化器循环/正式训练 | Phase 1 manifest `train` run `0001_83992/stt/0/go2_realsense_d435i`，initial/anchor 0/8；全量准入bbox sidecar与源文件SHA-256校验；policy admission校验；raw RGB、canonical `0,+3,…,+21` expert；`simulated_uwb`；无perception cache或locked数据 | 20260911 | 官方checkpoint tensor/参数覆盖均100%（437/437；34,299,463/34,299,463）；冻结DA3 22,059,008，adapter 100,736，新policy 2,117,907；单卡Fusion/GRU/adapter梯度`5.740385e-3`/`6.065018e-3`/`5.033064e-5`；输出waypoint `[1,8,2]`、stop `[1,1]`、UWB bias `[1,720]` | 10项专项及全仓220项unittest通过；CPU stub、单H100和一步8×H100 DDP报告均`passed`，8卡核心梯度同样非零；`gsplat`仅为未调用3DGS head的可选依赖警告 | NEXT-025通过；只证明接口、输入隔离、权重加载和waypoint-only梯度闭环，不代表正式效果。按冻结边界进入NEXT-026，不增加Flow、多尺度DA3、part-token/memory bank或occupancy/depth decoder |
| EXP-025 | 2026-09-11 | 在任何正式训练前给用户提供可视化检查，即使随机初始化策略输出很差也先确认数据、几何和主链可读 | 基于`b420517`的未提交NEXT-026可视化工作树 | 官方DA3-SMALL；policy seed 20260911；0 optimizer step；1512×1090 dashboard；initial/anchor 0/106以获得0.9077m非平凡expert轨迹 | 与EXP-024相同的已准入SAGE3D `train` run；raw RGB、一次sidecar初始化框、canonical expert与`simulated_uwb`；无cache、后续bbox或locked数据 | 20260911 | 初始waypoint SmoothL1 `0.099862`、stop概率`0.5035`、UWB image gamma/rho-FoV均`0.6471`；随机预测聚集原点而expert前进0.9027m，符合未训练预期 | PNG 1512×1090可完整解码；初始化框对准人物，history连续；Target/UWB热图及未clamp的2σ椭圆可见，Scene Attention呈随机纹理；预测/专家轨迹方向和8点数值完整显示；本机/远端PNG SHA-256同为`b3f935c2...` | 工程可视化已通过自动/人工版式检查，但用户准入尚待确认；正式训练保持未启动 |
| EXP-026 | 2026-09-11～12 | 在用户准入可视化后运行Architecture v1正式Phase 2 open-loop训练，确认继续训练能否修正短轨迹并稳定辅助目标 | NEXT-026训练工作树；`next026_phase2_world_action_long_v1` | 8×H100；既有4,096步checkpoint后追加32,768步；有效累计36,864步；raw RGB主链、World-Action/inverse等冻结配置；每模式1,024个SAGE3D `val`样本正式评测 | 已准入SAGE3D train/val；canonical `0,+3,…,+21` waypoint；`simulated_uwb`；无EVT-Bench训练、cache或`test_locked` | 20260911 | 正常模式ADE/FDE `0.087408/0.151858 m`，预测/专家平均路径`0.319615/0.417073 m`，路径比`0.766329`；visual+UWB ADE/FDE `0.086265/0.150237 m`；finite/stop 100% | 相比4k基线ADE/FDE改善35.7%/34.1%，路径比`0.602→0.766`；固定样例长度覆盖约68.1%，仍偏短；best SHA-256 `a1dc37ff...` | 正式长训有效且成为最低误差模型，但“偏短”未完全解决；进入受控loss精修，不以训练step数代替固定val和可视化 |
| EXP-027 | 2026-09-12 | 比较普通续训、terminal/delta和直接path-length约束能否自然纠正短轨迹 | `waypoint_geometry_v1`、`waypoint_length_v1`、`low_lr_refine_v1`、`waypoint_length_mild_v1`、`waypoint_terminal_mild_v1` | 均从EXP-026最低误差checkpoint初始化；固定128×4快速抽查后仅对候选运行1,024×4正式评测；部署图不变 | 与EXP-026相同SAGE3D train/val和输入隔离；不使用EVT-Bench或locked数据 | 20260911 | terminal+delta 8k ADE/FDE `0.093739/0.163034`、路径比`0.761224`，全面退化；直接长度权重1.0的1k～4k路径比约0.87～1.04且ADE/FDE退化；普通低LR 1k正式`0.089286/0.153984`、路径比`0.808002`；温和长度权重0.1正式`0.088633/0.152235`、路径比`0.875005` | 直接长度约束在固定图产生锯齿/回退，即使aggregate长度接近专家也被否决；terminal-only正式路径比仅`0.777470`且ADE/FDE仍退化；所有失败checkpoint和指标保留 | 不能把路径长度当独立优化目标；普通续训存在Pareto权衡，直接长度loss可被绕路利用。转EXP-028逐horizon方向进度监督 |
| EXP-028 | 2026-09-12 | 用沿专家方向的逐horizon进度loss纠正收缩，同时避免path-length锯齿漏洞 | `omtrackvla/models/end_to_end.py`径向进度loss；`phase2_end_to_end_v1_waypoint_radial_progress.yaml`；`next026_phase2_waypoint_radial_progress_v1` | 从EXP-026 checkpoint低LR训练1,000步；loss权重0.25；8×H100；500/1,000步固定128×4抽查；1,000步运行1,024×4正式评测与同一固定图 | 与EXP-026相同；不改Deployment主链，不用EVT-Bench训练，不访问locked | 20260911 | 正式正常模式ADE/FDE `0.088484/0.153559 m`、路径比`0.800828`、末点半径覆盖`0.884029`；visual+UWB `0.087205/0.151814 m`、路径比`0.806152`；finite/stop 100%；checkpoint SHA-256 `d728ca67...` | 固定128抽查ADE/FDE优于源模型、路径比`0.769→0.805`；正式相对源模型路径比+4.50%，ADE/FDE +1.23%/+1.12%；固定图0个向后step，无path-length方案的回退；模型/评测/训练13项unittest通过 | 当前最佳自然长度Pareto候选；不覆盖EXP-026最低误差模型，最终取舍留给消融和closed-loop，不继续用同一val无边界调权重 |
| EXP-029 | 2026-09-12 | 公平复核ABL-V1-04并修复GRU联合随机训练的优化瓶颈 | `next026_converged_probe_v1`、`next026_gru_identity_probe_v1`、`next026_gru_curriculum_probe_v1`；8×H100；固定SAGE3D `val`每模式1,024 | direct主GRU和single-step各36,864步等预算；按policy-step拆分第0/1步；近恒等GRU direct 4,096步负对照；single-step checkpoint后重置GRU并用GRU/policy/adapter=`3e-4/3e-5/3e-6`训练4,096步；清零hidden反事实 | 同一raw RGB、simulated UWB、masks与辅助loss；不用EVT-Bench训练、cache或`test_locked` | 20260912 | direct GRU `0.119734/0.205298`、single-step `0.090825/0.156395`；近恒等direct 4k `0.176821/0.298233`；课程GRU `0.089646/0.154162`、路径比`0.746039`，finite/stop 100%；清零hidden为`0.090231/0.155013`、路径比`0.741733` | 课程首次waypoint-only梯度Fusion/GRU/adapter=`3.3001e-2/8.7724e-3/1.3767e-2`；固定图前向单调、无锯齿/回退，终点仍偏短；single-step dashboard旧标题误写GRU，renderer已按`temporal_fusion`修正 | GRU有小幅真实递归收益，但必须通过课程训练才能显现；4k 13臂不能作为正式消融结论，NEXT-026继续开放 |
| EXP-030 | 2026-09-12～13 | 完成ABL-V1-08的Architecture-v1 Phase-1初始化与OSNet identity teacher二维消融 | `next026_abl08_converged_v1`；teacher off/on各含36,864-step single-step与其后的4,096-step GRU课程；固定SAGE3D `val`每模式1,024 | 四臂共享Architecture v1、DA3官方初始化、Phase-2采样/评测口径；teacher只在Phase 1作为可移除监督，OSNet不进入部署；每臂固定dashboard与waypoint-only梯度 | 已准入SAGE3D raw RGB、simulated UWB；无perception cache、EVT-Bench waypoint训练或`test_locked` | 20260911 | off single/GRU ADE/FDE `0.086013/0.148800`、`0.085573/0.148125`；on single/GRU `0.084867/0.147118`、`0.084341/0.146654 m`；路径比分别`0.755717/0.768817/0.754872/0.758918` | finite与safe-stop四臂100%；teacher-on matched single-step改善1.33%/1.13%，teacher-on GRU再改善0.62%/0.32%；四张固定图无回退/锯齿但终点仍偏短；DA3加载100%，Fusion/GRU/adapter waypoint-only梯度非零 | ABL-V1-08完成；当前最优为teacher-on+GRU课程，旧ResNet18 Phase 1仍只作历史baseline，不能伪装成权重迁移 |
| EXP-031 | 2026-09-13 | 在Habitat中验证Architecture-v1模型是否能在无GT target point条件下实际控制机器人，并审计输入、安全、视频和重获 | `next027_full50_v6`；固定episode；teacher-on+GRU checkpoint；waypoint worker独占GPU7；50次`env.step()`；本次无UWB | 决策首帧仅RGB+bbox、后续仅强制刷新的RGB；GT pose/panoptic仅在动作后计分；内部visibility/bbox只用于安全门；逐帧PNG后隔离FFmpeg编码 | SAGE3D/Habitat固定非locked开发episode；未用EVT-Bench训练或`test_locked` | 20260911 | 50步，following/visible rate `52%/54%`，collision `0`，target loss/reacquisition `1/0`，robot/target path `2.106/6.244 m`，路径比`0.3373`；热身后mean/P95推理`16.30/16.34 ms`；waypoint/proximity-hold/uncertain-stop `11/17/22`步 | 首次不可见为第28步；第35/43步模型框落到墙柱，但低置信安全门令动作全零；前27步强制刷新RGB全部有效；低置信非零动作违规0；H.264视频50/50帧解码，本地/远端JSON与MP4 SHA-256分别为`d31b3eb9...`/`bd776353...` | NEXT-027仅安全子门通过；跟随率低且墙角遮挡后无重获，最终gate失败。下一步是Phase 3恢复数据/DAgger，不得以无UWB盲搜伪造重获 |
| EXP-032 | 2026-09-13 | 验证Habitat能否从NEXT-027模型真实失败状态生成分布受控的expert 8点，并证明该样本可通过waypoint主损失修正Architecture v1主链 | `next007_expert_relabel_probe_v1/v2/v4`、`next007_phase3_single_sample_smoke_v2`；重放v6前28个模型动作后才启用标签侧GT/navmesh；expert平移/slew上限0.10；4帧history；teacher-on+GRU checkpoint；无优化器step | v1过快负对照；v4要求轨迹长度/末点半径、相对静止anchor进度、方向余弦、碰撞和replay误差全部通过；输入仅初始化RGB+bbox、step25～28 RGB、无效UWB及相机标定，GT只在expert label侧 | Habitat固定非locked `val` episode 26，仅用于smoke；`formal_training_eligible=false`；未访问`test_locked`，未用EVT-Bench | 20260911 | v1 path/terminal `4.2205/3.5902 m`被拒；v4 `0.704529/0.703875 m`、最大段0.112492 m、相对静止anchor少落后0.342163 m、方向余弦0.533952、collision 0、replay camera/distance误差0。waypoint-only loss `0.0400602`，Fusion/GRU/adapter梯度`0.122840/0.0406215/0.0416455`；当前预测/专家路径`0.437407/0.704529 m` | 8项专项unittest与py_compile通过；v4 report/sample SHA-256 `1de251a8...`/`cc1e701b...`；smoke report/PNG `12897fb7...`/`75946a8f...`，人工图确认人物已出画且绿色预测仍偏短 | NEXT-007在一个真实失败状态上的最小可行性通过：可重放、可标注、可反传，且没有GT输入泄漏。不能声称Phase 3数据集或恢复能力已完成；下一步采集多个隔离`train` episode失败状态后再做小批量优化smoke |
| EXP-033 | 2026-09-13 | 完成ABL-V1-09 Frozen Frontend公平基线并量化端到端视觉适配收益 | `sage3d_phase2_frozen_frontend_v1`完整cache；单GPU global batch 32；2-step smoke后36,864步；每模式1,024 val；固定64帧render | cache train/val/viz_val `5578/733/389`，train另5个canonical rejected显式skip；冻结OSNet/cache+MLP，不进入Architecture v1主方法；同batch、step、split和四条件模式 | SAGE3D非locked split；不用EVT-Bench waypoint训练；不访问`test_locked` | 20260911 | baseline ADE/FDE `0.168628/0.282747 m`、路径比`1.339015`；Architecture v1最优`0.084341/0.146654 m`，相对改善49.98%/48.13%，finite均100% | `CACHE_COMPLETE`、`TRAINING_COMPLETE`、`ABL09_COMPLETE`及summary生成；旧`torchrun` shebang导致首次eval退出126，失败记录归档后以当前Python模块方式续跑；训练未重复 | ABL-09完成，明确支持Architecture v1端到端主线；冻结前端只保留为较弱历史baseline |
| EXP-034 | 2026-09-13 | 用多个隔离train episode验证模型访问状态relabel与waypoint-only小批量恢复优化 | STT index 0、DT 200、AT 401各12步模型闭环；post-step-9；21步未来目标预演；4帧RGB history；3样本8个optimizer step；teacher-on+GRU checkpoint | 相机/距离精确replay；独立Habitat预演与expert实例；坐标/未来轨迹/路径/方向/相对静止anchor/碰撞硬门；模型输入不含GT pose/depth、后续bbox或UWB | Habitat `train`；三样本`formal_training_eligible=true`；不用EVT-Bench；不访问`test_locked` | 20260913 | STT/DT/AT相对静止anchor改善`0.488/0.499/0.495 m`，方向余弦`0.959/0.999/0.987`，碰撞0；smoke loss `0.014094→0.002434`，Fusion/GRU/adapter梯度`0.076001/0.011950/0.012074` | 19项专项测试；三条rollout均exit 0且12/12 fresh RGB；三份sample及before/after PNG生成；`checkpoint_saved=false`、`formal_training_started=false` | NEXT-007 train-split最小闭环和小批量优化smoke通过；正式Phase 3停在用户可视化检查门前 |
| EXP-035 | 2026-09-15 | 验证可观测性修正能否在独立recovery-val同时降低waypoint误差和假可见 | `v2_013_phase3_observable_safe_stop_pilot`；64步；独立recovery-val；小预算pilot | Architecture v2 8帧、0.1秒；不访问`test_locked`；不覆盖父checkpoint | 20260915 | recovery ADE变差，4个false-visible均未减少；可见概率虽下降但仍高于部署阈值0.005 | 正常结束但未生成选中checkpoint，失败产物完整保留 | 失败；不能靠整体压低概率解决状态可观测性，转多场景采集和严格分区 |
| EXP-036 | 2026-09-15 | 用多场景DAgger样本验证Architecture-v2 Phase 3 waypoint与安全状态能否在严格隔离验证集共同改善 | v2_014/v2_014b采集、v2_015/v2_015b重标、v2_016分区及128步训练 | index300/700的STT/DT/AT六条完整rollout，另加AT100/1100；20个train、13个recovery-val；场景/状态交集均0；index1300完全留出 | Habitat `train`；14 expert+6 safe-stop train，7 expert+6 safe-stop val；不访问`test_locked` | 20260915 | recovery ADE `0.23934→0.22053`、expert ADE `0.37993→0.36507`、safe-stop path `0.14875→0.10620 m`，SAGE3D ADE `0.05763→0.06172 m`；false-visible仍为5，门要求≤4 | 采集8/8完整成功；重标18/20成功，AT300 step77和STT700 step105按质量门拒绝；v2_016没有best，step128 SHA-256 `329b589f...` | waypoint与停车轨迹有真实改善，但可见性唯一失败，不能晋升；转纯head校准以隔离因果 |
| EXP-037 | 2026-09-15 | 在冻结全部waypoint主链的条件下校准visibility/stop heads，并证明离线变化不是轨迹头漂移 | `v2_017_phase3_safety_head_calibration`；从v2_016 step128开始；仅两个安全head；32步被选中 | 与EXP-036相同train/recovery-val；部署visibility阈值固定0.005；不访问`test_locked` | 20260915 | false-visible `5→4`、false-invisible `3→1`、visibility accuracy `38.46%→61.54%`；stop accuracy仍`0.53846`；waypoint所有指标漂移0 | 冻结参数哈希前后`edf90f95...`；best SHA-256 `ab8d8063...`；四类专用可视化显示修复及残留错误，waypoint逐坐标差0 | 仅离线gate通过；stop未解决且必须等待全新index闭环，不得直接部署 |
| EXP-038 | 2026-09-15 | 在完全未见index1300上区分v2_016 waypoint学习与v2_017 safety校准的闭环因果 | v2_006b/v2_016/v2_017 × `polar_reactive/se2_waypoint` × STT/DT/AT；GPU3串行18条；visibility阈值0.005；UWB missing | Habitat `train/index1300`；同一scene `00560-gjhYih4upQ9`、episode 3严格配对；每条最多130步；无GT target point；不访问`test_locked` | 20260915 | 全部SR 0、碰撞0。polar TR为`0.1346/0.1275/0.1531`；se2 TR为`0.0781/0.0916/0.0350`。v2_017 se2三条机器人总路程`0 m`、visibility above-threshold帧`0/0/0`；v2_016对应总路程`2.5997 m` | 18/18完成、0失败；汇总SHA-256 `0804543a...`；门禁在polar因false-invisible率升高失败，在se2又因TR下降超过0.02失败；97帧STT对比视频完整解码 | v2_017拒绝晋升；离线13点校准明显过拟合。index1300从此转为已消费诊断集，下一轮必须另留全新闭环holdout |

建议每个实验至少记录：

- 唯一配置文件或完整参数快照；
- 代码 commit/hash 与未提交修改；
- 数据版本、采样比例和过滤规则；
- checkpoint 路径及 checksum；
- clean、occlusion、distractor、noise、delay、recovery 等分项指标；
- 与最近有效 baseline 的唯一变量差异。

## 10. 失败与负结果

| FAIL-ID | 关联实验 | 现象 | 可复现条件 | 根因证据 | 已尝试修复 | 结果 | 防止重复方式 |
|---|---|---|---|---|---|---|---|
| FAIL-TEMPLATE | EXP-XXX |  |  |  |  |  |  |
| FAIL-001 | EXP-001 | DA3实际推理尚不能在`omtrackvla`环境启动 | 导入DA3官方API所需运行时 | 官方仓库固定commit `3d835ec1...`；原环境缺`huggingface_hub`、`safetensors`及若干导出依赖，且本机没有模型权重 | 部署独立源码、Python target运行时和SHA-256固定的DA3-SMALL权重；不改原conda环境 | EXP-002已真实加载权重并完成GPU推理；历史失败保留，NEXT-009接口gate解除 | 固定源码/模型revision、权重hash和隔离路径；不得把可选`gsplat`警告误判为几何probe失败 |
| FAIL-002 | EXP-002 | 首次真实DA3 pose转换出现近90度平移方向和yaw符号错误 | 直接将DA3 OpenCV w2c与Intern/Habitat相机外参组合 | 首次报告translation/yaw均值误差0.254m/0.171rad；四种变换数值对照仅“OpenCV/Habitat桥接+canonical基变换”同时恢复方向和符号 | 增加`diag(1,-1,-1,1)`相机轴桥接、`(right,forward)→(forward,left)`及非单位真实形态外参回归测试 | 同clip经robust scale后降至0.00496m/0.00519rad | 单位外参测试不足；后续坐标适配测试必须包含真实轴向、非零相机高度、平移和旋转 |
| FAIL-003 | EXP-003、EXP-004 | SAGE3D中后段黄色GT框频繁落在人物另一侧的墙、门或窗上，另有`visible=true`但框在画面外 | Phase 1 smoke人工查看及384帧分层detector审计 | 抽取器定义`right=[-sin(yaw),cos(yaw)]`，实际为left，却使用`u=cx+fx*right/fwd`；水平镜像反事实将冲突69降至14；5个源框裁剪后零面积 | 以正确右轴和完整相机平移重投影，增加画面相交及RGB对齐depth的support/近遮挡判定，生成外置全量侧车并独立审计 | 源`bbox/visible`继续禁止；修复侧车7,105/7,105完成、0失败并通过准入，样本冲突率从原18.21%降至2.45% | 坐标投影测试必须覆盖非中心目标、左右两侧、出画、遮挡和多相机；adapter必须fail-closed校验完整性、准入和源/episode hash |
| FAIL-004 | EXP-005 | DA3使用单一全局metric scale时，held-out admitted bad rate为24%，超过冻结上限20% | 31个`val`和31个互斥`viz_val` clips，global scale 1.879207、relative IQR 0.462809、confidence阈值2.024682 | D435i与ZED的相机族尺度分布不同；其余gate均通过，失败集中在`heldout_bad_rate`而非推理完整性 | 改为仅按manifest中已知相机族分层，在`val`分别冻结D435i/ZED尺度并重新拟合统一confidence阈值 | 分层规则在`viz_val`将bad rate降至5.26%，随后唯一locked结果16.67%并通过 | 禁止回退到global scale；policy固定`intern_camera_family`、两类尺度、split用途和所有阈值，CLI偏离即在推理前失败 |
| FAIL-005 | EXP-006、EXP-007 | Phase 1 v1把全部absent样本预测为visible；v2仍有77.61% absent误报 | 固定val的4,096个身份样本，其中67个absent | v1按源帧数比例采样压低TpT负例，且当前帧在自身可见性判定前污染身份记忆；v2修复后二类概率趋于饱和，固定0.5运行阈值不适合低误报目标 | 身份源等概率采样、absent loss 10×、排除当前帧记忆；仅用train split冻结0.999阈值 | v3 val FPR 0.16418、visibility accuracy 0.86401，均通过原gate | 禁止在val搜索运行点；阈值必须在train冻结并写入checkpoint，评测只读取 |
| FAIL-006 | EXP-007 | v2的B1-PROBE相对改善为-1.27198，尽管直接几何指标优于v1 | 256个train和512个val probe样本，原始feature直接使用固定ridge正则1e-3 | v1/v2/random feature平均范数分别47.33/27.76/13.12，固定绝对正则比较受尺度和条件数支配；使用各自train统计标准化后，v2/random MSE为0.1033/0.3382 | 对trained与random表示分别使用各自train均值/方差标准化，再以同一ridge和val标签比较 | v3相对改善+0.69457，通过未降低的`>=0` gate | linear probe必须仅用train统计预处理；报告记录标准化方法，禁止用val统计 |
| FAIL-007 | EXP-010、EXP-011 | 全部train序列训练后，单一score 0.95/margin 0.06使可见召回仅36.30%，E2E仅33.20% | ResNet50+OSNet+融合头在TpT两个完整`viz_val`序列 | 连续跟踪一次保守拒绝后即进入全局搜索；全局搜索和连续跟踪共用严格阈值，错误代价不对称却使用同一运行点 | 仅按train-calibration记录中的`global_search`分区冻结双阈值；不在`viz_val`搜索 | E2E升至46.30%、可见召回48.62%、wrong-target降至60、输出正确率90.68% | 后续状态相关阈值必须在train上冻结并写入模型；`viz_val`只做固定评测，失败版本与指标继续保留 |
| FAIL-008 | EXP-020 | KPR完整val虽较严格静态替换提高约10倍，但precision仅74.67%～79.05%、重现仅4.14%～5.92%，且动态记忆一次也未更新 | 官方KPR冻结特征、train-only anchor/global阈值、显式tracklet floor和3/4帧重获；5条val全帧 | 37条train中KPR正样本anchor P50为0.4643，但继承的memory identity/anchor阈值仍为OSNet尺度0.82/0.72，导致`memory_updates=0`；低tracklet floor又让`0013/0038`错误目标靠空间连续性持续 | 已完成KPR距离、tracklet floor和多帧重获接入；尚未校准图库写入/漂移终止 | v2/v3均未达到88% precision safeguard，NEXT-023继续阻断 | 后续backend接入必须连同记忆写入与漂移阈值做train-only尺度审计；报告显式检查非零memory update和分序列漂移，不能只看anchor ROC或总E2E |
| FAIL-009 | EXP-027 | 直接监督总path length时，aggregate路径比显著改善甚至接近1，但固定图出现左右锯齿和局部回退，轨迹不自然 | 从EXP-026 checkpoint低LR精修；path-length权重1.0或0.1；同一SAGE3D train/val与固定initial/anchor 0/106图 | 总路径长度只约束相邻段长度之和；模型可以通过横向摆动或折返增加总长，而不增加沿专家方向的末点进度 | 尝试terminal+delta、terminal-only和普通低LR；最终实现沿专家方向的逐horizon有符号进度loss，并增加“总路径相同但回到原地”的单测 | 径向进度候选路径比+4.50%，固定图所有step向前；正式ADE/FDE代价约1.2%，作为Pareto候选保留 | 禁止仅按path length ratio选模；必须同时报告ADE/FDE、逐horizon半径、向后step/回退及固定可视化；负结果不得覆盖 |
| FAIL-010 | EXP-029 | 13臂direct 4k初步矩阵除single-step外几乎全部落在相同短轨迹平台，若直接排序会把初始化/收敛差异误当成模块因果 | DA3官方权重+随机policy直接初始化；每臂4,096步；主对照曾错误与带不同初始化链的旧4k模型比较 | GRU从第0 policy step即形成额外优化瓶颈；4k不足以让多数臂越过收缩解，且旧对照混入Architecture-v1 Phase 1初始化差异 | 补跑direct GRU/single-step各36,864步，增加分policy-step与清零hidden诊断；尝试近恒等direct初始化并保留其失败结果，最终采用两段课程 | single-step在等预算direct比较显著胜出；课程GRU再小幅反超且hidden反事实成立 | 所有`next026_formal_v1` 4k结果改称diagnostic；不得用于论文正式模块收益表，正式矩阵须使用收敛日程与一致初始化口径 |
| FAIL-011 | EXP-031 | 闭环初版看似每步执行动作，但策略持续收到旧RGB，图像变化与机器人/目标真实运动不一致 | 对比`env.step()`返回观测、传感器时间序列和动作后直接读取；逐步记录RGB绝对差 | 当前Habitat包装器的`env.step()`返回缓存的pre-step RGB，不能代表动作后的最新传感器帧 | 每次动作后显式调用`env.sim.get_sensor_observations()`，下一决策只消费该刷新RGB；保留stale/fresh差分诊断 | v6前27步目标可见期的强制刷新RGB全部非零变化，输入审计通过 | 闭环runner必须强制刷新并报告旧/新帧差异；不能因画面能渲染就假设策略输入是新的 |
| FAIL-012 | EXP-031 | 60步闭环在第51次`env.step()`稳定触发Habitat原生abort，导致JSON/视频不完整 | v3用原视频路径、v4关闭imageio/OpenCV并逐步保存PNG；同一固定episode | 与视频编码无关；第51帧PNG已落盘而第51个动作未完成，崩溃边界是模拟器原生step调用 | 当前开发episode固定50步；PNG spool后隔离FFmpeg，结果显式写requested limit和termination reason | v6退出码0，50步JSON完整；H.264 768×384@8fps共50帧可解码 | 不把50步上限伪称任务自然结束；更长episode须更换/修复Habitat运行契约后另测 |
| FAIL-013 | EXP-031 | 模型在墙角遮挡后丢失人物，随后高响应区域落到墙柱且没有主动重获 | v6无UWB；第3/11/19步正常跟随，第27步进入遮挡，第28步起不可见，第35/43步墙柱误框 | Phase 2主要从expert访问状态做open-loop训练，缺少模型走偏/遮挡后的访问状态和恢复监督；无UWB时又没有可靠搜索方向 | 先用低置信全停保证安全；下一步验证任意状态expert relabel并采集closed-loop失败状态，做Phase 3恢复训练/DAgger | 0碰撞、0低置信非零违规，但following rate仅52%、reacquisition 0 | 安全与能力分开判定：当前不能继续盲调安全门或宣称闭环成功，必须补恢复数据闭环 |
| FAIL-014 | ABL-V1-09 | Frozen Frontend cache在4,910个文件处自行退出，三个分片首先报`missing admitted sidecar metadata` | 7分片遍历Phase 1的SAGE3D run split；根索引含7,110个目录，准入sidecar仅含7,105个accepted episode | cache builder按run从根index取全部episode，未执行canonical acceptance边界；缺失的不是损坏sidecar，而是源明确rejected episode | 检查全部5条拒绝记录与`_ACCEPTED`；只允许双证据明确拒绝时结构化跳过，其余missing继续fail-closed；合并器增加完整记账 | 20项远端专项测试通过；原4,910 cache保留，2026-09-13 18:50已在GPU1～7断点续跑 | 最终episode数以合并manifest记账为准；不得再把索引目录数当可训练数，也不得用无条件skip修复 |
| FAIL-015 | EXP-032 | 首版模型失败状态expert虽无碰撞且replay精确，但8点path 4.2205 m、末点3.5902 m，远快于SAGE3D 0.7 s轨迹分布，不能作为恢复标签 | NEXT-027 v6前28步真实模型动作 + OracleNavmeshFollowerV6默认速度 | oracle控制器面向闭环跟随，默认单步平移幅度不等于Architecture v1训练标签的短时速度分布；原gate只检查有限值、8点和碰撞，没有长度/方向/相对静止anchor进度 | 将forward/lateral/slew限制为0.10；增加path、terminal、最大段、最终目标位置、相对静止anchor距离改善和方向余弦审计；保存raw RGB后做waypoint-only backward | v4 path 0.704529 m、terminal 0.703875 m、相对静止anchor改善0.342163 m、方向余弦0.533952、0碰撞，单样本主链梯度全部非零 | DAgger expert标签不能只因“oracle能跑”就准入；必须同时匹配时域/动作分布，并证明动作在移动目标场景下有相对跟随收益 |
| FAIL-016 | EXP-034 | AT train index 400的post-step-9/11/12和21-step前瞻版本均未通过相对静止anchor进度/方向门 | 12步完整fresh-RGB模型rollout；精确replay；修正后的底盘坐标；reactive与未来目标expert | 人物快速横穿且当前位置到未来位置的navmesh最短路在21步窗口内先绕障碍；有效expert轨迹虽无碰撞，终点仍比机器人原地不动多落后`0.082 m`，方向余弦`-0.094` | 不放宽gate；保留所有失败report；改采STT 0、DT 200、AT 401三个独立train状态，并让未来预演和标签环境使用独立config/dataset实例 | 三个替代样本全部通过且相对静止anchor改善约0.49m；AT-400未进入sample或训练 | 模型访问状态不是天然可训练样本；绕行或过快横穿状态必须由质量门拒绝，不能靠改阈值硬收 |
| FAIL-017 | EXP-035 | v2_013训练64步后recovery ADE变差，4个false-visible一个也未减少 | 单一小规模可观测性pilot；固定独立recovery-val与部署阈值0.005 | 仅把可见概率整体压低不能区分真正可见、近期丢失和长期安全停车状态，且样本场景覆盖不足 | 不选择checkpoint、不放宽gate；转多场景采集、显式状态重标和scene/state双隔离 | v2_013正常结束但无best；失败报告保留 | safety概率必须按可观测状态和场景泛化验证，不能以logit方向变化代替分类运行点改善 |
| FAIL-018 | EXP-036 | v2_016同时改善recovery/expert waypoint与safe-stop path，却因false-visible仍为5而未生成best | 20 train/13 recovery-val、6/2个隔离场景、128步同配方训练 | waypoint与visibility并非同一优化问题；继续更新全模型可能为修复一个阈值指标破坏已经改善的轨迹主链 | 从step128启动只训练visibility/stop heads的v2_017，并要求所有waypoint参数哈希和指标不变 | v2_017离线false-visible降至4且waypoint漂移0，但后续闭环仍失败 | 辅助安全头可以隔离校准，但离线小样本通过不构成部署晋升证据 |
| FAIL-019 | EXP-037～038 | v2_017在13个离线recovery-val样本通过，却在新index1300的三条se2-waypoint闭环全程停车，机器人总路程0 | UWB missing、visibility threshold 0.005；STT/DT/AT同episode；v2_017每条94/94、97/97或94/94步均无一次超过阈值 | 20/13个离散锚点的纯head校准发生场景过拟合；head权重改变使新场景概率中位约`0.00023～0.00026`，低于冻结阈值，严格visual failsafe遂持续输出零动作 | 未放宽阈值、未关闭安全门、未在index1300重训；保留v2_016和v2_017全部结果并拒绝晋升 | polar执行器因独立搜索状态机仍有`0.1531` TR，但se2 TR降至`0.0350`且SR仍0，整体gate失败 | 下一轮必须用更广的连续模型访问状态训练/校准，并在训练前冻结新的闭环holdout；同一index不得用于调参后再报泛化 |

失败原因应区分：

- 数据或标签错误；
- 坐标系/时间同步错误；
- 身份漂移；
- 条件噪声或置信度失配；
- 专家状态与 rollout 状态分布偏移；
- 训练不稳定或灾难性遗忘；
- 闭环控制或仿真接口问题。

## 11. 修改记录

| CHG-ID | 日期 | 修改对象 | 修改内容 | 原因 | 验证方式 | 关联决策/实验 |
|---|---|---|---|---|---|---|
| CHG-001 | 2026-09-04 | `PROGRESS.md` | 创建项目进展文档，记录当前任务定义、训练阶段提案和数据需求 | 建立持续可追溯的研发记录 | 文档检查 | DEC-001～DEC-007 |
| CHG-002 | 2026-09-07 | `PROGRESS.md` | 固定一次视觉初始化、UWB-only冷启动、自动视觉绑定与双信号失效停车；重写Stage 0数据边界并将Stage 2拆为2A/2B | 消除输入定义和训练监督来源歧义 | 全文一致性检查 | DEC-013～DEC-022 |
| CHG-003 | 2026-09-07 | `PROGRESS.md` | 新增逐Phase benchmark、数据隔离、gate、可视化及8×H100一键流水线规范，并统一Phase 1/2/3对外编号 | 保证协作者可复现且每阶段结果可人工检查 | 文档结构与一致性检查 | DEC-023～DEC-027 |
| CHG-004 | 2026-09-07 | `OmTrackVLA/scripts`、`configs/pipeline`、`docs` | 新增单节点8×H100实际启动器、集中配置和使用/接口说明 | 将文档中的一键流水线约定落成可执行基础设施 | 远端`bash -n`及三Phase dry-run通过 | DEC-027 |
| CHG-005 | 2026-09-07 | OmTrackVLA Git历史 | 创建`wam`分支；先提交当前模块化精简基线，再单独提交8×H100流水线 | 将此前工作区状态冻结为可复现起点 | 102项pytest、Shell语法、diff check及凭据模式扫描通过；工作区干净 | DEC-028 |
| CHG-006 | 2026-09-07 | 4090 GitHub连接与`origin/wam` | 配置OmTrackVLA仓库专用Deploy Key别名，将origin切换为SSH并推送`wam` | 允许集群机以仓库级权限拉取和推送 | SSH认证成功；`ls-remote`与本地HEAD均为`45af91e2` | DEC-028 |
| CHG-007 | 2026-09-07 | 仓库根目录`PROGRESS.md` | 将外层项目进展文档纳入`wam`，新增协作者接手入口、实际仓库/数据快照、顺序工作包和明确阻塞项，并修正pipeline配置后缀 | 让协作者只读本文件即可知道当前事实、下一项工作与验收产物 | 路径/数据计数核对、Markdown检查、Git diff和测试 | DEC-027～DEC-028 |
| CHG-008 | 2026-09-07 | `PROGRESS.md` | 将正式外部数据范围纠正为完整InternData-N1、Sage3D extracted和TpT clean v2，记录当前容量、规模、schema事实及已发现风险 | 用户确认正式数据根；旧文档错误地把三个TpT目录列为候选，且InternData-N1旧统计漏扫 | g0014只读路径/mtime/`du`/parquet/JSON元数据复核及Git diff | NEXT-001 |
| CHG-009 | 2026-09-07 | `docs/data_inventory.md`、`configs/data_inventory.json`、`scripts/audit_data_inventory.py`、测试 | 完成WP-0：冻结三套正式数据的schema、规模、accepted规则、split unit、禁用字段与时钟风险；实现可复现只读审计 | 训练前必须有可执行、可追溯且不改源数据的盘点 | 全量元数据/路径检查；Intern/SAGE/TpT分别48/108/47个媒体解码；manifest对比；单元测试 | NEXT-001 |
| CHG-010 | 2026-09-07 | `example_datasets/`、`.gitignore` | 审计旧样例视图并只发布结构说明/审计记录，不发布原始媒体和绝对symlink | 旧视图解引用约1.80 GB且含旧TpT、生成crop/cache；公开仓库未找到数据再分发许可，源机无Git LFS | 27,750文件、扩展名/最大文件/symlink/凭据/嵌套仓库检查；Git ignore检查 | NEXT-001 |
| CHG-011 | 2026-09-07 | `example_datasets/`、`scripts/export_example_datasets.py`、`.gitignore`、测试 | 按用户确认改为发布可审计微型真实子集：三套数据各16连续帧，含必要结构/元数据切片和GitHub预览；增加校验和与数据声明 | 协作者需要直接认识真实数据外观和结构，而不是只看schema文档 | 5,854,999字节/99文件；98文件SHA-256通过；48 RGB、32 depth、3 preview和2 Parquet全部重读；无symlink/嵌套仓库/凭据特征/禁入派生文件；预览人工核对 | DEC-029 |
| CHG-012 | 2026-09-07 | `docs/data_contract.md`、`configs/data_contract.json`、`scripts/validate_data_contract.py`、`docs/data_inventory.md`、测试 | 完成WP-1：冻结模型输入、路由元数据、policy/identity监督、坐标/时间、缺失值、四种条件模式、安全约束和source admission gate | loader和训练模块必须共享一个可执行边界，不能各自猜测bbox、UWB、waypoint或源时钟语义 | 契约自检、有效policy/identity样本、禁止后续bbox、UWB年龄/协方差、轨迹mask、RGB故障停车、路径越界/只读存在性及source gate测试；全仓122项unittest通过 | DEC-030；NEXT-002～004 |
| CHG-013 | 2026-09-08 | `omtrackvla/{data,geometry,models,training,evaluation}`、Phase 1配置、manifest、脚本、文档和测试 | 实现WP-2首个可执行闭环：只读identity/geometry adapter、显式SE(2)转换、带GRU历史身份记忆的共享视觉双流baseline、分布式训练、B1评测、视频渲染和严格gate | 将流水线从dry-run推进到真实Phase 1，同时保持源准入和输入/标签边界 | 133项unittest；真实三源adapter smoke；单卡2-step和8×H100每卡1-batch DDP；B1-ID/B1-GEO/B1-PROBE；8帧MP4重读；正式gate预期失败；Phase 1 pipeline preflight通过 | DEC-031～032；EXP-001；NEXT-009～011、015～018 |
| CHG-014 | 2026-09-08 | DA3隔离运行时/权重、`omtrackvla/geometry/se2.py`、probe脚本、测试和文档 | 部署固定DA3-SMALL并完成真实GPU probe；修复OpenCV/Habitat轴桥接和canonical基变换；报告scale、置信度和feature形状 | 单元合成测试未覆盖真实相机坐标约定，首次实际pose暴露轴向错误 | 固定权重SHA；4帧真实推理；四种变换对照；非单位相机外参回归；全仓测试 | DEC-033；EXP-002；FAIL-001～002；NEXT-009～010 |
| CHG-015 | 2026-09-08 | `PROGRESS.md` | 同步协作者事实表与Phase 1最新实现状态，更新日期和DEC-027状态，并增加现有可视化位置及判读边界 | 消除“Phase 1已实现”与旧表“训练模块不存在”的内部矛盾，防止将数据预览或2-step smoke误判为正式模型结果 | 全文一致性检查、路径核对、135项测试和流水线dry-run | DEC-027、DEC-031～033；EXP-001～002 |
| CHG-016 | 2026-09-08 | `scripts/audit_sage3d_bboxes.py`、`docs/sage3d_bbox_audit.md`、测试和`PROGRESS.md` | 新增SAGE3D bbox只读分层审计、独立person detector对照、水平镜像反事实、时间/mode-camera统计及最差样本拼图；记录准入阻断和修复顺序 | 用户发现SAGE3D黄色GT疑似错位；正式训练前必须区分模型误差与源标签错误 | 384帧/128 episodes/18分层H100审计；5项单测；人工查看72个最差样本；源目录零写入 | DEC-034；EXP-003；FAIL-003；NEXT-021 |
| CHG-017 | 2026-09-08 | `omtrackvla/data/sage3d_sidecar.py`、sidecar build/audit脚本、Phase 1 adapter/config、`.gitignore`、文档和测试 | 实现正确投影、深度可见性、原子/可恢复的外置侧车生成、完整性与质量准入；训练只读准入侧车并在任何缺失/hash不符时fail closed；重新启用SAGE3D identity流并忽略大型运行产物 | 修复FAIL-003且不覆盖只读源，阻止未完成、被篡改或意外提交的标签进入训练/Git | 全量7,105 episode/2,131,500 step、0失败；独立384帧gate通过；真实adapter读取2,990个单run anchor；算力机149项unittest通过；本地拼图人工复核 | DEC-034～035；EXP-004；FAIL-003；NEXT-021 |
| CHG-018 | 2026-09-08 | `omtrackvla/geometry/da3_admission.py`、多场景审计脚本、冻结policy、文档和测试 | 实现确定性多场景选择、相机族尺度标定、退化/置信度过滤、held-out gate、开发可视化和locked admission；所有CLI值对policy fail closed并记录policy hash | 单clip同轨标定不能证明跨场景尺度；global scale实测超过bad-rate上限；正式pseudo label必须有不可回退的准入证据 | 真实31+31开发审计、人工`viz_val`复核、唯一31+32 locked审计；算力机159项unittest通过 | DEC-033、DEC-036；EXP-005；FAIL-004；NEXT-010 |
| CHG-019 | 2026-09-08 | Phase 1 data/model/train/eval、gate及pipeline cleanup | 身份源改为等概率采样，absent visibility loss 10×，当前帧不再写入自身判断前的身份记忆；增加invisible样本gate；pipeline失败退出清理PID lock | 修复v1恒预测visible及失败残留`.pipeline_lock` | v2正式4,096步、67个invisible样本；bbox/几何提升；161项unittest和Shell语法通过；FPR/probe负结果继续保留 | EXP-006～007；FAIL-005～006 |
| CHG-020 | 2026-09-08 | Phase 1模型配置、B1评测与`viz_val`渲染 | 将train-only校准的0.999 visibility运行点写入checkpoint；linear probe使用各自train feature统计标准化；视频固定包含8个absent案例并显示PRED/GT可见性 | 修复v2阈值失配、probe尺度偏置和短视频几乎看不到absent案例 | 提交`5103f88`；163项unittest；v3正式12/12 gate；64帧960×380 MP4完整解码并人工抽查absent帧 | DEC-037～038；EXP-008；FAIL-005～006 |
| CHG-021 | 2026-09-08 | `PROGRESS.md` | 追加正式Phase 1 v1/v2失败与v3通过记录，更新当前状态、下一工作包、可视化和产物索引 | 保留负结果并让协作者从单一文档获得最新接手点 | Markdown/diff检查；指标、commit、checksum和运行目录与算力机产物逐项核对 | EXP-006～008；WP-2 |
| CHG-022 | 2026-09-09 | `rgb_person_perception.py`、候选融合模型/训练/连续序列评测、overnight脚本、融合权重、测试和`PROGRESS.md` | 接入冻结Faster R-CNN与OSNet，增加不可覆盖anchor、正/负图库、候选诊断、train-only小型融合头、跟踪/重获双运行点及TpT完整序列指标/五类可视化 | 避免从零训练检测/ReID，并用可分解指标定位“检测不到”与“候选中选错”；修复单阈值过保守 | 37 train序列、399,247候选；5,928帧`viz_val`；174项unittest；评测JSON显式记录双运行点；80张图片零解码失败；融合JSON SHA-256 `03a78884675b20c09ab8c6a7c5bd945355678e74bd057804a6bd823e45db06f7` | DEC-039～040；EXP-009～011；FAIL-007 |
| CHG-023 | 2026-09-09 | SAGE3D Phase 2 policy几何/时钟helper、只读审计、gate、数据契约、说明和测试 | 固化world到base x-forward/y-left变换、30 Hz、8点`0,+3,…,+21`重算及源waypoint禁用；将SAGE3D policy角色从blocked改为条件准入 | Phase 2必须先证明expert轨迹坐标与时间，且发现源waypoint与WP-1 anchor约定偏移一帧 | 256个`val/viz_val` episode、13分层、11/11 gate；未运行/查看locked；报告SHA-256 `ec90f757...` | DEC-041；EXP-012；WP-3 |
| CHG-024 | 2026-09-09 | Phase 2 data/model/train/eval/render/gate、3个配置、冻结感知cache及8卡分片/merge脚本 | 接通四模式SAGE3D policy；冻结前端cache绑定全部准入哈希并拒绝partial误用；safe-stop硬归零；输出ADE/FDE/分模式/安全指标及逐张PNG+MP4 | 复用现成检测/ReID后仍需可训练的waypoint闭环，并防止bbox标签、非视觉confidence或旧cache泄漏进policy | 算力机192项unittest；Phase 2 8×H100 preflight；train/val/viz 1-episode smoke；2 GPU真实分片和manifest merge；16 PNG/MP4解码及人工抽查 | DEC-042；EXP-013；WP-3 |

| CHG-025 | 2026-09-09 | `PROGRESS.md` | 记录Phase 1与冻结感知前端的同协议连续跟踪对照，将Phase 2拆为2A感知适配与2B waypoint SFT，并使旧fusion v3 cache失效 | 用户要求先训练感知以提高跟踪准确率，再进入下一阶段；现有46.30% E2E尚有明确提升空间 | 文档事实、实验计数、非locked边界和准入门槛一致性检查 | DEC-043；EXP-014；NEXT-022～023 |
| CHG-026 | 2026-09-09 | Phase 2A训练器、独立gate、时序feature contract及7卡流水线 | 冻结detector/OSNet；加入hard-negative排序、sequence-balanced BCE、train-only双阈值、val选模及固定viz gate；补充跨rollout帧隔离、空hard-pair与precision fail-closed | 建立感知适配与waypoint训练之间的可执行边界，并防止数据泄漏或弱模型静默进入2B | 算力机198项unittest；合成权重写入/重载smoke；v1正式val按预期拒绝 | DEC-043～044；EXP-015；NEXT-023 |
| CHG-027 | 2026-09-09 | Phase 2A基线保持式初始化、val排序选模、新列梯度掩码及运行点来源配置 | 将legacy融合函数精确映射到新归一化空间；epoch 0参与选模；支持只更新新时序列及继承既有双运行点 | v1排序退化、v2阈值退化分别暴露两种独立回归路径，需同时约束 | 算力机199项测试通过v2初始化；新增测试验证legacy logits等价及legacy参数训练不变 | DEC-044～045；EXP-016；NEXT-023 |
| CHG-028 | 2026-09-09 | Phase 2A 7卡流水线的正式train时钟与rollout隔离 | train stride由4改为1并校验正整数；默认每个run使用独立baseline rollout root，防止误复用旧stride记录 | 新增时序状态以tracker调用步演化，训练与评测stride不一致会形成隐含时钟偏移 | shell语法、全仓测试及新v4独立输出root检查 | DEC-046；EXP-017；NEXT-023 |
| CHG-029 | 2026-09-09 | Phase 2A train-only on-policy dataset aggregation流水线 | 支持以候选fusion额外生成同stride train rollout，将多个独立record root合并训练；显式校验权重路径与epoch参数 | v4证明固定基线状态上的离线排序改善不能保证在线跟踪改善 | shell语法、跨root帧键隔离测试、全仓测试；v5输出root独立 | DEC-047；EXP-018；NEXT-023 |
| CHG-030 | 2026-09-10 | `kpr_reid.py`、KPR/tracklet评测与train-only校准脚本、测试和`PROGRESS.md` | 动态加载官方KPR/SOLIDER冻结推理，保留共同可见部位距离；加入结构化part embedding、连续tracklet floor和多帧重获；完成37条train校准及两个完整val版本 | 逐帧OSNet fusion家族已停止，需要比较更强冻结ReID并转向显式序列关联 | 官方crop smoke；76项相关unittest；28,888个train帧；12,457个val帧；两个完成标记和overnight report | DEC-048～049；EXP-020；FAIL-008；NEXT-023 |
| CHG-031 | 2026-09-10 | EVT人物识别对照、可恢复低并发runner、`scripts/summarize_evt_reid.py`及逐步评测原始计数 | 将EVT作为主评测并避免Habitat高并发失败污染模型结论；区分同一pre-action RGB的GT可见性与post-action task visibility；报告micro计数、分任务指标、候选上限、图库污染和配对轨迹一致性 | OSNet 6条完整报告；KPR 4条partial与理论上界；本地2项summarizer测试及语法检查；算力机脚本语法检查 | DEC-050～051；EXP-021；NEXT-022～023 |
| CHG-032 | 2026-09-10 | `PROGRESS.md`、本机`EVT_Bench_ReID汇报表_2026-09-10.csv` | 将EVT快速选型整理为21字段、6行的UTF-8 BOM汇报表，覆盖OSNet总体/分任务、KPR partial/理论上界、实验状态与限制；进展文档升级为v29并登记产物 | 为后续学术汇报提供可直接用Excel打开、且能追溯到EXP-021原始计数的结构化摘要 | CSV成功解析为21列、6行；各行字段数一致；中文编码、KPR空值及理论上界状态检查通过 | DEC-050～051；EXP-021 |
| CHG-033 | 2026-09-10 | 全量EVT runner/汇总器、通用8GPU runner、`oracle_modular_batch.py`及`PROGRESS.md` | 增加OSNet/KPR后端参数透传、逐步结果保存、官方max-step写入manifest、28分片可恢复全量val启动器，以及强制4,215/方法完整性、配对轨迹、micro/macro指标和UTF-8 BOM CSV的汇总gate；暂停提前启动的Phase 2B | 修正将6条快速预实验用于最终选择的证据不足，确保正式汇报覆盖全量val且失败不能被静默跳过 | 本地4项纯汇总测试；算力机Bash/py_compile及13项相关unittest通过；tmux实跑28 worker均启动并写入持久化episode结果 | DEC-052；EXP-022；NEXT-024 |
| CHG-034 | 2026-09-11 | `evt_reid_full_val_v1`故障恢复、严格配对复核、正式报告、本机CSV及`PROGRESS.md` | GPU 1、2、4～7在4-worker和单worker健康测试中均无法完成Habitat/EGL episode，故保持28分片并改由GPU 3的4 worker断点续跑；严格gate捕获1个故障GPU黑帧结果，归档后在GPU 3真实重跑；重新生成报告并冻结OSNet | 不把进程存活、JSON存在或未报错误当有效结果；最终前端必须建立在全量、完整、同轨迹证据上 | 两方法各4,215/4,215、0错误/缺失；4,215组轨迹一致；`REPORT.json` SHA-256 `33e294191bf3b679...`，`REPORT.csv` SHA-256 `df902ab41a13d0cc...`；`COMPLETE`存在；本机CSV 8行30列可解析且hash一致 | EXP-022；NEXT-022～024 |
| CHG-035 | 2026-09-11 | `PROGRESS.md`端到端主线纠偏与协作者交接入口 | 明确最终I/O和禁止推理输入；记录Phase 1→Phase 2实际迁移为0；将OSNet/cache/MLP降级为`Frozen Frontend Baseline`；新增统一模型、联合训练和closed-loop三个P0工作项 | 用户明确要求真正的RGB历史到waypoint端到端方法，现文档却仍让协作者优先恢复冻结cache，方向与研究目标冲突 | 对Phase 1/2模型、训练器和配置做静态数据流/checkpoint/梯度边界审计；全文检索旧“下一步恢复cache”表述并更新；不改历史实验记录 | DEC-053～054；EXP-023；NEXT-025～027 |
| CHG-036 | 2026-09-11 | 仅更新`PROGRESS.md`，冻结`Architecture v1` | 记录DA3-SMALL L11、单Target/World/UWB Token、GRU、8×2回归、显式SE(2)瓶颈、几何UWB Gaussian bias及World-Action B；列明部署/训练路径、tensor shapes、总loss和9组强制消融；纠正旧ResNet18 Phase 1不可伪装成DA3-v1权重迁移 | 用户确认架构收敛，要求开工前形成唯一实现规范并暂不大规模训练 | Markdown结构/字段检查、部署禁用项检索、shape/loss/ablation一致性复核；本次未改代码、未启动训练、未访问`test_locked` | DEC-055～057；Architecture v1；NEXT-025～027 |

| CHG-037 | 2026-09-11 | `omtrackvla/models/end_to_end.py`、`omtrackvla/data/end_to_end.py`、NEXT-025 smoke/config/doc及测试 | 实现DA3 L11 Target/Scene Attention、camera-token pair显式SE(2)瓶颈、UWB camera-geometry Gaussian patch bias与独立`z_uwb`、Fusion MLP、GRU和waypoint/stop；冻结DA3并在最后两层注入adapter；增加已准入SAGE3D train batch的cache-free loader、权重/参数/waypoint-only梯度报告 | 将Architecture v1落成最小可执行主链，并排除辅助loss替代waypoint主损失进入视觉backbone的假阳性 | py_compile；10项专项及全仓220项unittest；真实SAGE3D CPU stub、单H100官方DA3及一步8×H100 DDP smoke；forward禁用输入静态审计、diff check及报告SHA-256复核；未访问`test_locked` | DEC-053～057；EXP-024；NEXT-025 |
| CHG-038 | 2026-09-11 | `omtrackvla/evaluation/end_to_end_render.py`、`scripts/render_end_to_end_v1_pretrain.py`、测试及NEXT-025说明 | 新增正式训练前定性dashboard：并列展示初始化/history RGB、Target/Scene Attention、独立UWB camera-geometry prior、UWB 2σ、预测/专家轨迹、stop/SE(2)原始值及输入隔离元数据；输出PNG和机器可读JSON | 用户要求在初始权重很差时也先可视化检查，防止正式训练后才发现bbox、坐标、UWB或渲染语义错误 | 2项renderer单测、真实H100渲染、PNG回读/尺寸/hash、本机实际目视检查；0 optimizer step，未访问`test_locked` | EXP-025；NEXT-026 pre-training gate |
| CHG-039 | 2026-09-11～12 | Architecture v1 Phase 2正式训练/eval/render配置与训练期heads checkpoint加载 | 支持从Phase 2 checkpoint继续加载training-only dynamics heads；完成4k到36,864有效步长训、每模式1,024正式评测和固定dashboard | 用户准入初始图后要求正式训练，并要求持续抽查是否有效 | 训练完成标记、checkpoint/hash、固定128与正式1,024评测、PNG解码和本机目视检查；未访问`test_locked` | EXP-026～027；NEXT-026 |
| CHG-040 | 2026-09-12 | waypoint delta/terminal/path-length及逐horizon方向进度训练loss；评测增加路径长度比和逐horizon error/radius | 定量定位预测轨迹偏短，并用可视化识别总长度loss的锯齿漏洞；方向进度loss只影响训练，不改变Architecture v1 Deployment图 | 用户指出绿色预测明显短于专家，且要求“不行就优化” | 13项模型/评测/训练unittest；固定128×4与正式1,024×4对照；同一样例检查向后step；所有失败臂保留 | DEC-058；EXP-027～028；FAIL-009 |
| CHG-041 | 2026-09-12 | GRU近恒等显式重置、checkpoint后重置约束、GRU独立optimizer参数组、课程配置、single-step renderer文案及模型测试 | 修复随机GRU从第0步就压低Fusion表示、使4k和36k direct主链落后single-step的问题，同时保持冻结的Fusion→GRU→waypoint Deployment结构 | direct近恒等4k负对照；single-step→GRU课程4,096步；固定1,024×4、分步/清零hidden反事实；waypoint-only梯度与固定图检查 | 课程GRU `0.089646/0.154162 m`、路径比`0.746039`；清零hidden可测退化；12项相关unittest、py_compile及配置/manifest复核通过；未访问`test_locked` | DEC-059；EXP-029；FAIL-010；NEXT-026 |
| CHG-042 | 2026-09-12～13 | Architecture-v1同构Phase-1 teacher off/on预训练、对应single-step/GRU课程训练、正式评测与固定渲染 | 将ABL-V1-08从旧4k初始化诊断升级为四个收敛臂，分别量化同构Phase 1、OSNet teacher与GRU课程收益 | 四臂训练完成标记、checkpoint、1,024×4 metrics、waypoint-only梯度、固定PNG及统一summary；未访问`test_locked` | 最优teacher-on+GRU ADE/FDE `0.084341/0.146654 m`；teacher与GRU均有小幅正收益，四臂残余问题仍是轨迹偏短 | EXP-030；ABL-V1-08；NEXT-026 |
| CHG-043 | 2026-09-13 | Frozen Frontend baseline训练采样/scheduler/评测、ABL-09正式配置、自动编排、汇总器、审计文档与测试 | 在旧OSNet/cache+MLP仅作消融的边界内建立可恢复、同预算且指标定义与E2E一致的正式对照；明确anchor相位差而不伪称exact sample ID一致 | 200-step warmup+cosine、scheduler checkpoint、逐epoch连续窗口、LR/梯度JSONL；ADE/FDE/path length/ratio/horizon error；远端244项全仓及新增2项汇总测试通过，Bash/py_compile/diff/hash通过 | cache恢复任务与等待型正式tmux均正常；最终训练/评测/渲染待完整cache后自动执行；GPU0不使用，EVT-Bench不作waypoint训练，未访问`test_locked` | EXP-030；ABL-V1-09；NEXT-022；NEXT-026 |
| CHG-044 | 2026-09-13 | `omtrackvla/evaluation/end_to_end_closed_loop.py`、NEXT-027运行/启动脚本及10项专项测试 | 新增独立GPU waypoint worker、严格决策API审计、动作后RGB强制刷新、模型visibility/bbox安全门、逐步指标/输入溯源、PNG frame spool与隔离FFmpeg；结果记录请求步数和退出原因 | 将open-loop模型接到真实Habitat控制循环，并依次修复旧RGB、近距碰撞、第51步原生abort和视频完整性问题 | 本地/远端10项专项测试；v6退出码/JSON；50张PNG及50帧H.264独立解码；本地/远端SHA-256一致；人工contact sheet抽查 | DEC-060～061；EXP-031；FAIL-011～013；NEXT-027 |
| CHG-045 | 2026-09-13 | `build_sage3d_perception_cache.py`、cache merge/loader记账和3组测试 | 区分根index catalog与canonical accepted；对双证据source rejection生成结构化skip，对可疑missing sidecar继续硬失败；分片及合并强制`requested=cached+skipped` | 修复ABL-09在4,910个cache处退出，同时防止无条件跳过掩盖数据损坏 | 本地`py_compile`/diff check；远端20项专项测试；已归档原失败日志并保留完成cache断点重挂 | DEC-062；FAIL-014；ABL-V1-09；NEXT-022/026 |
| CHG-046 | 2026-09-13 | NEXT-007 expert relabel探针、Phase 3单样本loader/backward smoke、tmux运行脚本、8项测试和可视化 | 精确重放模型前28步后生成分布受控expert；增加相对静止anchor进度与方向门；保存初始化RGB、T=4连续失败history及输入/标签边界；加载现有Architecture v1 checkpoint并仅用waypoint loss反传 | 把NEXT-027墙角失败转成可审计训练样本，同时防止过快expert、GT输入泄漏、val样本误入正式训练和辅助loss冒充主链梯度 | v1负结果、v4通过report/sample、单H100 backward report及预测/专家PNG；Fusion/GRU/DA3 adapter非零梯度；0 optimizer step；未访问`test_locked` | DEC-063；EXP-032；FAIL-015；NEXT-007 |
| CHG-047 | 2026-09-13 | ABL-09评测启动修复与完成审计 | 将失效的环境`torchrun`入口改为当前`PYTHON_BIN -m torch.distributed.run`；保留首次exit 126记录，从已完成36,864步checkpoint继续评测/render/summary | 环境迁移后torchrun shebang仍指向不存在的`/h100/.../python3.9`，训练已完成但自动链在评测入口退出 | `bash -n`、19项专项测试、模块入口`--help`、1,024×4评测、64帧render、summary与complete hash | ABL-09完整闭环；未重复训练、不用GPU0、不访问`test_locked` | EXP-033；ABL-V1-09；NEXT-022/026 |
| CHG-048 | 2026-09-13 | NEXT-007 crash可恢复rollout、前瞻expert、独立环境实例、通用train候选采集器和3样本minibatch smoke | 成功step后原子写partial；修正底盘变换列轴；标签侧以机器人固定预演未来目标；预演/标签各自构造config与dataset；三任务并行采集并只用waypoint loss优化 | 原生abort会丢失前缀；AT-400 reactive expert追旧位置；同一Habitat实例重复reset和复用config分别造成3.26cm replay偏差与第二实例缺RGB键 | 19项专项测试、三条12步fresh-RGB rollout、replay/坐标/未来轨迹/碰撞硬门、3样本8-step loss/梯度及before/after PNG | 三个train样本与minibatch smoke通过；无正式checkpoint，等待用户可视化准入 | EXP-034；FAIL-016；NEXT-007 |
| CHG-049 | 2026-09-15 | v2_013～v2_016可观测性训练、候选采集/重标、严格分区验证器和多场景runner | 增加8帧Architecture-v2模型访问状态加载、可见/近期丢失/长期停车标签、采集失败保留、scene/state双重不重叠检查、waypoint/安全/干净回退联合gate | 早期单场景恢复样本不能支持闭环泛化，且错误expert或episode尾部样本必须显式拒绝 | 三组共30项回归测试；8条完整rollout；18/20重标准入；20/13分区交集为0；v2_016的128步评测与checkpoint哈希 | v2_016仅false-visible一项失败，未生成best | DEC-064；EXP-035～036；FAIL-017～018 |
| CHG-050 | 2026-09-15 | `train_v2_phase3_safety_heads.py`、v2_017配置/诊断与专用四类可视化 | 仅开放visibility/stop heads，逐tensor验证其余参数冻结；导出逐样本base/calibrated概率、stop、waypoint和参数delta；可视化修复的hard safe-stop/false-invisible及残留两类错误 | 隔离安全运行点校准与waypoint主链变化，避免辅助loss改善冒充轨迹能力提升 | 32步离线gate、冻结hash、waypoint零漂移、四张PNG与2400×940 contact sheet；本地/远程SHA-256一致 | 离线通过但闭环未通过，产物只作诊断 | DEC-065；EXP-037；FAIL-018～019 |
| CHG-051 | 2026-09-15 | v2_018三模型未见index1300闭环runner、tmux启动器和汇总gate | 在GPU3顺序运行18条，记录每条进度/退出码/result/video；聚合SR/TR、碰撞、假可见/假不可见、路径、重获和时延，并比较v2_006b/v2_016/v2_017 | 离线小样本校准必须在未见场景通过真实动作闭环，且需要把纯head变化与waypoint训练分开归因 | 远端py_compile/bash语法；18/18完成、0失败、同scene/episode配对；汇总和3个97帧STT视频本地哈希核对 | gate失败并自动拒绝v2_017晋升；未访问GPU0或`test_locked` | DEC-066；EXP-038；FAIL-019；NEXT-027 |

## 12. 下一步计划

NEXT-025、NEXT-026、ABL-08/09及Architecture-v2 Phase 3最小数据/训练/安全头闭环均已完成，但NEXT-027仍未通过。v2_016在严格分区离线集上改善waypoint和safe-stop path，却未过false-visible门；v2_017纯安全头校准虽离线通过，到了此前未见的index1300却令三条se2-waypoint全程停车、SR仍为0，已拒绝晋升。下一步不是继续扫0.005阈值或在index1300重训，而是扩大跨场景、连续时序的模型访问状态数据，显式覆盖可见→遮挡→近期丢失→长期丢失边界；保持干净Phase 2混合、冻结新的calibration划分和另一个全新闭环holdout，再做一次有界pilot。正式Phase 3大训练仍未开始。不得静默加入Flow、多尺度DA3、part-token/memory bank、occupancy/depth decoder；旧ResNet18与冻结OSNet cache继续只作baseline；真实UWB适配仍等待设备日志；`test_locked`不得重跑或查看。

| 优先级 | ID | 工作项 | 依赖 | 预期产出 | 状态 |
|---:|---|---|---|---|---|
| P0 | NEXT-001 | 盘点现有数据及其字段、规模、路径和质量 | 数据目录访问 | `docs/data_inventory.md`、机器可读manifest和完整 DATA 表 | 已完成；全量元数据/路径检查和分层媒体解码通过；严格划分转NEXT-015 |
| P0 | NEXT-002 | 将已选变换约定落成robot、target、UWB和waypoint坐标系规范 | 传感器/仿真接口 | 含公式、单位和时间语义的坐标系规范 | 已完成；见`docs/data_contract.md`，具体source adapter仍须逐gate确认 |
| P0 | NEXT-003 | 设计一次视觉初始化与内部目标身份记忆的训练样本表示 | 输入数据格式 | 目标身份条件规范 | 已完成；初始化是history index 0的一次事件，后续bbox仅作label |
| P0 | NEXT-004 | 定义 Phase 2 最小训练样本 schema | NEXT-001～003 | 样本字段与缺失值规则 | 已完成；schema v1及只读验证器已落库 |
| P0 | NEXT-009 | 在候选几何backbone上验证camera pose、depth、confidence及中间feature接口 | 模型权重与视频样本 | backbone/pseudo-motion可用性报告 | 已完成接口探测；DA3-SMALL固定源码/权重和隔离运行时已在H100真实输出有限depth/conf/pose/intrinsics及layer 5/11 feature，见`docs/da3_geometry_probe.md`；pseudo-label准入转NEXT-010 |
| P0 | NEXT-010 | 定义DA3 pose到统一SE(2) pseudo trajectory的转换、尺度校准和置信度过滤 | NEXT-002、NEXT-009 | 几何伪动作规范 | 已完成；global scale因24% bad rate被拒绝；D435i/ZED尺度2.074511/1.806175与confidence阈值2.947961已冻结，`viz_val`复核和唯一一次32-clip locked admission均通过，见`docs/da3_multiscene_admission.md` |
| P0 | NEXT-011 | 定义FutureNav式forward/inverse/单步next-state目标及feature teacher | NEXT-009～010 | World-Action辅助loss规范 | 首版已实现；inverse、motion-conditioned forward和action-free next-feature loss已进入Phase 1 baseline；DA3 layer 5/11 feature接口及pseudo-motion准入已确认，正式训练中的teacher接入与消融待运行 |
| P0 | NEXT-012 | 固定主流UWB产品适配接口并盘点实际设备字段、频率、延迟和LOS/NLOS能力 | UWB设备/SDK或日志 | UWB输入与标定规范 | 待开始 |
| P0 | NEXT-013 | 定义UWB-only冷启动到自动视觉绑定的数据采集与标注协议 | NEXT-002～004、NEXT-012 | tag—track配对样本规范及歧义标签 | 待开始 |
| P0 | NEXT-014 | 定义目标视觉失联、RGB故障、UWB失效及安全停车/恢复状态机的标签语义 | 控制与安全接口 | 失效模式和评测规范 | 待开始 |
| P0 | NEXT-015 | 建立Phase 1/2/3的train、val、viz_val和test_locked manifest | NEXT-001～004 | 版本化benchmark及固定episode/seed列表 | Phase 1已完成：Intern 2980/372/186/187，SAGE 729/91/46/46，TpT 37/5/2/3；Phase 2复用按run隔离的SAGE划分且不消费locked；Phase 3待后续source解锁 |
| P0 | NEXT-016 | 冻结每个Phase的指标、可视化布局、baseline和初始gate阈值 | NEXT-008、NEXT-015 | benchmark配置与验收规范 | Phase 1已完成；Phase 2已实现ADE/FDE/分模式/安全停车、逐张PNG+MP4和open-loop开发gate，正式baseline及closed-loop退出gate待完成；Phase 3待实现 |
| P0 | NEXT-017 | 实现8×H100一键训练、评测、渲染、gate和断点续跑入口 | 训练代码、NEXT-015～016 | `run_pipeline_8xh100.sh`及可复现产物目录 | Phase 1正式8卡run已完成；Phase 2已接入并通过8卡preflight和单卡最小闭环，正式全量run待完成；Phase 3待接入 |
| P0 | NEXT-018 | 实现启动器约定的training/evaluation/render/gate模块及9个Phase/benchmark/gate配置 | 模型、数据schema、NEXT-004、NEXT-015～016 | 正式preflight通过并完成最小Phase 1训练 | Phase 1的4入口/3配置及正式规模闭环已完成；Phase 2的4入口/3配置及smoke已完成；Phase 3的4入口/3配置待完成 |
| P0 | NEXT-019 | 为4090机器配置GitHub认证并推送`wam`分支 | GitHub HTTPS token或SSH key | `origin/wam`及协作者拉取命令 | 已完成 |
| P0 | NEXT-020 | 将`PROGRESS.md`纳入仓库并建立协作者顺序工作包 | 当前项目事实与数据初盘 | GitHub可见的单一协作入口 | 已完成 |
| P0 | NEXT-021 | 修复并重新准入SAGE3D bbox/visible身份监督 | EXP-003、相机外参、RGB对齐depth、模块化person detector/ReID | 不改源数据的版本化修复侧车、投影/遮挡测试、独立审计报告和Phase 1 admission gate | 已完成；7,105/7,105 episode、2,131,500步、0失败，完整性与冻结384帧质量gate通过；Phase 1仅通过准入侧车重新启用SAGE3D，源bbox/visible保持阻断 |
| P1 | NEXT-022 | 完成冻结OSNet感知cache并训练Frozen Frontend waypoint baseline | DEC-042～043、DEC-051～054、EXP-013～014/021～023、NEXT-023～024、8×H100 | 完整train/val/viz cache、模块化baseline checkpoint/metrics/report/PNG+MP4和open-loop gate | 已完成；cache `5578/733/389`并显式skip 5个train source rejection，36,864步、1,024×4和64帧render完成；ADE/FDE `0.168628/0.282747 m`，仅作ABL-V1-09较弱baseline；不用EVT waypoint训练，不消费`test_locked` |
| P0 | NEXT-023 | 训练并准入Phase 2A目标身份时序融合前端 | DEC-039～040、DEC-043～052、EXP-009～022 | 冻结detector/ReID的可复现配置、EVT指标、失败案例和前端选型记录 | 已完成；逐帧时序融合v1～v5和KPR负结果均保留，最终由NEXT-024全量EVT选择OSNet；TpT继续作跨域压力审计 |
| P0 | NEXT-024 | 完成EVT-Bench全量`val` OSNet/KPR正式人物识别对照并冻结Phase 2B前端 | DEC-052、EXP-022、NEXT-023、GPU 1～7 | 两方法各4,215/4,215、0错误/缺失、配对轨迹一致的JSON与UTF-8 BOM CSV；总体/分任务micro与episode-macro指标、失败状态及明确胜者 | 已完成；OSNet 74.2978% vs KPR 71.1748%总体Micro-F1，完整性和4,215组配对轨迹gate通过，正式选择OSNet；远端`REPORT.json/CSV/COMPLETE`和本机正式CSV均已生成 |
| P0 | NEXT-025 | 实现Architecture v1严格端到端统一模型与最小反向传播闭环 | DEC-053～057、EXP-023、WP-1数据契约、DA3官方源码/权重 | `EndToEndFollowPolicy`及配置/原始RGB loader/smoke：一次初始化`RGB+bbox` + 后续RGB history + UWB/masks → `[B,8,2]` waypoint + stop；无cache/后续GT bbox/depth/target pose推理输入；shape、UWB投影边界、SE(2)语义、DA3权重/参数清单及waypoint-only梯度报告 | 已完成；EXP-024的CPU stub、单H100官方DA3和一步8×H100 DDP均通过，waypoint-only loss对Fusion、GRU及DA3后层adapter梯度非零；未正式训练、未迁移旧ResNet18 checkpoint、未访问`test_locked` |
| P0 | NEXT-026 | 运行正式Architecture v1端到端Phase 2 open-loop训练、固定可视化和强制消融 | NEXT-025、SAGE3D policy admission、TpT/Intern辅助流、8×H100 | 正式checkpoint、ADE/FDE/分模式/identity/stop/ego/World-Action指标、PNG+MP4、gate；完成第3.1.7节`ABL-V1-01～09` | 主训练、长度修正、ABL-04、ABL-08和ABL-09均完成；当前最优teacher-on+GRU为`0.084341/0.146654 m`，Frozen Frontend为`0.168628/0.282747 m`；ABL-01～03/05～07的4k结果仍只作diagnostic。真实UWB仍标记`simulated_uwb`，不访问`test_locked` |
| P0 | NEXT-027 | 建立无GT point的Habitat端到端closed-loop benchmark与阶段退出gate | NEXT-026、固定EVT/SAGE/Habitat episode与控制接口 | 跟随成功率、距离误差、碰撞、目标丢失/重获、安全停车、轨迹效率及视频；模型控制轨迹，不使用GT target point/reactive-controller固定轨迹 | 进行中且仍失败；v6单episode安全子门通过但跟随/重获失败。v2_018又在此前未见index1300完成18条三模型对照：SR全0、0碰撞，v2_017令se2-waypoint三任务总路程0并拒绝晋升。须先扩连续多场景Phase 3数据并在新holdout复测，不能用open-loop ADE、无碰撞或polar局部TR改善替代最终成功 |
| P1 | NEXT-005 | 收集真实UWB误差、偏置、漂移、延迟、丢包和置信度校准统计 | 定位模块日志 | 噪声模型报告 | 待开始 |
| P1 | NEXT-006 | 定义 Phase 3 噪声矩阵和难度课程 | NEXT-005 | 扰动配置规范 | 待开始 |
| P0 | NEXT-007 | 验证仿真是否支持任意访问状态的 expert relabel | NEXT-027失败状态、仿真环境 | DAgger可行性结论、模型访问状态导出、expert接管轨迹及输入泄漏审计 | 最小闭环及v2_014～v2_017多场景小预算pilot已完成；8条rollout、18个准入重标、20/13严格分区和waypoint主链训练均有证据。但v2_017在新场景过拟合，正式Phase 3仍未开始；下一步扩充连续状态与场景覆盖并冻结新的calibration/closed-loop holdout |
| P2 | NEXT-008 | 定义 clean/occlusion/distractor/noise/delay/recovery 验证切片 | 数据盘点 | 评测矩阵 | 待开始 |

## 13. 产物索引

| 产物 | 版本/Hash | 位置 | 说明 |
|---|---|---|---|
| 论文训练方法导读 | `e0bbbb71f29d882b64a5ca9f4efd1ffc41793e17c25b0850f13e6dcfba135f05` | `/data/nas_ray/home/zeying.gong/algorithm/repos/Tracking_TRAINING_DETAILS_v2.md` | TrackVLA 等五篇工作的训练方法整理 |
| 8×H100流水线脚本 | `1890585212f093af141371dc404d283bdabc3550a1ecc493c57162f022e3508c` | `/data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA/scripts/run_pipeline_8xh100.sh` | 已分配单节点上直接bash；三Phase训练/eval/render/gate编排 |
| 8×H100流水线配置 | `6ca438b92c65e76212e0e2cea1eff65014e895120c50f40755ca14bc8b5d01ea` | `/data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA/configs/pipeline/h100_8gpu.env` | 模块、配置、数据、GPU和checkpoint契约 |
| 集群训练说明 | `2caaa97e17e29f28ea43a4230ebe475319b38347f58f6318d19abe5f58d2b676` | `/data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA/docs/cluster_training.md` | 运行、恢复、模块接口及产物说明 |
| WAM模块化基线提交 | `724c87658b505b08dfb24942265e1791ef2be83b` | `OmTrackVLA origin/wam` | 当前精简仓库快照；102项测试通过 |
| WAM流水线提交 | `45af91e2239a315730eb6248cc289df6c7f3710a` | `OmTrackVLA origin/wam` | 三阶段8×H100启动器；工作区干净 |
| WP-0数据清单 | `schema v1` | `docs/data_inventory.md`、`configs/data_inventory.json` | 三套正式外部数据的字段、规模、质量、split unit与禁止输入 |
| WP-0只读审计 | `script v1` | `scripts/audit_data_inventory.py` | 全量元数据/路径核对与确定性分层媒体解码；源数据零写入 |
| 微型真实样例与旧视图审计 | `schema v2` | `example_datasets/` | 每套正式数据16连续帧、预览、manifest与校验和；同时保留旧symlink视图审计和发布边界 |
| WP-1统一数据契约 | `schema v1` | `docs/data_contract.md`、`configs/data_contract.json` | 输入/标签隔离、canonical坐标/时间、四条件模式、缺失值、source gate和最小policy/identity clip schema |
| WP-1只读验证器 | `script v1` | `scripts/validate_data_contract.py` | 校验契约及JSON/JSONL样本；可选只检查记录实际引用的文件，不扫描数据根 |
| Phase 1固定划分 | `schema v1`, `254c7c6c...` | `configs/manifests/phase1_v1.json`、`scripts/build_phase1_manifest.py` | 按Intern scene、SAGE run、TpT sequence隔离train/val/viz_val/test_locked；不扫描媒体 |
| Phase 1最小闭环 | `baseline v3`，commit `5103f88` | `omtrackvla/{data,geometry,models,training,evaluation}`及3个Phase 1配置 | 只读双流adapter、共享ResNet-18、B1-ID/B1-GEO/B1-PROBE、含absent的render和12项gate；正式8卡闭环已通过 |
| Phase 1 smoke产物 | `EXP-001` | `results/wp2_phase1_memory_smoke/phase_1`（不进Git） | 带GRU history的2-step开发checkpoint、metrics、report、gate失败原因和8帧可视化；不得当作正式baseline |
| Phase 1正式baseline历史 | `EXP-006～008` | H100 `outputs/training/phase1_baseline_v{1,2,3}_*`（不进Git） | v1/v2失败和v3成功均保留；v3 best checkpoint SHA-256 `9b8025af29e1c252...`，metrics `d4d6df3f791c01b9...`，gate `61536cfd2e80dfa8...`，视频 `061314568b76c9b...` |
| Phase 1正式复核视频 | 64帧、960×380、4fps；SHA-256 `061314568b76c9b88db22526c1e5ce7310a69d2fc477ef09490607831bdb6343` | H100 v3运行目录及`Desktop/OmTrackVLA_visual_review/phase1_baseline_v3_5103f88_phase1_viz.mp4` | 前32帧身份、后32帧几何；身份段24 visible + 8 absent；本地/远端hash一致且64帧完整解码 |
| DA3几何probe | 官方commit `3d835ec1...`；模型revision `e08cab65...` | `scripts/probe_da3_geometry.py`、`docs/da3_geometry_probe.md`；运行JSON在`results/wp2_da3_probe/`（不进Git） | 真实4帧H100输出已验证；修正后单clip scale/translation/yaw为2.07116/0.00496m/0.00519rad；正式准入证据见下一项 |
| DA3多场景准入 | policy `da3-small-intern-multiscene-v1`，SHA-256 `b4bfbe49...` | `omtrackvla/geometry/da3_admission.py`、`scripts/audit_da3_multiscene.py`、`configs/gates/da3_multiscene_v1.json`、`docs/da3_multiscene_admission.md`；运行产物在`results/wp2_da3_multiscene_v2_{viz,locked}/`（不进Git） | global-scale负结果、D435i/ZED分层尺度、confidence gate、开发可视化和唯一locked准入；locked 14项通过且无图片，禁止重跑/查看 |
| SAGE3D bbox审计 | `EXP-003` | `scripts/audit_sage3d_bboxes.py`、`docs/sage3d_bbox_audit.md`；H100运行产物在`results/wp2_sage_bbox_audit_v2/`（不进Git） | 384帧分层审计确认投影水平符号和可见性错误；含detector对照、水平镜像反事实和72例拼图 |
| SAGE3D修复侧车 | `sage3d-bbox-depth-v1`、`EXP-004` | 代码见`omtrackvla/data/sage3d_sidecar.py`、`scripts/{build,audit}_sage3d_sidecar.py`和`docs/sage3d_bbox_sidecar.md`；H100全量/审计产物为`results/sage3d_bbox_sidecar_v1{,_audit}/`（不进Git） | 7,105 episode、2,131,500步外置标签；完整性/质量准入、SHA-256绑定和4张最差案例拼图；Phase 1 adapter只接受该准入版本 |
| 冻结检测/ReID双运行点融合 | `EXP-009～011`；融合SHA-256 `03a78884...` | `omtrackvla/rgb_person_perception.py`、`omtrackvla/models/candidate_fusion.py`、`omtrackvla/training/train_candidate_fusion.py`、`configs/models/candidate_fusion_resnet50_tpt_train37_dualop_v3.json`；H100评测位于`outputs/evaluation/pretrained_identity_v4_resnet50_fusion_dualop_viz/` | 37序列train-only融合；TpT完整`viz_val` E2E 46.30%、precision 90.68%、absent FPR 4.16%、wrong-target 60；五类各16张验收图 |
| SAGE3D Phase 2 policy准入 | `sage3d-policy-se2-30hz-v1`；报告SHA-256 `ec90f757...` | `omtrackvla/data/sage3d_policy.py`、`scripts/audit_sage3d_policy.py`、`configs/gates/sage3d_policy_v1.json`、`docs/sage3d_policy_admission.md`；H100报告`results/sage3d_policy_v1_audit/admission.json` | 256个非locked episode、13分层、11/11 gate；源waypoint偏移一帧，训练必须按`0,+3,…,+21`重算 |
| Phase 2 open-loop smoke | `EXP-013` | H100 `outputs/training/phase2_frozen_frontend_smoke/phase_2/`；本地逐帧复核`Desktop/OmTrackVLA_visual_review/phase2_frozen_frontend_smoke/` | 20-step开发checkpoint；正常模式ADE/FDE 0.22561/0.38741m；safe-stop 100%；16张四模式PNG和MP4；partial gate按预期仅样本量失败 |
| Phase 1与冻结前端同协议对照 | `EXP-014` | H100 `outputs/evaluation/phase1_v3_tpt_viz_same_protocol/metrics.json`及`outputs/evaluation/pretrained_identity_v4_resnet50_fusion_dualop_viz/comparison_metrics.json` | 5,928帧E2E分别为2.59%与46.30%；只使用`viz_val`开发对照，未访问`test_locked` |
| Phase 2A首轮失败迭代 | `EXP-015` | H100 `outputs/training/phase2a_temporal_fusion_v1/`及`outputs/evaluation/phase2a_temporal_fusion_v1_val/aggregate.json` | val误跟大降但E2E/重获退化，已fail closed；未运行viz、未进入2B |
| Phase 2A权重防退化/阈值失败迭代 | `EXP-016` | H100 `outputs/training/phase2a_temporal_fusion_v2/`及`outputs/evaluation/phase2a_temporal_fusion_v2_val/aggregate.json` | epoch 0权重正确保留但阈值重拟合过严，val再次fail closed；未运行viz、未进入2B |
| Phase 2A新增列/固定运行点失败迭代 | `EXP-017` | H100 `outputs/training/phase2a_temporal_fusion_v3/`、`outputs/evaluation/phase2a_temporal_fusion_v3_{val,viz}/` | val显著提升但viz E2E与wrong-target gate失败；定位为train stride 4与部署stride 1时钟偏移候选原因 |
| Phase 2A同stride离线/在线偏移迭代 | `EXP-018` | H100 `outputs/training/phase2a_temporal_fusion_v4/`、`outputs/evaluation/phase2a_temporal_fusion_v4_{baseline_rollouts,val}/` | 离线排序改善但在线val退化，触发train-only dataset aggregation；未运行viz、未进入2B |
| Phase 2A on-policy聚合失败迭代 | `EXP-019` | H100 `outputs/training/phase2a_temporal_fusion_v5/`、`outputs/evaluation/phase2a_temporal_fusion_v5_{onpolicy_rollouts,val,viz}/` | val提升但viz E2E/precision/wrong-target/reappearance均失败；未生成完成标记、未进入2B |
| KPR Phase 2A冻结对照 | `EXP-020` | H100 `outputs/evaluation/kpr_phase2a_v1_train_stride4/`、`outputs/evaluation/kpr_phase2a_overnight/` | 37条train校准及两个5-sequence val版本均完成；v3 E2E 14.51%但precision 74.67%，且memory update为0，未准入也未访问viz/test_locked |
| EVT人物识别快速选型 | `EXP-021`、schema v2 | H100 `outputs/evaluation/evt_reid_compare_v1/{OSNET_REPORT.json,osnet/,kpr/}`；代码`scripts/summarize_evt_reid.py` | OSNet 6条/213步完整，precision 95.31%、recall 85.92%；KPR 4条partial与不可超越OSNet F1的上界；未访问`test_locked` |
| EVT人物识别6条预实验汇报表 | `pilot report v1`、21字段/6行 | 本机`C:\Users\59783\Desktop\OmTrackVLA\EVT_Bench_ReID汇报表_2026-09-10.csv` | UTF-8 BOM；汇总EXP-021的OSNet总体及STT/DT/AT、KPR partial和子集理论上界；只作0.1423% val覆盖率的预实验留档，不得作为正式benchmark或最终选型 |
| EVT人物识别全量val评测 | `EXP-022`、schema v1 | H100 `outputs/evaluation/evt_reid_full_val_v1/{REPORT.json,REPORT.csv,COMPLETE}`；本机`C:\Users\59783\Desktop\OmTrackVLA\EVT_Bench_ReID全量Val正式汇报表_2026-09-10.csv`及`C:\Users\59783\Desktop\OmTrackVLA\EVT_Bench_ReID全量Val正式报告_2026-09-10.json`；代码`scripts/run_evt_reid_full_val_7gpu.sh`、`scripts/summarize_evt_reid_full_val.py` | OSNet/KPR各4,215条、官方300步、逐步感知计数和4,215组配对轨迹gate全部完成；总体Micro-F1为74.2978% vs 71.1748%，正式选择OSNet；远端/本机报告hash一致 |
| Architecture v1冻结规范 | `v1 (2026-09-11)` | 仓库根目录`PROGRESS.md`第3.1节 | DA3-SMALL L11、Target Cross-Attention、World token、UWB Gaussian patch bias与独立`z_uwb`、显式SE(2)瓶颈、GRU、8×2 waypoint、Deployment/Training-only边界、tensor shapes、loss及`ABL-V1-01～09` |
| Architecture v1端到端smoke | `EXP-024`；model/report SHA-256 `d984a15a...`/`68ddbe7d...` | `omtrackvla/{models,data}/end_to_end.py`、`scripts/smoke_end_to_end_v1.py`、`configs/phases/phase2_end_to_end_v1.yaml`、`docs/next025_end_to_end_smoke.md`；运行报告在`outputs/training/next025_{stub,single_gpu,8gpu_ddp}_smoke/report.json` | 已准入SAGE3D train raw-RGB batch；DA3 checkpoint 100%加载；waypoint-only loss对Fusion/GRU/DA3 adapter非零梯度；单卡及一步8×H100通过；无正式训练、无辅助loss、无`test_locked` |
| Architecture v1训练前可视化 | `EXP-025`；PNG/JSON SHA-256 `b3f935c2...`/`beb97ae9...` | H100 `outputs/visualization/next026_pretrain_v1/`；本机`C:\Users\59783\Desktop\OmTrackVLA_visual_review\next026_pretrain_v1\`；代码`omtrackvla/evaluation/end_to_end_render.py`与`scripts/render_end_to_end_v1_pretrain.py` | 0 optimizer step的真实raw-RGB dashboard；显示bbox/history、Target/Scene/UWB空间图、UWB 2σ、预测/专家轨迹、stop与SE(2)；等待用户准入后才能正式训练 |
| Architecture v1 Phase 2最低误差模型 | `EXP-026`；SHA-256 `a1dc37ff953b58d1311f5c5ed335e34ae2d47f914ee6e98bf9d1821100ba48d5` | H100 `outputs/training/next026_phase2_world_action_long_v1/` | 有效累计36,864步；正式ADE/FDE `0.087408/0.151858 m`、路径比`0.766329`；作为最低误差基线保留 |
| Architecture v1径向进度候选 | `EXP-028`；SHA-256 `d728ca67c5588b75e989b386394a8fcdff690add8cd276c22b509e4ddfdb4a0ef` | H100 `outputs/training/next026_phase2_waypoint_radial_progress_v1/`；本机`C:\Users\59783\Desktop\OmTrackVLA_visual_review\next026_refine_comparison\` | 正式ADE/FDE `0.088484/0.153559 m`、路径比`0.800828`；固定图无回退；作为更长且自然的Pareto候选保留 |
| Architecture v1 GRU课程与ABL-04复核 | `EXP-029` | H100 `outputs/{ablations/next026_converged_probe_v1,ablations/next026_gru_identity_probe_v1,training/next026_gru_curriculum_probe_v1}/`；本机`C:\Users\59783\Desktop\OmTrackVLA_visual_review\next026_ablation_formal_v1\` | direct 36k公平对照、失败的近恒等direct 4k、成功的single-step→GRU课程、waypoint-only梯度、清零hidden反事实和固定PNG；4k 13臂明确只作diagnostic |
| Architecture v1 ABL-08收敛消融 | `EXP-030` | H100 `outputs/ablations/next026_abl08_converged_v1/`；本机`C:\Users\59783\Desktop\OmTrackVLA_visual_review\next026_ablation_formal_v1\abl08_converged\` | teacher off/on × single-step/GRU课程四臂、正式1,024×4 metrics、固定PNG、waypoint-only梯度及summary；最优ADE/FDE `0.084341/0.146654 m`，未访问`test_locked` |
| Frozen Frontend ABL-09 | `EXP-033`；summary SHA-256 `942df1dd789c4053...` | H100 `outputs/ablations/next026_abl09_frozen_frontend_formal_v1/`；代码`configs/{phases,benchmarks}/next026_abl09_*`、`scripts/run_next026_abl09_frozen_frontend.sh`、`scripts/summarize_next026_abl09.py` | cache、2-step smoke、36,864步、1,024×4、64帧render和summary全部完成；ADE/FDE `0.168628/0.282747 m`，约为Architecture v1最优的两倍；不用GPU0/EVT waypoint训练/`test_locked` |
| NEXT-027闭环v6 | `EXP-031`；JSON/MP4 SHA-256 `d31b3eb966179e37...`/`bd776353b6bfc515...` | H100 `outputs/evaluation/next027_full50_v6/`；本机`C:\Users\59783\Desktop\OmTrackVLA\results\next027_full50_v6\` | 50步无GT point端到端控制、输入审计、逐步JSON、50帧H.264、contact sheet；安全子门通过，跟随/墙角重获gate失败，不能作为NEXT-027成功结论 |
| ABL-09 canonical rejection修复 | `DEC-062`、`FAIL-014` | `scripts/build_sage3d_perception_cache.py`、`scripts/merge_sage3d_perception_cache.py`、`omtrackvla/data/phase2.py`及测试；H100原失败日志归档于cache train logs | 仅双证据明确source rejection可结构化跳过，其他missing sidecar仍fail-closed；保留4,910个既有cache并断点重挂 |
| NEXT-007失败状态expert relabel v4 | `EXP-032`；report/sample SHA-256 `1de251a88fb39f3e...`/`cc1e701b2e0e6190...` | H100 `outputs/evaluation/next007_expert_relabel_probe_v4/`；本机`C:\Users\59783\Desktop\OmTrackVLA\results\next007_expert_relabel_probe_v4\` | v6前28步模型动作精确重放；标签侧expert 8点；保存初始化图与step25～28 raw RGB；轨迹分布、方向、相对静止anchor进度、碰撞及GT输入边界通过；val样本只作smoke |
| NEXT-007 Phase 3单样本backward | `EXP-032`；report/PNG SHA-256 `12897fb74eb07c1a...`/`75946a8fd3160fb3...` | H100 `outputs/evaluation/next007_phase3_single_sample_smoke_v2/`；本机`C:\Users\59783\Desktop\OmTrackVLA\results\next007_phase3_single_sample_smoke_v2\` | 当前checkpoint在真实墙角失败RGB上预测0.437407 m、expert 0.704529 m；waypoint-only loss对Fusion/GRU/DA3 adapter梯度非零；无optimizer step、无GT模型输入、未开始正式Phase 3训练 |
| NEXT-007 train三样本恢复smoke | `EXP-034` | H100 `outputs/evaluation/{next007_train_{rollout,relabel}_*,next007_phase3_train_minibatch_smoke_v3}/`；本机`C:\Users\59783\Desktop\OmTrackVLA_visual_review\next007_phase3_smoke_v3\` | STT/DT/AT三个train访问状态、前瞻expert、3样本8-step waypoint-only smoke及6张before/after图；loss下降83%，Fusion/GRU/adapter梯度非零；无正式checkpoint |
| v2_016多场景Phase 3 pilot | `EXP-036`；step128 SHA-256 `329b589f02f192a3...` | H100 `outputs/training/v2_016_phase3_multiscene_observable_pilot/`及`outputs/evaluation/v2_016_phase3_multiscene_partitions/` | 20/13 train/recovery-val严格scene/state隔离；waypoint与safe-stop path改善，false-visible一项失败，未生成best |
| v2_017安全头校准与可视化 | `EXP-037`；best SHA-256 `ab8d8063307b150a...` | H100 `outputs/training/v2_017_phase3_safety_head_calibration/`、`outputs/evaluation/v2_017_phase3_safety_calibration_visualization/`；本机`C:\Users\59783\Desktop\OmTrackVLA_visual_review\v2_018_unseen1300_gate\` | 只更新visibility/stop heads，其他参数哈希不变；离线false-visible 5→4、false-invisible 3→1，但stop accuracy未改善；四类contact sheet显示waypoint完全相同 |
| v2_018未见index1300闭环gate | `EXP-038`；summary SHA-256 `0804543aa7b3842d...` | H100 `outputs/evaluation/v2_018_phase3_unseen1300_closed_loop_gate/`；本机`C:\Users\59783\Desktop\OmTrackVLA_visual_review\v2_018_unseen1300_gate\` | 三模型×双执行器×三任务共18条、0失败、0碰撞、SR全0；v2_017 se2三任务全程停车且TR退化，gate失败并拒绝晋升；含三段97帧STT对比视频 |
| 项目进展记录 | `v46 (2026-09-15)` | 仓库根目录`PROGRESS.md` | 本文件；记录v2_013～v2_017小预算训练/校准及v2_018未见闭环负结果；正式Phase 3大训练未开始，index1300不再可作未见gate，未访问`test_locked` |


## Takeover correction: 2026-09-14

User authorized unified takeover with the original full product scope. See `docs/phase3_takeover.md` and `outputs/takeover/DECISION.json`. The 8-anchor/512-step pilot completed but paired 50-step following-frame rate regressed from 0.52 to 0.32; it is not promoted. Keep the original Phase 2 parent. Legacy offsets/30 recovery labels are not admitted to v2: measured simulation clocks differ and can vary per step. Stateful sequence / FP32 auxiliary-loss backward preflight passed on official DA3; 26 CPU regression tests passed. One real train candidate now records the full reset-to-anchor prefix and resamples expert states on measured time to 0..0.7s. It remains `formal_training_eligible=false`. Formal v2 optimizer training has NOT started; sequence loader, independent scene validation, candidate admission, checkpoint/resume and stage gates must be integrated first. Completed pilot entrypoints now preserve artifacts and reject duplicate training.
