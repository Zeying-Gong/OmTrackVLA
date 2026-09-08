# 端到端目标人物跟随模型：项目进展

> 本文件用于持续记录已确认决策、数据状态、训练与实验尝试、失败原因、项目修改和下一步计划。
> 原则：事实、提案和待确认事项必须分开；实验与失败记录尽量追加，不覆盖历史。
> 最近更新：2026-09-08

## 0. 当前快照

- 任务：端到端目标人物跟随。
- 目标条件：视觉侧只输入一次“初始化图像 + 目标 bbox”，后续不接收外部逐帧 bbox；UWB 是持续但可能缺失的空间条件。理想启动时，用户佩戴 UWB 并在机器人面前露面，系统据此建立视觉身份与 UWB 标签的绑定。
- 冷启动：支持没有视觉初始化的 UWB-only 启动。机器人先依据 UWB 保守接近；目标进入视野后，模型利用 UWB 投影位置、RGB 中的人物候选与时间连续性完成自动视觉绑定，并在内部保存目标身份参考。
- 缺失情形：目标不在画面时可依靠 UWB 粗定位；UWB 不可用时可在已有视觉绑定的前提下纯视觉跟随；当目标无法被视觉定位且 UWB 同时失效时，默认安全减速至停止，等待任一信号恢复，不主动盲目搜索。
- 主要观测：后续 egocentric RGB 历史。
- 输出：机器人未来的局部 waypoint trajectory。
- 明确不需要：自然语言描述、VQA、语义 CoT。
- 当前状态：WP-0数据审计和WP-1统一数据契约均已完成；WP-2已实现Phase 1固定split、只读source adapter、共享视觉baseline、真实train/eval/render/gate入口和单卡小样本闭环。DA3-SMALL实际权重、隔离运行时和4帧GPU probe已跑通，并据真实结果修复OpenCV/Habitat轴桥接；384帧SAGE3D bbox独立检测器审计确认抽取器存在水平符号错误和错误可见性判定，修复侧车准入前禁止其bbox/visible进入身份监督；TpT物理时钟、真实UWB、DA3多场景尺度/置信度准入及正式规模训练尚未完成，因此WP-2仍在进行中。
- 集群入口：仓库内的`scripts/run_pipeline_8xh100.sh`已创建，可在已分配的单节点8×H100上直接bash启动，负责环境预检、三Phase衔接、断点续跑、eval/render/gate与产物管理；Phase 1模块和3个配置已接通并通过单卡preflight，Phase 2/3的6个配置与实现仍缺失，因此`--phase all`会明确失败。OmTrackVLA使用`wam`分支并通过GitHub `origin/wam`协作。
- 当前方案：采用 3 个正式阶段——Phase 1身份与几何World-Action预训练、Phase 2目标人物跟随监督训练、Phase 3噪声与闭环恢复训练。
- World-Action 路线：优先评估 DA3 等视觉几何基础模型。利用其从视频恢复的相机轨迹作为显式 pseudo ego-motion，而不是再学习 WALA 式 latent action；借鉴 FutureNav 的 forward/inverse dynamics 与单步 future-state prediction。普通无任务 ego 视频只训练几何与状态转移辅助能力，不直接提供 policy trajectory 监督。
- 当前主要风险：UWB 冷启动后的首次视觉绑定；首帧/首次绑定目标身份如何长期保留；UWB 时间延迟、坐标转换与误差标定；pseudo ego-motion 的尺度及可执行性；专家状态与模型实际访问状态存在分布偏移；TpT 视频时钟与 GT/ODOM 时钟语义尚未统一；SAGE3D除严格执行三条件accepted规则外，还必须修复投影bbox水平符号、画面相交和遮挡可见性。

## 0.1 协作者接手入口：当前该做什么

本节是协作者开始工作的入口。先核对事实和数据，再实现训练；不要把流水线dry-run误认为模型已经可以训练。

### 工作目录与安全边界

- GitHub开发分支是`wam`。在H100上使用干净共享目录`/data/nfs/share/wam_tracking/OmTrackVLA`；旧目录`/data/nfs/share/OmTrackVLA`含有大量未提交实验代码和数据，暂时只读，不执行`git clean`、`git reset --hard`或`git add -A`。
- 开始前运行`git status --short`并记录`git rev-parse HEAD`。正式流水线默认拒绝dirty worktree。
- Git只保存代码、配置、小型manifest、文档，以及已审计的`example_datasets/samples`微型真实样例。除此之外的完整数据、checkpoint、视频及运行日志不得提交；它们通过`OMTRACKVLA_DATA_ROOT`和`outputs/training/`管理。
- 本文件中的“已确认”是任务约束；“提案”和“待开始”可以通过实验修改，但修改时必须记录证据、失败原因和关联commit。

### 已核对的仓库与数据事实（2026-09-07，g0014/NFS复核）

| 路径/对象 | 实际状态 | 可直接用于什么 | 不能假设什么 |
|---|---|---|---|
| `scripts/run_pipeline_8xh100.sh` | 已实现编排、preflight、8进程启动、断点续跑、eval/render/gate和产物检查；Phase 1已接入真实模块 | Phase 1正式入口及全Phase dry-run | 不代表Phase 2/3已经实现 |
| `configs/pipeline/h100_8gpu.env` | 已实现，实际配置格式是Bash env，不是YAML | 集中定义模块、Phase配置、GPU和产物契约 | 不包含具体模型或数据schema |
| `omtrackvla/` | 已包含原模块化oracle/感知代码，以及Phase 1的`data`、`geometry`、`models`、`training`和`evaluation`实现 | Phase 1双流baseline、train/eval/render/gate及后续可复用组件 | 当前baseline完成接口闭环，不代表正式规模效果或Phase 2/3 policy已完成 |
| `configs/phases/`、`configs/benchmarks/`、`configs/gates/` | Phase 1各有1个配置并已接通；Phase 2/3共6个配置尚缺 | 当前可运行`--phase 1`；后续按Phase补齐 | 不得用空配置或占位结果绕过gate，也不得运行`--phase all`宣称完成 |
| `data/datasets/track/{STT,DT,AT}/{train,val}/*.json.gz` | Git跟踪的6个episode文件；每类train 7,257、val 1,405，共25,986 episodes | 启动Habitat仿真、生成观测/rollout、建立固定episode划分 | 文件本身不含已录制RGB历史、逐帧bbox、UWB日志或expert future-waypoint张量 |
| episode字段 | 已看到`scene_id`、机器人初始pose、`main_human_semantic_id`、humanoid名称与人物waypoint字段；还含`instruction` | 识别目标人物、构造仿真真值和生成样本 | `instruction`不得进入本项目模型输入；人物waypoint不自动等于机器人expert trajectory |
| `data/scene_datasets` | 本机软链接到`/data/nas_ray/home/zeying.gong/datasets/scene_datasets`，目标存在；该链接不进Git | Habitat场景资产 | 新clone/H100会自动拥有同一路径 |
| `data/humanoids`、`data/versioned_data` | 本机存在且被Git忽略，约573 MB和57 MB；主humanoid目录有100个人物资产目录 | Habitat人物资产 | 它们是训练样本或会随Git下载 |
| `/h100-2/vln_n1/traj_data` | 完整的 InternData-N1 展开目录；12个group、196,536 episodes；3,730个scene目录中3,725个含正式episode，另5个仅含未索引depth残留 | Phase 1导航几何、pose/action与future-state预训练 | 不是人物跟随数据；自然语言task不得进入模型；5个无metadata/parquet的残留目录不得入manifest；旧报告的85,124 episodes是漏扫结果 |
| `/data/nfs/share/OmTrackVLA/data/sage3d_extracted` | 约160 GiB（`du -s -B1`为171,095,801,856字节）；912 runs、7,110个索引episode、其中7,105个canonical accepted；2,132,276 steps | RGB/depth、robot/target pose和8点ego waypoint；修复后可作为Phase 1/2监督候选 | 无真实UWB；156,420个尾部step无未来waypoint；抽取器把左轴当右轴造成bbox水平镜像，且`visible`未检查画面相交/遮挡，当前bbox/visible禁止直接训练；见`docs/sage3d_bbox_audit.md` |
| `/data/nfs/share/OmTrackVLA/data/tpt_bench_clean_v2` | 约17 GiB（`du -s -B1`为17,604,104,192字节）；47序列、141,326帧；parquet与RGB逐帧对应 | Phase 1B身份保持、遮挡/干扰人与可见性监督 | 无expert waypoint/UWB；`vid_pts_ms`与GT/ODOM时钟的整段时长约差4.48倍，冻结horizon前必须核实 |
| `example_datasets/samples` | 三套正式数据各16个连续真实帧；5,854,999字节、99文件；含48 RGB、32个16-bit depth、2个16行Parquet及3张预览 | 协作者在GitHub检查真实外观、目录结构、深度编码和标签形状 | 不是训练/评测划分；预览黄框和逐帧标签不得作为模型输入；不授予上游数据额外权利 |
| 仓库Git跟踪的`data/` | 31个文件，约19.7 MB；WAM提交没有新增大数据 | 小型episode元数据和Spot机器人资产 | `??`本地数据已经上传GitHub |

### 协作者按顺序执行的工作包

1. **WP-0：数据审计，已完成。** 正式外部数据范围固定为`/h100-2/vln_n1/traj_data`（InternData-N1）、`/data/nfs/share/OmTrackVLA/data/sage3d_extracted`和`/data/nfs/share/OmTrackVLA/data/tpt_bench_clean_v2`。产物为`docs/data_inventory.md`、`configs/data_inventory.json`和`scripts/audit_data_inventory.py`；审计只读、检查全量元数据/路径并按group/mode-camera/sequence分层解码媒体，不生成target crop。另有`example_datasets/samples`保存每套16帧的可公开浏览微型真实样例，仅用于理解数据。
2. **WP-1：冻结数据契约，已完成。** `docs/data_contract.md`和`configs/data_contract.json`已冻结一次性初始化事件、RGB历史、UWB与路由元数据、8点底盘系expert轨迹、辅助GT及`null + valid/mask`缺失值；`scripts/validate_data_contract.py`只读强制执行输入/标签隔离、四种条件模式、坐标/时钟不变量及源准入gate。逐帧bbox只能在label域，不能出现在model input域。
3. **WP-2：打通Phase 1最小闭环，进行中。** 已确认InternData-N1可提供pose ego几何流，TpT可提供identity流，SAGE3D identity流则因投影bbox/visible审计失败而暂时阻断；已实现真实的`omtrackvla.training.train`、`omtrackvla.evaluation.{evaluate,render,gate}`入口、3个Phase 1配置和单卡小样本闭环，能产出B1-ID、B1-GEO、B1-PROBE与固定`viz_val`视频。DA3-SMALL的固定源码、权重和隔离依赖已部署，真实probe已验证depth/confidence/pose/intrinsics/feature并修复坐标转换。剩余工作是完成SAGE3D只读修复侧车和准入审计、用多场景held-out样本冻结DA3尺度/置信度过滤证据，然后运行正式规模baseline与8卡流水线。
4. **WP-3：打通Phase 2基本跟随。** 从Habitat episode元数据实际渲染RGB并由oracle生成robot expert future waypoints，或接入经WP-0确认的现成expert数据；实现四种模态模式。没有真实UWB日志时，可从同步robot/target pose生成明确标记为`simulated_uwb`的数据，但不得声称已覆盖真实UWB误差。
5. **WP-4：打通Phase 3恢复。** 先完成3A离线受控扰动，再验证仿真可从模型访问状态查询expert后实现3B DAgger。始终混入Phase 2干净专家数据，并用相同scene/seed对比Phase 2与Phase 3。
6. **WP-5：每个工作包都回填。** 在第9～11节追加实验、失败和修改记录；写明命令、commit、数据manifest、checkpoint、指标和产物路径。失败也要记录，不覆盖历史，不只汇报总loss。

### 当前明确阻塞项

- Phase 1所需4个Python入口和3个Phase/benchmark/gate配置已实现；Phase 2/3仍缺6个配置及相应实现，因此当前只允许`--phase 1`通过正式preflight，`--phase all`失败是预期行为。
- 四种目标条件模式、canonical坐标/时间接口和最小样本schema已由WP-1冻结，但这不等于源适配已通过：SAGE3D抽取器的投影bbox水平符号及可见性已确认错误，修复侧车通过独立检测器/深度复核前不得进入身份loss或评测；TpT没有expert waypoint且物理时钟未解决；InternData-N1不是人物跟随数据。严格数据划分仍由NEXT-015冻结。
- 当前没有真实UWB日志、设备标定或误差统计被纳入仓库；在获得这些数据前，只能验证接口和仿真UWB，不能完成产品级UWB鲁棒性结论。

### 当前可视化与判读边界

| 可视化 | 位置 | 现在可以检查什么 | 不能据此判断什么 |
|---|---|---|---|
| InternData-N1预览 | `example_datasets/samples/previews/intern_data_n1.jpg` | egocentric RGB连续性、视角与场景外观 | 无人物目标，不能检查身份跟踪或跟随效果 |
| SAGE3D预览 | `example_datasets/samples/previews/sage3d_extracted.jpg` | 仿真人物外观、连续帧和GT投影bbox是否合理 | 黄框是离线GT，不是模型预测，也不是后续模型输入 |
| SAGE3D bbox审计 | `docs/sage3d_bbox_audit.md`及H100的`results/wp2_sage_bbox_audit_v2/` | 黄框投影错误、水平镜像反事实、独立person detector框及mode/camera/时间分层统计 | detector是审计证据，不自动等于目标GT；当前源bbox/visible未准入训练 |
| TpT clean v2预览 | `example_datasets/samples/previews/tpt_bench_clean_v2.jpg` | 真实机器人鱼眼画面、人物尺度变化和GT bbox | 黄框是离线GT；不能证明时钟已对齐或模型已学会重识别 |
| Phase 1 smoke视频 | H100共享工作副本的`results/wp2_phase1_memory_smoke/phase_1/visualizations/phase1_viz.mp4` | render链路、`PRED/GT`标识、身份/几何面板和视频编码是否工作 | 仅2-step开发smoke，正式gate预期失败，不是可报告的模型效果 |
| Phase 1 smoke指标 | 同目录的`metrics.json`、`report.md`、`gate.json` | 指标字段、报告和失败原因是否完整 | 不能作为baseline数值或Phase 1验收结果 |
| DA3单clip probe | `docs/da3_geometry_probe.md`及H100的`results/wp2_da3_probe/report.json` | 坐标转换、输出shape和单clip误差 | 同clip校准与评测，尚不能证明跨场景尺度/置信度准入 |

正式Phase 1完成前，NEXT-010还必须产出多场景depth/confidence、预测/GT鸟瞰轨迹和误差曲线；正式8卡baseline必须另行产出固定`viz_val`视频，不能复用上述smoke视频充当验收。

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
| DEC-030 | 2026-09-07 | WP-1统一使用anchor时刻底盘系（x前、y左、z上，米/弧度）、归一化xyxy bbox、含anchor的8点绝对局部XY轨迹和显式时间偏移；序列只在history index 0消费一次外部bbox，UWB tag ID仅作路由；所有缺失值用`null + valid/mask` | 消除不同源的坐标、时间、缺失值和输入/标签边界歧义，同时阻止未确认source semantics静默进入训练 | 已确认；SAGE3D policy、TpT physical clock、real UWB继续由机器可读gate阻断 |
| DEC-031 | 2026-09-08 | Phase 1固定按Intern `group/scene`、SAGE3D `run`、TpT `sequence`隔离划分，种子为20260907；训练adapter只能内存裁剪初始化目标，不得向源数据写crop/cache | 防止scene/人物/episode泄漏，并修复旧外部loader会在源目录生成`_target_crops/_target_refs`的问题 | 已实现；manifest只枚举split unit和必要元数据，不扫描全量媒体 |
| DEC-032 | 2026-09-08 | Phase 1先以无下载权重的共享ResNet-18建立可执行双流baseline；DA3-SMALL作为Apache-2.0几何teacher候选，实际输出验证前不得生成pseudo label | 先验证数据、分布式训练和评测闭环，同时不把尚未安装/验证的DA3接口伪装成已完成 | baseline smoke与DA3单clip真实接口probe已完成；多场景pseudo-label准入仍未通过 |
| DEC-033 | 2026-09-08 | DA3使用官方commit `3d835ec1...`和DA3-SMALL revision `e08cab65...`的独立源码、Python target目录和校验权重；pose必须经过OpenCV/Habitat相机轴桥接、Intern底盘外参、canonical基变换和显式metric scale | 不污染现有训练环境；真实probe证明直接混用OpenCV与Habitat外参会产生近90度平移和yaw符号错误 | 已确认；单clip修正后平移/yaw均值误差为0.00496m/0.00519rad，但同clip标定与评测不能用于正式准入 |
| DEC-034 | 2026-09-08 | SAGE3D源`bbox/visible`在修复侧车通过独立person detector、深度和人工拼图审计前，不得进入Phase 1身份loss或指标；修复与伪标注不得覆盖只读源，detector/ReID仅用于离线监督或审计，不构成后续推理bbox输入 | 384帧分层审计确认抽取器水平轴符号错误、画面外零面积框和未验证遮挡可见性；直接训练会把背景当目标并污染身份记忆 | 已确认；TpT身份流和Intern几何流不受影响，SAGE3D修复转NEXT-021 |

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
| DATA-009 | Sage3D：`/data/nfs/share/OmTrackVLA/data/sage3d_extracted` | 有，2,132,276帧，另有等量depth | 可从首个可见投影bbox构造 | 有逐帧投影bbox/visible标签；源episode含目标语义信息 | 有逐帧robot pose/yaw | 有逐帧target pose/target_local；无真实UWB | 有8点ego waypoint；1,975,856帧有效 | 已抽取仿真run | Phase 1B、Phase 2、Phase 3A | 约160 GiB；7,110个索引episode中7,105个canonical accepted、5个rejected；2个accepted与源`success`不一致；按run隔离 |
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
| FAIL-003 | EXP-003 | SAGE3D中后段黄色GT框频繁落在人物另一侧的墙、门或窗上，另有`visible=true`但框在画面外 | Phase 1 smoke人工查看及384帧分层detector审计 | 抽取器定义`right=[-sin(yaw),cos(yaw)]`，实际为left，却使用`u=cx+fx*right/fwd`；水平镜像反事实将冲突69降至14；5个源框裁剪后零面积 | 已完成只读审计、反事实统计和可视化；尚未生成正式修复侧车 | 源`bbox/visible`禁止直接训练，修复进入NEXT-021 | 坐标投影测试必须覆盖非中心目标、左右两侧、出画、遮挡和多相机；首帧中心样本不能作为唯一验证 |

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

## 12. 下一步计划

| 优先级 | ID | 工作项 | 依赖 | 预期产出 | 状态 |
|---:|---|---|---|---|---|
| P0 | NEXT-001 | 盘点现有数据及其字段、规模、路径和质量 | 数据目录访问 | `docs/data_inventory.md`、机器可读manifest和完整 DATA 表 | 已完成；全量元数据/路径检查和分层媒体解码通过；严格划分转NEXT-015 |
| P0 | NEXT-002 | 将已选变换约定落成robot、target、UWB和waypoint坐标系规范 | 传感器/仿真接口 | 含公式、单位和时间语义的坐标系规范 | 已完成；见`docs/data_contract.md`，具体source adapter仍须逐gate确认 |
| P0 | NEXT-003 | 设计一次视觉初始化与内部目标身份记忆的训练样本表示 | 输入数据格式 | 目标身份条件规范 | 已完成；初始化是history index 0的一次事件，后续bbox仅作label |
| P0 | NEXT-004 | 定义 Phase 2 最小训练样本 schema | NEXT-001～003 | 样本字段与缺失值规则 | 已完成；schema v1及只读验证器已落库 |
| P0 | NEXT-009 | 在候选几何backbone上验证camera pose、depth、confidence及中间feature接口 | 模型权重与视频样本 | backbone/pseudo-motion可用性报告 | 已完成接口探测；DA3-SMALL固定源码/权重和隔离运行时已在H100真实输出有限depth/conf/pose/intrinsics及layer 5/11 feature，见`docs/da3_geometry_probe.md`；pseudo-label准入转NEXT-010 |
| P0 | NEXT-010 | 定义DA3 pose到统一SE(2) pseudo trajectory的转换、尺度校准和置信度过滤 | NEXT-002、NEXT-009 | 几何伪动作规范 | 进行中；OpenCV w2c→c2w→OpenCV/Habitat桥接→base→canonical SE(2)已由真实probe修正，单clip robust scale后误差0.00496m/0.00519rad；多场景held-out尺度、退化过滤和confidence阈值待冻结 |
| P0 | NEXT-011 | 定义FutureNav式forward/inverse/单步next-state目标及feature teacher | NEXT-009～010 | World-Action辅助loss规范 | 首版已实现；inverse、motion-conditioned forward和action-free next-feature loss已进入Phase 1 baseline；DA3 layer 5/11 feature接口已确认，但teacher pseudo label仍受NEXT-010 gate阻断 |
| P0 | NEXT-012 | 固定主流UWB产品适配接口并盘点实际设备字段、频率、延迟和LOS/NLOS能力 | UWB设备/SDK或日志 | UWB输入与标定规范 | 待开始 |
| P0 | NEXT-013 | 定义UWB-only冷启动到自动视觉绑定的数据采集与标注协议 | NEXT-002～004、NEXT-012 | tag—track配对样本规范及歧义标签 | 待开始 |
| P0 | NEXT-014 | 定义目标视觉失联、RGB故障、UWB失效及安全停车/恢复状态机的标签语义 | 控制与安全接口 | 失效模式和评测规范 | 待开始 |
| P0 | NEXT-015 | 建立Phase 1/2/3的train、val、viz_val和test_locked manifest | NEXT-001～004 | 版本化benchmark及固定episode/seed列表 | Phase 1已完成：Intern 2980/372/186/187，SAGE 729/91/46/46，TpT 37/5/2/3；Phase 2/3待后续source解锁 |
| P0 | NEXT-016 | 冻结每个Phase的指标、可视化布局、baseline和初始gate阈值 | NEXT-008、NEXT-015 | benchmark配置与验收规范 | Phase 1首版已实现B1-ID/B1-GEO/B1-PROBE、固定视频布局和初始gate；阈值须在正式baseline后复核冻结 |
| P0 | NEXT-017 | 实现8×H100一键训练、评测、渲染、gate和断点续跑入口 | 训练代码、NEXT-015～016 | `run_pipeline_8xh100.sh`及可复现产物目录 | Phase 1真实模块已接入且单卡preflight通过；正式8卡run与Phase 2/3仍待完成 |
| P0 | NEXT-018 | 实现启动器约定的training/evaluation/render/gate模块及9个Phase/benchmark/gate配置 | 模型、数据schema、NEXT-004、NEXT-015～016 | 正式preflight通过并完成最小Phase 1训练 | Phase 1的4入口/3配置及单卡最小闭环已完成；正式规模8卡与Phase 2/3的6配置待完成 |
| P0 | NEXT-019 | 为4090机器配置GitHub认证并推送`wam`分支 | GitHub HTTPS token或SSH key | `origin/wam`及协作者拉取命令 | 已完成 |
| P0 | NEXT-020 | 将`PROGRESS.md`纳入仓库并建立协作者顺序工作包 | 当前项目事实与数据初盘 | GitHub可见的单一协作入口 | 已完成 |
| P0 | NEXT-021 | 修复并重新准入SAGE3D bbox/visible身份监督 | EXP-003、相机外参、RGB对齐depth、模块化person detector/ReID | 不改源数据的版本化修复侧车、投影/遮挡测试、独立审计报告和Phase 1 admission gate | 进行中；水平符号根因已确认，需正确重投影、画面相交/深度遮挡判定，并对残余歧义使用detector+OSNet/DINO；通过前SAGE身份loss保持阻断 |
| P1 | NEXT-005 | 收集真实UWB误差、偏置、漂移、延迟、丢包和置信度校准统计 | 定位模块日志 | 噪声模型报告 | 待开始 |
| P1 | NEXT-006 | 定义 Phase 3 噪声矩阵和难度课程 | NEXT-005 | 扰动配置规范 | 待开始 |
| P1 | NEXT-007 | 验证仿真是否支持任意访问状态的 expert relabel | 仿真环境 | DAgger 可行性结论 | 待开始 |
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
| Phase 1最小闭环 | `baseline v1` | `omtrackvla/{data,geometry,models,training,evaluation}`及3个Phase 1配置 | 只读双流adapter、共享ResNet-18、B1-ID/B1-GEO/B1-PROBE、render和gate；正式训练待完成 |
| Phase 1 smoke产物 | `EXP-001` | `results/wp2_phase1_memory_smoke/phase_1`（不进Git） | 带GRU history的2-step开发checkpoint、metrics、report、gate失败原因和8帧可视化；不得当作正式baseline |
| DA3几何probe | 官方commit `3d835ec1...`；模型revision `e08cab65...` | `scripts/probe_da3_geometry.py`、`docs/da3_geometry_probe.md`；运行JSON在`results/wp2_da3_probe/`（不进Git） | 真实4帧H100输出已验证；修正后单clip scale/translation/yaw为2.07116/0.00496m/0.00519rad；多场景准入待完成 |
| SAGE3D bbox审计 | `EXP-003` | `scripts/audit_sage3d_bboxes.py`、`docs/sage3d_bbox_audit.md`；H100运行产物在`results/wp2_sage_bbox_audit_v2/`（不进Git） | 384帧分层审计确认投影水平符号和可见性错误；含detector对照、水平镜像反事实和72例拼图 |
| 项目进展记录 | `v14 (2026-09-08)` | 仓库根目录`PROGRESS.md` | 本文件；新增SAGE3D bbox审计失败、根因、准入阻断与NEXT-021修复计划 |
