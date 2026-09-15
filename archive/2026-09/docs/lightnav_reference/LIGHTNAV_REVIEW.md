# LightNav-0 对 OmTrackVLA 原始需求的参考价值审查

LightNav-0 值得作为**独立对照模型和端侧工程参考**，目前不能直接替换 OmTrackVLA 并宣称满足原始需求。它公开了真实的语言条件导航推理、轨迹解码、历史视觉缓存、ROS 控制和 Jetson Thor 启动代码；缺少本项目要求的目标 RGB+bbox / 坐标 / 混合三模式输入、UWB 绑定与降级、可校验的同一目标重识别和有界主动搜索执行链路。继续增加当前 waypoint 训练量，也不会自动补齐这些接口和控制能力。

本审查只读公开源码和本地已有审计材料，未安装依赖、执行下载的代码或测试、下载权重、启动 GPU 工作、修改 OmTrackVLA 运行代码。作者公开成绩与本项目实测结果分别标明。

## 1. 固定版本与证据边界

- 仓库：<https://github.com/lightorigins/LightNav-0>。
- 审查 commit：`c6f40e3220edbf7011e4f17eaf2c865416737d4d`，下文 GitHub 引用均固定到此版本。
- 可读源码：`source_snapshot/`；逐文件 Git blob、SHA-256、字节数在 `SOURCE_MANIFEST.json`。
- 已提取 205 个文本文件。Git 对象已下载，但 Windows checkout/archive 被 MuJoCo 资产文件名中的冒号阻断；通过 `git ls-tree` 和 `git cat-file --batch` 提取源码，未执行仓库代码。演示资产和 `docs/assets` 媒体未纳入文本快照。目录内 `source_snapshot.tar` 为失败尝试留下的 **0 字节文件，不是有效源码归档**。
- README 指向 [LightOriginsHQ/LightNav-0 模型页](https://huggingface.co/LightOriginsHQ/LightNav-0)。最初两次访问遇到 TLS 接收错误，后由接管主进程用 Python HTTPS 成功读取元数据，再按固定模型 revision `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64` 读取模型卡、config、eval_config、动作 tokenizer manifest 和权重索引。元数据显示非门控，模型卡声明 Apache-2.0。原访问失败记录保留；本次没有下载模型张量，不能宣称权重内容已校验。小文件原始字节与哈希见 `HF_SMALL_FILES_READ_RESULT.json`。

## 2. 作者声称什么，公开代码实际提供什么

| 内容 | 作者材料 | 源码核对结论 | 对本项目的含义 |
|---|---|---|---|
| 统一视觉导航模型 | Qwen3-VL-4B-Instruct、语言任务指令和 RGB 历史；指向 token + 3 个 RVQ action token 解码为 10 个 SE(2) 路点 | `prompts.py`、`inference/samples.py`、`tracking.py` 提供相应输入构造及解码路径 [1][2] | 可以研究非人物目标和 ObjectNav，但语言描述不能等价于明确指定的目标图片/bbox或坐标 |
| 跟踪与导航共用 checkpoint | README 报告 STT SR 91.7%、DT SR 82.6%，并报告多个导航基准结果 | 仓库提供 EVT、Habitat 评测接入；本次没有运行这些基准 [1] | 不能把作者成绩记成本项目 SR；也不能说已覆盖本项目高密度动态碰撞指标 |
| 训练方法与数据规模 | README 描述分阶段训练和数据/环境规模收益 | 当前发布主要是推理、评测和部署包；在审查源码中未找到完整训练入口、optimizer/backward 训练循环及训练语料 [3] | 可以借鉴方法，不能声称接入后就获得可复现的训练数据流水线或 1000+ 合格环境 |
| 长历史与缓存 | 新近帧保细节、远期帧更稀疏/池化 | `slowfast.py` 实现分层采样；`vit_cache.py` 提供按会话的 tubelet LRU 缓存 [4] | 是可参考的性能设计；必须与模型训练时输入合同一致，不能直接改变现有模型四帧历史而期待等效 |
| Thor 端侧部署 | 明确写机器人本机 Thor 运行客户端+服务端，localhost 通信 | 存在 `serve_thor.sh` 和 LLM-only FP8 补丁 [5][6] | 不是只能云端部署；但仍需在我们的 Thor、相机及完整控制链上实测 |
| 机器人控制 | ROS2 客户端、MPC、Go2/TRON 适配器 | 有时间戳、请求匹配、里程计对齐、运动学 MPC、命令 watchdog [7][8] | 可借鉴工程模块，但不能把该 MPC 称为通用动态避障规划器 |
| 主动搜索和身份 | 语言导航可输出探索轨迹；输出可见性与指向位置 | 未发现完整的 LOST→SEARCH→同一目标确认→恢复跟随状态机，也未发现 UWB 或独立视觉 ReID 身份确认链 [2][9] | 缺口与本项目 `active_search_review.md` 一致，仍需独立补齐 |

这里“未找到”限定于固定版本的已审查发布源码，不推断作者内部是否已有未发布实现。

## 3. 接口差异：不能直接接上旧适配器

| 接口 | LightNav-0 | OmTrackVLA 当前/原始合同 | 必须做的适配 |
|---|---|---|---|
| 目标指定 | 当前 RGB + `instruction` 文本 | 初始目标 RGB+bbox；机器人坐标目标；二者混合 | 为独立基线明确语言任务来源；实现三模式需要新增训练与编码路径，不能仅将坐标拼成一句 prompt 后宣称完成 |
| UWB | WebSocket 协议没有目标坐标、质量、协方差、age、绑定字段 | 需要合法坐标变换、时序对齐、目标绑定及模态切换 | 保留独立传感器输入合同与缺失/冲突判定；point-only 指目标条件，环境障碍感知仍须可用 |
| 动作形状 | 通常 H=10，`[forward_m,left_m,yaw_ccw_rad]`，累积机器人局部位姿 | 当前策略统一 8 点 xy，秒偏移 `[0,.1,.2,.3,.4,.5,.6,.7]` | 明确坐标轴、旋转、轨迹参考时刻与底盘模型；不能 reshape 或截断后当作等义输出 |
| 起点 | 第一行是未来一步，无前导原点 | 当前 8 点合同包含 t=0 原点 | 不得把 LightNav 第一个预测点当原点丢掉，或把 OmTrackVLA 原点当首个运动目标 |
| 时间 | 协议未给路点逐点时间；机器人 MPC 的采样时间由控制配置决定 | 当前训练与采集以固定秒偏移为合同；EVT 时钟已需实测对齐 | 单独标定轨迹空间跟踪和速度约束，或取得可靠时间语义再重采样；不得直接照搬 EVT 的 `.375/.25/(pi/20)` 换算 [9] |
| 可见/停车 | grounding token 解码出的 bool；零轨迹表示 stop | visibility/stop 概率、身份诊断及独立控制门 | bool 不是校准的同一目标置信度；`visible=true` 与“同一目标被确认”分开计账 |
| 目标记忆 | 会话 RGB 历史，reset 清帧/帧 ID/缓存 | 固定原目标引用 + 因果历史 + 丢失/重获状态 | 保留 task/session 隔离；不能用新候选替换最初指定目标来制造重获 |
| 多目 | 当前服务协议每次一张图 | 原要求支持多目、也允许纯前方单目 | 多目需要相机 ID、标定、时间同步及训练覆盖；不是把多张图随意拼接 |

上述 OmTrackVLA 时间合同依据本地 `time_contract_review/时间契约审查.md`。源协议明确路点是**空间轨迹**，不能凭一个 10 点数组推定 1 秒或者与现有 0.1 秒标签相同。

## 4. 最值得复用的代码与限制

### 请求生命周期与采集时刻对齐

`robot_deploy/src/vln_client/vln_client/vln_client.py` 用单个在途请求关联 `seq`、episode 和原始 `CameraFrame`；结果返回后核对序号，任务改变后拒绝旧会话结果，并有 3 秒响应超时。`vln_node.py` 将原相机采集时间放入 `capture_stamp_ns`。这是本项目 worker 阻塞等待和过期结果问题的直接工程参考。[7]

MPC 在相机采集时刻匹配里程计，将当时的机器人局部轨迹投影到 odom；异步 solver 结果有 generation 校验。这比将旧图像的轨迹直接当成机器人此刻局部轨迹更适合移动底盘。[8]

**仍需补强后才能采用：**

- 当前 `_block_reason()` 只检查 odom 接收超时、运行状态及是否有轨迹，没有限制轨迹采集年龄；旧轨迹仍可能被重复求解并发布带新时间戳的速度命令。下游 watchdog 无法单凭新命令时间识别上游旧图像。需要贯通 capture→request→result→path→command 的有效期。
- 请求超时 3 秒是原库配置，不符合本项目实时控制期限，不能照抄；采集时钟、处理时钟及 monotonic 时钟应明确转换和回退规则。
- 接收消息解析了 episode，但 MPC 此处主要用 stamp/seq 排序，不能将该节点单独视为完备的任务授权/代际边界。
- 参考库的当前 ROS 图片编码常量是 **480×270**，Thor 文档性能条件写 **640×360**；复现实验必须记录实际送入模型的尺寸，不能混用两组数字。

### 轨迹跟踪 MPC 与执行 watchdog

`mpc.py` 是 CasADi/IPOPT 的单轮模型轨迹跟踪器：优化状态/参考误差和控制量，约束前进速度、偏航速度、加速度。公开优化问题没有障碍物几何、距离场、动态对象预测或扫掠区域约束。原实现不允许倒退，不包含独立侧移控制；若当前底盘允许全向移动，也要重新选择运动学模型。[8]

Go2 适配器确实调用厂商 `ObstaclesAvoidClient.Move`，因此不能说仓库所有路径都没有避障。但这是特定机器人提供的能力，不是可直接搬到 Thor 任意底盘的通用避障模块，更不是对原项目碰撞率的证明。[10]

`safety.py` 中模式/来源选择、有限数值和 command age watchdog 可借鉴；其纯函数只比较 `age <= watchdog`，不显式要求 age 非负且有限。采用时应拒绝负年龄、未知时钟、旧授权覆盖新否决，并保证独立执行进程到期即停。Tracking 模式的 `stop` 不用于“任务完成”，ObjectNav 的 stop 才完成任务；这种任务语义不能直接覆盖 OmTrackVLA 的保护停车/搜索状态。[8][10]

### 历史压缩、ViT 缓存与端侧量化

`slowfast.py` 可参考近期密集、远期稀疏、固定早期 anchor 的因果历史组织方式。`vit_cache.py` 的 key 为绝对 episode 帧 ID 对及网格大小，按 session 独立保存，适合减少重复视觉编码。[4]

复用前需保证同一帧 ID 的像素/预处理不变、reset 清缓存、模型/精度/图像尺寸变更时失效。它依赖 Qwen3-VL 的双帧 tubelet 与 deepstack 特征组织；OmTrackVLA 的视觉编码和 GRU 不能直接换成该缓存类。若输入历史合同改变，需要训练和对照验证。

Thor 启动器检查 GPU 名称后禁用三个不兼容 SM110 的 FP8 kernel；`fp8_llm_only` 补丁使 `visual.*` 线性层保持 BF16，并在没有匹配任何视觉层时报错。此实现绑定 vLLM 私有接口，应固定其验证版本；它适用于该 Qwen3-VL 路径，不是任意模型通用加速开关。[5][6]

## 5. Thor 性能：可参考，但尚未证明达标

作者文档报告：AGX Thor、单会话、640×360、64 帧历史；同段 12 帧贪心回放服务端中位 BF16 268 ms、LLM-only FP8 178 ms。另报告相机→客户端→服务端→MPC 完整闭环 4.5–4.8 Hz。这些是作者数据，本次没有在我们的硬件重测。[5]

**178 ms 是服务端中位数，不是本项目从采集到指令生效的 ≤200 ms 证明。** 完整环频率也不能直接当作单帧端到端延迟或 P95。至少应分别记录采集、预处理、排队、视觉编码、模型解码、控制、下发/生效时间，报告完整分布、超期比例和长时间运行表现；最终统计口径随原项目验收协议冻结。

文档还报告全模型 FP8 在 348 步回放中 visible 翻转 91 次、stop/go 翻转 121 次；LLM-only 与 BF16 的 stop 一致率 97.4%、路点差 p50 4.4 cm。这说明即使加速配置较好，也不能省略任务成功率、重获身份与碰撞回归。[5]

推理包要求 Python 3.11，固定 transformers 5.8.0、vLLM 0.19.1、nvidia-cutlass-dsl 4.5.2；Thor 文档指定 aarch64 CUDA 构建。现有 OmTrackVLA/Habitat Python 3.9 环境应保持独立。端侧安装采用离线 wheelhouse 和本地权重，不需要现场网络。[3][5]

## 6. 仍须由 OmTrackVLA 完成的功能

1. **三种目标条件的统一训练/推理合同。** 目标 RGB+bbox、真实机器人坐标流和混合输入分别有正确编码、有效位、单位、时间、目标绑定；切换不清错目标、不把 GT 坐标输入策略。
2. **任意物体及三路 Navigation。** 补非人物目标类别/实例、目标出视野、未知区域探索、ObjectNav/PointNav/混合导航的任务和数据；冻结独立场景验收。语言模型能输出物体名不等于完成指定实例跟踪。
3. **丢失后的主动搜索闭环。** 继续依据 `active_search_review.md` 分开冷启动无目标、已初始化后丢失和环境传感器失效；接入受限 SEARCH、原目标多帧确认、运动许可、累计预算、任务重置和执行 watchdog。原目标可见性不足应触发状态转移，不能永远被全局 visibility 门挡住，也不能无条件沿预测路点运动。
4. **真实访问前缀的训练标签。** 补可见性、可信 bbox、候选身份、可执行/应停状态与实际搜索/重获前缀。单独降低 recovery waypoint ADE 不足以证明搜索控制和原目标重获改善。
5. **静态/动态避障与可运动性证据。** 由合法实时传感器提供环境观测，结合底盘几何、制动、时间戳判定许可；UWB 仅定位目标，指向 token 仅是预测，二者都不能当作空闲区域证明。仿真 GT/navmesh 只能在评价端使用，不得伪装为策略输入。
6. **场景与最终指标。** 1000+ 环境、至少 500 个且占比≥50%的 Real2Sim、1000+ 自定义高密度/遮挡评测实例需要清单、来源许可、加载证据、去重及分割。LightNav demo 只随仓提供 ProcTHOR `val_2` 场景，不能填补场景数量。最终仍须按原要求验收 SR≥90%、静态碰撞≤1%、动态碰撞≤3%、Thor 端到端≤200 ms。

## 7. 许可与评测协议不能混用

根原创代码为 Apache-2.0，但 `THIRD_PARTY_NOTICES.md` 明确列出例外：VLN-CE/Habitat 适配部分 MIT；`evt_bench/trackvla_client_agent.py` 的 `evaluate_agent` 改编部分为 **CC BY-NC-SA 4.0**；MuJoCo 内随附 ProcTHOR/THOR 资产 CC BY 4.0；模型权重适用其 checkpoint 自身许可。后续成功读取的固定版本模型卡独立声明 Apache-2.0，这一结论来自模型卡，不是从根代码许可推断。CasADi、机器人 SDK、相机 SDK 和外部场景各有自己的条款。[11]

EVT runner 还有实际协议差异：`first_init` 只从 shard 第一个 episode 读取 instruction，并对该 shard 所有 episode 复用；源码备注称目标 ID 也遵循 upstream 的冻结方式，结果依赖 shard 数。OmTrackVLA 已在独立评测中处理逐 episode 的目标语义 ID 与初始化分母，不能整体导入此 runner 回退这些修复。可单独适配模型客户端，在冻结的本项目 evaluator 下测量，同时另保留作者原协议结果作区分。[12]

## 8. 建议落地顺序与用户配合

近期继续推进已接管的 OmTrackVLA：补真实前缀监督、接独立搜索状态机的时间/身份/许可接口与执行 watchdog，再做同 checkpoint、同场景的控制器对照。LightNav 作为平行参考，优先借鉴请求生命周期、采集时刻 odom 对齐和缓存设计。已确认公开模型卡的许可声明与非门控状态；下一步仍需独立环境方案、权重下载校验和冻结小规模基线，当前未安装依赖或启动其训练。模型 eval_config 实际规定 256×448 模型输入、4 fps 和 64 帧分层历史，与相机编码分辨率是不同环节；这些条件都要随对照记录。

需要用户提供的条件很具体，当前代码修复和仿真审计不依赖用户手动运行训练：

| 用户提供/协调 | 为什么需要 | 时间 |
|---|---|---|
| 机器人/底盘型号，速度控制和急停接口，外形尺寸、速度/制动限制；现有里程计/IMU接口 | 把模型路点变成有单位、可到期的真实动作，并校准扫掠区域 | 实机适配前 |
| Thor/RDK 的实际设备型号、内存、系统/JetPack版本及可用访问方式 | 生成能安装的离线部署包和实测完整延迟；参考项目不能代替本机测试 | 端侧性能阶段 |
| 相机数量/位置/接口/标定；UWB型号、坐标系、更新率、质量字段、目标绑定方式，已有短样例数据或日志 | 完成三模式时间对齐、绑定、失效和模态切换 | 接入真实传感器前 |
| 已有扫描场景/数据清单及许可，缺少部分的来源或采购协调 | 对 1000+/500+ Real2Sim 数量形成可核验分母；无须现在重采全部数据 | 场景扩充阶段 |
| 受控实机测试场地和在场操作人员，代表性非人物目标与高密度场景 | 验收搜索重获、动态避障和端侧延迟 | 仿真门槛通过后 |

原始验收范围已经选定，不需要因参考仓库再次要求用户二选一。现阶段最有价值的是用户告知现有硬件和数据清单；缺哪一项就标记具体缺口，工程与仿真工作继续进行。

## 固定版本源引用

[1]: https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/README.md#L50-L86
[2]: https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/src/lightnav/tracking.py#L61-L170
[3]: https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/pyproject.toml#L5-L65
[4]: https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/src/lightnav/inference/vit_cache.py#L1-L78
[5]: https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/docs/JETSON_THOR.zh.md#L5-L110
[6]: https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/src/lightnav/inference/vllm_utils.py#L254-L327
[7]: https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/robot_deploy/src/vln_client/vln_client/vln_client.py#L335-L454
[8]: https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/robot_deploy/src/vln_mpc/vln_mpc/mpc_node.py#L502-L745
[9]: https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/docs/PROTOCOL.md#L17-L164
[10]: https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/robot_deploy/src/robot_adapters/go2_adapter/go2_adapter/unitree_client.py#L232-L368
[11]: https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/THIRD_PARTY_NOTICES.md#L1-L151
[12]: https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/evt_bench/trackvla_client_agent.py#L259-L315

其他精确代码入口：

- [Thor launcher，SM110/FP8配置](https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/scripts/serve_thor.sh#L44-L55)
- [SlowFast采样层级](https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/src/lightnav/slowfast.py#L24-L40)
- [MPC完整目标和约束](https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/robot_deploy/src/vln_mpc/vln_mpc/mpc.py#L119-L173)
- [MPC轨迹新鲜度门](https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/robot_deploy/src/vln_mpc/vln_mpc/mpc_node.py#L613-L657)
- [Go2命令选择与age判断](https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/robot_deploy/src/robot_adapters/go2_adapter/go2_adapter/safety.py#L11-L32)
- [ROS传出采集时间](https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/robot_deploy/src/vln_client/vln_client/vln_node.py#L186-L223)
- [当前ROS编码尺寸与超时](https://github.com/lightorigins/LightNav-0/blob/c6f40e3220edbf7011e4f17eaf2c865416737d4d/robot_deploy/src/vln_client/vln_client/vln_client.py#L21-L28)
