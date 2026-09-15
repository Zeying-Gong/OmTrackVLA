# 仿真阶段进展（2026-09-14 19:18）

人物实例标签修复已通过真实渲染和独立原始数据核验；修正后的原 48 例仿真采集已启动。当前仍没有 SOTA 成绩，也尚未放行新的 optimizer。旧数据和任务 checkpoint 不覆盖、不冒充实例修复后的结果。

此前真实 Habitat 人物的 visual_scene_nodes 为空，旧编号分配没有生效。更改公开根/骨骼节点同样无法触及 skin 绘制树。当前修复使用每个实例的私有 AO 模板、同一份不可变 URDF/GLB/动作资产，并 force_reload=True，避免按原 URDF filepath 缓存导致 semantic ID 沿用。实际 API 为 sim.metadata_mediator.ao_template_manager；两个未注册 enum getter 改为读取其真实 Configuration 字符串。旧 V6/V7 接口失败记录完整保留；没有执行有潜在字段边界问题的双语义字段探针。

V8 在固定 DT、AT 场景中覆盖全部 8 个人物、初始 placeholder 与原任务替换路径。独立复核通过 542 个封存文件（139,979,974 bytes）、144 个原始 capture、72 次恢复检查、14 个执行源文件和 537 个独立引用。AT 中 8 个演员原先各自产生 5,948 个 ID1075 像素；修复后主目标仍1075，干扰演员各自为2002..2008，单独放在主目标位置时不再产生1075。DT 同样逐人通过。

原版/修正版在相同物理状态下，baseline 和全部单人隔离 RGB 逐像素完全相同；人物轮廓、空背景、相机投影/外参完全相同，初始物理状态最大差为0。各轮原生位姿恢复仍有已记录的浮点残差，满足预先固定的1e-6位姿界限和有界RGB残差协议，不能把初始跨环境“零差异”混称为所有恢复状态都零差异。实际传感器 semantic_target 均为 SemanticSensorTarget.SEMANTIC_ID。

root 已采纳精确 helper/factory 的渲染资格，资格 SHA 25c96e6727ad94a5fc547cf2a3be2a9c6a66655f962412d17d0f44afe417eba2。它仅证明实例构造与渲染身份，未授予动态 teacher 质量或 optimizer 准入。对照图见 docs/simulation_sota/VERIFIED_IDENTITY_FIX.png。

修正后的 48 例批次保持原场景、成员顺序、种子、V5 控制器与失败分母；48/48 远端 CPU 合同检查已通过。Prepared SHA 809501641fee9206ecf7d2fc7b4703f5e29c90d937a033be4978a61a0ad83455，bundle SHA 6feb88bd00271b83821c12438e181e9c53197e06d2c9def5070693ee33d89d7d。root 已单次启动 GPU3 串行采集，首批案例已保存真实帧并回收 worker，原有 guard 失败仍保留。每帧保存实际 AO 属性/源资产引用、相机与语义通道，reset 刷新采用原 detector 和相关 metric 链。批次完成后需再做控制文件封存和独立标签/时间/动作复算。

纯视觉序列接口也有独立 CPU 候选：forward 只有初始化 RGB、初始框和 [B,S,4,3,H,W] 历史 RGB 三输入，S 保留真实 policy-call 序列；每次从 reset 重建状态，内部固定禁用 UWB/binding。13+2 项真实 CPU 测试通过，验证逐步输出、burn-in/TBPTT 梯度、episode 隔离和权重更新后的初始 memory 重算。原 UWB encoder 的常量分支仍保留，不能称为删除 UWB 架构。候选未安装到生产，仍缺合格的序列 loader、真实静态校准与导航训练集成。

LightNav 已完成 181 个依赖的离线安装与独立核验，14/14 必需模块真实 CPU import 通过。GPU7 的 FP32/BF16 64×64 运算误差均为0，vLLM 两个扩展导入成功；该检查最后的16MiB Torch allocator假设过低，实测峰值33,644,544bytes，因此原 worker exit1完整保留，不能把整次检查改称成功。没有运行完整模型或 vLLM 自定义 kernel。GPU worker 已回收。

固定版本约9.70GB权重仍在分块传输，已补充针对短 HTTP EOF 和 TLS EOF 的三次同块有界恢复。证书、其他TLS、HTTP协议、长度、磁盘与SHA失败仍拒绝；已验分块重用，旧失败状态保留。截至本快照前检查，远端至少50块（838,860,800bytes）已通过SHA。只有全部578块、整文件官方SHA与safetensors结构复核通过，才能进入完整模型加载。

旧 V3/V5 成功率、旧 B1 开发 IoU 及三组任务 checkpoint 仍是旧实例编号链下的历史结果。修正版与强基线必须在同一环境/拆分重新运行，不能把修正后的成绩和他人旧原生语义链分数直接做SOTA排名。模型名义4fps、仿真实际世界时间、1/40秒动作积分也须明确分开；不能仅凭WebSocket序号声称时间一致。

项目仍按任意目标、视觉/坐标/混合三模式、丢失主动搜索和同目标重获推进，当前只需仿真。接下来完成修正48数据和固定dev4的独立检查，再 fresh 训练感知与导航，最后运行纯视觉闭环及同条件强基线。当前人物实例修复通过，不代表任意目标能力或三模式最终验收已经完成。
