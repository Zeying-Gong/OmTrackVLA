# WA 行人跟踪项目台账｜2026-10-07

## 当前结论

新学生61377完整评测3652/4215成功，SR86.642942%，初始化失败0。三类SR均超过同协议LightNav；STT仅净多3条。用户新目标：STT严格超过91.7%，即至少1289/1405（比当前净增13）；DT至少1173/1405、AT至少1203/1405。碰撞继续记录但不是本轮主要优化目标；不改物理与成功判据。达到后再分析UWB有无的影响。
这是用户授权的评测集内适配，不是未见测试泛化。WA为RGB+首帧GT框+理想模拟极坐标UWB，无文本；LightNav为RGB+文本，不同输入条件。
已核验24分片完成、4215唯一task/key、每类1405、权重身份、78个源文件哈希与指标重算。2026-10-07全4215条与双教师的初始RGB/动态状态配对审计通过；4215首帧JPEG哈希与视频流/时长元数据检查通过，未逐帧解码所有视频。上述结果真实性审计完成，但STT>91.7%的新目标尚未达到。

## 模型与训练

输入输出和架构沿用[2026-10-05台账](PROJECT_LEDGER_20261005.md)：冻结DINOv2-S/14、64个MetaQuery、16层ActionExpert、七步SE(2) Flow Matching、目标几何辅助和JEPA潜在动力学。无Qwen/VLM；JEPA仅训练辅助，不是在线MPC。4帧因果历史；首帧GT框仅作模板，不持续提供GT框或未来GT路径。
损失保留动作Flow+0.5×几何+0.1×潜在动力学；没有RL、架构或成功阈值改动。
双教师从相同状态分别连续运行：成功分支优先；双方成功选following_rate高者；平局LightNav。双方失败不作示范；LightNav回退分支保留基准成绩但不作示范。
4215组配对：LightNav选择2865、Oracle1125、双方失败225；剔除5个被选中回退分支，3985条示范。候选458001窗口，转换为因果七步SE(2)后436816有效窗口。
独立从59866模型和优化器step22707开始，不从60502续训。61377/72474使用8×A800，新增1epoch、累计2epochs；最终step59065，2026-10-06 07:27:18北京成功结束。
每卡batch2、梯度累积2、有效batch32、seed42、history-repeat概率0.25；教师repeats1。原726631窗口+教师436816窗口，实际八rank曝光726626原数据+436814教师，教师37.545%；尾部丢7窗口（原5、教师2）。不盲目追加epoch3。
训练配置：[dual_teacher_train_a800_v1.yaml](../jobs/dual_teacher_train_a800_v1.yaml)。训练冻结commit：865124050fe7a3b15402630969b9aa00cb942672。

## 已完成训练：受限 STT 难例分支 61609/72803

2026-10-07 10:35:58 北京提交，15:50:25 成功结束，实际 8×A800-SXM4-80GB。独立继承59866模型及优化器，新增37009次更新、累计2epochs、最终step59716；不是对61377追加epoch3。模型、loss、controller、physics及成功判据不变。
原726631窗口和全部教师436816窗口各一次，对91条有效难例的10413个教师窗口额外重复2次（总3次）。实际八rank曝光1184272：原数据726631、教师457641，教师占38.643234%；难例31239、难例前2秒9810。教师STT/DT/AT曝光148547/156048/153046；每rank148034；1个非难例AT/Oracle尾部窗口丢弃。逐位置、逐源窗口、分组、rank与冻结计划完全一致。
训练冻结 `source_hard_stt_train_v1` commit `8d8efe3aa8a7ce9714913b6f65e5e3c8196eae06`；配置 [hard_stt_train_a800_v1.yaml](../jobs/hard_stt_train_a800_v1.yaml)，SHA256 `99f8fc7491b4f7c4c347c73b0ba4f1bf363ff292c76de0d4b29fd9850df19184`。
新权重：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61609/task_72803/wa_hard_stt_train_a800_v1/checkpoint.pt`
SHA256：`c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52`。
完整训练审计：`/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/hard_stt_training_audit_61609_v1.json`，SHA256 `59d5b468eec4283db1a1d87ad7b9a9d1bc2123739cffc38cebfc9a1e14dcc727`。24198源哈希、checkpoint实际加载、模型及优化器继承、最终步数、曝光、日志和heldout指标验收通过。仅证明训练产物合格，不证明闭环目标达成。

| 61609离线模式 | 窗口数 | ADE m | FDE m |
|---|---:|---:|---:|
| image |73368|0.265734|0.463479|
| point |73368|0.253940|0.442441|
| mixed |73368|0.253890|0.441945|

监控纠错保留：早前内联脚本误读输出子目录console.log，将缺失当空日志；其“无fatal”判断撤回，train.jsonl步数证据不受影响。已改查task_72803/console.log及.md-ak/workload.log，必需日志缺失直接报错；终态全文检查未发现指定fatal/OOM签名，原xFormers及NCCL警告保留。未重启训练或覆盖产物。
[TensorBoard](http://127.0.0.1:16006/#scalars) 的 `hard_stt_61609` 是已结束训练曲线，不代表仍在训练。查看电脑执行 `ssh -N -L 16006:127.0.0.1:6006 devpod-a800`；既有scalar API曾与NAS实际数据核验。
候选/拟合依据：[HARD_STT_CANDIDATE_20261007.json](HARD_STT_CANDIDATE_20261007.json)。40项训练CPU测试及4更新开发检查已完成；88早期窗口拟合仅作依据，不是SR。

## 正在运行：61609 三任务24卡完整闭环

[预检报告](STUDENT61609_PREFLIGHT_20261007.json)：98项相关CPU测试执行通过；新权重真实RTX4090短接口检查通过（2次合成RGB推理，不是Habitat轨迹/端侧时延）；真实数据24路定义预检通过，STT/DT/AT各1405，72个产物哈希回读一致。定义预检未加载模型、reset模拟器或渲染首帧，不能冒充真实闭环。
独立冻结 `source_student61609_eval24_v1` commit `192b57f5e270acfffd8c7c1a4590cb1b257d92a3`，工作树干净。配置：[STT](../jobs/student61609_stt_a800_v1.yaml)、[DT](../jobs/student61609_dt_a800_v1.yaml)、[AT](../jobs/student61609_at_a800_v1.yaml)，各8A800、1405条，共24卡/4215条。2026-10-07 16:13北京已提交：STT61653/72847、DT61654/72848、AT61655/72849。配置与预检已备份GitHub `b27a39b9`；旧61377/60502结果不复用或覆盖。
启动审计[STUDENT61609_STARTUP_20261007.json](STUDENT61609_STARTUP_20261007.json)：北京时间16:17:19，实际24×A800-SXM4-80GB及24个模型ready契约通过；完整有效STT25、DT23、AT23，共71条，重复/半行/契约错误/初始化失败均0。24路均有实际trace/steps，尚无COMPLETE；这不是全量SR。真实task根console/workload及worker/server日志未见指定fatal，Gym/xFormers警告保留。
三个新输出根分别为：

- STT：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61653/task_72847/wa_student61609_stt_a800_v1`
- DT：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61654/task_72848/wa_student61609_dt_a800_v1`
- AT：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61655/task_72849/wa_student61609_at_a800_v1`

协议仍mixed/zero/seed7/learned_yaw_guard_v1、固定语义修复与7例初始框修复；RGB+首帧GTBBox+理想UWB，无文本。A800实时显示29空闲时选择其优先；资源提交前重查，不能承诺立即调度。61377基线评测为RTX4090，新评测A800，不声称跨硬件逐位一致；需全量初始RGB/状态配对及成功增退分析。
17:26:27最新进度：STT923/1405、DT794/1405、AT813/1405，共2530条；初始化失败0，尚无完成分片。本轮10分钟只读观察13个快照从2256增至2530（+274），三任务/24路均推进；调度器三次查询均RUNNING，观察进程退出不是评测结束。见 [本轮连续观察记录](STUDENT61609_WATCH_20261007T0916Z.json)，[前轮记录](STUDENT61609_WATCH_20261007T0858Z.json)保留。实际task根与worker/server日志未发现四种指定致命签名，不代表没有任何警告；HM3D语义描述加载、mesh、Gym和xFormers历史警告保留。
固定各任务8分片首条共24条起点配对通过，包含失败而非挑选成功；144份教师证据重新核验，raw sensor SHA一致、动态state atol1e-6。见 [24条配对抽查](STUDENT61609_STARTPAIR_BOUNDED24_20261007.json)，不是全4215条/媒体验收。
另完成相同固定24条的[录像元信息与一致性抽查](STUDENT61609_MEDIA_BOUNDED24_20261007.json)：视频/complete/动作记录/结果步数一致，时长为帧数÷20；实际路径均属此次新任务，文件hash稳定。不是逐帧视觉审查、HTML链接验收或4215条全量媒体验收，20fps回放时长也不是模拟耗时。
新[MP3D边界抽查](STUDENT61609_MP3D_BOUNDARY_20261007.json)：17:08:36另一次快照STT747/DT631/AT648，其中MP3D63/0/0；各STT分片首个已完成MP3D共8条（含失败、排除旧24）与双教师起点配对通过，48份教师文件验真。按manifest和实际资产路径识别MP3D，不猜场景ID；DT/AT的16路尚无MP3D，明确待查。七例初始框repair静态计划通过，但运行仅见1例，另6例待查；不是全部修复例或全4215条验收。
后续[DT/AT MP3D边界抽查](STUDENT61609_MP3D_DTAT_BOUNDARY_20261007.json)：17:19:01–03的独立快照为DT733/AT747，其中MP3D50/63；新增固定15对（DT7/AT8）起点全部通过，90个教师文件hash、原始审计payload及选中row一次保存。DT lane3当时尚无MP3D，仍待查；不重复STT8例。七例repair当时3例运行标记正确、4例待完成，但3例均不在这15例中，不冒称完成了其起点配对。
61609尚无全量SR；完成后独立验收STT≥1289、DT≥1173、AT≥1203，再决定UWB对照，不盲续训。

## 已完成的61377完整闭环结果（均1405条/类）

| 类别 | WA成功 | WA SR% | LightNav成功 | LightNav SR% | Oracle SR% | WA比LightNav百分点 |
|---|---:|---:|---:|---:|---:|---:|
| STT |1276|90.818505|1273|90.604982|91.174377|+0.213523|
| DT |1173|83.487544|1128|80.284698|86.120996|+3.202847|
| AT |1203|85.622776|944|67.188612|87.259786|+18.434164|

| 类别 | WA TR% | LightNav TR% | WA HumanCollision CR% | LightNav CR% | WA macroTR% |
|---|---:|---:|---:|---:|---:|
| STT |87.238169|86.565345|4.341637|2.989324|92.061019|
| DT |79.234993|79.436705|6.619217|6.192171|81.922280|
| AT |85.036584|75.759135|5.124555|7.758007|87.903467|
TR=sum(following_step)/sum(max(total_step,reference_step))；每类52条缺reference时使用实际步数。macroTR另列。CR为曾距目标人小于0.5m，不是墙/门框或一般障碍物碰撞率。WA与双教师各类初始化失败均0；LightNav回退例数STT4/DT5/AT2保留其成绩。
同key互斥成功数：STT WA独自59/LightNav独自56；DT145/100；AT312/53。
原60502保留3627/4215=86.049822%；本轮多25条成功（STT0、DT15、AT10），+0.593120个百分点。新模型三类SR仍低于Oracle；不称所有指标全面领先。

## 机器、任务与产物

代码分支仍为OmTrackVLA/wa。开发入口devpod-4090，共享百度持久NAS；本轮无跨NAS迁移。训练8A800，评测三任务各8RTX4090，共24卡，全部SUCCEEDED；不复用旧模型行，不因SSH断连重跑。
STT61423/72526结束2026-10-06 11:43:32；DT61424/72527结束12:12:37；AT61425/72528结束13:21:10（北京时间）。评测冻结commit：5ed88af27ee637129af44b11727f244e312eecbd。
新checkpoint：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61377/task_72474/wa_dual_teacher_train_a800_v1/checkpoint.pt`
SHA256：`b5236a21f2d2695780029503c97e339c8350dc4f7337d05ec7e8692b9b9c327f`；step59065，累计2epochs。
父checkpoint：`/data/nas_ray/project/md-ak/users/zeying.gong/job_59866/task_70728/wa_history_repeat_v1/checkpoint.pt`
父SHA256：`ab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d`；模型+优化器继承。
原60502checkpoint：`/data/nas_ray/project/md-ak/users/zeying.gong/job_60502/task_71381/wa_recovery_mix_a800_v1/checkpoint.pt`；SHA256 `20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331`；保留未覆盖。
教师缓存：`/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/dual_teacher_se2_cache_20261006_v1`
教师选择与审计根：`/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/dual_teacher_complete_audit_20261006_v1`
STT结果：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61423/task_72526/wa_student61377_stt4090_v1`
DT结果：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61424/task_72527/wa_student61377_dt4090_v1`
AT结果：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61425/task_72528/wa_student61377_at4090_v1`
各根含PARTITION_COMPLETE.json和combined_episodes.jsonl，逐行artifact_root保留视频出处。不上传checkpoint/数据/视频到GitHub，仅备份代码、配置及关键结果。

合并审计：`/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/student61377_full_audit_20261007_v1`；summary SHA256 `30dfdbd9263ca0ba3615cb1845938de235c7e1b15b2141edb5ac584622f6bbf3`。
新学生视频浏览：`/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/student61377_review_20261007_v1`；audit SHA256 `7d2000efd58135753c9a4b5a8fe5bd3c228593ad2427f6278fc66841a293261d`。HTML、audit.json及首条视频HEAD均返回HTTP200。

在查看页面的电脑执行：`ssh -N -L 18798:127.0.0.1:18798 devpod-4090`，打开 [新学生4215条结果](http://127.0.0.1:18798/)。旧18797仍为60502结果，不替换或混用。

## 下一步与边界

Goal与20分钟持续跟进已启用，验收STT至少1289、DT至少1173、AT至少1203。难例覆盖、分组拟合、受限采样接入及开发检查已完成；61609训练最终真实性验收及三任务预检已完成。三任务各8卡完整闭环已提交并有实际结果，继续监控原任务直至最终配对及媒体审计，不提交集群smoke或复用旧模型结果。只在这次闭环结果支持时继续有限尝试；不盲目续训epoch3。
达到后先测同权重有/无UWB依赖，再区分无UWB重训的纯视觉能力；两者不是同一实验。当前无真实UWB噪声/丢包/多径、真实机器人、Thor/RDK时延或产品级避障验收。
UWB只读接口审计保留：[UWB_INTERFACE_AUDIT_20261007.json](UWB_INTERFACE_AUDIT_20261007.json)。image路径不发送测量字段且屏蔽坐标token；控制器使用模型预测geometry，不直接读实测UWB。后续不以“mixed坐标填零”冒充无UWB，不把推理时移除与无UWB训练混为一谈。
后续入口准备：[UWB_EVAL_MODE_PREPARATION_20261007.json](UWB_EVAL_MODE_PREPARATION_20261007.json)。显式image模式已贯通server、worker、ready、结果、完成标记、合并及视频/HTML；默认mixed不变，image禁止复用旧结果和实测UWB控制器。98项相关CPU测试执行通过（含21项新模式/发布测试），已有61377全4215条/78源文件hash只读回归通过、指标未变；合法False失败保留完整分母。尚未加载真实最终权重验证无UWB端到端行为，未跑image轨迹、未产生UWB消融结果。正式对照仍需先达SR门槛，再做真实接口/observer扰动/数据预检、同权重全量配对；不把LightNav比较当作有无UWB效果。
旧台账及旧评测修复记录保留。本次备份不修改冻结源码、模型或已完成产物；只为新权重准备独立全量评测，不重复旧模型任务。
