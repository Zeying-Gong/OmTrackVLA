# WA 行人跟踪项目台账｜2026-10-07

## 当前结论

61609 新分支完成4215条完整闭环：STT **1279/1405（91.032028%）**、DT **1178/1405（83.843416%）**、AT **1207/1405（85.907473%）**，合计3664/4215（86.927639%），初始化失败0。相对61377分别净增3、5、4条，DT/AT维持并提升；STT仍差10条才能达到至少1289（严格超过91.7%）。**Goal仍为ACTIVE，不能标记达标；UWB消融尚未启动。**
三类SR均超过本项目同协议LightNav，但这是用户授权的评测集内适配，不是未见测试泛化。WA为RGB+首帧GT框+理想模拟极坐标UWB，无文本；LightNav为RGB+文本，不同输入条件。
已核验三个调度任务SUCCEEDED、24分片完成、4215唯一task/key、每类1405、78个源文件哈希；全4215条与双教师的原始RGB/动态状态起点配对通过。4215首帧JPEG哈希与视频流/时长元数据检查通过，未逐帧解码所有视频。指标、逐例增退及固定难例审计均通过；碰撞继续记录但不是本轮主要优化门槛。

## 下一轮准备进度｜2026-10-07 20:14

固定预算重平衡的代码已接入运行时，66项新旧CPU测试及全量8路采样索引核验通过。开发机RTX4090完成16个真实样本、4次模型与优化器更新，四类采样组均有消费；只做前置检查，未保存新checkpoint。代码与诊断已推送GitHub `wa`：`dd1fec9e2cba0450be7a33387c8a2e550db4cd91`。20:20冻结独立训练源码与配置，通过原始权重完整SHA、环境和依赖检查，随后20:23提交 **61715/72909（8×A800）**。20:33:41实际已到 **step24150／新增1443步，共计划37009步**，源码与父模型契约通过；尚未完成训练，不能写成新SR。
下一轮计划保持总曝光1,184,272次和DT/AT数据不变：困难STT前2秒保持3次，后段由3次降为2次；补充10个新退步episode的886个有效教师窗口，并为其余1248个成功STT episode选6257个唯一时序分位窗口加一次曝光。选中anchor窗口共12514次，全部anchor episode池共122679次，二者口径不同。
仍从59866模型及优化器独立开始、新增1epoch／累计2，不续训61609第三轮、不加LR、不改loss／控制器／物理／阈值。性能是否提升必须看训练后的完整4215条配对闭环。详情见[运行时前置检查](STT_ANCHOR_RUNTIME_PREFLIGHT_20261007.json)、[冻结发布记录](STT_ANCHOR_RELEASE_20261007.json)与[正式配置](../jobs/stt_anchor_train_a800_v1.yaml)。配置含完整命令与NAS输出路径，冻结源码为 `source_stt_anchor_train_v1`（`dd1fec9e`），配置SHA `4a5b465a9533c779a7990aa754575caee34dc302fa0238ecd62d3471cb3879ef`。

新任务输出：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61715/task_72909/wa_stt_anchor_train_a800_v1`；最终 `checkpoint.pt` 和SHA待训练终态审计，勿与同样预计step59716的61609权重混淆。
[启动与实际进度证据](STT_ANCHOR_STARTUP_61715_20261007.json)。新TensorBoard选择 `stt_anchor_61715`：先运行 `ssh -o ExitOnForwardFailure=yes -N -L 16006:127.0.0.1:6006 devpod-a800`，打开 [localhost曲线](http://127.0.0.1:16006/#scalars)。已验证实际SSH转发与标量；事件walltime为导出时间，非训练发生时间。

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

## 已完成：61609 三任务24卡完整闭环

[预检报告](STUDENT61609_PREFLIGHT_20261007.json)：98项CPU测试、真实权重短接口、24路真实数据定义检查通过。独立冻结 `source_student61609_eval24_v1` commit `192b57f5e270acfffd8c7c1a4590cb1b257d92a3`，运行期间不修改；配置：[STT](../jobs/student61609_stt_a800_v1.yaml)、[DT](../jobs/student61609_dt_a800_v1.yaml)、[AT](../jobs/student61609_at_a800_v1.yaml)。三项各8×A800、各1405条，全部重新运行，没有复用旧模型结果。

| 任务 | Job/Task | 北京时间2026-10-07成功结束 | 成功数 | SR% | 比61377净增 | 新增成功／退步 |
|---|---|---|---:|---:|---:|---:|
| STT |61653/72847|18:26:50|1279|91.032028|+3|15／12|
| DT |61654/72848|18:50:16|1178|83.843416|+5|26／21|
| AT |61655/72849|18:45:01|1207|85.907473|+4|25／21|

| 任务 | TR% | HumanCollision CR% | macroTR% | 初始化失败 |
|---|---:|---:|---:|---:|
| STT |87.522621|3.985765|92.064468|0|
| DT |79.634758|6.049822|82.210367|0|
| AT |85.074148|5.053381|88.071922|0|

完整记录：[STUDENT61609_FINAL_20261007.json](STUDENT61609_FINAL_20261007.json)，包含调度终态、监控、审计命令、逐例新增成功/退步和NAS产物hash。最终4215条完整分母均保留。STT原91个有有效教师窗口的难例恢复15、仍失败76；额外2个零窗口难例仍失败，不混入91分母。STT的15项新增成功全部来自原91难例，但12个原成功样本退步（7目标人Collision、1Lost、4Normal未成功），抵消了大部分收益。当前STT仍有88个教师曾成功而学生失败的样本（86有窗口、2零窗口），教师起点成功不等于从学生失败状态也可恢复，不能据此保证下一轮会成功。
协议仍mixed/zero/seed7/learned_yaw_guard_v1、固定语义修复及7例首框修复。新评测A800，61377基线RTX4090；起点配对通过但不声称跨硬件逐位等价或把全部变化单独归因为采样。没有训练epoch3、改loss/物理/阈值或新增LightNav。

三个运行根：

- STT：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61653/task_72847/wa_student61609_stt_a800_v1`
- DT：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61654/task_72848/wa_student61609_dt_a800_v1`
- AT：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61655/task_72849/wa_student61609_at_a800_v1`

合并根：`/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/student61609_full_audit_20261007_v1`。summary SHA `223b4b8140e79842408bc7104d16f75b7b4ff1d409354ee547b00e4b49f937f7`；combined SHA `0ab45e1b35bb0b8809fcc77fcaabca59b35bab0839507d716c371ce2d5a6f358`。
目标审计：`artifacts/student61609_goal_20261007_v1/goal_report.json`，SHA `92e388281c3074dee918c411f61f3624e545f19d6c9a06a83393fee2817b93a8`，165个源文件哈希验证、状态PASS但goal NOT_MET，差额10/0/0。固定难例审计：`artifacts/student61609_hard_outcomes_20261007_v1/hard_stt_outcomes.json`，SHA `b03ec11a2932ed317d9ebedbd420342f9ac26551b7a39872f4f99e072e681736`，168源hash通过。
新视频页：`artifacts/student61609_review_20261007_v1`，audit SHA `39d0376cccf925c1735264f686c3908cec67cbb2b9cbebc583c06e3da2b36c73`。19:04实际本机转发HTML/首视频HTTP200；远程HTML/audit/summary/首视频也均200。执行 `ssh -o ExitOnForwardFailure=yes -N -L 18799:127.0.0.1:18799 devpod-4090`，打开 [61609完整4215条结果](http://127.0.0.1:18799/)。18798仍为61377，18797仍为60502，不替换。
原Gym/xFormers/SSD语义加载警告保留；指定fatal/OOM签名未发现，不写完全无异常。先前各部分监控、24条抽查、MP3D边界和STT乐观上界报告均保留为历史证据；最终完整审计不再把当时“待查”误写为当前缺失。

## 已完成的固定窗口诊断与下一轮准备

[诊断证据](STUDENT61609_DIAGNOSTIC_20261007.json)：仅新61609模型在固定88个教师状态窗口上回放，普通历史/重复历史各一次，176条预测、1713输入哈希均通过。两次确认4090第6卡空闲后执行，597.856秒、峰值1.482GiB；没有训练或新增闭环轨迹。原61377误差来自既有固定报告，没有重新选窗或加载旧模型。完整新七步预测和标签存NAS，标签不进入策略输入。

| 普通历史组别 | 窗口 | 61377 ADE m | 61609 ADE m |
|---|---:|---:|---:|
| 难STT／碰撞 |24|0.487173|0.467611|
| 难STT／其他 |16|0.394402|0.383305|
| 原成功STT对照 |16|0.252272|0.255145|
| DT对照 |16|0.290127|0.310269|
| AT对照 |16|0.400361|0.371023|

这些是小规模教师状态拟合，不是SR或因果结论。DT窗口误差升高但完整闭环成功数仍提高，不能单凭拟合误差挑选最终模型。
NAS结果：`/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/teacher_group_fit_61609_replay_20261007_v1.json`，SHA256 `3db61f67a648258593382dd2adbfd508a2d8dbe0aeb077648b0e8b51f54d7f73`。

12条STT退步均不在原三倍hard组；10条已有成功教师的972候选窗过滤后886有效窗，前2秒357窗，本轮均只曝光一次；另外2条双方教师失败不能强行补标签。新旧24个trace哈希及48个info/review文件已核验。原先临时汇总986为算术错误，已明确修正为886；首次主校验器将info路径误写为扁平文件名而失败，改用真实scene/episode_info路径后通过，没有改变原始结果。
退步并非一种原因：Hax/16全程TR=1但终点0.981m低于原成功下限1m；部分样本后退指令已经存在但位移小；VLzq/197与/238持续向前却推进不足；Vt2/123及ac26/258的可见性没有恢复。VLzq/238最后动作已执行，但Lost分支在写info前退出，149动作只有148条info，不能伪称记录了末post状态。日志中的facing实为距离+detector共同决定的human_following，不是纯视觉标志。
76条原hard持续失败中，32条前40步内结束且全部为目标人Collision；38条Collision末5步平均均已后退。18条Lost和6条Normal末段有明显前进命令但实际水平速度<0.1m/s，没有接触几何证据，不能直接断定门框。已有75/76条有前2秒有效示范，不支持一概归因缺少启动标签。

据此先准备**一个固定预算再平衡候选**：base与全部teacher保留一次；3270早期hard额外2次、7143后期hard额外1次、10条退步示范886窗额外一次，另6257次均衡分给1248条原成功STT。额外20826次、总预算与61609相同；DT/AT原示范不减，仍从59866模型与优化器独立分支、1新epoch/累计2。原LR、loss、模型、控制器、物理、成功阈值不改。
独立v2计划已完成：14项CPU测试通过，24111个源文件hash及原资格重核；1248条锚点至少6个有效窗，选出的6257个索引全部唯一。八rank索引模拟各148034、合计1184272；仅丢弃与61609相同的非早期AT/Oracle尾部窗口。早期hard保持9810曝光，后期hard14286；15个既有增益样本总曝光3032、76个持续失败21064、10个退步1772、稳定成功锚点122679。这里均为计划与模拟，不是新训练实际消费。
[候选准备证据](STT_ANCHOR_PREPARATION_20261007.json)：NAS目录`artifacts/stt_anchor_candidate_61609_20261007_v1`；report SHA`bbe21c78d1b38731f812866158ea1492c0950d6f05843bfbc8d95fd4da3ff8bd`，plan文件SHA`98eb72e3aaf466046ec35e42f96317be43147e455e0c15c136958cae53377ca4`。工具首次把合法0.0/1.0结果拒为类型不合，已在创建输出前失败；修复严格二值兼容并补NaN/Inf等拒绝测试后通过，未放宽教师或标签过滤。
这是待验证方案，向后兼容runtime仍在接入，**尚未正式训练或证明有效**。旧v1只能统一倍数重复hard，新异质采样必须明确版本并验实际8rank逐窗口曝光，不能假称已有兼容。若仍不能解决偏离状态，再考虑有界学生失败前状态的双教师成功恢复示范，不继续无依据地堆整段重复或epoch3。

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
61377基准checkpoint：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61377/task_72474/wa_dual_teacher_train_a800_v1/checkpoint.pt`
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
61377基准视频浏览：`/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/student61377_review_20261007_v1`；audit SHA256 `7d2000efd58135753c9a4b5a8fe5bd3c228593ad2427f6278fc66841a293261d`。HTML、audit.json及首条视频HEAD均返回HTTP200。

在查看页面的电脑执行：`ssh -N -L 18798:127.0.0.1:18798 devpod-4090`，打开 [61377基准4215条结果](http://127.0.0.1:18798/)。旧18797仍为60502结果，不替换或混用。

## 下一步与边界

Goal与20分钟持续跟进保持ACTIVE，验收STT至少1289、DT至少1173、AT至少1203。61609训练与4215条真实性/媒体审计全部完成，三类净增但STT仍差10条。继续固定窗口拟合和增退轨迹诊断，依据证据选择下一项有限实验；不盲目追加epoch3或直接加大难例权重。没有新正式任务，不重跑已经完成的61609闭环。
达到后先测同权重有/无UWB依赖，再区分无UWB重训的纯视觉能力；两者不是同一实验。当前无真实UWB噪声/丢包/多径、真实机器人、Thor/RDK时延或产品级避障验收。
UWB只读接口审计保留：[UWB_INTERFACE_AUDIT_20261007.json](UWB_INTERFACE_AUDIT_20261007.json)。image路径不发送测量字段且屏蔽坐标token；控制器使用模型预测geometry，不直接读实测UWB。后续不以“mixed坐标填零”冒充无UWB，不把推理时移除与无UWB训练混为一谈。
后续入口准备：[UWB_EVAL_MODE_PREPARATION_20261007.json](UWB_EVAL_MODE_PREPARATION_20261007.json)。显式image模式已贯通server、worker、ready、结果、完成标记、合并及视频/HTML；默认mixed不变，image禁止复用旧结果和实测UWB控制器。98项相关CPU测试执行通过（含21项新模式/发布测试），已有61377全4215条/78源文件hash只读回归通过、指标未变；合法False失败保留完整分母。尚未加载真实最终权重验证无UWB端到端行为，未跑image轨迹、未产生UWB消融结果。正式对照仍需先达SR门槛，再做真实接口/observer扰动/数据预检、同权重全量配对；不把LightNav比较当作有无UWB效果。
旧台账及旧评测修复记录保留。本次备份为新完整结果、训练和checkpoint路径、诊断代码及后续门槛；不修改冻结源码、权重或旧结果，不重复已完成任务。
