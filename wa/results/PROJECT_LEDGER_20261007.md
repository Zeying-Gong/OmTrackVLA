# WA 行人跟踪项目台账｜2026-10-07

## 当前结论

新学生61377完整评测3652/4215成功，SR86.642942%，初始化失败0。三类SR均超过同协议LightNav；STT仅净多3条。用户新目标：STT严格超过91.7%，即至少1289/1405（比当前净增13）；DT至少1173/1405、AT至少1203/1405。碰撞继续记录但不是本轮主要优化目标；不改物理与成功判据。达到后再分析UWB有无的影响。
这是用户授权的评测集内适配，不是未见测试泛化。WA为RGB+首帧GT框+理想模拟极坐标UWB，无文本；LightNav为RGB+文本，不同输入条件。
已核验24分片完成、4215唯一task/key、每类1405、权重身份、78个源文件哈希与指标重算。全量逐例初始状态配对及视频审计尚未收尾，不能写整体验收完成。运行中曾通过64条STT/DT起点抽检、64视频元数据/首帧哈希检查及AT八分片首条配对/媒体检查；抽检不替代全量审计。

## 模型与训练

输入输出和架构沿用[2026-10-05台账](PROJECT_LEDGER_20261005.md)：冻结DINOv2-S/14、64个MetaQuery、16层ActionExpert、七步SE(2) Flow Matching、目标几何辅助和JEPA潜在动力学。无Qwen/VLM；JEPA仅训练辅助，不是在线MPC。4帧因果历史；首帧GT框仅作模板，不持续提供GT框或未来GT路径。
损失保留动作Flow+0.5×几何+0.1×潜在动力学；没有RL、架构或成功阈值改动。
双教师从相同状态分别连续运行：成功分支优先；双方成功选following_rate高者；平局LightNav。双方失败不作示范；LightNav回退分支保留基准成绩但不作示范。
4215组配对：LightNav选择2865、Oracle1125、双方失败225；剔除5个被选中回退分支，3985条示范。候选458001窗口，转换为因果七步SE(2)后436816有效窗口。
独立从59866模型和优化器step22707开始，不从60502续训。61377/72474使用8×A800，新增1epoch、累计2epochs；最终step59065，2026-10-06 07:27:18北京成功结束。
每卡batch2、梯度累积2、有效batch32、seed42、history-repeat概率0.25；教师repeats1。原726631窗口+教师436816窗口，实际八rank曝光726626原数据+436814教师，教师37.545%；尾部丢7窗口（原5、教师2）。不盲目追加epoch3。
训练配置：[dual_teacher_train_a800_v1.yaml](../jobs/dual_teacher_train_a800_v1.yaml)。训练冻结commit：865124050fe7a3b15402630969b9aa00cb942672。

## 完整闭环结果（均1405条/类）

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
各根含PARTITION_COMPLETE.json和combined_episodes.jsonl，逐行artifact_root保留视频出处。新综合HTML尚未发布；旧18797属于60502，不冒充新结果。不上传checkpoint/数据/视频到GitHub，仅备份代码、配置及关键结果。

## 下一步与边界

先完成全量起点/视频审计与新页面；随后分析STT失败而教师成功的样本，有限改进，验收STT至少1289、DT至少1173、AT至少1203。未授权由本记录自动提交新训练；不盲目epoch3。
达到后先测同权重有/无UWB依赖，再区分无UWB重训的纯视觉能力；两者不是同一实验。当前无真实UWB噪声/丢包/多径、真实机器人、Thor/RDK时延或产品级避障验收。
旧台账及旧评测修复记录保留。本次备份不修改运行源码、模型或已完成产物，不重新提交评测。
