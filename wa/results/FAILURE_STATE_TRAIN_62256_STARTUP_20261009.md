# WA 62256 八卡训练启动核验

截至2026-10-09 00:50:24北京时间，Job62256/Task73511已经开始实际训练，不再只是排队。累计step23175，本轮从59866的step22707起新增468次更新，占计划38545次的1.2142%。尚无已验收新checkpoint、完整heldout或闭环SR。

## 身份和运行契约

- 调度器于00:44:42将Job和Task变为RUNNING，worker shell于00:44:46开始执行。先前CPU/GPU/placement排队记录保留，不把K8s Job的早期Running当成当时已训练。
- 输出根：`/data/nas_ray/project/md-ak/users/zeying.gong/job_62256/task_73511/wa_failure_state_train_a800_v1`。
- 冻结训练源码：`199385cd9c826c8f21308ad99b6a8375396d9c90`；配置：[failure_state_train_a800_v1.yaml](../jobs/failure_state_train_a800_v1.yaml)，SHA256 `08553eb73e8d2db4ae2fd37ab1c322bc7fbf77ac4a3ba3a7c59ae2508a2f2e69`。
- worker识别8张NVIDIA A800-SXM4-80GB，单卡81920MiB、cc8.0、driver550.163.01。8个UUID不同，见原sibling launch JSON。
- 配置world_size8、每卡batch2、累积2、有效batch32、seed42、JEPA world_weight0.1、history_repeat_probability0.25，diagnostic=false。
- parent为59866的原checkpoint，SHA256 `ab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d`。冻结trainer先载入model和optimizer并检查step22707，之后才生成config和训练记录；本轮不额外读取大checkpoint进行独立optimizer张量审计。
- 正式新增1epoch、累计2epochs；不是从best61609继续第三epoch。模型、loss、LR方案、控制器、物理和成功判据未改。
- 实际worker日志记录24项输入SHA通过及WLA288源码核验。environment记录冻结commit且dirty为空。

## 真实训练进度

| 观察时间（北京时间） | 证据 | 累计step | 本轮更新 |
| --- | --- | ---: | ---: |
| 00:47:08 | NAS已有train.jsonl，527字节 | 22708 | 1 |
| 00:49:19 | 主线程读取日志末条 | 23000 | 293 |
| 00:50:24 | 20条完整JSONL，末条elapsed202.7051秒 | 23175 | 468 |

首条loss0.116216667、grad_norm0.842128694；末条loss0.111114323、grad_norm0.636187613、peak_allocated_gib8.7016487。该快照所有已记录loss、flow、geometry、world、grad_norm、LR、elapsed和峰值显存均有限。原trainer在首次更新及累计step为25倍数时记录，20条日志不等于只训练20步。日志中的loss是DDP累积组均值，不能视为闭环成功率。

计划总曝光1,233,424 positions，其中base726631、原teacher457641、新recovery49152。当前`failure_state_exposure.json`仍为计划/初始化记录，不是完成后的实际曝光；不从阶段更新数推断三源实际分项。终态须核对actual_exposure及每rank消费，无丢弃、重复或越界。

## 启动和快照证据

下表路径除sibling launch外，均相对上述输出根。静态启动JSON不代表训练完成。两项日志SHA只绑定当时已读取的前缀字节；后续正常追加会改变完整文件SHA。

| 文件 | 字节 | SHA256 |
| --- | ---: | --- |
| sibling `wa_failure_state_train_a800_v1.launch.json` | 10813 | 846a06f4f3f68d7abb8e39a77fb01f870433b1eb1fdbb559a44b93b40f2eb601 |
| config.json | 3783 | 5ff152cbff4e72183606c3884f4f2e7493be2cc4d18d77627d59c45a6408e5b0 |
| environment.json | 59556 | deadca571fd363ca97ff03d2564930e55ad1aeccc2eb973e8bb1ae3b5def428f |
| failure_state_exposure.json | 38270 | 51892cb22777723821480fe0da7fa392aee7d35d17e4e785cdc8737f4726c0ad |
| train.jsonl读取前缀 | 10549 | 446e8cdd3635d614895feae387c7b7d3bbba898ff35048d17abd68ad77137f38 |
| Task73511/console.log读取前缀 | 27544 | abe833a800a3beee5f98130bf6f98f7a461c1c18b485a49a49b0c17cc5ed883d |

00:50:24的console前缀中，Traceback、CUDA OOM、ChildFailedError、Segmentation fault、NCCL error模式命中均为0。保留48次“xFormers is not available”和8次NCCL设备映射警告，不写“完全无异常”。训练记录已越过初始化并持续增长，不因警告文字判定死锁，也不修改正在运行的源码。

该时点checkpoint.pt、metrics.json、actual_exposure_epoch1.json和worker_postcheck.json均不存在，这是运行早期状态而非丢失产物结论。后续checkpoint存在也不等于完成，因为原流程在heldout之前写最终权重。

另一次独立只读核验于00:52:21通过：26项CLI、24个pin与worker及console相同；冻结源码241文件及WLA288文件SHA匹配。日志进一步增长到step23450、phase743（31条完整记录），loss/grad/LR均有限且组合符合flow+0.5geometry+0.1world。未重新读取大checkpoint，不把启动核验称为终态验收。

## 01时10分持续训练快照

2026-10-09 01:10:17北京时间，日志已到累计step26075、本轮3368/38545次更新（8.7378%）。读取136条完整记录、无半行；记录step序列符合首次22708及后续25倍数，phase_step均等于step减22707。所有已记录loss分量、梯度、LR、elapsed和显存峰值有限，loss与flow+0.5geometry+0.1world的最大浮点差为2.45883711724737e-8。末条loss0.1622880548、grad_norm1.0026556253、peak8.9229012GiB、elapsed1387.6603秒。调度器01:07只读查询仍为Job及Task RUNNING。

train.jsonl读取前缀71783字节，SHA256 `247fc80211779a304461e39da605c8375fd4964dbbfe73dc0ea4a6f3bc84f8e6`；console前缀95112字节，SHA256 `6c87ff91b02276e7d27279bf9f461c350f64f83102081a5124bb48c64f92099d`。同样五个fatal模式命中0，xFormers警告文字48次。两项SHA仅绑定快照前缀，训练继续追加会改变整个文件SHA；有限日志不代替终态模型张量审计。

中间文件`step-0024000.pt`与`step-0026000.pt`均已出现，各4358281329字节；本轮只读取文件元数据，未载入、哈希或验收它们。最终`checkpoint.pt`、`metrics.json`、`actual_exposure_epoch1.json`、`actual_exposure_epoch1.npz`和`worker_postcheck.json`仍不存在。保留现有训练，不用中间权重提前替代计划终态评测。

已有终态审计器和固定152样本candidate入口不需重写或重测；[审计器报告](FAILURE_STATE_TRAIN_AUDITOR_20261008.md)补充了绑定62256的完整待执行CPU命令，[固定拟合报告](FAILURE_STATE_GROUP_FIT_20261008.md)已有后续candidate命令。另一只读agent的devpod-4090入口发生kex关闭（exit255），该入口已停止重试；主线程devpod-a800读取正常，不因此认定训练中断或网络全局不可用。

## 后续验收和资源边界

维持现有62256，不重复提交、迁移或另开4090正式训练。4090开发GPU6于00:47:18/00:47:48满足短检查余量（24053MiB，利用率0%/5%），但未运行额外诊断；该快照不等于独占资源或4090训练兼容已验证。

完成后核验38545新增更新、最终step61252、三模式各73368完整heldout、8rank真实曝光、parent及checkpoint来源，再做固定152样本拟合和STT/DT/AT各1405闭环评测。最终目标仍STT>=1289、DT>=1173、AT>=1203，随后同权重无UWB配对；best61609及历史失败保留。

本报告是运行中启动核验，不是终态训练审计、H100复现、ModelScope上传或新SR结论。
