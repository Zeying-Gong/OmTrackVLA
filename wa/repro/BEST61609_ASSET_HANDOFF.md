# WA 61609 跨机器资产交接
补充：[环境重建元数据包](BEST61609_ENVIRONMENT_METADATA_20261009.md) 已按另一固定 ModelScope revision 上传并读回验 SHA；[公开来源与原 NAS 827 文件哈希清单](BEST61609_PUBLIC_ASSETS_20261009.md) 可供协作者自行获取资源后核对。二者均不代表 H100 已具备完整运行环境。

目标是让另一台 8×H100 机器评测同一份 WA 61609，而非重训。**2026-10-09 最新状态：四个预备归档和六个原始清单已上传到非公开 ModelScope 数据集，并按固定 revision 全部下载回 NAS 验 SHA**，见 [固定版本、十文件哈希及剩余缺口](BEST61609_MODELSCOPE_REVISION_20261009.md)。双环境、场景与人物/机器人资产尚未完成异机交接，也未验证 H100 闭环。原命令、完整路径和协议见 [复现指南](BEST61609_REPRODUCTION.md)。

## WA 成品权重与旧初始化权重

WA 保留 WLA 来源的 MetaQuery、ActionExpert 和目标几何头结构，但不运行 Qwen 或语言 LoRA。冻结入口 `192b57f5` 的 `eval_server.py:72–78` 先构造模型并加载旧 WLA/JEPA 初始化，再用 WA checkpoint 覆盖全部非 encoder 状态。因此，这些初始化文件是旧启动实现的依赖，不是另一个在线 WLA 模型。

2026-10-08 对真实 step59716 checkpoint 的 CPU mmap 只读检查得到：

| WLA 来源模块 | 61609 覆盖位置 | 张量数 | key、shape、dtype |
| --- | --- | ---: | --- |
| ActionExpert | policy.action_expert | 261 | 完全匹配 |
| MetaQuery | policy.adapter.metaquery | 1 | 完全匹配 |
| 目标几何头 | policy.target_head | 6 | 完全匹配 |

checkpoint 共494个非 encoder 状态张量，其中 policy486、robot_action4、robot_state4；模型张量合计1,452,579,160字节。包含94个 `policy.world.*` 张量。冻结 DINO encoder 不在此 checkpoint 中，必须另外交接或按已固定官方哈希获取。

`training.py:58–59` 与 `adapters.py:161–167` 的在线预测只经过编码器、WA 条件融合、动作专家和目标头；不调用 JEPA world predictor。JEPA 辅助训练过的模型不等于在线世界模型规划。原 JEPA 初始化文件的读取也属于构造流程，不能据此认定它是在线算法必需输入。

2026-10-08 23:04已完成独立直接加载函数的[A800等价检查](../results/STANDALONE61609_EQUIVALENCE_20261008.md)：669个完整状态张量和619模块非state属性一致，包含JEPA普通Tensor mask；旧/新各6次固定合成输入Session预测逐值一致。新函数实际只加载WA与DINO，不读WLA/JEPA初始化文件。保留原192b快照、所有模型模块和原计算方法；不改物理、控制器或评价标准。随后[RPC及全量接线](BEST61609_STANDALONE.md)已实现且A800真实RPC通过；完整闭环、H100实机及资产上传仍需完成，不能把有限等价检查当成全量复现。

## 已找到的权重

| 文件 | 原始字节 | 交接用途 |
| --- | ---: | --- |
| WA61609 checkpoint | 4,358,277,705 | 必须保留的原成品与来源锚点，含优化器 |
| DINOv2-S/14 | 88,283,115 | 在线视觉编码器，对方已报告哈希通过 |
| WLA step0043203 | 4,118,468,737 | 仅原入口初始化；暂不作为新精简包必传文件 |
| JEPA mz_jepa-wm | 211,639,615 | 原入口初始化及训练辅助，不是在线 predictor 调用 |

本轮重新计算三个待交接文件的 SHA256，均符合原记录：WA `c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52`；WLA `0b8f036fd282474d8c9efe9e34e1f4dc56b9282c44295bcda1f16edcf48460e1`；JEPA `a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc`。未导出或覆盖这些文件。若另导出不含优化器的推理权重，必须给新文件独立 SHA、原 c510 来源及逐张量等价报告，不冒用原 SHA。

## 固定评测证据的精确交接范围

以下为2026-10-08 22:30北京时间的固定清单 stat 结果；不是已打包上传。教师媒体沿用已有逐文件哈希，本轮没有重哈希或重新解码全部媒体。

| 范围 | 文件数 | 字节 |
| --- | ---: | ---: |
| manifest、三份val、BBox计划、基线、七份probe报告 | 13 | 6,993,573 |
| 教师selections、8430分支各三份初态文件 | 25,291 | 2,317,549,686 |
| 上述合计 | 25,304 | 2,324,543,259 |
| 再附原配对审计路径与哈希清单 | 25,305 | 2,330,389,716 |

三份 val 的真实路径为 `/data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA/data/datasets/track/{STT,DT,AT}/val/val.json.gz`，由原 manifest 固定身份。

七例 BBox 证据只需 `artifacts/initial_bbox_repair_v2/plan.json`、它指向的61171 `combined_episodes.jsonl` 和七份 probe `report.json`，无需 probe 图片或基线录像。

教师初态仅需每分支 `pair_start.json`、`observations.json` 和 observations 首条引用的 PNG，不需要教师动作、后续帧、视频、教师推理模型或训练缓存。精确25290路径和预期哈希来自 `artifacts/student61609_full_audit_20261007_v1/student_teacher_pair_audit.json` 的 `teacher_artifact_hashes`，其 SHA 为 `dc98149bdcfb6baf05d1fae167de572b476679f48384dd91ea92e57851645844`。不要整包复制61264/61269任务目录。

JSON保留原字节与绝对路径，不能重写后跳过哈希校验。接收端要么在授权隔离环境保持布局，要么另做可审阅的路径适配。首PNG为场景渲染内容，公开再分发不能由代码仓库公开性推定。

## 双环境并非两个独立的小目录

本轮限定范围统计的环境与overlay约22.36GB未压缩，包含目录元数据，尚非最终最小包或压缩包实测：

- `probe_env` 约41MB，是继承 system-site-packages 的 venv；Python绝对链接到约8.46GB的 `envs/wla_evt_torch28`。只复制41MB会遗漏模型运行环境。
- `envs/habitat` 约8.37GB，实际包含Habitat-Sim0.3.1和Magnum/Corrade二进制。
- `torch_overlay` 自身仅4810字节，七个链接目标约5.24GB，实际位于 `envs/janusvln_base`。须按确定目标交接，不能只打包软链接，也无需泛复制整个Janus环境。
- `official_runtime` 约223MB、Xvfb bundle约13.7MB、Habitat-Lab约3.19MB及所需外部评测代码另外保留。当前实际路径没有 BENCH/build 或 BENCH/src_python，不把它们列为已找到资产。

尚无已验证的可搬移环境包；限定检查未找到可用conda-pack、docker/podman或现成镜像归档。Python环境包不包含目标主机NVIDIA驱动、EGL/OpenGL、glibc和libstdc++兼容性保证，H100还需真实接口验证。当前精确包版本信息见 [环境清单](../results/STUDENT61609_ENVIRONMENT_INVENTORY_20261008.json)，它不是完整二进制安装锁。

## 发布渠道与安全范围

现有 GitHub OmTrackVLA 仓库为public。源码、说明、元数据清单和哈希放 GitHub；不要把大权重塞进Git历史。GitHub普通仓库拒绝超过100MiB的文件，二进制发布需另用其支持渠道，见 [GitHub官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

ModelScope现有备份仓库已核实为 private；本次使用它的新 `handoff/best61609_20261009_v1/` 前缀，上传账号和固定 revision 的十文件读回均已验证，详见[本次交接](BEST61609_MODELSCOPE_REVISION_20261009.md)。不得依赖默认可见性；不得把上传账号能下载等同于协作者已有权限。SDK能力见 [ModelScope Hub](https://github.com/modelscope/modelscope_hub)。

原模型环境中发现user-site ModelScope1.39.1及hub0.4.0，但导入因缺cryptography失败。上传工具应安装到单独工具环境，不向正在使用的训练或仿真环境补包。凭据不进入命令行参数、源码、Git、压缩包或日志；通过隐藏输入或授权凭据机制登录。

资产按 allowlist 分包、生成包内文件和压缩包各自SHA，上传后按固定revision下载校验，再交给协作者。不包含home配置、SSH、token、训练缓存或无关任务。DINO/JEPA上游按固定提交获取并保留许可证；JEPA快照含CC-BY-NC4.0及第三方许可，不能将整套依赖统一改称Apache。HM3D/MP3D及人物资源先核验接收方已有资产和许可，仅补获准交接的缺项，不公开整库场景。

## 已生成的 NAS 私有交接包

2026-10-08 23:56:38北京时间，两个包在原NAS生成并通过逐文件和归档读回校验，状态为 `PASS_NAS_BUNDLE_NOT_UPLOADED`。目录为项目根下 `artifacts/best61609_private_bundle_20261008_v1`；打包耗时63.968秒，不占GPU，不修改原权重、科学JSON或PNG。此处记录归档文件SHA，不能拿归档SHA替代内部checkpoint的c510 SHA。

| 文件 | 字节 | SHA256 |
| --- | ---: | --- |
| best61609_weights.tar | 4358287360 | 23a9445518908411fca039f9b7330edadc2fa3e259a8896a6573d95927fa7331 |
| best61609_evidence.tar | 2374461440 | 4e8c8aa803a51b58e2d291e8cef15dcc05c3c79f50b64dbb1a48c58a3cee9faf |
| manifest.json | 12533791 | c9e193bd335b10af74866df7a5652e58aa5c4a135fccdc401f94a84e1e2c7894 |
| complete.json | 1152 | 9dae3b87c38eae88b8051eca1c008fc3ea7dbc8d3ce2e4eb65d6630ed6a4d62d |

权重包只含原WA61609文件，内部4,358,277,705字节，SHA `c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52`。证据包含25,305文件、原内容2,330,389,716字节。归档成员使用相对 `data/nas_ray/...` 名称；JSON中的原绝对引用未改写。**没有包含DINO、WLA/JEPA初始化权重、外部架构源码、双环境、场景、视频或训练缓存**。DINO仍单独需要；接收方已报告其官方哈希通过。

打包工具为 `wa/tools/pack_best61609_assets.py`（SHA `de437ffe79315583ec02269604df135b1d0d514080dc9954dd138759c3432876`）；固定清单工具为 `wa/tools/best61609_asset_manifest.py`（SHA `6027b97b0d6961f4a14ca26e07ea98f34cb1ef7cb0d793737a4a61fec6ef67b4`）。34项CPU测试通过16.904秒，包含符号链、非普通文件、重复/额外成员、SHA变更、PAX长路径、填充隐藏数据、尾部截断和完成标记原子发布。首轮29项有一项错误消息正则不匹配，拒绝行为正确；修正fixture/断言并增加5项负例后通过，未放宽产品校验。

真实命令在checkout执行：`PYTHONNOUSERSITE=1 PYTHONPATH=. R/probe_env/bin/python -u -m wa.tools.pack_best61609_assets --output R/artifacts/best61609_private_bundle_20261008_v1`，将R替换为本文原项目绝对路径。既有输出不允许覆盖，也不要为查看结果重新执行打包。日志为该目录路径加 `.log`，SHA `627dadfb09ef8bf28b1915712faaac1206b39e73edbbb54f35d2afa0acdc9f78`。

2026-10-09 00:00:23北京时间，独立审计使用标准库tarfile而非生产校验函数，22.578秒内读完两包并逐一验证25,306成员与完整归档SHA，全部匹配；重复、链接、额外成员、危险路径和稀疏文件均为0，归档与清单读取期间未变。没有解压或重读原NAS媒体。

该段原为两个包刚生成时的状态。**最新交付状态**见[固定 ModelScope revision 交接](BEST61609_MODELSCOPE_REVISION_20261009.md)：已上传并从固定提交读回验 SHA；私有协作者权限、运行环境和 H100 完整复现仍未验证。不得把“上传账号能下载”写成“协作者已能下载”。

[独立RPC及启动器](BEST61609_STANDALONE.md)已通过A800有限等价检查。H100完整运行、外部架构源码与双环境的可移植交接、场景一致性检查仍未完成；这两个包不是完整环境镜像，也不构成新的闭环SR结果。

## 私有 WLA 结构源码包

2026-10-09 00:30:49北京时间已在项目NAS生成`artifacts/best61609_wla_source_bundle_20261009_v1`，状态`PASS_PRIVATE_WLA_SOURCE_BUNDLE_NOT_UPLOADED`。归档只含288个原WLA Python文件（5,528,170字节）及一份原61609环境来源JSON（50,234字节），共289成员；后者是元数据，不是Python运行环境。源码集合及逐文件SHA与61609记录完全相同，原始字节不变。此包不包含旧WLA权重、DINO/JEPA源码、场景、二进制环境、缓存或Git目录。

| 文件 | 字节 | SHA256 |
| --- | ---: | --- |
| best61609_wla_source.tar | 6103040 | 9685c93d33403c1c7eca3ecb9c7216cc9de2062078b7c1b2da790a2fabc67f14 |
| manifest.json | 134345 | 4efb7499025ad082239bc25f66c847bbdd56064ca5e20f6323edc18014a8b6ab |
| complete.json | 1184 | a328feef0d525e00df11a2c1730d76a6099cf39803f9d28c8126b25506a46182 |

来源锚点为原`job_61609/task_72803/wa_hard_stt_train_a800_v1/environment.json`，SHA `a76ba7506cac9a9a39db839fe554e7d0382c45ca76285fa58130f4bcba2546ad`。打包器`wa/tools/pack_best61609_wla_source.py` SHA `f186cea96b642e1ddd8f913f687397a2d7a5e984c1bdb7cdd0dc76993b8a127e`，复用前述已固定的安全归档函数；AST检查通过，实际运行exit0，打包验证0.216秒，不占GPU。只有新源码包生成，没有重打两个大权重/证据包或重跑评测。

00:32:34独立标准库tarfile审计通过，289成员名单、全部内容SHA、归档SHA及读取期间稳定性均匹配；重复、额外、链接、危险路径和稀疏成员均为0。有一个合法空`src/md_wla/cli/__init__.py`，其空内容SHA与原记录一致。没有调用生产验证器、解压或重读大资产。

接收方先在独立暂存目录核验tar及manifest，确认289个成员都是预期普通文件；不得直接向系统根或已有项目盲目解包。归档成员为相对`data/nas_ray/...`路径、文件权限0600，内容均为源码或JSON，不含可执行脚本。按已授权的隔离路径布局恢复后，外部比较完整288文件SHA清单。原快照未包含LICENSE，不将其公开上传GitHub，不把私有交接解释为开源授权。

DINO/JEPA应从官方仓库取得完整真实checkout并固定提交，见[源码获取步骤](BEST61609_STANDALONE.md#架构源码获取与核验)。直接删除`.git`、只挑若干模型文件或不加检查地全量pip安装上游依赖，均不能代替当前评测的源码和环境准入。

上述WLA结构包完成时，Habitat/EVT外部评测源码尚待交接；该部分随后生成下面的独立小包。双环境、场景及H100完整运行仍待核验。既有大权重/证据包和WLA结构包均保持不变；随后已上传，固定版本见[本次交接](BEST61609_MODELSCOPE_REVISION_20261009.md)。

## 私有 Habitat 与 EVT 源码配置包

2026-10-09 01:26:45北京时间，项目NAS的`artifacts/best61609_evt_source_bundle_20261009_v1`生成成功，耗时0.8137秒。包内373个文件、原始16,816,185字节：339份源码/配置加34个Xvfb `.deb`，保留3个合法空文件。不占GPU、不解包、不运行Xvfb安装器、不修改原源码；状态`PASS_PRIVATE_EVT_SOURCE_BUNDLE_NOT_UPLOADED`。

| 内容 | 文件数 | 范围 |
| --- | ---: | --- |
| WLA评测桥接 | 8 | text_action_v2/v3控制及agent、full common与manifest |
| BENCH桥接及辅助配置 | 11 | trained_agent、humanoid_infos、可选顶层import源码、run_evt_ten、两个Xvfb脚本 |
| evt_bench | 6 | 仿真器、动作、传感器与指标注册源码 |
| Habitat-Lab | 314 | 当前完整源码/配置树，排除bytecode缓存 |
| Xvfb运行包 | 34 | 原`.deb`文件；未执行安装 |

其中10个已有依赖逐文件SHA符合冻结192b的两份原清单；清单本身SHA也与192b一致。其余文件是本次当前源码快照，**不能声称373文件都已有61609历史逐文件等价证据**。原文件内容、版权头和现存元数据保留；快照缺少独立LICENSE文件，不推定整个包具有公开再分发许可。

| 文件 | 字节 | SHA256 |
| --- | ---: | --- |
| best61609_evt_source.tar | 17479680 | 9a16415b0513a0a6d1bc893f0e10e63fe1b271667413738a0863543af298295f |
| manifest.json | 188018 | e81ec4cf10db1f53cb6c016ab84a7dcaac2a303d8516fa08efe067b7db2e8c86 |
| complete.json | 1342 | 258267c89a7fc94cd75b1f3dd2eea6bd4f755b1630faabd7a9eba1156e7d3637 |

打包器`wa/tools/pack_best61609_evt_source.py` SHA `c5d0b8f2cd23a0474484f029704d54905f7be6368a7e988f9111137c1c387a49`，测试SHA `9bb9629295627089497c82f274e05782b68d3a677288e92ef66e3fcce1e7e4c1`。10项CPU测试PASS0.033秒，包括空文件、链接/特殊文件、固定哈希、重复/危险路径、源集合变化、失败无完成标记及已有输出不覆盖。逐文件内容在打包时重新验证，归档读回后再核对来源集合与全部SHA。日志为目录名加`.log`，SHA `fdad7e80ec2c89c350c77d78ee036f8e03197555af3434010c384768b3a283aa`。生成命令为在checkout运行`PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 R/probe_env/bin/python -B -u -m wa.tools.pack_best61609_evt_source`，R按本文原NAS路径展开；已有包不可重跑覆盖。

01:29:49北京时间，独立审计仅用标准库tarfile读取新包的三个文件，0.117797秒、exit0：373成员的名单、大小和逐文件SHA全部匹配，三个归档产物的大小/SHA及读取期间稳定性通过。链接、特殊文件、重复/额外成员、危险路径和稀疏成员均为0；尾部9728字节全零。没有调用生产校验函数、解包、重读原始来源或上传。

归档成员统一0600，接收方应先在新隔离暂存目录核验全部成员和哈希，再恢复到获准布局。启动器会直接执行`BENCH/scripts/runtime/run_xvfb.sh`，因此**仅该脚本**需在核验后的隔离目标执行`chmod u+x "$BENCH/scripts/runtime/run_xvfb.sh"`；manifest的`executable_restore_allowlist`记录了完整相对归档路径。不得对整个包递归开放执行权限或直接覆盖系统根。`install_xvfb_bundle.sh`由shell source调用，无需据此给其他文件执行权限。

`humanoid_infos.json`按工作目录读取，评测worker仍须以BENCH为cwd。保留可选import源码不意味着要加载OmTrackVLA/DA3/VLM权重。这个包**不含Python二进制环境、WA/DINO权重、场景、人物URDF/动作PKL或机器人资产**；它们仍须分别核验和交接。配置中的`data/default.physics_config.json`在当前来源未找到，本轮没有虚构或补写该文件。包现已上传并固定读回验 SHA，但不是H100兼容或完整闭环复现证明。
