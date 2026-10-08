# WA 61609 跨机器资产交接

目标是让另一台 8×H100 机器评测同一份 WA 61609，而非重训。源码和配置已在 GitHub；大权重、运行环境和固定评测证据仍在原 NAS。尚未上传资产或验证 H100 运行。原命令、完整路径和协议见 [复现指南](BEST61609_REPRODUCTION.md)。

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

ModelScope适合单独的私有资产仓库，使用显式 `visibility="private"` 并在上传前读回确认；不得依赖默认可见性。官方SDK支持私有仓库和文件/目录上传，见 [ModelScope Hub](https://github.com/modelscope/modelscope_hub)。用户提供的账号名尚需认证核实；本轮没有创建仓库、认证或上传。

原模型环境中发现user-site ModelScope1.39.1及hub0.4.0，但导入因缺cryptography失败。上传工具应安装到单独工具环境，不向正在使用的训练或仿真环境补包。凭据不进入命令行参数、源码、Git、压缩包或日志；通过隐藏输入或授权凭据机制登录。

资产按 allowlist 分包、生成包内文件和压缩包各自SHA，上传后按固定revision下载校验，再交给协作者。不包含home配置、SSH、token、训练缓存或无关任务。DINO/JEPA上游按固定提交获取并保留许可证；JEPA快照含CC-BY-NC4.0及第三方许可，不能将整套依赖统一改称Apache。HM3D/MP3D及人物资源先核验接收方已有资产和许可，仅补获准交接的缺项，不公开整库场景。

当前交付状态：源码和说明可获取；权重原文件与证据范围已找到；直接加载函数的完整状态和有限Session等价检查通过。[独立RPC及启动器](BEST61609_STANDALONE.md)已实现，A800真实RPC和32CPU测试通过；全量新入口/H100、环境可移植验证及私有资产上传仍未完成。
