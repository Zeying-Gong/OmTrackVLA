# WA 61609 无旧初始化权重评测入口

此入口评测原WA61609，只需WA成品checkpoint和DINO权重，不读取旧WLA/JEPA初始化权重。模型结构、Session、控制器和评测标准不变。新入口已在A800通过有限等价和真实RPC检查；H100全量复现、环境和资产交接仍待完成。

原冻结源码192b和原命令保留。**新入口源码固定为 `716e948294a916318b2ee83c0abe9c19b2999aa6`**，不要在192b工作树中寻找或覆盖这些新增文件。

## 需要的文件

| 类别 | 必需内容 |
| --- | --- |
| WA权重 | 原61609 step59716 checkpoint，SHA `c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52` |
| DINO权重 | DINOv2-S/14，SHA `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |
| 结构源码 | 同一WLA模块快照、固定DINO与JEPA源码；结构代码仍需要，不等于另加载WLA/VLM模型 |
| 完整闭环资产 | 原双环境、manifest、三份val、场景、七框修复和教师初态证据，见[原复现指南](BEST61609_REPRODUCTION.md) |

旧4.12GB WLA构造权重与约212MB JEPA初始化文件不用随新入口传输。WA checkpoint已经含训练后的JEPA、MetaQuery、ActionExpert和融合/几何/机器人接口参数；仅JEPA预训练和DINO不能还原这些WA训练结果。原WA文件未裁剪或改变SHA，包含的optimizer在推理时不用。DINO encoder不在WA文件内，仍需另提供。

## 固定代码与开发接口

在自己的持久工作区新建独立工作树；已存在时核对身份，不能覆盖。下面仍采用原进程内路径布局：

```bash
WA_PROJECT=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928
git -C "$WA_PROJECT/checkout" fetch origin wa
git -C "$WA_PROJECT/checkout" worktree add --detach "$WA_PROJECT/source_best61609_standalone_v1" 716e948294a916318b2ee83c0abe9c19b2999aa6
cd "$WA_PROJECT/source_best61609_standalone_v1"
```

完成环境和资产哈希检查、确认开发GPU安全余量后，可用下列有限两请求检查。`READY`必须是新的持久NAS文件名；不在集群提交smoke。

```bash
: "${READY:?set a new persistent diagnostic JSON path}"
: "${CUDA_VISIBLE_DEVICES:?set an authorized developer GPU}"
PYTHONNOUSERSITE=1 PYTHONPATH=. "$WA_PROJECT/probe_env/bin/python" -u -m wa.wm.standalone_eval_server \
  --root "$WA_PROJECT" \
  --encoder-weight /data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth \
  --wla-source "$WA_PROJECT/dependencies/wla_v1" \
  --checkpoint /data/nas_ray/project/md-ak/users/zeying.gong/job_61609/task_72803/wa_hard_stt_train_a800_v1/checkpoint.pt \
  --checkpoint-sha256 c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52 \
  --checkpoint-step 59716 --mode mixed --noise-mode zero --ready "$READY" --developer-check
```

无 `--developer-check` 时服务在127.0.0.1随机端口监听，原 `/reset`、`/predict` 协议不变，ready保留原顶层身份字段并增加源码/加载来源。只绑定本机回环，不为外部公开服务。

## 完整评测接线

新增 `wa.wm.eval_full_standalone_launch`，仅改变server命令和额外身份检查。CPU测试逐字验证原启动器的8分片、worker参数、PYTHONPATH、Xvfb、进程清理和汇总逻辑保持不变。原 `wa/scripts/eval_full_mixed.sh` **仍启动旧入口**。

在新的隔离worker包装中保留原shell的所有export、两份依赖SHA检查和Xvfb初始化，只将末尾模块 `wa.wm.eval_full_mixed_launch` 改为 `wa.wm.eval_full_standalone_launch`。工作目录及PYTHONPATH使用上述新工作树，WA身份、mixed/zero、语义修复、七框计划和教师索引环境变量按原指南完整保留。不要仅替换一个Python命令而漏掉原shell准备，也不要把会修改系统链接的Xvfb初始化直接跑在共享开发机。

H100需本平台调度器和真实双环境兼容检查，不能使用含A800型号断言的旧YAML。只有8卡时顺序跑STT、DT、AT三个完整1405任务，不改变分母。当前未执行新入口的完整闭环或H100运行。

## 已有验证

- [加载等价报告](../results/STANDALONE61609_EQUIVALENCE_20261008.md)：669完整状态张量和619模块非state一致，旧/新各6次Session输出逐值一致，仅WA+DINO两次权重读取。
- 32项CPU测试PASS：15加载器＋17RPC/启动器测试，主线程独立重跑0.322秒。包含SHA/step、完整覆盖、非法参数、ready来源和原子无覆盖发布；旧入口未改。
- 新RPC真实检查：A800 GPU3，UUID `GPU-8feb0a89-838a-a61a-cbd3-ec9205b9c344`。5GiB分配器cap、180秒启动期限，实际21.876秒。reset两次200；mixed首/后续预测200且与等价报告完全一致；重复时间返回预期500。测试后只停止自有PID3801568，exit-15为主动SIGTERM，非推理崩溃。
- 本次有一个预期非法请求Traceback和xFormers警告，日志原样保留，不称“完全无异常”。

真实RPC产物在 `WA_PROJECT/artifacts/standalone61609_rpc_20261008_v1`：

| 产物 | 字节 | SHA256 |
| --- | ---: | --- |
| report.json | 40999 | f52af645dc7a6b21a5737e2ce26a8a87b0df24e391c88ea1b92059802fae5af4 |
| ready.json | 35040 | 7cd2660223738375b2303b0c639a9256f663533697316820052615a844bbe97b |
| server.log | 1460 | 23225e90ba9c5cd49fe4bb757c6cc9a1694260bb250192ff5c274a0c713151d5 |

这些检查未重新评测SR、未训练，也不验证H100、反向传播或8卡NCCL。权重和环境尚未上传ModelScope；授权私有资产仓库、打包、下载验哈希后才能提供完整接收步骤。
