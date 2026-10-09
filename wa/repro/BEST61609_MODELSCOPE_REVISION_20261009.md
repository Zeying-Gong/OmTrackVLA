# WA 61609 ModelScope 私有资产交接（2026-10-09）

这是评测旧最佳 WA 61609 的部分资产交接，不是另一台机器已复现的证明。源码和说明在 GitHub `wa` 分支；本页锁定四份私有归档及六份原始清单的 ModelScope 提交。不得公开数据集或转发个人凭据。

- 私有数据集：[a597836509/wa-evt-jepa-private-backup-20260929](https://www.modelscope.cn/datasets/a597836509/wa-evt-jepa-private-backup-20260929)
- 固定 revision：`8dd25a3af7f61c71d02dd6918cd9fe398a15c398`
- 文件前缀：`handoff/best61609_20261009_v1/`
- 上传账号 `whoami()` 为 `a597836509`；固定 revision 读回为 private，前缀下恰好十个文件。上传账号从该 revision 将十件全部下载回原 NAS 新目录，独立核对大小与 SHA256，结果 `ALL_TEN_REMOTE_DOWNLOADS_SHA_OK`。这不证明协作者账号有权限。

| 前缀内文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `best61609_weights.tar` | 4358287360 | `23a9445518908411fca039f9b7330edadc2fa3e259a8896a6573d95927fa7331` |
| `best61609_evidence.tar` | 2374461440 | `4e8c8aa803a51b58e2d291e8cef15dcc05c3c79f50b64dbb1a48c58a3cee9faf` |
| `best61609_wla_source.tar` | 6103040 | `9685c93d33403c1c7eca3ecb9c7216cc9de2062078b7c1b2da790a2fabc67f14` |
| `best61609_evt_source.tar` | 17479680 | `9a16415b0513a0a6d1bc893f0e10e63fe1b271667413738a0863543af298295f` |
| `private_manifest.json` | 12533791 | `c9e193bd335b10af74866df7a5652e58aa5c4a135fccdc401f94a84e1e2c7894` |
| `private_complete.json` | 1152 | `9dae3b87c38eae88b8051eca1c008fc3ea7dbc8d3ce2e4eb65d6630ed6a4d62d` |
| `wla_manifest.json` | 134345 | `4efb7499025ad082239bc25f66c847bbdd56064ca5e20f6323edc18014a8b6ab` |
| `wla_complete.json` | 1184 | `a328feef0d525e00df11a2c1730d76a6099cf39803f9d28c8126b25506a46182` |
| `evt_manifest.json` | 188018 | `e81ec4cf10db1f53cb6c016ab84a7dcaac2a303d8516fa08efe067b7db2e8c86` |
| `evt_complete.json` | 1342 | `258267c89a7fc94cd75b1f3dd2eea6bd4f755b1630faabd7a9eba1156e7d3637` |

权重归档内部原 WA checkpoint SHA256 为 `c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52`，step 59716；外层 tar SHA 不能替代内部 SHA。新独立启动入口源码固定 commit `716e948294a916318b2ee83c0abe9c19b2999aa6`，见 [无旧初始化权重入口](BEST61609_STANDALONE.md)。它只需 WA 成品权重和 DINOv2-S/14 权重，不需要传旧 WLA 或 JEPA 初始化权重；WLA 结构源码及固定上游 DINO/JEPA 源码仍需恢复。

协作者须先用自己的获授权 ModelScope 账号测试私有库访问；访问失败则申请库权限，不使用聊天记录或他人 token。在持久隔离目录用 ModelScope Hub API 对上述十件逐一指定 `revision` 和 `expected_sha256` 下载，再独立计算大小和 SHA。不要从可移动的 `master` 下载后称其为本次版本。原始 JSON 保持字节不变；先检查 tar 成员与清单，再在隔离目录解包，不向系统根或已有项目盲目覆盖。原绝对路径引用未改写，异机需审阅后适配并重核哈希。

## 尚未闭合的异机运行门禁

1. DINOv2-S/14 在线编码器权重不在四包中；接收方先前报告 SHA `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` 已通过，目标机器仍须复核文件与加载接口。
2. 双 Python 环境、Habitat-Sim 二进制及 CUDA/EGL/OpenGL 兼容环境未打成可搬移镜像；环境清单不是二进制安装锁。8×H100 被 CUDA 识别不等于仿真和模型可运行。
3. 固定评测涉及 87 个 HM3D 和 14 个 MP3D 场景；人物 URDF/动作 PKL、机器人资产均不在四包中。原 NAS 有候选内容不等于接收方已有授权与相同文件，须逐项核许可、路径、哈希；不可公开再分发场景。
4. Habitat/EVT 归档有源码、配置及 Xvfb `.deb`，没有上述二进制环境/场景/人物机器人文件；`data/default.physics_config.json` 在来源中未找到，不伪造。WLA/EVT 快照未附独立 LICENSE，不从 WA 仓库公开性推定再分发许可。
5. 本页只证明上传账号按固定 revision 可读且十件字节一致；协作者账号访问、H100 真实 RPC、完整 4215 条闭环结果均未验证。61609 的 1279/1178/1207 是历史结果，不是新机器复现。

原始路径、包内 allowlist 与安全恢复边界见 [资产交接](BEST61609_ASSET_HANDOFF.md) 和 [复现指南](BEST61609_REPRODUCTION.md)。
