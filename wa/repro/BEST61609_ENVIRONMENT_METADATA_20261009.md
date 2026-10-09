# WA61609 环境重建元数据交接（2026-10-09）

本页交付的是原 A800 NAS 的**环境元数据快照**，不是 Python/Conda 二进制环境、可直接运行的镜像，也不是 H100 上已复现的证明。它与 [61609 独立运行说明](BEST61609_STANDALONE.md)、[私有权重/源码/证据交接](BEST61609_MODELSCOPE_REVISION_20261009.md)及[场景、人物、机器人资产获取与 SHA 核验](BEST61609_PUBLIC_ASSETS_20261009.md)配合使用；不能单靠这个小包运行 4215 条闭环评测。

ModelScope 私有数据集：[a597836509/wa-evt-jepa-private-backup-20260929](https://www.modelscope.cn/datasets/a597836509/wa-evt-jepa-private-backup-20260929)；此次元数据包**独立固定 revision** 为 `20a3a73a09b7dea3e02fe4c54e07804d1f0498da`，仓库内前缀为 `handoff/best61609_environment_metadata_20261009_v1/`。它与权重四包所用 revision 不同，不能把两个提交号互换。上传账号在该 revision 将四件下载回原 NAS 独立目录并逐件计算 SHA256，结果 `ALL_FOUR_REMOTE_DOWNLOADS_SHA_OK`；这仅证明上传账号当时可读且字节一致，不证明协作者账号有私有库权限。

| 前缀内文件 | 字节 | SHA256 |
| --- | ---: | --- |
| `best61609_environment_rebuild_metadata.tar` | 931840 | `484a49618b50b78421f3efa1573b5c8a3d5d6c50040d3d131c29744a92ad7a70` |
| `manifest.json` | 1247 | `b3dd8dd9437295919fc68cc647456e383893870c770a51decb0f65f7c94df502` |
| `complete.json` | 331 | `0896b766e6273a75688c24430d1a191599382136d6d8e706165f6740ed5fb60e` |
| `SHA256SUMS.txt` | 269 | `fad6cb1f86d99d6a163818fa4ae0a98ae0816f6b587243bb3ce8c6e9ba19c9dc` |

压缩包内恰有 7 个小文件：`README.md`、`check_runtime.py`、`conda_package_identities.json`、`environment_inventory.json`、`native_library_hashes.json`、`runtime_paths.json`、`runtime_symlinks.json`。`complete.json` 的状态仅为 `PASS_METADATA_TAR_INDEPENDENT_READBACK_ONLY`；清单状态明确是 `METADATA_ONLY_NOT_BINARY_ENVIRONMENT_NOT_H100_VERIFIED`。其中 Conda 条目是包身份记录而非带下载 URL 的可安装锁；本机路径和符号链接记录不能跨机器原样生效。`check_runtime.py` 只测试导入和 GPU 可见性，不能证明 Habitat 渲染、物理、初态、评测指标或 4215 条结果等价。

## 协作者获取与校验

先由协作者使用**自己的获授权账号**在目标机器通过 ModelScope 的交互式登录完成认证；不要在聊天、命令参数、日志或 Git 中复制个人 token。ModelScope 官方 `ms-hub` 支持私有数据集的 `login`、固定 `--revision`、`--repo-type dataset` 与 `--include` 下载。建议放在新的持久隔离目录，先核对本机 CLI 帮助和权限：

```bash
ms-hub login
ms-hub whoami
ms-hub download --help
ms-hub download a597836509/wa-evt-jepa-private-backup-20260929 \
  --repo-type dataset \
  --revision 20a3a73a09b7dea3e02fe4c54e07804d1f0498da \
  --include 'handoff/best61609_environment_metadata_20261009_v1/*' \
  --local-dir /path/to/new-persistent-isolated-download
```

下载后进入上述前缀所在目录，先验证 `SHA256SUMS.txt` 自身的固定 SHA，再执行其中三个文件的 SHA 校验：

```bash
cd /path/to/new-persistent-isolated-download/handoff/best61609_environment_metadata_20261009_v1
sha256sum SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt
tar -tf best61609_environment_rebuild_metadata.tar
```

第一条输出须等于表中 `fad6cb1f...c9dc`，第二条须三项均 `OK`；任何缺件、权限失败、大小或 SHA 不符都应停止，不能用可移动的 `master` 或文件名相同代替该固定 revision。确认 tar 成员安全且确有需求后，才在新的隔离目录解包和检查；不要向系统根目录、已有 Conda 环境或工作仓库直接覆盖。

## 边界与历史失败

原始完整二进制运行环境打包尝试未成功：来源树中有嵌入的测试用私钥，宽泛的内容扫描还有假阳性；失败记录保留在原 NAS `artifacts/best61609_runtime_bundle_20261009_v1/failed.json`，没有以忽略扫描、泄露密钥或宣称成功来绕过。此次只上传不含环境二进制的重建元数据；并未上传原场景、人物/机器人资产，也没有验证 H100 CUDA/EGL/OpenGL/驱动、Habitat 双环境、WA 权重加载与固定 4215 轨迹。协作者仍需按各自来源许可取得[逐文件固定资产](BEST61609_PUBLIC_ASSETS_20261009.md)，并在目标机器独立重建/实测环境。不要把这份元数据包称为“可直接解压运行的环境”。
