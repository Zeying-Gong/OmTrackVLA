# WA61609 场景、人物与机器人资产获取/核验（2026-10-09）

这是原 NAS 的逐文件 SHA256 **清单**，不是资产文件、公开再分发许可或异机复现证明。固定 `manifest.json`（SHA `a1f515534153ccaa720f787f01ea23b9097c174fe25020f756c7d7947eedf98f`）含 STT/DT/AT 各 1405、共 4215 条；场景 87 HM3D v0.2 val + 14 MP3D。原三份 `val.json.gz` 引用的人物名字并集恰为 `humanoid_infos.json` 中全部 100 名。

- [仅路径与大小的预清单](BEST61609_ASSET_EXPECTED_FILES_20261009.json)：417538 字节，SHA `29961cd6bbab155274c34d1f0cd3fb62513372a6b68c0232e728a21897c504f4`；历史初步盘点，`sha256: null`，不可用于字节级验收。
- [逐文件 SHA256 固定清单](BEST61609_ASSET_HASHES_20261009.json)：468832 字节，SHA `28a8310032b2ff0add984d86a2407da604552e781843bb459603bdb9f7c90ca5`；827 个路径、大小、SHA 和建议的相对恢复布局。原 NAS 独立只读 readback：827/827、额外文件 0、缺件 0、哈希不符 0。
- [只读核验器](verify_best61609_assets.py)：从接收方自己取得的资源逐文件比较，不下载、不安装、不写入资源。

| 原 NAS 组别 | 场景/人物 | 文件数 | 字节 | 获取说明 |
| --- | ---: | ---: | ---: | --- |
| HM3D v0.2 val | 87 场景 | 238 | 5070254533 | `.basis.glb`/navmesh；有些场景还含 semantic GLB/TXT |
| MP3D Habitat | 14 场景 | 57 | 3553043977 | GLB/house/navmesh/semantic PLY 等原目录文件 |
| Habitat humanoid 扩展集 | 100 名 | 500 | 598735385 | 每名 URDF/GLB/ao_config/动作 PKL；另记录 FBX 来源网格 |
| `humanoid_infos.json` | 100 名映射 | 1 | 32722 | 已在私有 EVT 源码归档内；此表只固定哈希 |
| Spot+arm | 1 机器人 | 25 | 3624467 | URDF、网格、纹理、README/LICENSE |
| HM3D 根配置 | — | 2 | 25413 | 原目录一份真实文件、一份指向 train 配置的符号链接；清单记录解析后字节 |
| 补充 navmesh | 4 场景匹配 | 4 | 127996 | 原 NAS 另存；运行必要性未证，默认核验器视为可选 |

原项目工作目录为 `OmTrackVLA`，其 `data/scene_datasets` 是指向数据集根的符号链接；清单中 `restore_path` 均相对该工作目录。协作者可以用自己的持久数据目录及获准的符号链接实现同一布局，但不能据绝对路径相同推定文件相同。请先在接收方检查许可，再从**权利人官方入口**自行获取：

1. HM3D：按 [Matterport 官方 v0.2 下载表](https://github.com/matterport/habitat-matterport-3dresearch/blob/main/README.md#-downloading-hm3d-v02)申请访问、接受学术非商业条款。Habitat-Sim [官方数据指南](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d)列出 API token 与下载器流程；本次需 `val` 的 Habitat 格式及对应 semantic annots/configs。不要在聊天、脚本或 Git 中填写/保存个人 token。下载后只取清单要求的 87 场景并逐文件验 SHA。
2. MP3D：按 [Matterport3D 官方仓库](https://github.com/niessner/Matterport#data)使用机构邮箱签署 Terms of Use、取得独立访问权限；Habitat-Sim [官方指南](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset)说明 `download_mp.py --task habitat -o <path>`（其文档指出原下载脚本需 Python 2.7），只需 Habitat zip。将 14 场景恢复到 `data/scene_datasets/mp3d/` 后核对 GLB/house/navmesh/semantic PLY 原 SHA。开源下载脚本的 MIT 许可不等于 MP3D 数据可任意再分发。
3. Spot+arm：Habitat 官方 [数据下载器定义](https://github.com/facebookresearch/habitat-sim/blob/main/src_python/habitat_sim/utils/datasets_download.py)有 `hab_spot_arm` UID，来源 `ai-habitat/hab_spot_arm` 的 `v2.0`；可在目标环境执行 `python -m habitat_sim.utils.datasets_download --uids hab_spot_arm --data-path data/`，仍须逐文件比对。其 [数据卡](https://huggingface.co/datasets/ai-habitat/hab_spot_arm)有额外 LICENSE/归属说明。
4. 人物：Habitat 官方下载器有 `habitat_humanoids` UID，来源 [ai-habitat/habitat_humanoids](https://huggingface.co/datasets/ai-habitat/habitat_humanoids)，但其数据卡明确只提供 **12 个**预制 avatar；WA 固定评测实际使用 **100 个**，因此该公开下载**不足以恢复本次人物集**。另有 [TrackVLA 作者仓库](https://github.com/ShaoanWang/TrackVLA) 的 `download_humanoid_data.py` 指向 `https://0x0.st/80mQ.zip`，README 给出 [humanoids.zip 候选镜像](https://drive.google.com/file/d/1aE_wyvPqvOuVmF8px2vTO3trr70DKf1l/view)。作者仓库公开的 [`humanoid_infos.json`](https://github.com/ShaoanWang/TrackVLA/blob/main/humanoid_infos.json) 与原 NAS 文件逐字节 SHA256 一致：`4d230a3f39b80dae0641e7179660f045a49e06de065838dad3f51c50d8c18659`。这证明 **100 名索引的公开来源**，不证明两个 ZIP 候选含全部 100 名，更不证明 URDF/GLB/动作/FBX 字节相同；2026-10-09 只读 HEAD 检查 `0x0` 返回 HTTP 404，Google Drive 从原开发机连接超时，未下载大包或检查 ZIP 中央目录。协作者可以在获准访问后尝试作者镜像，先检查内容/许可，再用本清单逐文件核验；若任一文件不匹配，应如实报告，不能以官方 12 个或名字替代后宣称复现。数据卡还区分 avatar 的 CC-BY-NC 条款与 motion 数据许可；人物完整来源和再分发权仍需逐项核定。

收齐资源后在接收方运行（`PROJECT_ROOT` 指向包含 `humanoid_infos.json` 与 `data/` 的目录）：

```bash
python wa/repro/verify_best61609_assets.py \
  --manifest wa/repro/BEST61609_ASSET_HASHES_20261009.json \
  --project-root "$PROJECT_ROOT"
```

校验器内置固定清单 SHA `28a8310032b2ff0add984d86a2407da604552e781843bb459603bdb9f7c90ca5`，清单内容若改变会先拒绝；还检查 4215 episode、87+14 场景、100 人名、827 文件各组计数/字节和角色。只允许项目根目录内的文件；若场景目录以符号链接指向项目外，接收方必须显式添加对应的 `--allow-scene-root <canonical-directory>`（可重复），**仅场景**可用，不能以此放行人物或机器人外链。原 NAS 的场景目录存在两层链接，因此原地完整核验命令为：

```bash
python wa/repro/verify_best61609_assets.py \
  --manifest wa/repro/BEST61609_ASSET_HASHES_20261009.json \
  --project-root /data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA \
  --allow-scene-root /data/nas_ray/home/zeying.gong/datasets/scene_datasets \
  --allow-scene-root /data/nas_ray/home/zeying.gong/datasets/versioned_data/hm3d-0.2/hm3d \
  --strict-all --strict-extras
```

默认 `MISSING`、`SIZE_MISMATCH`、`HASH_MISMATCH`、`READ_ERROR`、`UNAPPROVED_PATH` 失败；原 FBX 来源网格与补充全局 navmesh 缺失列为 `OPTIONAL_MISSING`，叶目录额外文件列为 `extra_files` 警示。若需与原 NAS 全文件清单精确一致，加 `--strict-all --strict-extras`：原 NAS 上此模式已 827/827 PASS，但不能据此自动认定任意 H100 的 Habitat、物理配置、初 RGB 或 4215 闭环等价。

此清单没有将受限原始场景、人像或机器人网格放进 GitHub/ModelScope；也没有复制原始资产。官方可获取不代表公开可再分发，更不代表获取后的版本自动等于本次哈希。
