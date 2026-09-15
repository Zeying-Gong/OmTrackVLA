# 原 Phase2 checkpoint 的训练来源与官方 val 场景审查

调查日期：2026-09-14。结论：**目前不能认证原 Phase2 checkpoint 排除了官方 val 的 101 个场景，也不能用它作来源已满足该条件的正式 SOTA 排名模型。** 已经查到其 Phase1 祖先所引用的训练准入清单与官方 val 存在明确场景重合；SAGE3D 的场景映射还有缺口。

这不等于已经证明这些重合场景全部实际进入过 optimizer。应分别记录：当前被引用的训练准入集合、训练时文件/父权重的历史绑定、实际抽样记录。本次能核验前者及部分哈希链，不能补出缺失的实际采样账本。现有模型仍可用于内部开发、闭环故障定位和回归对照；后续 train4 场景无交集不能清除祖先训练来源的不确定性。

## 1. 已固定的模型身份与证据范围

原 checkpoint 路径：

```text
/data/nfs/share/wam_tracking/OmTrackVLA/outputs/ablations/next026_abl08_converged_v1/teacher_on/gru_curriculum_4k/checkpoints/best.ckpt
```

SHA256：`32c8f2f73277cfab8c3c7c22a53994061f7178498cfa003e1635b9576641bc9c`。此身份取自已固定的 [Phase2 val4 baseline protocol](../closed_loop_val4_review/phase2_val4_baseline_protocol_v3.json)，本次没有重新打开或反序列化模型权重。

读取范围是 run 的 config/manifest、SAGE sequence/source/split/sidecar 元数据、一个属于 train run 的原始 episode 元数据，以及已有官方公开成员表。没有导入 Torch、启动 GPU、改动远端文件/数据，或访问路径含 `test_locked` 的文件；split manifest 中的锁定分区仅保留计数，不打开其样本。

机器可复核结果：[phase2_official_scene_comparison.json](phase2_official_scene_comparison.json)，SHA256 `0067a2b7fcb0427f903d424662493e04c31d0f13224f7bdbb4a9228018533a1b`。本机重算脚本：[summarize_phase2_scene_comparison.py](summarize_phase2_scene_comparison.py)。

## 2. 初始化来源确实延伸到 Phase1

从各 run 的 `initialization.checkpoint` 和 `checkpoint_stage` 得到以下声明链：

```text
next026_architecture_v1_phase1_osnet_teacher
  → next026_abl08_converged_v1/teacher_on/single_step_36k
  → next026_abl08_converged_v1/teacher_on/gru_curriculum_4k
  → 原 Phase2 32c8f2…
```

| run | 当前 run_manifest SHA256 | 与训练来源有关的字段 |
|---|---|---|
| `teacher_on/gru_curriculum_4k` | `adc25d6c8bd016677fd1c13781890e8489191a6b2565eba97abc86c2d362e156` | 从 single_step36k 载入；GRU near_identity；4096 步预算 |
| `teacher_on/single_step_36k` | `fc3dd838e63e79fe9a7827639416b4df6e739063d58ae1d5b3e08bda26135cd3` | 从 Phase1 osnet_teacher 载入；36864 步预算 |
| `next026_architecture_v1_phase1_osnet_teacher` | `4bda56ff20a643fa5c29b6235fa3db86721e0c6b52ab734af71fa53c4b47144a` | 声明 InternData-N1、SAGE3D、TPT 三类数据；4096 步预算 |

Phase1 config 指向 `configs/manifests/phase1_v1.json`，几何训练使用 InternData-N1，身份训练使用 SAGE3D 与 TPT，`max_units_per_dataset: null`。当前训练实现 `omtrackvla/training/end_to_end_phase1.py` 也明确构造 `split="train"` 的 Intern 几何和身份数据集。这能解释清单中训练单元的用途；当前源码不自动等于已经封存的历史训练代码。

父 checkpoint 引用目前只有路径、phase/stage、匹配参数统计，没有父 checkpoint 的内容 SHA。因此上述链是 run 元数据声明及现存文件的审计链，尚不是每一步父权重均由内容哈希封闭的历史证明。

## 3. 已确认的声明集合重合：15 个训练单元、9 个物理场景标识

官方比较对象来自固定 TrackVLA Git revision：`c69f9d98a73c8c9d16c7cf49351354c32d8bc0cd`。本地 [public_episode_membership.json](public_episode_membership.json) 的 SHA256 是 `8759aaae1f7d461d748c0953efce1c2f2f0b7cc53b2b564849e168693eff0780`。

STT、DT、AT 官方 val 各 1405 episodes、各 101 canonical scenes，三任务场景并集也是 101。当前 Phase1 所引用的 `phase1_v1.json` 文件 SHA256 为 `0a4b14051eb4d0fa42538bc9b0747040812978a5c03800ecd957818f5cef7ee5`。其 InternData-N1 train 中有 2980 个 `group/scene` 单元；以下 Matterport3D 场景与官方 `mp3d/<scene>/<scene>.glb` 直接对应，仅进行大小写统一：

| canonical scene | Phase1 train 中的相机组 |
|---|---|
| `2n8karjn3hm` | `matterport3d_d435i`、`matterport3d_zed` |
| `e9udofap3sh` | 两组 |
| `jmbyfde2qkz` | 两组 |
| `vlzqgdo317f` | 两组 |
| `ac26zmwg7at` | 两组 |
| `prba3pwrgk9` | 两组 |
| `xca2tqtssaj` | `matterport3d_d435i` |
| `yfuzgdq5vwj` | `matterport3d_d435i` |
| `e9zr4mvmww7` | `matterport3d_zed` |

合计 15 个 group/scene 单元、9 个 canonical scene。这 9 个场景对应每个官方任务 val 中的 **525 个 episodes**；此计数表示公开评测集合的覆盖，不是已证明的训练泄漏 episode 数。

已确认的是“当前引用清单允许这些场景进入祖先 Phase1 训练”。尚未取得 Phase1 历史 manifest 内容哈希绑定及逐步采样记录，不能声称 15 个单元或 525 个评测 episodes 实际用于 optimizer，更不能把场景重合扩写为轨迹/图像逐样本重复。`split=train`、`test_locked_used=false` 也不能代替这项独立场景比较。

## 4. SAGE3D 哈希链可核验，canonical 场景身份还不能核验

两段 Phase2 的 run_manifest 都引用同一个 sequence index：

```text
outputs/indexes/architecture_v1_sage3d_sequences_v1.json
```

| 检查对象 | 结果 |
|---|---|
| sequence index 文件 | 2,813,231 bytes；文件 SHA `df802a2e0b656d456682c230c066b166fe6dc76e57e4867adcad4c9a4a30416d` |
| index 内部 canonical seal | `a24fc1830d04adf4cafd5aedde56feb79336c7d083df704b5fca2aeaa429c440`；按源码删除 `index_sha256` 后 canonical JSON 重算一致，且与两个 run_manifest 一致 |
| source index | `/data/nfs/share/OmTrackVLA/data/sage3d_extracted/index.json`；文件 SHA `fe84b71ae82ce69faec5045aefa4d891506b73710f76bf65ee3edaf03e0fe47e` 匹配 |
| split manifest | `phase1_v1.json` 的文件 SHA 与 index 中绑定值一致 |
| sidecar manifest | SHA `f103f0348826fffbebbb6c8d5e207640281fc6d6b3277abfaece2d68448de2e0` 匹配 |
| policy admission | SHA `ec90f7579d7f9503d45c0dcf96023fad5f5a79f880ec17e4142ebd7bb2621802` 匹配 |

内部 seal 与带缩进/带 seal 字段的完整文件 SHA 是不同口径；它们不同不构成文件被改的证据。

sequence train 有 **729 runs、5578 episode/camera descriptors、507598 sequences**。上游 source index 在这 729 个 runs 中有 5584 个 episode records，差额 6 对应 index 记录的 `missing_sidecar=5`、`no_visible_initialization=1`。本次没有把 507598 序列数当作独立物理场景数。

完整 sequence train 条目的字段为：`run, mode, episode, camera, relative, frames, initial_index, anchor_start, anchor_end, anchor_count, sidecar_sha256`。完整 source index 条目的字段为：`run, mode, ep, cam, path, steps`。两者都没有原始数据集命名空间、canonical scene ID 或源场景资产哈希。

进一步检查一个明确属于 train 的样本：

```text
0001_83992/dt/6/go2_realsense_d435i
```

其结果 JSON 只有 `episode_id=6`、`ori_episode_id=22` 等信息；保存的 `tmp_episodes/dt/0001_839920/episode_6.json` 中 `episode.scene_id` 仍是 **`0001_839920`**，不是 HM3D/MP3D canonical ID。该临时 episode 文件 SHA 为 `ce9216d1795ac04d2bb9cbb5866a7619e59035f6e04178b0cee87a8ab78da120`。run index 指向 `formal_runs_0001_83992.tar.gz`，但没有提供可核验的原始场景映射。

因此，把 `0001_83992` 与官方 11 位 scene ID 做字符串集合相交得到 0，没有排除场景重合的意义。本次结论是“已检查的全量索引与代表性 source 元数据不足以建立映射”，不是“整台服务器一定不存在映射”。需要原始生成配置或 run/episode 到 scene asset 的权威映射，不能靠猜测数字后缀来补齐。

## 5. 哪些已有 task checkpoint 不能直接作为干净替代

本次只读调查了 43 份已存在的 run_manifest，未打开它们的 checkpoint。

| 已存在的分支 | 观察到的来源 | 能否据此直接认证排除官方 val |
|---|---|---|
| `teacher_on/single_step_36k` 与 `teacher_on/gru_curriculum_4k` | 明确继承同一个 osnet_teacher Phase1；后者即原模型 | 不能 |
| `teacher_off/single_step_36k` 与 `teacher_off/gru_curriculum_4k` | 继承另一个 `next026_architecture_v1_phase1_pretrain`，声明相同三类数据；并非 osnet_teacher 权重 | 不能以关闭 teacher 替代来源审查 |
| `next026_phase1_init_phase2_core_v1` 及已查到的 world_action/refine 后续分支 | 继承上述普通 Phase1 分支 | 不能 |
| `next026_direct_phase2_core_v1` | `initialization.kind=direct_phase2`、`checkpoint=null`，但仍使用同一个未建立 canonical 映射的 SAGE index | 排除了这条 Phase1 初始化链；仍不能证明 SAGE 与官方 val 无交集 |

这里是已观察分支的判断，不是对所有历史 checkpoint 的穷举认证。现存 run_manifest 缺少完整父权重内容哈希，因此不能把路径链当成已校验每份权重的等价关系。

## 6. 最短可验证的新训练路线

**建立新模型分支，从固定官方 DA3 backbone 加全新任务参数开始；任务适配先使用身份已明确的官方 train 场景。** 现有 `omtrackvla.training.end_to_end_v1` 已有不传 `--init-checkpoint`/`--resume-from` 的 `direct_phase2` 构造路径。这里指重建新的模型实例和随机 adapter/策略头；不能只重置最后一层，却保留原 Phase1 学过的 bbox、身份、GRU 或 adapter 参数。现有启动 shell 默认占用 GPU0–7，后续实现需要新的资源配置，不能原样运行旧 8 卡脚本。

这条路线具备以下已经实算的候选来源：固定官方 STT/DT/AT **train 各 7257 episodes、各 703 canonical scenes**，与官方 val 的 101 场景交集分别为 **0**。三任务的逐文件 SHA 和比较结果均保存在机器结果中。实际新增采集还应排除项目永久开发场景，使用独立训练池和固定 episode/trajectory 指纹。

若希望继续利用 SAGE/Phase1 数据，应在新训练之前完成：

1. SAGE 的每个准入 descriptor 绑定 `original_dataset_namespace + canonical_scene_id + source_scene_asset_sha256 + source_episode/trajectory_sha256 + run_alias_mapping_sha256`；无法映射的条目不进入新准入集合。不能因为没有检测到重合就默认准入全部 5578 条。
2. 新 Phase1 manifest 按物理场景分组，统一 d435i/zed 等相机别名，排除官方 101 场景及永久开发场景；对 Intern 先排除上述明确匹配的 15 单元，但仅删这 15 个并不足以证明其余未知命名空间全部无重合。TPT 等来源同样需要明确的数据来源身份；最短路线可暂不引入这些上游任务数据。
3. 从同一官方 backbone 重新训练该 Phase1，再初始化新的 Phase2，或者先直接训练新 Phase2。不能将旧任务 checkpoint 在“清洁子集”上继续训练后称为已经消除祖先暴露。
4. 每个新 checkpoint 同时保存父权重内容 SHA、训练代码/config SHA、scene/episode allowlist SHA、实际采样记录或可完全重建的采样规则/版本与 RNG 状态；训练时及评测前自动核查场景交集。

官方 DA3 自身的外部预训练数据也需在实验中如实披露；本次没有审计其全部预训练语料。因此新路线可以建立“本项目任务适配没有使用官方 val 场景”的可验证边界，不能额外声称“整个模型从未在任何预训练中见过任何同场景”。是否满足正式比较口径还需按 benchmark 对预训练与外部数据的规则报告。

## 7. 尚缺的证据字段

| 缺口 | 为什么影响结论 | 应补证据 |
|---|---|---|
| Phase1 当时准入清单的历史绑定 | 当前路径所指清单不能单独证明当时完整使用了相同字节 | run/checkpoint 中的 manifest 和源 index 内容 SHA，或同期封存副本 |
| 父 checkpoint 的内容绑定 | 路径与 stage 可能复用，无法独立校验完整参数祖先 | 每一级 `parent_checkpoint_sha256`、config/code SHA |
| 实际训练样本暴露 | 被允许的场景不等于 optimizer 实际抽到的样本 | 每步/每 epoch sample IDs 与场景集合，或完整可重放采样状态和数据版本 |
| SAGE 原始 canonical 场景 | 当前 run 与重命名 scene ID 不在官方命名空间 | 每个训练 descriptor 到原始 scene/asset 的可信映射及来源哈希 |
| 全来源统一物理场景规则 | 相机组/运行批次切分可让同一场景跨分区 | 跨来源 alias 表、未知项拒绝策略、固定官方 val blacklist |

## 8. 原始调查产物

- [phase2_remote_provenance_inventory.json](phase2_remote_provenance_inventory.json)：原 run、config、sequence index 摘要。
- [phase2_remote_source_chain.json](phase2_remote_source_chain.json)：三段初始化链与 source/split/sidecar/admission 哈希核验。
- [phase2_remote_scene_metadata.json](phase2_remote_scene_metadata.json)：SAGE 元数据结构及 Phase1 train 的 15 个重合单元。
- [phase2_remote_rebuild_routes.json](phase2_remote_rebuild_routes.json)：43 份 task run 元数据摘录、SAGE 临时 episode 场景字段。
- [phase2_official_scene_comparison.json](phase2_official_scene_comparison.json)：固定官方场景集合、9 个声明交集及官方 train 的零交集核验。

本次完成了来源审查和新分支路线设计；没有创建训练数据、重新训练模型或授予正式排名资格。
