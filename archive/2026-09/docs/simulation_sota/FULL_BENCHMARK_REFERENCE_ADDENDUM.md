# 完整 EVT 清单与参考轨迹长度补充

2026-09-14。本补充细化 `SIMULATION_SOTA_PROTOCOL_REVIEW.md` 中“完整参考长度必须可用”的计划。实际核查发现，上游固定版本并没有覆盖全部官方验证 episode 的参考长度文件，因此不能把它当成已经满足的前提。

固定 TrackVLA commit `c69f9d98a73c8c9d16c7cf49351354c32d8bc0cd` 的 STT、DT、AT 验证集各有 1,405 集、101 个场景。已生成完整成员与分片清单：每任务 30 个 shards，前 25 个各 47 集，后 5 个各 46 集，集数无缺失、无重复。所有 101 个场景的几何与 navmesh 文件在按 Habitat `data/scene_datasets` 解析的路径中存在；这只验证文件元数据和 glTF 文件头，尚未验证逐场景渲染与导航。

首次资产检查误把官方 scene_id 当成仓库相对路径，报告了 101 项缺失；该次失败保留在 `official_val_asset_inventory.json`。修正后的 `official_val_asset_inventory_v2.json` 为本次有效路径检查结果，不能用首次结果推断服务器缺少场景。

公开 `track_episode_step` Git tree 有 1,400 个文件，其中与固定官方验证 `(scene, episode_id)` 精确匹配的只有 **1,353** 个，另有 **52** 个验证 episode 无对应文件、47 个公开文件不属于该验证集合。已经下载并按 Git blob SHA1、文件 SHA256 和长度值校验所有 1,353 个匹配文件；52 项缺失清单明确保留，没有补值或替换 episode。

新的完整比较规格明确报告：

1. 主指标 SR 使用完整 1,405 集，参考长度缺失不改变成功率分母。
2. 若报告遵循公开 analyzer 的完整 1,405 集 TR，1,353 集用 `max(actual_steps, reference_steps)`，固定缺失的 52 集用 `actual_steps`。必须披露 1,353/1,405 的参考覆盖率和此回退规则；不能称“全部 episode 均有固定参考长度”。
3. 同时报告共同 1,353 集的参考加权 TR，所有方法使用完全相同的子集。这个子集数字不是完整 1,405 集 TR。

上述完整 TR 沿用公开的缺失回退语义，因此缺参考 episode 的分母依赖各模型实际结束时刻；这是比较局限。若以后补齐参考长度，必须冻结独立协议并对所有方法统一重算，不能把新旧数字混排。已有作者成绩还受到视角、初始化信息、旧分片语义等差异影响；完整清单本身不是对其分数的复现。

制品：

- 完整清单：`evt_full1405_manifest_v1/manifest.json`，SHA256 `39521bb5688346c17cf7e75c1c03a5e5ac2d604688e5397fdd4a49adf8de62c1`。
- 有效资产清单 SHA256：`3c0ab6a64372fc491ba1cb0f6bd5d1763741184ddb7f9bbd23d5f641cacad56a`。
- 参考文件 Git tree：`5d8725d1815b9c8ae2c88db3749593cfd0f9524e`。
- 固定源码压缩包 SHA256：`276d2a7b4e6c61d89ff809e3df3fcde5867b55d333af88cb12bdecd890289196`。
- 原始参考及下载回执：`official_reference_lengths_v2/`。

这些是协议与资产准备，尚未执行正式 benchmark。逐集 avatar、指令、semantic ID 刷新与所有方法的共同适配器仍需实现和测试；当前不授予 SOTA 结论。
