# OmTrackVLA Architecture v2 提案：保留空间 token 的轨迹策略

状态：`proposal + CPU shape/backward prototype`。它是 Architecture v1 的并行候选，不能覆盖或改名现有 v1 checkpoint，也未获准正式训练。

## 1. v1 当前实际数据流

1. 首帧 `RGB+bbox` 经 DA3-SMALL L11、投影和 RoIAlign 得到一个 `target_memory [B,256]`。
2. 四帧 RGB 经同一个 DA3 得到每帧 `20×36=720` 个 L11 patch token 和 camera token；部署只取当前帧 patch。
3. target cross-attention 在当前 720 patch 上查询目标，显式 UWB Gaussian patch bias 加到 attention logit，输出单个 `z_target [B,256]`。
4. scene attention 把 720 patch 压成单个 `z_scene`；最后两个 camera token 经监督 SE(2) 瓶颈得到 `z_ego`；二者再压成单个 `w_t`。
5. UWB 数值、协方差、质量、age、valid 独立编码为 `z_uwb`。
6. `[z_target,w_t,z_uwb,masks]` 经 Fusion MLP 压成一个 256 维向量，GRUCell 维护一个 256 维跨调用状态，MLP 一次回归 7 个未来 xy 点并前置原点，同时输出 stop。

GRU 并非原则上错误。当前风险来自它位于过早空间压缩之后：720 个场景位置先变成两个向量，GRU 无法恢复已丢掉的左右通道、障碍边界和多条可行路径。训练序列通常只有两个 policy call，也不足以让 GRU 学到长遮挡恢复。直接 SmoothL1 回归整条路径还容易把多种合理绕行平均成偏短轨迹。

## 2. v2-core（推荐先做）

```text
DA3 history L11 patch tokens [B,T,720,768]
          │ L11 projector + explicit 2-D patch position
          v
current scene tokens [B,720,256]
   ├─ immutable reference appearance → visual target attention → z_target_vis
   ├─ UWB 20×36 Gaussian bias      → geometry pooling       → z_target_geo
   ├─ UWB continuous encoder                              → z_uwb
   ├─ camera-token pair → supervised SE(2)                → z_ego
   └─ 16 learned scene queries cross-attend 720 patches   → scene_1..16

[reference, z_target_vis, z_target_geo, z_uwb, z_ego, masks, scene_1..16]
          → 2-layer token Transformer
          → [STOP query, waypoint-time query_1..7] cross-attention decoder
          → bounded Δxy_1..7 → cumulative 8×2 waypoint + stop
```

关键变化：

- UWB 仍严格保留两路，不退化成 token-only：显式 20×36 几何图单独池化 `z_target_geo`，连续量另走 `z_uwb`。
- 外观目标与 UWB 几何目标在进入策略时仍是两个 token；视觉/UWB 冲突可测量，不再在一次 target attention 里被静默抵消。
- 16 个 scene latent 保留多个空间区域，替代单个 `z_scene/w_t` 摘要；不是跨 episode memory bank。
- 不使用 GRU。短期时间信息仍来自 DA3 的四帧联合编码和显式 SE(2)；更长历史须作为后续独立消融，不能先藏进 v2-core。
- 七个未来点各有独立时间 query，通过共同 context 和 query self-attention 协调；预测有界局部增量并累积，天然比 14 维一次回归更容易约束连续、无回退轨迹。

原型代码为 `omtrackvla/models/end_to_end_v2.py`。当前只验证 policy decoder 的 shape、双路 UWB、增量边界和 waypoint 梯度，不表示已接通 DA3 或已有成绩。

## 3. 从 LightNav 和 NavDP 借什么、不借什么

LightNav 固定源码审查可直接支持三点：近期密集/远期稀疏的 SlowFast 历史、按 session 的视觉 token cache、以及显式 SE(2) 轨迹配合带时间戳控制器。v2-core 先采用“显式整条轨迹 query”；长历史和缓存等 v2-core 胜过 v1 后再加。它的 Qwen3-VL、语言 prompt、RVQ action token 和 64 帧输入不适合直接搬到一次 bbox 初始化 + UWB 的人物跟随合同。

NavDP 的合理参考是把整条未来轨迹作为条件生成对象，并允许训练期 privileged teacher 提供可通行性/碰撞指导；GT depth、navmesh、目标 pose 仍不得进入部署。当前环境没有可复核的 NavDP 源码快照且外网 DNS 不通，因此这里称为 “NavDP-style trajectory diffusion”，不宣称源码复现。

v2-core 不立即加入 diffusion。只有同时满足以下条件才进入 `v2-DP` 独立消融：

1. waypoint-query deterministic head 在同一固定验证集仍出现左右多模态平均或明显短轨迹；
2. 训练集能提供同状态多条可行轨迹，或至少有可靠 collision/feasibility teacher；
3. 4～8 次去噪的 P95 延迟满足部署预算；
4. 同 checkpoint/input 下优于 v2-core，而不是只生成更花哨的轨迹。

## 4. 目标绑定与闭环状态

`uwb_only` 的正确含义是“UWB 是目标定位/绑定来源”，不是关闭整幅 RGB。RGB 仍负责墙、可行走区域和人物候选。闭环丢失时应保留 scene RGB，撤销陈旧的可变 target observation，回到不可变 reference + UWB geometry 重新选择候选。v2 把 `z_target_vis` 与 `z_target_geo` 分开，才能对二者的一致性做监督和部署门控。

不可变的首帧 reference 不能被新候选覆盖；可变 observation 可在 UWB/外观/时序一致且候选唯一时更新。SEARCH、重获和执行 watchdog 仍属于显式控制状态，不由 Transformer 暗中代替。

## 5. 最小验证顺序

1. `V2-000`：CPU token decoder shape/backward；waypoint loss 对 scene、target、UWB、ego 输入梯度均非零。
2. `V2-001`：接真实 DA3 L11 与现有双路 UWB，只跑 0-step/1-step smoke；报告 DA3 权重覆盖、冻结/新增参数、梯度。
3. `V2-002`：冻结 DA3 的 head-only 小预算，对照 v1 的 ADE/FDE、路径长度、逐 horizon 半径、左右方向灵敏度与 stop；不用 EVT-Bench、不访问 `test_locked`。
4. `V2-003`：只有 head-only 有正信号，才开放 DA3 后层 adapter；固定可视化后再扩大训练。
5. `V2-DP`：仅在 v2-core 仍被多模态平均限制时评估小步数 diffusion decoder。

任何阶段都保留 Architecture v1 和既有 checkpoint 作为 baseline，不把结构变化伪装成 v1 续训。

## 6. 时序输入补充冻结（NavDP / LightNav 源码核查后）

Architecture v1 的 4 帧是 SAGE3D 30 Hz 连续帧，只覆盖约 0.10 秒；v2-core 不继承该设置。源码核查与微基准详见 `docs/navdp_lightnav_temporal_review.md`。

v2-core 的时序合同改为：

- 8 帧 RGB，与 NavDP 的 `memory_size=8` 对齐；
- SAGE3D 每 3 个原始帧采一张，即 10 Hz，offset 为 `[-21,-18,-15,-12,-9,-6,-3,0]`；
- 历史覆盖 0.7 秒，与 8 点 waypoint 的 `0.0..0.7 s` 时间合同对齐；
- 每帧 720 个 DA3 patch 先压缩成 16 个 scene latent，8 帧形成 128 个时空 latent；
- 当前帧 patch 继续独立承担目标外观 attention 和 UWB Gaussian geometry pooling；
- 采用真实相对时间 embedding，不用 GRU 保存隐式时间；
- 不在 v2-core 立即增加 LightNav 全 episode SlowFast 历史。8 帧版本通过后，才评估 2--4 秒稀疏 Slow 分支。

H100 物理 GPU 1 的 DA3-only、batch=1、BF16 微基准中，4/8 帧中位前向约为 12.07/13.00 ms，peak allocated 约为 0.204/0.288 GiB。该数字只用于准入 smoke，不代表 Thor 完整延迟。
