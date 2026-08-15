# e2e_policy 修改记录：3D mRoPE 替换 2D RoPE（2026-08-15）

## 一句话总结

将 eVGGT 主干的 2D RoPE 替换为 **Cosmos3/Qwen3VL 风格的 3D mRoPE（temporal / height / width）**，
并把主干输入从"单帧 patch"升级为"**时间帧堆叠 (T, H, W)**"，历史帧与当前帧一起进入 backbone。

## 参考来源

参考实现来自远端 Cosmos3 代码（未改动其原文件）：

- `/h100-2/cosmos-framework/cosmos_framework/data/generator/sequence_packing/mrope.py`
  - `get_3d_mrope_ids_text_tokens`、`get_3d_mrope_ids_vae_tokens`
- `/h100-2/cosmos-framework/cosmos_framework/model/generator/reasoner/qwen3_vl/qwen3_vl.py`
  - `Qwen3VLTextRotaryEmbedding.forward` + `apply_interleaved_mrope`

## 改了什么

### 1. 新增 `e2e_policy/rope3d.py`（新建，不动原 rope2d.py）

完整移植 Cosmos/Qwen3VL 的 3D mRoPE：

| 函数 | 作用 |
| --- | --- |
| `get_3d_mrope_ids_vae_tokens(grid_t, grid_h, grid_w, temporal_offset, reset_spatial_indices=True)` | 视觉 token 的本地 3D 网格位置 id，T-major 展平；默认 spatial 轴从 0 开始（Qwen3VL 设计） |
| `get_3d_mrope_ids_text_tokens(num_tokens, temporal_offset)` | text 型 token 位置 id：三个轴共享单调递增 id `(n,n,n)` |
| `build_3d_mrope_cos_sin(position_ids, head_dim, theta=10000.0, mrope_section=None)` | 把 `(3, N)` 位置 id 转成 cos/sin。频率布局从分段 `[TT..HH..WW]` 重排成交错 `[T H W T H W ...]`；`emb = cat([freqs, freqs])` 翻倍到 head_dim。`mrope_section` 默认按 head_dim//2 三等分（head_dim=96 时为 `[16,16,16]`），三者之和必须等于 head_dim//2 |
| `apply_rotary(q, k, cos, sin)` | Qwen3VL 交错式应用（`cos`/`sin` 做 `::2`/`1::2` 交错后再 `rotate_half`） |

**约定说明（与 Cosmos3 一致）：**
- 视觉 token 坐标：帧 `t` 取 `temporal_offset + t`，高/宽取各自网格坐标。
- 一个 segment 结束后，`temporal_offset = max(all_positions) + 1`，下一个 segment（如 text/context）从非重叠位置开始。
- text 型 token 三轴 id 相同 → 等效 1D 单调位置，用于 context/query 等非视觉 token。

### 2. 修改 `e2e_policy/evggt.py`：主干支持时间帧堆叠

- `FrameGlobalBackbone.forward(frame_tokens, context_tokens)`：
  - 输入改为 **`frame_tokens: (B, T, P, dim)`**（T 帧 × P patch）。
  - 序列布局：`[帧 patch (T*P)] [context tokens (Cctx)] [policy queries (Q)]`。
  - **Frame Attention**：只允许同帧内的 patch 互相 attend（+ 各自 self），跨帧/跨类别被 mask 掉。
  - **Global Attention**：全 token 全量 self-attention。
  - 3D mRoPE：patch 用 `(t, h, w)` 网格位置 id（t 递增 = 帧序号），context/query 用 text 型单调 id 接续在视觉 grid 之后。
  - `fused_patches` 返回**最后一帧（当前帧）**的 patch 特征 `seq[:, (T-1)*P : T*P]`。
- 新增构造参数 `rope_theta=10000.0, mrope_section=None`。

### 3. 修改 `e2e_policy/policy.py`：喂时间帧堆叠

- `cur_patches` 从 `(B, P, C)` 变为 `(B, T, P, C)`：
  - 有 `history_rgb`：`[历史帧 (B,K,P,C) + 当前帧 (B,1,P,C)]`，当前帧在最后一帧（t 最大）。
  - 无 `history_rgb`：`(B, 1, P, C)` 单帧兜底。
- `history_enc.scene_tokens` 不再被调用（历史帧以原始 patch 形式进 backbone，而非 mean-pool 记忆 token）；`motion_tokens` 仍保留进 context。
- `cur_patches_teacher` 取最后一帧 `cur_patches[:, -1]`，与 future teacher `(B,P,C)` 形状对齐（前向动力学 loss 用）。

### 未改动

- `rope2d.py`（保留，供对照/回退）
- 其余 `encoders.py / heads.py / losses.py / dino_backbone.py / dino_vit.py / smoke_test.py`

## 验证结果

在远端 H100（`omtrackvla` 环境，cuda）跑通：

```
python e2e_policy/smoke_test.py --size 224 --iter 3 --device cuda
```

- trainable params: 13.08M，total: 35.13M
- loss: 1.6110 → 1.6030 → 1.6801，grad_norm 正常，SMOKE OK。
- 位置 id 自检：`(T=4, H=W=14)` 时 t 轴 = [0,1,2,3]，h/w = [0..13]；text token 从 14 起（=max+1）；cos/sin shape `(789, 96)`。

## 运行命令

```bash
cd /data/nfs/share/OmTrackVLA
PYTHONPATH=/data/nfs/share/OmTrackVLA /data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python \
  e2e_policy/smoke_test.py --size 224 --iter 3 --device cuda
```

## 下一步（数据管线）

- 新建 `e2e_policy/data_collate.py`：把 `tools/data_loader/unified.py` 的 sample dict 转成模型 batch
  （current/window → DINO patch 时间堆叠、target crop、pointgoal、trajectory 对齐 horizon=8、future RGB、valid/conf mask）。
- 新建 `train_e2e.py`（不动原 `train.py`）。
- 确认 `tools/data_loader/self_test.py` 三源（sage3d / tpt / navdp）正常。

---

## 追加：评审修复（#2-#9）+ 数据管线 + 端到端训练（2026-08-15）

### 评审问题（10 项全部属实）

1. **#1 无训练入口** → 新建 `train_e2e.py`（见下）。
2. **#2 forward 在缺 future 时 KeyError** → `policy.py` 的 forward/inverse 仅在 `future_rgb and trajectory` 同时存在时启用。
3. **#3 inverse 未来特征用了 DINO mean-pool+detach** → 保留 detach（teacher），但改为经 `future_pool`（2 层 MLP）投影后的 `h_future`，shape 与 `h_t` 对齐。
4. **#4 lambda_target 死代码** → `losses.py` 新增 `target_loss()`（pos smooth-L1 按 mask + vis BCE），`compute_total_loss` 在 `"target_state"` 存在时计算 `L_target`。
5. **#5 零 token 非 learnable** → `policy.py` 新增 learnable `null_target` 参数（trunc_normal 初始化）。
6. **#6 动作 mean 压时序** → `heads.py` ForwardDynamics 改为 `act_embed(Linear) + act_gru(GRUCell)` 时序编码，逐 step 的 GRU 递归保留 chunk 内顺序与间隔。
7. **#7 分支覆盖/深度** → 未列在 10 项内，随数据管线一并处理（depth 走 monkey-patch，见下）。
8. **#8 假定方 patch 网格** → `evggt.py` `forward(frame_tokens, context_tokens, grid=None)` 用真实 `(Hp,Wp)`（来自 `dino_backbone.grid_size(h,w)`）替代 `round(P**0.5)`。
9. **#9 yaw 无归一化** → `heads.py` WaypointHead 的 sin/cos 做非原位 L2 归一化（`n=sqrt(s²+c²)`，避免 autograd 版本冲突）。
10. **#10（原列表第 10 项）** → 涉及目标表征，纳入数据管线设计（target 图像 + target_local 双监督）。

### 数据管线

- **`data_collate.py`（新建）**：
  - `waypoints_to_actions(wpts, horizon=8)`：绝对 ego 航点 → `(H,4)` 增量动作 `(dx,dy,sin,cos)`，与 `losses._cumulative_se2_error` 的坐标约定一致（每步平移转到累计 heading 前帧）；无 heading 时从位移推 yaw；不足 pad 末点。
  - `collate_batch(samples, ...)`：输出 `mkw`（current_rgb/target_image/target_valid/target_confidence/target_type/history_rgb/pointgoal/task_type/goal_spec/future_rgb）与 `loss_batch`（trajectory/action_valid/action_confidence/target_state/target_state_valid/stop_label/stop_task_mask）。
  - future_rgb 支持 path 字符串与 np.ndarray 两分支。
- **`train_e2e.py`（新建，不动原 `train.py`）**：
  - `MixedOmniDataset(sage, tpt, navdp)` 采样，`get_sample(as_arrays=True)`。
  - 默认 `use_forward_dyn=False use_inverse_dyn=False`（辅助任务默认关），target_state 默认开（`--no-target-state` 关闭）。
  - `_patch_unified_2d_depth()` monkey-patch：NavDP depth 是 2D `(270,480) uint16`，`unified.resize_keep_aspect` 只支持 3 通道；patch 在 2D 分支自行 letterbox（不除 255），3D 走原函数。

### 验证结果（H100，真实数据）

```
cd /data/nfs/share/OmTrackVLA
PYTHONPATH=/data/nfs/share/OmTrackVLA /data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python \
  e2e_policy/train_e2e.py --steps 30 --batch 4 --device cuda
```

- 数据集：`len=13991`（pf_units=13927，navdp_eps=32）。
- 30 steps 全程跑通，loss 下降：step0 loss=0.919 → step25 loss=0.541，`L_wp` 0.278 → 0.011。
- **300 steps 收敛验证**（batch 8，真实数据）：loss 0.887 → 0.075（step 290），`L_wp` 0.267 → ~0.01-0.03；
  训练期间 checkpoint 保存于 `e2e_policy/ckpt_e2e.pt`（133.5 MB）。

  | step | loss | L_wp |
  | --- | --- | --- |
  | 0 | 0.887 | 0.267 |
  | 100 | 0.141 | 0.032 |
  | 200 | 0.158 | 0.021 |
  | 290 | 0.075 | 0.026 |
- 开关验证：
  - 默认（trainable 11.29M）：`L_waypoint + L_target + stop`。
  - `--no-target-state`（11.14M）：无 `L_target`。
  - `--use-forward-dyn`（13.51M）：`L_dino` 进入 loss。
  - `--use-inverse-dyn`（11.75M）：`L_inv` 进入 loss。

### 排查记录

- **depth 2D 报错**：`cv2.resize` 会把 `(270,480,1)` 压回 2D，且原函数末尾 `/255.0` 对 depth 不正确；`_rk` 改为 2D 分支完整自做 letterbox、不除 255。
- **TpT `traj=None`**：`waypoints_to_actions` 加 `if not wpts: return None`，无轨迹样本 action_valid=0。
- **forward_dyn 不触发**：`train_e2e.py` 补 `mkw["trajectory"] = batch["trajectory"]`，否则 `policy.forward` 缺 trajectory 分支直接跳过。
