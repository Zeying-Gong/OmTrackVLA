# NavDP / LightNav 时序输入源码核查与 Architecture v2 建议

状态：源码核查完成，参数建议用于 Architecture v2；不修改或覆盖 Architecture v1 checkpoint。

## 1. 固定源码版本

- NavDP：`https://github.com/InternRobotics/NavDP.git`
  - commit：`bff5eb8857f50e082243fb455d312409daabc3aa`
  - 本地只读副本：`C:\Users\59783\Desktop\OmTrackVLA_reference_sources\NavDP`
- LightNav-0：`https://github.com/lightorigins/LightNav-0.git`
  - commit：`c6f40e3220edbf7011e4f17eaf2c865416737d4d`
  - 完整抽取源码：`/data/nfs/share/wam_tracking/baselines/lightnav0_c6f40e3/source`
  - checkpoint eval config revision：`826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`

只读取源码与小型配置，没有运行第三方模型或下载第三方模型权重。

## 2. NavDP 实际使用多少帧

NavDP 发布代码的部署参数为：

- `memory_size=8`：FIFO 保存最近 8 张 RGB；不足 8 张时在历史前端补零。
- 当前 depth 只输入 1 张，不是 8 张历史 depth。
- RGB 和 depth 均缩放至 `224x224`。
- 每张 RGB 经 DINO/Depth-Anything-V2 视觉编码得到 256 个 patch token；8 张 RGB 与当前 depth 合计 2304 个 token。
- 128 个 learned queries（`memory_size * 16`）经两层 Transformer decoder 压缩这些视觉 token。
- 策略条件维度 384、8 heads；部署端使用 16 层 Transformer decoder。
- 轨迹为 24 个 `[dx, dy, dyaw]` 增量；DDPM 使用 10 个去噪步。
- 默认并行采样 16 条轨迹，再由 critic 打分选出最好的一条。

源码入口：

- `baselines/navdp/policy_agent.py`：8 帧 FIFO。
- `baselines/navdp/policy_backbone.py`：8 帧 RGB + 当前 depth、128 个视觉压缩 query。
- `baselines/navdp/policy_network.py`：24 步增量轨迹、10 步 diffusion、16 候选及 critic。
- `baselines/navdp/navdp_server.py`：部署参数固定为 8/24/16-layer/8-head/384-dim。

NavDP 的公开推理代码没有给出固定的历史秒数：benchmark 的 planning thread 是“推理完成后再 sleep 0.1 秒”，所以实际采样间隔还包括推理和通信耗时。不能把它解释成 8 张连续 30 Hz 原始帧。

## 3. LightNav 实际怎样组织历史

固定 checkpoint 的 `trackvla` 和 `vlnce` 配置均记录：

- `video_fps=4`
- `num_history_frames=64`
- `predict_horizon=10`
- 模型输入 `256x448`
- post-ViT spatial pooling

但是启用 SlowFast 时，代码不会简单维护“最近 64 帧 ring buffer”，而是保留整个 episode，并按年龄分层抽样：

1. current：age 0..1，连续 2 帧，空间 pool 1。
2. fast：age 2..33，连续 32 帧，空间 pool 2。
3. mid：age 34..89，每 6 帧取一对相邻帧，空间 pool 2。
4. long：age 90 到 episode 开头，均匀取 14 对相邻帧，空间 pool 4。
5. anchor：固定保留 episode 最初 2 帧，空间 pool 4。

真正值得借鉴的不是“64”这个数字，而是：

- 最近信息高时间/空间分辨率；旧信息稀疏、低空间分辨率。
- 每个片段携带绝对 frame id，并由 `frame_id / fps` 产生真实时间戳。
- 以相邻双帧 tubelet 为键做 session 级 ViT LRU cache；滑窗重复帧不重复编码。
- reset 会同时清历史、绝对 frame id 和视觉 cache，避免跨 episode 污染。
- 轨迹输出采用显式 SE(2) 和控制时间语义，异步控制器按相机采集时刻对齐 odometry。

源码入口：

- `src/lightnav/slowfast.py`
- `src/lightnav/inference/policies.py`
- `src/lightnav/inference/vit_cache.py`
- `src/lightnav/inference/samples.py`
- `robot_deploy/src/vln_mpc/vln_mpc/mpc_node.py`

LightNav 的 Qwen3-VL、文本 prompt、RVQ action token 和 64 帧大历史不直接移植到 OmTrackVLA。

## 4. 对 OmTrackVLA 的直接结论

当前 SAGE3D 策略合同为 30 Hz，v1 使用连续 4 帧，因此历史跨度只有 `(4-1)/30 = 0.10 s`。这不足以稳定估计行人横向运动、机器人转向，更不足以承担遮挡恢复。

Architecture v2-core 建议冻结为：

- `history_size=8`
- `history_stride_raw=3`
- 原始输入 30 Hz，策略历史采样为 10 Hz。
- 历史 frame offsets：`[-21,-18,-15,-12,-9,-6,-3,0]`。
- 历史覆盖 0.7 秒，恰好对应 waypoint 时间合同 `0.0,0.1,...,0.7 s`。
- 每帧保留真实相对时间 embedding，不只使用序号 embedding。
- DA3 输出的每帧空间 token 先各自压缩为 16 个 scene latent，再将 8 帧共 128 个 latent 交给时序 Transformer；当前帧 patch 仍单独用于目标外观 attention 和 UWB Gaussian geometry pooling。
- 不使用 GRU。

这与 NavDP 的“8 帧、每帧约 16 个压缩 token、Transformer 轨迹生成”在容量上接近，但保留 OmTrackVLA 已冻结的 DA3、显式 UWB 双路、目标 reference 和 SE(2) 约束。

暂不在 v2-core 加 LightNav 的全 episode 历史。长历史需要跨帧 SE(2) 对齐、严格时间戳和缓存一致性；直接塞入旧 RGB token 会混淆已经移动的机器人坐标系。若 8 帧版本通过离线和短闭环门槛，再单独增加 2--4 秒 Slow 分支消融。

## 5. DA3 4/8 帧微基准

物理 GPU 1，DA3-SMALL，输入 `[1,T,3,280,504]`，BF16 autocast，warm-up 2 次，计时 5 次，仅 backbone：

| T | 中位前向时间 | peak allocated | peak reserved |
|---:|---:|---:|---:|
| 4 | 12.07 ms | 0.204 GiB | 0.238 GiB |
| 8 | 13.00 ms | 0.288 GiB | 0.340 GiB |

这是 H100 上的局部微基准，不是 Thor 端到端延迟，也不包含 v2 token resampler、trajectory decoder、图像预处理或控制。它只说明 8 帧值得进入 0-step/1-step smoke。

DA3 后层包含跨 view global attention，因此 LightNav 的单 tubelet cache 不能原样套用并保证数值等价。v2-core 先不引入该 cache；后续若需要缓存，应单独验证分帧冻结特征与 DA3 联合编码的精度差异。

