# train4 perception sidecar loader 实现报告

新增 `omtrackvla/data/perception_sidecar.py` 和 `tests/test_perception_sidecar.py`。模块不改原样本、manifest、现有 loader 或 model inputs，不读取 panoptic，不创建模型、optimizer 或仿真环境。标签可用于显式检查，`for_optimizer=True` 在所有公开消费入口均拒绝；formal/product 资格始终为 False。

## API

```python
store = PerceptionSidecarStore(
    verification_path,
    verification_sha256="aec0690b2d2a3727d7e3ba80afce07670876568e49ed2028e48d6545d70cfc5d",
    plan_path=plan_path,
    plan_file_sha256="dd9fcd6b00249a802410f1638216f56a64af51ef998fd1fc9d6566317e080780",
    artifact_root=repository_root,
)
original = recovery_item.as_batch().batch  # 完整 [B=1,S,...] 前缀
supervision = store.supervision_for(recovery_item.identity.sample_path, learning_steps=4)
combined = overlay_recovery_sequence(original, supervision, for_optimizer=False)
```

`artifact_root` 可以是远端 repository，也可以是保持原始文件字节的本地镜像；模块只做冻结绝对路径到镜像根的明确映射，不改 JSON 内路径。verification/hash 必须由调用者显式提供，不能从待验文件自身推导后冒充外部 pin。

`supervision.as_numpy()` / `.as_tensors()` 返回单序列 `[S]` / `[S,4]` 监督；后者仅生成 CPU Torch tensors。overlay 返回全序列 `[1,S]` / `[1,S,4]` 新字典，新增字段为：

| 字段 | dtype | 含义 |
|---|---|---|
| `target_visible` | float32 | 原初始化目标的 anypixel 可见性 |
| `visibility_label_valid` | bool | 默认仅末 4 个真实调用有效 |
| `target_bbox` | float32 | inclusive extrema / 原图宽高；无效位置存有限零 |
| `bbox_label_valid` | bool | 仅可见且非退化 bbox 有效 |

单像素、单行或单列目标仍有可见性监督，其 bbox mask 为 False。前置 burn-in 标签存零且无效。`supervision_mask` 为原 anchor 与已验证感知窗口的并集，`waypoint_mask` 和 `target_waypoints` 保持原对象、仅 anchor 有效。`identity_label_valid`、stop/binding/ego masks 保持全 False，不虚构对应 targets。

`combined` 的所有 model input 值和对象均来自 `original`。overlay 还逐个核对原 RGB、reset 初始化、四帧历史、UWB 缺失值和缩放后的相机参数，拒绝给同长度的另一条序列贴标签。按原 `burn_in_steps` 切出 loss labels，完整 model inputs 仍从 reset 开始 unroll；不要把新增标签后的字典再交回原先仅接受 anchor 标签的 groups helper。

与新的 `phase3_sequence_loss` 协同：visibility 使用独立 `visibility_label_valid`，bbox 使用 `bbox_label_valid & target_visible`；旧 `identity_label_valid` 不再阻止这两种已验证监督。loss 修改由另一代理负责，本模块未编辑该文件。

## 绑定与拒绝范围

验证链为外部 verification SHA → plan 文件 SHA/内部 seal → 完整 4 源与 370 帧状态 → 各 source/status/launch/labels 的 SHA → 原 admitted manifest 与永久 train 角色 → 19 个唯一样本及其实际 prefix RGB/hash/world time。样本以原绝对路径和 SHA 匹配，不能按 basename 或仅按长度匹配。

只允许这些原始前缀的 reset→anchor 观测。旧 4 个短轨迹 train 样本、val、未列样本、terminal、未来帧、非连续/错时观测、变更源 assignment、错误 target ID、部分验收与资格升级均拒绝。独立验收报告已经重算 raw semantic 证据；本模块依赖其外部 pin，不再执行 panoptic 重导出。

## 验证结果

本机专用测试共 19 项：**18 项通过，1 项真实 Torch overlay 测试因本机没有 Torch 而跳过**，耗时 10.806 秒；实现及测试通过 Python 3.9 语法解析。测试覆盖输入不变、跨 sample 绑定、来源/标签/样本篡改、退化 bbox、无效 aux、时钟、未来 history、val/terminal/optimizer 拒绝。

另一代理在远端隔离候选目录完成真实 schema 检查：store 正确收录 19 个样本；STT 0 anchor027 为 28 calls、burn-in 24，末 4 个 visibility 是 `[1,1,1,0]`、bbox mask 是 `[True,True,True,False]`，前 24 调用没有感知监督，optimizer 请求被拒绝。该检查没有使用 Torch/GPU 或覆盖运行模块。

提交给父任务统一安装和真实 Torch 综合测试的快照：

- 模块 SHA：`c893a2f15ef7f7069d4f8436528f99428c7ed807f2cb313507534ded0ea898de`
- 测试 SHA：`08a5eb66fe85f9f79649935f947a2a17ac1ffcf7a508fe638b695d1535008025`

随后已完成远端安装及集成验证：专用 19 项测试全部通过，包括本机跳过的 Torch overlay；连同相关序列和 trainer 回归共 65 项不同测试通过。19 条真实前缀全部完成输入对象与数值不变、时间与标签对齐、原 anchor waypoint 保留检查。

GPU7 上原 Phase2 的真实 BF16 反向亦通过：STT anchor027 完整 28 次调用，burn-in 24；visibility/bbox 两个头都有有限非零梯度。损失分别为 1.51743054 和 0.06158895。没有构造优化器、更新或保存权重，父 checkpoint SHA 不变。原始结果为 `outputs/takeover/perception_supervision_preflight_v1/{cpu,backward}.json`，文件 SHA 分别为 `f9e9c03ed2a54828b8b664dd0f3aabaacaeacff3f0f0f719a17fed8fda23de32`、`5f9a6a94f701b070c0eb9c86f23103790267c614ad0fcad62447b440310721b1`。

这是接口与梯度验证；optimizer 资格仍关闭，尚无感知优化后的精度或闭环提升结论。
