# WA 61609 直接加载与原入口等价检查

2026-10-08，独立构造器 `wa.wm.standalone_inference.load_standalone` 直接加载原 WA61609 checkpoint，不再读取旧 WLA 或 JEPA 初始化权重。完整模型结构保留，冻结 DINO 仍单独加载。A800 上的完整状态和有限接口等价检查通过；不是 H100 或4215条闭环复现结果。

## 权重和代码身份

- 原成品 checkpoint：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61609/task_72803/wa_hard_stt_train_a800_v1/checkpoint.pt`，step59716，SHA256 `c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52`。未导出、覆盖或训练。
- DINO：`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`。
- 新加载器：`wa/wm/standalone_inference.py`，SHA `a457c595a790231cea65320173081cafb861decaaad1750a4e5070b76e0e6a2c`。
- 15项CPU测试：`wa/tests/test_standalone_inference.py`，SHA `4b1c4da52731076f75a0797c1761ac84b4039b99c24e03694703796a84c64e42`；独立重跑PASS，0.236秒。
- 独立审计器：`wa/tools/audit_standalone_inference.py`，SHA `fdf6999851aa8a8b528d4afa3f93fa89e2751858dc6a79425963d616f02fbc16`；另有6项CPU辅助检查通过。
- 当前 training/loaders/adapters/robot_data/eval_server/sim_uwb 六个核心文件与原评测源码 `192b57f5e270acfffd8c7c1a4590cb1b257d92a3` 逐文件SHA一致。没有修改原加载器、冻结评测源码或运行中的训练源码。

## 真实检查结果

| 检查 | 结果 |
| --- | --- |
| 完整 state_dict | 669张量逐值一致，含494 WA状态与175 DINO状态 |
| 模块结构及非state属性 | 619模块一致；仅完整模型和JEPA模块两个构造器子类不同，继承原计算方法 |
| 非持久buffer及普通Tensor | 一致，包含不在state_dict的JEPA attn_mask（1024×1024 bool） |
| 旧新源码清单 | 288个WLA结构源码哈希一致；DINO和JEPA源码保持固定提交 |
| 新入口实际 torch.load | 仅WA61609与DINO两个文件；其他权重路径被审计拦截器禁止 |
| Session预测 | 每模型6次，共12次；mixed有首框、image有首框、mixed无首框各首/后续两请求，zero噪声 |
| 输出比较 | 完整payload一致；336个轨迹/几何/历史时刻数值有限，最大差0；排除耗时字段 |
| 非法请求 | 重复时间、后续BBox、image无首框均拒绝 |
| 训练辅助模块调用 | world、robot_action、robot_state均0次 |

使用固定合成渐变PNG和当前理想UWB测试Session，不是模拟器完整轨迹，不证明一般输入或其他硬件上的数值等价。完整WA参数和非权重属性未删减；JEPA仍在checkpoint/模型结构中，但上述在线预测不调用它。

## 运行与产物

在devpod-a800 GPU3执行，UUID `GPU-8feb0a89-838a-a61a-cbd3-ec9205b9c344`，A800-SXM4-80GB，PyTorch2.8.0+cu128。两次运行前快照为0%利用率、79263MiB空闲；没有停止其他进程，也不是独占GPU声明。旧、新模型顺序移入GPU。分配器cap为5GiB，不包含CUDA context等外部分配；超时上限900秒，实际47.282秒、exit0。

旧入口峰值allocated/reserved为1,571,047,424/1,595,932,672B；直接入口为1,570,234,368/1,598,029,824B。保留3条xFormers unavailable警告；日志未命中审查的5类fatal模式。

项目根 `R=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928`：

- 完整报告：`R/artifacts/standalone61609_equivalence_20261008_v1.json`，770822B，SHA `d62f50513100dd64c17f8866ca249d8e40019e5efebc81bfaf3614a1f4b5dd2d`。
- 日志：同目录 `standalone61609_equivalence_20261008_v1.log`，1153B，SHA `74e933a3a661515d4fb06156cad22019da836075aac6ab90d1e2556b6d9f8de9`。
- 原报告保留在NAS；本摘要和代码进入Git。独立只读复核重新检查源哈希、输出、计数和上述产物哈希，没有重复加载模型。

完整审计命令（复查时使用新的输出名，不覆盖v1；旧WLA文件只供reference比较）：

```bash
cd "$R/checkout"
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. timeout 900 "$R/probe_env/bin/python" -u -m wa.tools.audit_standalone_inference \
  --root "$R" --reference-source "$R/source_student61609_eval24_v1" \
  --encoder-weight /data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth \
  --wla-source "$R/dependencies/wla_v1" \
  --wla-checkpoint /data/nas_ray/project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt \
  --checkpoint /data/nas_ray/project/md-ak/users/zeying.gong/job_61609/task_72803/wa_hard_stt_train_a800_v1/checkpoint.pt \
  --checkpoint-sha256 c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52 \
  --checkpoint-step 59716 --gpu --allocator-cap-bytes 5368709120 \
  --output "$R/artifacts/standalone61609_equivalence_20261008_v1.json"
```

## 复现所需与未完成项

新加载函数所需权重是WA成品和DINO；WLA模块实现、JEPA与DINO结构源码仍需固定版本。已有61609 checkpoint内含完整训练后JEPA参数，不需另外下载JEPA预训练文件来替代它。只有JEPA和DINO权重不能还原WA学得的融合、动作专家和目标头。

独立RPC入口和全量启动接线需另行验证；不要删除旧依赖后直接运行192b旧命令。H100实机接口、双环境/场景和证据迁移、私有资产上传仍未完成。本检查不改变原SR、不证明训练反向/NCCL或新的模型效果。
