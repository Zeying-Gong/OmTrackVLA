# 永久 val4 感知补标与评估协议

本目录完成了可复现的 CPU 协议生成器、固定分母指标，以及复用已有同次 RGB/panoptic 捕获核心的独立 val 适配测试。21 个 CPU 测试通过。真实 val 补标、远程资产预检、独立 val 标签准入及新模型推理尚未执行；当前状态仍为 `frozen_design_pending_val_collector_and_remote_asset_preflight`。没有启动 GPU、优化器或训练，没有修改旧 train 冻结包。

该协议在已知原 val4 闭环结果后制定，在新 val 原始标签与新感知候选预测产生前冻结。不能称为原闭环实验之前的预注册。

## 固定数据与用途

只重放已完成 Phase2 baseline 组的动作，不采用 best16 或 last64 组轨迹。Phase2 checkpoint step 为 4096，SHA-256 为 `32c8f2f73277cfab8c3c7c22a53994061f7178498cfa003e1635b9576641bc9c`。

| 案例 | 原数据索引 | episode | 永久场景 | 原动作数 | reset 至 terminal 标签数 | 可预测观察数 |
|---|---:|---:|---|---:|---:|---:|
| dt_1300_episode3 | DT 1300 | 3 | gjhyih4upq9 | 90 | 91 | 90 |
| at_1700_episode0 | AT 1700 | 0 | ayhkzj2fehp | 46 | 47 | 46 |
| dt_1700_episode0 | DT 1700 | 0 | ayhkzj2fehp | 45 | 46 | 45 |
| stt_1700_episode0 | STT 1700 | 0 | ayhkzj2fehp | 44 | 45 | 44 |
| 合计 | | | 2 个场景 | 225 | 229 | 225 |

原 Habitat 文件目录的 split 是 `train`，但这四例的永久角色始终是 `val`。它们只能评估，不能成为优化器输入、温度拟合或阈值校准数据。已有 train4 仍只用于训练或训练开发；不读取 test_locked，不调整永久场景分工。四例只有两个场景，不能给出广泛泛化的结论。

生成器校验原冻结 val4 协议、已独立审计的 Phase2 suite、suite 制品清单以及每例 result/status/launch/timing 的原始文件哈希；同时固定原动作、初始化、语义 ID 分配、模型配置与既有捕获核心。新 raw 标签总分母必须为 229，模型预测总分母必须为 225，不能替换失败案例或接纳部分数据。

## 采集与时间对齐

观察 `k` 是执行原动作 `k+1` 之前的输入。因此 `source.steps[k].policy` 对应观察 `k`；`k>=1` 的动作后 GT 对应 `source.steps[k-1].evaluation_only_after_action`。终止观察 `N` 必须保存 raw 标签，但不存在下一次预测，不能把动作后标签错配到同一行的动作前预测。

每例从原 reset RGB 和 panoptic-mask 初始化框开始，保持原 agent 0 目标与所有 distractor 的语义 ID 分配。按固定动作列表执行每个动作一次，在 reset 和每次动作后从**同一次传感器观察字典**同时获取 RGB 与 panoptic；保存无标注 RGB PNG、原 shape/dtype 的 panoptic NPY、文件哈希和数组哈希。语义像素只生成标签与审计数据，不能选择动作。

每帧记录并检查：捕获前后世界时间一致、相机变换稳定、世界时间单调、原始动作顺序、相机与 GT 距离/可见性、非终止帧的原 RGB mean/std/temporal statistics、精确自然终止步数。几何容差 `1e-4`，时间容差 `1e-8 s`，RGB 统计容差 `1e-5`，不能看到失败后放宽。

源 result 没有原始 worldtime，也没有完整无标注 RGB 前缀。其 `rollout_frames/frame_*.png` 经 `_draw_frame` 添加了标注与面板，不能作为模型原始输入逐像素对照。旧 anchor9 前缀源于另一条旧 rollout，同样不能强制对齐。新 val 捕获只能对源几何、GT、统计做已有证据允许的检查；统计相等不代表像素相等，不能伪称验证了原 worldtime 或原像素。

提前终止、渲染漂移、native crash、缺帧或错位都必须保留失败状态及已写证据。未来 supervisor 应保留四例固定分母，禁止重试、补选案例或覆盖输出。新增 raw 数据只有通过独立 val verifier 才能获得 `verified_permanent_val_perception_labels_evaluation_only` 准入。

## 在同一批新 raw RGB 上评估

Phase2 与后续候选都必须在同一份独立准入的新 RGB 数据上重新推理，才可直接比较感知预测。旧日志 Phase2 visibility/bbox 只用于诊断，不能默认与新 replay 图像相同。

每例清空 hidden state 与 target memory，使用原 reset 图像和一次初始化框，执行 reset 至 `N-1` 的全部因果调用与原四帧历史规则。UWB 维持缺失条件；后续 GT bbox、panoptic、目标语义 ID 不进入模型输入或缓存。候选动作不在模拟器执行，模型 gate 保持不变。固定候选 checkpoint SHA、代码/config/resize/precision/seed 后再评估；不根据该 val 结果反复拟合阈值。

每个预测必须包含观察索引、`source_rgb_array_sha256`、`source_world_time_s`、visibility 概率、归一化 bbox。重复、负数、未来或 terminal 预测索引直接判协议失败；缺失预测保留在固定分母内，错 RGB 或时间绑定会使该帧两个感知输出均无效。

## 冻结指标

- 可见性：主阈值 0.90，次阈值 0.50。报告 TP/TN/FP/FN、GT 正负样本上的无效输出数、包含无效失败的 accuracy/recall/specificity，precision 明确只统计实际正预测输出。
- 概率校准：仅有限数值 `[0,1]` 有效，bool 无效。Brier 使用完整分母，无效惩罚 1；log loss epsilon 为 `1e-6`，无效惩罚 `-log(epsilon)`。10 个等宽 ECE bin 只计有效概率，同时报告有效覆盖率与无效数，不能用低覆盖掩盖缺失预测。
- Bbox：分母是所有 GT 可见且 bbox_label_valid 的非终止观察，**不得用预测 visibility 过滤**。报告 mean IoU、IoU>=0.5/0.75 比例，以及 `visibility>=0.9 且 IoU>=0.5` 联合比例。无效、缺失、越界、退化预测框 IoU 为 0，不裁剪。
- 框采用既有 inclusive 像素极值除以宽高，再计算连续面积 IoU；归一化坐标不加 1。单像素或线状目标依然 visible=True，但没有有效 bbox 标签，单独报告该数量。
- 每例、pooled micro、四例等权 macro 同时报告；无定义指标报告 null 和 defined_case_count，不替换成好看的零或一。

语义像素可见性不是身份预测正确率。此评估不能证明人物闭环、主动找回、身份恢复、避障、安全停车、任意目标或 Thor 延迟已经达标，不能自动提升 checkpoint 或放宽 gate。

## 现有采集器可复用部分与剩余实施

旧 train collector 的 `verify_plan`、supervisor 和准入规则固定了四个 train 来源、370 观察和 19 个原样本，不能加一个 `--val` 参数绕过，也不能 monkeypatch train 常量。

`val_collection_adapter.py` 已验证可单独复用其 `HabitatReplay` 所需数据投影与中性的 `collect_episode` 捕获核心。它对 source/config/template 做哈希校验，给出空旧 prefix 参考，保持原动作、初始化和语义 ID。CPU harness 仅接受显式 `cpu_mock_only` 后端，真实 Habitat 后端被拒绝，没有 GPU CLI。Mock 使用合成 RGB，因此与真实 baseline RGB 统计不一致会按预期留下 failed_alignment；这验证失败保留机制，**不是成功采集真实 val 数据**。

独立 val launch contract 与新的 artifact manifest 绑定中性核心写出的原始 labels/raw 字节；不重写 label 文件伪造角色，也不复用旧 train 完成报告哈希。下一阶段需完成：

1. 只对两处永久 val 场景、选定 episodes 及必要 runtime/config/URDF/motion 依赖做远程资产闭包与哈希预检，固定 resolved config、episode、环境、原代码依赖。不能仅依靠本地协议 JSON 就启动。
2. 单独实现真实 val worker/supervisor、统一四例状态清单、故障/超时保留及全新输出目录；按原资源约束单 worker、GPU3/MAGNUM0、线程数 1。当前目录不提供这一真实执行入口。
3. 实现独立 val verifier：从 raw PNG/NPY 重新计算标签/数组哈希，检查每帧动作和 capture 证据、完整四例与所有制品哈希，不信任 collector 的 passed 字段；禁止部分准入。
4. 固定新候选，再实现同 RGB 上的 Phase2/候选完整因果推理和上述指标导出。闭环搜索等后续验收仍需独立任务与指标。

## 对 train loader/loss 的独立审查

已阅读新 `omtrackvla/data/perception_sidecar.py`、`phase3_sequence_loss` 与相关 loss 测试。loader 的必要边界为永久 train 角色、19 个唯一原样本、独立 verification/plan/source/labels 全链文件哈希、reset 至 anchor 的真实观察/时间/RGB 绑定、末四个 policy call 的监督、terminal 排除与四帧历史 reset padding 正确。重复 reset 图像仅是模型历史，不能被当作四个新观察标签。

发现真实 source 的 semantic assignment 比较曾因 `_sources` 在调用后才填充而退化成默认值自比较，已反馈 loader owner；owner 已直接在读取 source 后比对并删除 fallback，补测试。原未知 identity/binding/stop/ego 保持无效，raw panoptic 不能进入模型分支。开发标签准入不能变成 optimizer 授权，当前 `for_optimizer=True` 拒绝是正确边界。

loss 的新 visibility/bbox validity 必须独立；仅 key 缺失时回退 legacy identity mask，显式全 false 绝不能回退。bbox 另需 GT visible；无效 NaN 必须在算术前索引排除，不能 NaN*0。现有测试包含这些条件与 legacy loss/gradient 等价，方向正确。额外建议在有效 visibility 标签上检查有限 0/1；生产 sidecar 已保证此项，loss 层检查可更早定位手工 batch 错误。总 supervision 可覆盖末四帧，但 waypoint_mask 必须仍只覆盖原 anchor，不能创造前三帧 waypoint 标签。

## 本地用法

仅校验并打印协议，不写入：

```powershell
python -B .\generate_val4_protocol.py
```

新建冻结文件，已有文件拒绝覆盖：

```powershell
python -B .\generate_val4_protocol.py --output .\permanent_val4_perception_protocol_v1.json
```

CPU 检查：

```powershell
python -B -m unittest discover -s . -p "test_*.py" -v
```

最终源码/协议/报告哈希、测试日志与 Python 3.9 语法检查记录在 `VERIFICATION.json`。该记录证明本目录的 CPU 设计与测试，不是远程采集或独立 raw 标签准入凭据。
