# train4 感知标签 sidecar 采集说明

本轮实现独立的标签采集与来源核验，**不运行模型、不改变原始动作、不训练、不修改现有样本/manifest/val/runner**。GPU 采集必须显式加 `--execute`，普通运行仅校验固定输入并输出预检结果。GPU 3 串行运行四个隔离子进程，每例都保留结果或失败，不重试、不替换场景，输出目录存在即拒绝。

## 已完成的实现与验证

- `semantic_labels.py`：由同一帧 RGB/panoptic 生成原目标和干扰者的可见性、面积、bbox 标签。
- `collection_contract.py`：冻结来源、初始化、语义 ID、自然结束、样本前缀及计划引用完整性核验。
- `freeze_perception_plan.py`：读取已有源与配置，生成固定计划；不会创建 Habitat 环境。
- `collect_perception_sidecar.py`：按原动作完整重放，每个观测保存 PNG、原始 panoptic NPY、标签与审计证据。
- 本地及远端 CPU 测试 **47/47 通过**。覆盖错时/错图、GT 对齐偏差、相机/时钟变化、早终止、无终末、原始 NPY 形状、原动作不变、拒绝覆写、native crash 全四例计账，以及计划丢 pin/伪 prefix 的反例。
- 远端完整只读预检通过：1053 个固定输入文件，共 338,307,973 字节。包括源结果、源 launch/status、当前/永久 manifest、19 样本与媒体、配置入口和完整 resolved config、回放代码、选定场景及配置/episode 引用的资源、解释器、新采集代码、准入政策。
- 本子任务未启动 GPU。root 完成审查与第二次 1053 项输入核验后，已启动采集 supervisor（PID `2647169`，启动记录目录 `.codex_upload/perception_label_launch_v1`）。采集结果尚待独立核验；CPU 测试成功不代表仿真重放已匹配，更不代表新标签已准入训练。

远端 bundle：`/data/nfs/share/wam_tracking/OmTrackVLA/.codex_upload/perception_label_collection_v1`。

计划 `frozen_plan.json` 的内容 seal：
`789c1e4b5a668a1017e29d56aea174721ba08352921df22193877f9b96bd5697`

计划文件 SHA-256：
`dd9fcd6b00249a802410f1638216f56a64af51ef998fd1fc9d6566317e080780`

root 提前冻结的 `perception_sidecar_admission_policy_v1.json` 已纳入计划，SHA-256：
`4da01fd3ef238e46db9620948e8f1ca767576c37db648e0f83f5a3d12cce60b7`。

## 固定分母与准确的匹配范围

| 固定源 | 原动作数 | reset..terminal 观测 | 关联新样本 | 最长原 prefix anchor | 原 prefix 逐项比较次数 |
|---|---:|---:|---:|---:|---:|
| AT 401 | 85 | 86 | 3 | 64 | 140 |
| DT 200 | 104 | 105 | 5 | 64 | 149 |
| STT 0 | 84 | 85 | 5 | 32 | 128 |
| STT 3100 | 93 | 94 | 6 | 64 | 228 |
| 合计 | 366 | **370** | **19** | — | **645** |

四源必须为原始永久 train 角色并已自然结束。原短源的 4 个 train 样本在计划中明确排除精确匹配范围，不要求新长源重放伪装成旧短源。4 个永久 val 样本只用于固定 manifest 分母核验，采集器不读取其图像或生成其标签。

645 是重叠样本的逐项检查数，**去重后只有 228 个观测**有原存储 PNG 和 worldtime 对照。其余 142 个观测包括：138 个非终末观测有原策略输入 RGB mean/std/temporal 统计与源几何/可见性检查；4 个 terminal 观测只有相应 post-action 源审计，没有下一次策略调用统计。统计一致不能被称作像素完全相同。

每个 obs k 与动作的对应为：obs 0 是 reset；动作 k 执行后是 obs k；obs k<N 条件化原动作 k+1。源 `steps[k].policy` 的 RGB 统计对应 obs k，源 `steps[k-1].evaluation_only_after_action` 对应 obs k。terminal obs N 没有下一条动作，标签明确 `next_source_action_step=null`。source result 没有逐帧 worldtime 字段，因此时间证据只与现有样本的实际 prefix worldtime 比较，并对新记录检查严格单调；不假造源时间。

## 同次渲染与输出

reset 与每次原动作之后只调用一次显式 `env.sim.get_sensor_observations()`，从同一个返回字典复制 RGB 和原始 panoptic。记录渲染前后相机变换、渲染前后 simulator worldtime，核验采集期间世界与相机未改变。遵循现有 `probe_next007_recovery_sequence_v2.py` 的配置、reset、唯一语义 ID 指派及原动作回放步骤，没有修改该脚本。

每例输出：

```text
<固定output_root>/<run_id>/
  launch_contract.json
  collector.log
  nvidia_egl.json
  observations/rgb_0000.png ... rgb_NNNN.png
  observations/panoptic_0000.npy ... panoptic_NNNN.npy
  labels.jsonl
  worker_progress.json
  worker_result.json              # 正常跑完才有；部分失败保留progress及已有帧
```

批目录保存 `frozen_plan.json` 与固定四条记录的 `batch_status.json`。native crash/超时/错配会保留失败；错配不会改变任何下一步原动作。若源本应未结束但环境已提前结束，保存已取得帧后失败，不能补造后续观测。

NPY 保存传感器原 dtype 与原 shape，包括可能的 H×W×1。helper 的 `panoptic_array_sha256` 是 squeeze 后 H×W 的规范数组 hash；`raw_panoptic_file` 另存原 shape/dtype、原数组 hash 和 NPY 文件 hash。二者不能混为一谈。PNG 比对既核验原文件 hash，又对解码像素生成数组 hash，避免将编码差异误判为像素不同。

每个标签包含：environment_step、policy_call_index、worldtime、terminal、原目标/distractor 的 semantic ID（label-side only）、像素面积、inclusive xyxy、归一化 bbox、visibility/bbox 有效位、RGB/panoptic hash，以及 `source_audit.capture_evidence` 原始采集字段。独立验证器可从保存的原始数据重新计算，不依赖 `passed` 声明。

目标只要有一个渲染像素，visibility 就是真；退化为单像素/单行/单列时 bbox_label_valid=false。该标签不区分遮挡和出视野，不代表校准后的视觉身份置信度。stop、binding、motion permission、ego 与遮挡原因均保持不可用，不从 GT 或旧策略停车行为推导这些监督。未来 loader/loss 必须显式尊重 bbox_label_valid。

## 执行方式

只读预检，不启动 GPU：

```bash
cd /data/nfs/share/wam_tracking/OmTrackVLA
/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python -B \
  .codex_upload/perception_label_collection_v1/collect_perception_sidecar.py \
  --plan .codex_upload/perception_label_collection_v1/frozen_plan.json
```

root 本次采集固定四例所用入口（已启动，不应重复运行）：

```bash
cd /data/nfs/share/wam_tracking/OmTrackVLA
/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python -B \
  .codex_upload/perception_label_collection_v1/collect_perception_sidecar.py \
  --plan .codex_upload/perception_label_collection_v1/frozen_plan.json --execute
```

固定输出：`outputs/takeover/perception_label_sidecar_train4_v1`。内部通过已有 `run_egl.sh` 启动每个 worker，CUDA_VISIBLE_DEVICES=3、MAGNUM_CUDA_DEVICE=0、各线程数为 1，不加载策略权重。不要直接另起 worker，不要删除失败输出后原名重跑。

采集后由独立验证器重新读取原始 PNG/NPY/源文件复算标签及每个匹配条件。只有 4/4、370 观测、19 个唯一原前缀全部通过，才可标为 verified development perception labels；仍不授予 optimizer 输入权或正式训练/产品验收资格。任何阶段不得修改冻结样本、manifest、永久 val 或现有运行器来让结果通过。
