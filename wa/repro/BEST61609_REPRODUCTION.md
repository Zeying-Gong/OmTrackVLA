# WA 61609 最佳版本闭环评测复现指南

本指南复现已完成的 **61609 权重评测**，不训练新权重，也不运行当前 failure-state 新训练。Git 中已有实现、配置和结果；权重、场景资产、私有依赖与证据链需要单独交接，**不是只 clone 即可运行的自包含发布包**。

## 目标版本与参考结果

| 任务 | 原 Job / Task | 成功 / 总数 | SR | TR | macro TR | HumanCollision CR | 初始化失败 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STT | 61653 / 72847 | 1279 / 1405 | 91.0320% | 87.5226% | 92.0645% | 3.9858% | 0 |
| DT | 61654 / 72848 | 1178 / 1405 | 83.8434% | 79.6348% | 82.2104% | 6.0498% | 0 |
| AT | 61655 / 72849 | 1207 / 1405 | 85.9075% | 85.0741% | 88.0719% | 5.0534% | 0 |

原运行是三个独立 8×A800 任务，共24卡；4215条全部新评测，没有复用旧模型行。权威报告：[STUDENT61609_FINAL_20261007.json](../results/STUDENT61609_FINAL_20261007.json)，SHA256 `812d5d350754ff79f4e83edee3927cb84f9171474a3ed1a7bc504786601775cb`。STT 距研究目标1289仍差10，不将“复现成功”写成“研究目标全部完成”。

- 评测执行源码：`192b57f5e270acfffd8c7c1a4590cb1b257d92a3`。
- 配置及报告快照：`602c6aa3efb354388de2871f6614c350ff18d27c`，位于同一仓库的 wa 历史中。**三份61609 YAML不在较早的评测源码 commit 内**，因此分开固定。
- checkpoint：`/data/nas_ray/project/md-ak/users/zeying.gong/job_61609/task_72803/wa_hard_stt_train_a800_v1/checkpoint.pt`，step59716，累计2epochs。
- checkpoint SHA256：`c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52`。
- 原训练来源：59866模型与优化器恢复后新增1epoch，新增37009updates；实际曝光1184272=726631原始+457641旧教师。**仅复现推理不需要重训或下载原训练缓存，也不需要59866父权重。**

## 获取代码并固定两个版本

下面路径是原运行的进程内路径。只在自己的持久工作区、确认目录不存在后执行；已有目录则核对身份，不覆盖。

```bash
WA_PROJECT=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928
git clone --branch wa https://github.com/Zeying-Gong/OmTrackVLA.git "$WA_PROJECT/checkout"
git -C "$WA_PROJECT/checkout" worktree add --detach "$WA_PROJECT/source_student61609_eval24_v1" 192b57f5e270acfffd8c7c1a4590cb1b257d92a3
git -C "$WA_PROJECT/checkout" worktree add --detach "$WA_PROJECT/config_best61609" 602c6aa3efb354388de2871f6614c350ff18d27c
git -C "$WA_PROJECT/source_student61609_eval24_v1" status --short
```

从 `config_best61609/wa/jobs/` 获取三份配置，并逐文件确认：

| 配置 | SHA256 |
| --- | --- |
| student61609_stt_a800_v1.yaml | c394ae653205fb98b5ce08f34d7d9df61a56794dae03abfe468046bafea2a648 |
| student61609_dt_a800_v1.yaml | b2bb2794e7411e67b53d6558108d645601649b2ed4a8a8c1b63082a9ef063d1e |
| student61609_at_a800_v1.yaml | 2c9406d45d9c35cc03b3c6ed7138a8f3458b7d2bc8eae16909b427a3516025d4 |

不要用继承的根 README、早期 ResNet RUNBOOK 或9月29日 JEPA probe 指令代替本版本；不要用当前新训练冻结源 `199385cd` 代替61609评测源。

## 交接资产与环境

以下原路径相对 `WA_PROJECT`，除非另有说明。接收方应先给出 PRESENT / MISSING / HASH_MISMATCH 清单。缺少资产、许可或编译环境时停止相应操作，不以别的数据、默认模型或删除校验替代。

| 必需资产 | 原位置或获取方式 | 校验 |
| --- | --- | --- |
| 61609模型 | 上述 Job61609 checkpoint，由项目负责人授权交接 | c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52 |
| WLA构造权重 | /data/nas_ray/project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt | 0b8f036fd282474d8c9efe9e34e1f4dc56b9282c44295bcda1f16edcf48460e1 |
| WLA源码 | dependencies/wla_v1，交接原快照及其源码清单 | 不能以不明版本公共上游替代 |
| DINOv2-S/14 | /data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth | b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 |
| JEPA初始化权重 | models/jepa_wms/mz_jepa-wm.pth.tar | a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc |
| 上游源码 | upstream_audit/dinov2；upstream_audit/jepa-wms | commits 7764ea0f912e53c92e82eb78a2a1631e92725fc8；13cf1d9c7e476f53c17714d2e0f1dc239a883ce0 |
| 全量manifest | ../WLA-EVT-20260925/evt_full_20260926/manifest.json | a1f515534153ccaa720f787f01ea23b9097c174fe25020f756c7d7947eedf98f |
| 七例BBox修复计划 | artifacts/initial_bbox_repair_v2/plan.json | 6535f7b9a53e399ba7f2323c2d7f72d223146ee77f090223eb260f5885f6269a |
| 教师参考索引 | artifacts/dual_teacher_complete_audit_20261006_v1/combined_selections.jsonl | f45f21d71b63d7f2a0898e1e762c393a95b0f3eaf8291bd1affbc827d23354d8 |

DINO与JEPA初始化文件仍被当前模型构造器读取；不能因为有完整checkpoint便自行删除这些依赖。**不需要 Qwen/VLM 权重，不新增 LightNav/Oracle 推理。** 官方DINO/JEPA来源见[既有资产说明](../wm/H100_ASSET_INVENTORY.md)，其中历史训练数据迁移清单不是本次评测的必需清单。

还须交接：

1. manifest指向的三份 `val.json.gz`：STT SHA `8a96fe3be38ab78b0b8985e71645eef337d396be126bbd78ce1c87ffef5a0c63`；DT `1997a14b2a03fb1319c924a3d38fe5824f1b627130d77b7e3d137633f6b63398`；AT `ed4aee0d49c2bd4b0f75e9922c53bd41d7aa50da2aa35fd5f994205c4537b093`。路径按manifest原文，不猜目录名。
2. 获许可的 HM3D/MP3D 场景、semantic PLY、navmesh、机器人和humanoid资源；不是只有JSON就能运行。
3. `../OmTrackVLA-da3-polar-20260924` 的实际运行快照：Habitat-Lab、已编译Habitat-Sim/Magnum、`artifacts/official_runtime`、`torch_overlay`、Xvfb bundle和runtime脚本；以及 `../WLA-EVT-20260925` 的控制/全量评测依赖。
4. 修复计划引用的 `baseline` JSONL及7条 `probe_report` 文件。保留原字节和路径；`load_plan` 会重新验哈希和初态。
5. 教师索引引用的双教师初态证据链：每分支的 `pair_start.json`、`observations.json`、首PNG及验证器读取的其余引用。只给 selections JSON 不足以通过最终配对审计。
6. 两套可恢复环境：模型 `probe_env/bin/python`（Python3.11.15 / PyTorch2.8.0+cu128），仿真 `/data/nas_ray/home/zeying.gong/algorithm/envs/habitat/bin/python`（Python3.9.19）。仿真使用外部PYTHONPATH/二进制overlay，普通pip包表不等于完整环境锁。要求交付原镜像/安装记录/二进制快照及版本清单，不能假称一个 `pip install -r wa/requirements.txt` 已复建环境。

外部10个关键文件的精确哈希已在冻结源码的 `wa/wm/closed_loop_dependencies.sha256` 和 `wa/wm/full_eval_dependencies.sha256` 中；保持这两个文件原样，在资产就绪后执行 `sha256sum -c`。它们不是全部场景/运行环境的完整打包清单。大型文件、受许可场景和凭据不上传Git；跨NAS传输由项目负责人授权并在接收端核验。

补充的[当前环境清单](../results/STUDENT61609_ENVIRONMENT_INVENTORY_20261008.json)记录模型138项、Habitat普通环境88项及按原PYTHONPATH解析112项distribution。Habitat有效torch静态解析为overlay中的2.5.0+cu124，而普通环境metadata列出2.1.2+cpu；numpy来自official_runtime。清单保留冲突，是2026-10-08只读现场信息，不是历史完整二进制锁，也不能直接作为pip安装文件。

## 跨机器路径和权限

原脚本、外部common.py、manifest、repair和teacher JSON中均有绝对路径。最接近原运行的方式是在**经授权的隔离worker或容器内**提供相同路径布局；不要擅自挂载宿主机NAS、复制凭据或覆盖已有目录。

若目标机器不能保留路径，应另建一个可审阅的路径适配分支，列出每个原路径→新路径、文件内容哈希与JSON引用的证据链变化，保留192b57f5原始快照。**不能只替换YAML路径、重写科学JSON然后关闭哈希校验。** 在完成适配审阅和实机检查前标记 PORTING_NOT_VERIFIED，而非已复现。

历史Xvfb安装脚本会清理固定 `/tmp` 前缀，wrapper可能创建 `/usr` 链接。必须先审查这类系统操作的实际目标，只在独立worker/容器及具备授权的环境使用；不要直接在共享开发机照跑。三任务同时运行需三个隔离的8卡worker/容器，防止固定X display编号冲突。

## 不可变实验协议

- `mixed/zero`，4次flow采样；每episode重置Python/NumPy/Torch/CUDA seed7，推理种子元数据为7+step。不要直接采用eval_server默认的image/random。
- RGB历史＋episode0 GTBBox目标模板＋当前理想模拟极坐标UWB，无文本。UWB噪声0/延迟0；首帧后不给GT框、未来目标轨迹或语义ID。观察器保存的世界状态只供离线审计。
- 视觉历史目标时刻[-1.5,-1,-0.5,0]秒，因果选取；模型图像224×224，原RGB传感器384×384。
- 保留JEPA/MetaQuery/ActionExpert；ready必须 `world_predictor_inference=false`。JEPA是训练辅助，不是在线世界模型MPC。
- `learned_yaw_guard_v1`，保留原平移保护、学得yaw、动作尺度、实际dt和物理设置；不换回UWB heading。ctrl_freq40、ac_freq_ratio4、integration_dt0.025。
- `WA_SEMANTIC_PLY_FIX=mp3d_semantic_ply_v1`；配置SHA `1dc43d5488cdcfc0b66d998a63fa87da588a37f115d6970099032caece9776d6`。只修复MP3D语义PLY方向，不改RGB/碰撞坐标。
- 固定 `initial_rgb_mesh_bbox_v1` 只处理7例：STT/DT的pRbA3pwrgk9/27、/38；AT的/27、/38、/71。原框必须为零且首RGB匹配；不得扩展修复范围。
- 不设resume/targeted plan，不复用旧结果。每任务1405唯一task/key、8固定分片[176,176,176,176,176,175,175,175]。

## 新机器先检查再全量运行

先确认checkpoint和所有输入身份、双环境导入、被冻结的协议测试及三任务8分片真实数据定义 `--audit-only`。audit-only不加载模型、不render、不算闭环结果。随后在**开发环境**做有界真实权重接口检查，检查mixed/zero/4steps/59716身份和两次有限输出；不是在集群另提交smoke任务。参考原[预检记录](../results/STUDENT61609_PREFLIGHT_20261007.json)与[诊断记录](../results/STUDENT61609_DIAGNOSTIC_20261007.json)。

无缺项且实机检查通过后，提交完整任务。原百度平台、相同布局下：

```bash
MD_AI_KIT=/data/nas_ray/home/zeying.gong/algorithm/envs/md_ai_kit_submit/bin/md_ai_kit
"$MD_AI_KIT" submit "$WA_PROJECT/config_best61609/wa/jobs/student61609_stt_a800_v1.yaml"
"$MD_AI_KIT" submit "$WA_PROJECT/config_best61609/wa/jobs/student61609_dt_a800_v1.yaml"
"$MD_AI_KIT" submit "$WA_PROJECT/config_best61609/wa/jobs/student61609_at_a800_v1.yaml"
```

每份申请完整8卡和86400秒；先核对登录、实际资源、重复任务、镜像、目录、权限及GPU型号。不是给另一平台照搬百度调度器的指令。

其他平台应使用其调度器，给每个任务一个独立8卡worker。下面是**资产和同路径布局已验证后**的完整评测核心启动环境，按任务分别设置 `TASK=stt/dt/at` 和新的持久 `OUT`；不要让三个任务共享输出目录。原脚本会拒绝已有输出。这只是已核对的核心调用，不代表已在目标机器通过。

```bash
set -euo pipefail
: "${TASK:?set stt dt or at}"
: "${OUT:?set a fresh persistent output directory}"
case "$TASK" in stt|dt|at) ;; *) exit 2;; esac
WA_PROJECT=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928
cd "$WA_PROJECT/source_student61609_eval24_v1"
test "$(git rev-parse HEAD)" = 192b57f5e270acfffd8c7c1a4590cb1b257d92a3
test -z "$(git status --porcelain)"
test ! -e "$OUT"
unset WA_RESUME_PLAN WA_RESUME_PLAN_SHA WA_TARGETED_PLAN WA_TARGETED_PLAN_SHA
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export WA_EVAL_TASK="$TASK" WA_EVAL_MODE=mixed
export WA_STUDENT_EVAL=evaluation_set_adaptation_v1
export WA_EVAL_CHECKPOINT_SHA=c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52
export WA_EVAL_CHECKPOINT_STEP=59716
export WA_DIAG_CHECKPOINT=/data/nas_ray/project/md-ak/users/zeying.gong/job_61609/task_72803/wa_hard_stt_train_a800_v1/checkpoint.pt
export WA_EVAL_TEACHER_SELECTIONS="$WA_PROJECT/artifacts/dual_teacher_complete_audit_20261006_v1/combined_selections.jsonl"
export WA_INIT_REPAIR_PLAN="$WA_PROJECT/artifacts/initial_bbox_repair_v2/plan.json"
export WA_INIT_REPAIR_PLAN_SHA=6535f7b9a53e399ba7f2323c2d7f72d223146ee77f090223eb260f5885f6269a
export WA_SEMANTIC_PLY_FIX=mp3d_semantic_ply_v1
export WA_DIAG_CONTROLLER=learned_yaw_guard_v1 WA_DIAG_OBSERVER_STATE=1 WA_REVIEW_VIDEO=1
"$WA_PROJECT/probe_env/bin/python" -B - <<'PY'
import os, torch
from wa.wm.loaders import sha
from wa.wm.student_eval_contract import model_contract
from wa.wm.student_eval_finalize import load_teachers
from wa.wm.semantic_scene import provenance
model_contract(os.environ)
load_teachers(os.environ['WA_EVAL_TEACHER_SELECTIONS'])
assert sha(os.environ['WA_DIAG_CHECKPOINT']) == os.environ['WA_EVAL_CHECKPOINT_SHA']
assert provenance()['config_sha256'] == '1dc43d5488cdcfc0b66d998a63fa87da588a37f115d6970099032caece9776d6'
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names) == 8
print({'gpu_names': names, 'torch': torch.__version__, 'cuda': torch.version.cuda})
PY
bash wa/scripts/eval_full_mixed.sh "$OUT"
```

此核心命令允许记录实际GPU，**不像原A800 YAML那样保证A800型号**。若换4090/H100，必须记为跨硬件复现并完成兼容性检查，不能把硬件差异隐藏掉或保证bitwise相同；具体资源由目标机器负责人授权。只有8卡时可顺序跑三个完整任务，不改变样本、分片或分母。

## 汇总与验收

每任务8个 `COMPLETE.json`、1405条唯一结果和 `PARTITION_COMPLETE.json` 后，在配置/审计快照目录运行合并。变量必须指向此次的新结果；不指向归档的原成绩。

```bash
cd "$WA_PROJECT/config_best61609"
: "${STT_OUT:?}"; : "${DT_OUT:?}"; : "${AT_OUT:?}"; : "${MERGED_OUT:?}"
MANIFEST=/data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925/evt_full_20260926/manifest.json
TEACHERS="$WA_PROJECT/artifacts/dual_teacher_complete_audit_20261006_v1/combined_selections.jsonl"
"$WA_PROJECT/probe_env/bin/python" -B -m wa.tools.merge_student_partitions \
  --stt "$STT_OUT" --dt "$DT_OUT" --at "$AT_OUT" --manifest "$MANIFEST" \
  --teacher-selections "$TEACHERS" --mode mixed --checkpoint-step 59716 \
  --checkpoint-sha c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52 \
  --output "$MERGED_OUT"
```

合并器重新核对24个原分片、4215唯一key、模型/修复身份及真实初态配对。教师证据缺失时不能声称完整配对审计通过。原初态标准是raw sensor SHA（含shape/dtype/all channels）及动态状态atol1e-6，不是只对JPEG。

用 `"$WA_PROJECT/probe_env/bin/python" -B -m wa.tools.build_student_review --merged "$MERGED_OUT" --manifest "$MANIFEST" --teachers "$TEACHERS" --ffprobe <实际ffprobe路径> --output <新的review目录>` 检查每条首JPEG和视频元数据并建立HTML；这不等于逐帧解码全部录像。人工复核固定样例及任何异常，不删除失败。

若还需运行 `audit_student_goal.py`，其默认61377基线及原分片证据也必须单独交接；它用于研究门槛比较，**不是61609推理依赖**。只看 `audit_status=PASS` 不代表 `goal_status=MET`。本次61609参考结果应是NOT_MET，STT差10。

验收至少包括：

- 调度器终态、worker实际GPU及每分片完整产物；RUNNING/exit0均不单独代表成功。
- 4215唯一task/key、每任务1405；失败与初始化失败留分母。保存逐条success/collision/following_step/total_step及artifact_root。
- SR按原success计数；HumanCollision CR仅表示距目标人曾小于0.5m，不是墙、门框或一般障碍碰撞。
- TR = 100 × sum(following_step) / sum(max(total_step, reference_steps.get(key,0)))；macro_TR = 100 × mean(following_rate)，分开报告。每任务52条缺reference，保留原fallback，不重新选择分母。
- 与原1279/1178/1207逐key对照，保存成功增减和异常。不能为了“匹配分数”删样本、替换失败结果或调阈值。
- 保存执行源码与配置SHA、模型SHA、manifest/修复/教师索引SHA、双环境/驱动/GPU、命令、日志、summary、combined、配对审计和视频位置。
- 若提供远程HTML，先验证服务和SSH链路，再同时给实际端口转发命令与localhost链接。不要复用原机器18799等端口的旧页面当新结果。

该模型经过evaluation-set adaptation；本次复现不是未见测试集泛化，也不是真实UWB、机器人或产品避障验收。LightNav用不同输入，不能忽略输入差异。
