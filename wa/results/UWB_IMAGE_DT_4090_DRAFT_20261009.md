# Best61609 DT image/no-UWB RTX4090 配置草案（2026-10-09）

状态：**STATIC_DRAFT_NOT_SUBMIT_READY**。仅准备配置，未运行 4090 开发测试、未提交正式任务、无 DT image SR。

- 配置：`wa/jobs/student61609_dt_image_4090_v1.yaml`，SHA256 `3f2a0dbd124b402717ebe92f2a4a8a5474a2ac28ba8832ef5388cd1c80951c11`。基于已完成 STT image 同协议的 DT A800 配置 `wa/jobs/student61609_dt_image_a800_v1.yaml`（SHA256 `dc3d70724a6a4c2fdbaab9ee3af25bb19879a0eb19bcd58f8db8c6cffaca64b0`）。
- 机械差异仅为目标集群 `baidu_4090`、日志和任务身份、GPU 名称断言 `4090`、独立输出名；另有首行 `NOT_SUBMIT_READY` 注释。模型、source、输入模式、物理、控制器、指标、8 分片及完整执行脚本不变。
- 冻结评测源码 `source_student61609_eval24_v1` commit `192b57f5e270acfffd8c7c1a4590cb1b257d92a3`；固定 checkpoint `J/job_61609/task_72803/wa_hard_stt_train_a800_v1/checkpoint.pt` SHA256 `c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52`，step 59716。其中 `J=/data/nas_ray/project/md-ak/users/zeying.gong`。
- 配置请求单个 K8s shell task、8×RTX4090、timeout 86400 秒；运行 `WA_EVAL_TASK=dt WA_EVAL_MODE=image`，输出 `J/job_${MD_AK_JOB_ID}/task_${MD_AK_TASK_ID}/wa_student61609_dt_image_4090_v1`，完整 worker 命令在 YAML。
- DT 固定清单 1405 唯一 episode，8 分片 176×5+175×3；共享 GPFS、Python/Xvfb/双环境和 10 项依赖已静态检查。YAML parse、`bash -n`、与 A800 模板允许差异检查通过；这些不能代替实际 4090 worker 兼容性。

提交门禁：先在未干扰他人的 4090 开发卡上做有界 best61609 image RPC 和真实 DT Habitat 单步，核 source/权重 SHA、模式、无 UWB 输入、有限预测和显存。随后重核实际集群资源及本人任务查重，并让正式 worker 验证 8 个独立 GPU UUID、checkpoint SHA/step、Xvfb、场景和输出隔离。A800 仅余 1 卡、H100 项目所在独立 NAS 未就绪、4090 集群有空闲卡的观察均只是时点资源，不是已分配 8 卡。当前没有 Job/Task ID，不可称为正式评测结果。
