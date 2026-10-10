# Student62256 同权重有／无 UWB 全量配对评测

日期：2026-10-10（北京）。结论：同一 step61252 权重直接移除理想 UWB 后，STT／DT／AT 三项 SR 均下降。原 mixed 三项固定门槛已达标；image 模式单独未达标。这是 evaluation-set adaptation 上限实验，不是未见测试集泛化。

| 任务 | mixed 成功／1405 (SR) | image 成功／1405 (SR) | image−mixed SR | 参考归一 TR mixed→image | macro TR mixed→image | CR mixed→image | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STT | 1300 (92.53%) | 1124 (80.00%) | −176 / −12.53 pp | 88.56→83.67% | 92.27→89.14% | 2.35→5.91% | 0→0 |
| DT | 1184 (84.27%) | 454 (32.31%) | −730 / −51.96 pp | 80.09→49.85% | 82.20→58.50% | 5.20→14.52% | 0→0 |
| AT | 1212 (86.26%) | 707 (50.32%) | −505 / −35.94 pp | 85.77→69.61% | 88.33→77.55% | 4.06→10.25% | 0→0 |

逐例配对四格按“两组都成功／仅 mixed／仅 image／都失败”：STT 1118/182/6/99，DT 445/739/9/212，AT 685/527/22/171；每行共 1405。TR=sum(following_step)/sum(max(total_step,reference_step))，macro TR 另列。CR 是目标人距离曾小于 0.5 m，不是一般障碍碰撞。每任务 52 条缺参考步数依固定协议处理；invalid 均留在 1405 分母中。

## 运行身份和终态

- 同一 checkpoint：J/job_62256/task_73511/wa_failure_state_train_a800_v1/checkpoint.pt，4358277705 B，SHA256 40915b366ee5a2ef5967e2ce49a94d2b0f45f149955dd5cccf85ac3d6ab178fc，step61252。训练从 59866 的模型及优化器 step22707 独立再训练一轮；采样、历史失败及训练审计见 FAILURE_STATE_TRAIN_62256_TERMINAL_20261009.md。
- 冻结评测源码 R/source_student62256_eval24_v1 commit 5a23a982c7ef01f7fdec58faf27f6ea623ed9eeb。三份已提交的 image YAML 在 wa/jobs/student62256_{stt,dt,at}_image_4090_v1.yaml，配置 Git commit 6d7598bb7e3c65d8a4cb389d5b7835e7866667c5；SHA256 依次为 1c22c052f9802d03bac6eb22e9ed9b0f948df7a6b64b257b5855f54997e50758、04f82200fa66522f852cb7f012752e7ddaedbe2e9b385de9fd1653528b33b263、79c584cc1899585cedca760e64ae354be7e85fa26253711ca8fdc87db17f47c2。完整命令在 YAML。
- image STT Job63245/Task74578 SUCCEEDED 11:25:31；DT63246/74579 SUCCEEDED 11:24:06；AT63247/74580 SUCCEEDED 11:51:37，均为北京时间 2026-10-10、canonical baidu_bj_4090、各 8 张 RTX4090，仅提交一次。新输出分别为 J/job_63245/task_74578/wa_student62256_stt_image_4090_v1、J/job_63246/task_74579/wa_student62256_dt_image_4090_v1、J/job_63247/task_74580/wa_student62256_at_image_4090_v1。
- 每任务 8/8 shard COMPLETE.json 和根 PARTITION_COMPLETE.json；分片 176/176/176/176/176/175/175/175；4215 个唯一 (task,key) 与冻结 manifest 精确一致，无漏项或重复。24/24 server_ready 固定同一权重 SHA／step、image/zero/sampling4/seed7+step、无文本与在线 world predictor；每 Job console 记载 8 个不同 RTX4090 UUID、cc8.9。worker/server/console 未见致命错误或 OOM。
- mixed 对照沿用已审计的同权重、同 4090、同 1405 key/任务结果。STT62445/73719 在 8/8 shard 完成后依用户要求 STOPPED，CPU-only 根目录恢复及 provenance 已保存，不能改称调度 SUCCEEDED；DT62446/73720 和 AT62447/73721 正常 SUCCEEDED。详见 STUDENT62256_POSTSTOP_MIXED_AUDIT_20261010.md。

## 全局审计和范围

- 冻结结果审计源码 R/source_student61715_result_audit_v1 commit 32fa14084e71eacb6c9527e493580f089a5b7428 clean。merge_student_partitions --mode image 得到 R/artifacts/student62256_image_full_audit_20261010_v1/summary.json，状态 COMPLETE_FULL_VALIDATION，SHA256 2d3c7338318a29102077c38fcbd10482510b993a0604ec6699ccf772e6f6fd15。另独立逐行核查 semantic PLY：每任务 MP3D 721 条与非 MP3D 684 条标记符合场景，7 条初框修复分布 STT2／DT2／AT3，无额外修复；教师初态配对证据在合并审计中复核。
- build_student_review --mode image 得到 R/artifacts/student62256_image_review_20261010_v1/audit.json，PASS，SHA256 97c0b9433af10ecb4f46ab66cc97aa2c52ca3f2dbf4926035bc64de85f99d6c4。4215 对初态及首帧 JPEG 哈希、4215 个视频流元数据／时长通过；不包含逐帧解码。index.html 在同目录并链接 NAS 中原 24 分片视频。
- compare_uwb_modes 得到 R/artifacts/student62256_uwb_comparison_20261010_v1/comparison.json，SHA256 785e3799db25bca2699edc9d740ca5204b615585da023ec3ecd21eef1d1344a4，状态 PASS_STORED_EVIDENCE_COMPARISON_ONLY、ablation_release=false。4215 对初始 raw RGB SHA 精确相同、动态初态 1e-6 容差内、保存的教师初态证据一致；四格逐例键在 JSON 中。该 CPU 工具不重放 GPU、物理或视频；结论还结合调度终态、worker 日志、冻结源码／配置哈希及媒体审查，不以单个 PASS 代替真实性证明。
- image 推理请求只含 RGB、episode0 GT bbox 模板和时间，服务端拒绝额外 UWB／后续 GT 框，UWB token 与 polar/age 置零；mixed 另外使用当前理想极坐标 UWB。两组无语言或在线 world predictor，JEPA 仅用于训练辅助。这里只测试同权重直接去输入，未进行无 UWB 重训、真实 UWB、边缘时延或未见测试集评估。
