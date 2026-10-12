# Student63395 纯图像重训全量评测与存储证据审计

日期：2026-10-11（北京）。结论：新模型在固定 STT／DT／AT 各 1405 条闭环评测中分别成功 1142／477／753，invalid 均为 0；STT 81.2811% 严格超过 80% 门槛。三项评测及全局存储证据审计已完成。与旧模型直接去 UWB 的比较使用不同权重，不能解释为同权重 UWB 输入的因果消融。

| 任务 | 62256 mixed／UWB-on 成功 | 62256 同权重直接 image 成功 | 63395 纯 image 重训成功（SR） | 63395 参考步数 TR | macro TR | CR | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STT | 1300／1405 | 1124／1405（80.00%） | 1142／1405（81.2811%） | 84.6107% | 89.8010% | 6.1922% | 0 |
| DT | 1184／1405 | 454／1405（32.3132%） | 477／1405（33.9502%） | 51.0593% | 59.7236% | 15.0178% | 0 |
| AT | 1212／1405 | 707／1405（50.3203%） | 753／1405（53.5943%） | 71.9263% | 78.9392% | 9.8932% | 0 |

62256 mixed 与其直接 image 使用同一 checkpoint；其 mixed STT 在 8 个分片完成后按用户要求 STOPPED，随后只在 CPU 恢复根级来源标记，不能写成调度 SUCCEEDED。63395 相对直接 image 多成功 18／23／46 条，但它是从共同父模型独立训练出的另一套权重。TR 为 sum(following_step)／sum(max(total_step, reference_step))，macro TR 单列；每任务 52 条缺参考步数按原协议处理。CR 指目标人距离曾小于 0.5 m，不是一般障碍物碰撞。失败与 invalid 均保留在 1405 分母内。

## 训练、输入与运行身份

- 路径简写：R=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928；C=R/checkout；J=/data/nas_ray/project/md-ak/users/zeying.gong。
- 新训练 Job63395/Task74738 的权重为 J/job_63395/task_74738/wa_failure_state_image_only_a800_v1/checkpoint.pt，4,358,277,769 B，SHA256 ac8ca38158d89ab6dcd11a9f604b09d37a3db7db63075091cfd66458a2171efa，step61252。它从共同父 59866 的模型和优化器 step22707 独立续训一轮，新增 38,545 次更新，不是从 62256 再训练一轮或从 61609 续第三轮。实际曝光 1,233,424=726,631 base+457,641 retained teacher+49,152 recovery；image 1,233,424、point 0、mixed 0，推理显式 image／zero UWB。
- 冻结训练源码 R/source_failure_state_image_only_v1 commit 76635eb257e43ab2c8194e5b358cac023a3351a0；训练 YAML C/wa/jobs/failure_state_image_only_a800_v1.yaml SHA256 f5413cf82f95d6d0d032675445ecb58b7d307d59a907268d02400d34b6a4c0ea。独立离线训练审计 R/artifacts/failure_state_image_only_training_audit_63395_v1.json SHA256 619e35c26916c490c164ce956929e10bc82ce72f473643b76a97348c37411203，状态 TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY_IMAGE_INPUT；其“无闭环 SR”是训练结束时的离线审计范围，不覆盖后续正式评测。
- 冻结评测源码 R/source_student62256_eval24_v1 commit 5a23a982c7ef01f7fdec58faf27f6ea623ed9eeb；STT／DT／AT YAML 分别为 C/wa/jobs/student63395_{stt,dt,at}_image_4090_v1.yaml，SHA256 依次为 9abd61cea349df6bc6f03cf2d4f89d9a5571c854d5a79df5e7e81ed5366e5026、576130e01691563406ca13c5f48f075a5853dea0ba87b403f22189c16a1db6f4、f670dc38ddea8fb848a8d31b9e3189ddb292d2a38ddb382614d46627d816effd。固定清单 /data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925/evt_full_20260926/manifest.json SHA256 a1f515534153ccaa720f787f01ea23b9097c174fe25020f756c7d7947eedf98f；教师选择 R/artifacts/dual_teacher_complete_audit_20261006_v1/combined_selections.jsonl SHA256 f45f21d71b63d7f2a0898e1e762c393a95b0f3eaf8291bd1affbc827d23354d8。
- 正式 image／无 UWB 评测 STT Job63933/Task75322、DT63934/Task75323、AT63935/Task75324 均在 canonical baidu_bj_4090 K8s 终态 SUCCEEDED，各自 8 张不同 UUID 的 RTX4090／cc8.9、8/8 正确 server_ready、8/8 COMPLETE 与根 PARTITION_COMPLETE。每任务八分片 176/176/176/176/176/175/175/175，1405 个清单唯一 key，无重复、漏项、invalid 或致命 OOM。输出依次位于 J/job_63933/task_75322/wa_student63395_stt_image_4090_v1、J/job_63934/task_75323/wa_student63395_dt_image_4090_v1、J/job_63935/task_75324/wa_student63395_at_image_4090_v1。推理输入为 RGB、episode0 GT bbox 模板与时间，无 UWB、文本、后续 GT 框、未来目标路径或在线 world predictor；JEPA 仅为训练辅助。

## 全局审计与证据边界

- 新结果合并 R/artifacts/student63395_image_full_audit_20261011_v1/summary.json SHA256 a28d8d383c4542516401475aad13c70e1a66941195850e5115ec64b3de854b7c，状态 COMPLETE_FULL_VALIDATION；4215 条均来自新 24 分片、每任务与冻结清单精确一致，复用旧结果 0 条。原合并 summary 的通用训练模式限界句沿袭旧实现，不能用来描述 63395 的训练输入；下面比较审计内的 derived_new_summary 修正文案而不改指标或原文件。
- 不同权重逐例比较 R/artifacts/student63395_image_result_comparison_20261011_v1/comparison.json SHA256 4776433ef618d9c4b60080908f47c307bfee8fa647ceec284d643e749b28a7e7，状态 PASS_STORED_EVIDENCE_DIFFERENT_WEIGHTS_ONLY；4215 个 task/key 配对初始 raw RGB 哈希、九 agent 动态状态（容差 1e-6）及保存的教师初态证据一致。旧 62256 direct-image 权重 SHA256 40915b366ee5a2ef5967e2ce49a94d2b0f45f149955dd5cccf85ac3d6ab178fc；新 63395 权重如上，两者都为 step61252、但参数不同。
- 媒体审计 R/artifacts/student63395_image_media_trace_review_20261011_v1/audit.json SHA256 975fdb7e7437d0c56fce8ca04dbe64a7b0a575172a62eebfa00ed4287945b9c0，状态 PASS_STORED_MEDIA_TRACE_REVIEW。对 4215 段 review.mp4 全帧解码并核精确帧数，审计清单累计 440,187 帧；6195 张保存的 step0/100/… JPEG 与视频前面板配对，另核 4215 个首 JPEG、视频哈希与 trace/review/action。中间帧虽已解码，并无独立保存相机帧供逐像素配对。
- 这是 evaluation-set adaptation 上限与持久化产物一致性检查，不是未见 test 泛化、真实 UWB／边缘时延测试，也不是模拟器、物理或 RPC 重放。不同权重比较不能单独隔离“有无 UWB”训练效应；同权重直接移除输入的结论仍以 C/wa/results/STUDENT62256_UWB_PAIRED_20261010.md 为准。历史失败、旧权重及原 worker 产物均保留；报告仅引用 NAS 结果路径与哈希，不纳入大型权重、逐帧媒体或原始大 JSON。
