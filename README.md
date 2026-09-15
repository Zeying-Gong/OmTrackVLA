# OmTrackVLA

## 当前唯一任务

完成可审计的 Phase 3 失败状态恢复训练闭环：使用真实模型诱导失败状态、按测得模拟时钟重采样 expert，并在独立场景上验证。正式优化训练尚未开始。

## 验收标准

- 训练样本保留 reset 到 anchor 的完整前缀，模型输入不含 GT 目标点。
- expert 标签按实测时间重采样；坐标、时间和场景切分均可复核。
- 训练/恢复验证场景与状态严格隔离，候选样本和 checkpoint 可追溯。
- 先通过数据、单样本 backward、最小批次和安全 gate，才允许正式训练。
- 闭环评测同时报告 Success、Tracking、Collision、跟随帧率和安全停车；不得以离线 ADE 替代。

## 当前结论

- Architecture v1 的最佳开环候选仍是 ABL-08 收敛模型（ADE/FDE 0.084341/0.146654 m）。
- Frozen Frontend ABL-09 明显退化；NEXT-027 闭环及 v2_018 未见闭环 gate 均失败，不能晋升。
- Phase 3 小预算训练/安全头校准没有通过闭环 gate；保持原 Phase 2 parent。
- test_locked 未被访问；index1300 不能再作为未见 gate。

## 阻塞项

- 尚无通过准入的完整失败状态/恢复数据集与正式 Phase 3 optimizer 训练。
- 最新闭环证据显示跟随和重获失败，不能宣称端到端方案有效。

## 使用约定

- 默认只读本文件、PROGRESS.md 和 EXPERIMENTS.csv。
- 原始进度、方案、审计和失败日志位于 archive/2026-09/，仅按需查阅。
- 原始数据、权重、checkpoint、视频和运行输出不入 Git。
