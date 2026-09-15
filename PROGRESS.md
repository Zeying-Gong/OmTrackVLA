# PROGRESS

更新：2026-09-15

## 当前真实状态

1. 当前主线是可审计的 Phase 3 失败状态恢复闭环；正式 Phase 3 optimizer 训练尚未开始。
2. 保留原 Phase 2 parent。v2_016 多场景 pilot 与 v2_017 安全头校准均未通过晋升条件。
3. v2_018 的未见闭环 gate：18 条 rollout 零碰撞，但 Success 全为 0；v2_017 se2 三任务全程停车，拒绝晋升。
4. NEXT-027 v6 的 50 步无 GT-point 闭环安全子门通过，但跟随与墙角重获失败；不是成功结果。
5. ABL-08 是当前最佳开环参考（ADE/FDE 0.084341/0.146654 m）；ABL-09 frozen frontend 约为其两倍误差。
6. test_locked 未访问；index1300 已使用，不能再作为未见 gate。

## 下一步

- 完成失败状态候选的完整 reset-to-anchor 前缀、测时 expert 重采样、独立场景验证、checkpoint/resume 和 stage gates。
- 先运行数据准入、单样本 backward 与最小批次 smoke；全部通过前禁止正式优化训练。
- 闭环复核必须报告 Success、Tracking、Collision、跟随帧率及安全停车，并保存可追溯配置和汇总行。

完整过程、实验细节、失败日志和历史方案：archive/2026-09/。
