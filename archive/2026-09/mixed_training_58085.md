> Archived DA3 evidence. All earlier continuation/retraining instructions and live status claims are superseded by the 2026-09-28 retirement decision; Job58613 is STOPPED. See docs/tracking_route_decision.md.

# SAGE + EVT mixed training 58085

Status: RUNNING_WITH_OPTIMIZER_UPDATES, not completed or validated generalization.

Job58085/Task68820 RUNNING2026-09-26 15:09:02AsiaShanghai;submitted15:08:57;baidu_bj_a8004actualA800-SXM4-80GB driver550.163.01;2epochs/globalbatch8/lr5e-5/seed7;339939stepsPerEpoch679878totalUpdates;freshAdamW modelOnlyWarmstart57528epoch2;all4ranks fiveRequiredGradientGroupsFiniteNonzero;firstMeanLoss0.1858642101;visualAdapterDeltaL2=0.001382117742;step_00001.pt161562802bytes;latestObservedStep471loss0.1047398448elapsed95.15s;bothSAGEandEVTobservedInLogs;generalizationUNVERIFIED

## Release and audit

- Migration: 68,209 files / 62,851,248,760 bytes / 16 shards. All per-file SHA256 passed. No source changes; failed partials and earlier test logs retained.
- Transfer root: /data/nas_ray/home/zeying.gong/datasets/sage_mixed_transfer_20260926_v1; complete receipt SHA256 b0c2a05af421e14acef017b05509cc779a88779ee749948c3a12ddff3d7c2c60.
- Mixed release root: /data/nas_ray/home/zeying.gong/datasets/da3_mixed_release_20260926_v1/release. Release ID f5688de50698ce007fd96cd767b9a2273a393a94974e0644d292d0303d4f9a18.
- Train: 40,939 episodes / 2,719,509 windows / 1,456 scenes. SHA256 27e5a464ea4f7a86727f9448a2b2cebd996e6be4b4e523c03c7b373f8e8ccb81.
- Heldout: 3,824 episodes / 253,860 windows / 152 scenes. SHA256 0d267278ce4470cf50df31bec4f79cab1684fc797e952be4a69e1da9ec00e547.
- 49 CPU tests PASS; /data/nas_ray/home/zeying.gong/datasets/da3_mixed_release_20260926_v1/preflight/complete.json CPU_MIXED_PREFLIGHT_PASS. Real SAGE and EVT batch 2x32x3x224x224 / 2x7x2, finite tensors, 0.1-0.7s future times, strict checkpoint load and hash PASS.
- Original immutable source audits and releases unchanged. SAGE user-attested 30Hz/same-index RGB-pose mapping; render phase not instrumented and physical collision unverified. TPT remains excluded/quarantined. Source-family scene separation is provenance-based, not mesh-hash identity proof.

## Reproducible run

- Canonical source nas-h100 /data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA; base9ad4bd1554f97e3a43913bdcd3d4f30af797f6b7 + existing dirty changes. No commit/push.
- Frozen execution: /data/nas_ray/home/zeying.gong/datasets/da3_mixed_release_20260926_v1/code; 108 files; source.sha256 SHA256 35ed25a1148efa94aeb1629189d2cc863614a9290366c56ca5dac98a82c3daea. All files match canonical source and worker source_check.log passed.
- Config scripts/da3/baidu_a800_sage_evt_mixed_4gpu.yaml SHA256 b8fca88d03f5a740dd1b75f3e242547da0b937d0ec704a590fefcb7c80f9503e.
- K8s baidu_a800; 4 GPUs; image x5-builder:cuda12.8-isaac5.0.0-v2.test1; timeout259200s; Python internnav torch2.7.0+cu128.
- Warmstart /data/nas_ray/project/md-ak/users/zeying.gong/job_57528/task_68138/da3_evt_joint_train/training/epoch_2.pt SHA2563ca2b3ac99b47091fb551c1d1d1c67c2e19d6709d67dc8e1b82e4c8bab29f63c. Fresh optimizer, not checkpoint resume.
- Natural full-window coverage; batch2/rank; 2epochs; lr5e-5; seed7; UWB dropout.5/noise.05m/maxdelay0; 3 explicit DDP padding repeats/epoch. No architecture/loss changes.
- Actual task command is preserved in the frozen YAML. No scheduler smoke job.
- Output /data/nas_ray/project/md-ak/users/zeying.gong/job_58085/task_68820/da3_sage_evt_mixed.
- First step meanloss0.18586421012878418; five required gradient groups finite/nonzero across all4ranks; visual adapter deltaL2 .00138211774174124; step_00001.pt161562802bytes. Observed step471 meanloss.10473984479904175 elapsed95.15s. Both dataset sources observed in training logs.

## Remaining acceptance

- Full679878updates /2epochs and final checkpoint/completion receipt still pending.
- Training losses are not heldout or benchmark metrics. Learned Who and generalization UNVERIFIED.
- Downstream comparison restricted to matched10episodes; resolve controller1/40 vsactual.048s before closed-loop evaluation. No automatic full1405episode benchmark.
