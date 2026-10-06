# Current task
Updated: 2026-10-06 China
Status: A4_DEVELOPER_OFFLINE_ONLINE_PASS_FORMAL_PIPELINE_PENDING / A3_FULL4215_AUDITED / DA3_RETIRED

## User objective
- Improve actual frozen full4215 SR by reusing A3 experience and mixing offline learning with direct online adaptation of pristine A2.
- Repeated TEST learning/tuning/selection is authorized. Label TEST_SCENE_ADAPTATION_NOT_HELDOUT; no heldout-generalization or global-optimum claim.
- First exceed A2_60058 2635/4215=62.514828%, then B60994 2684/4215=63.677343%. Preserve the best full-tested executable candidate.
- Inference uses causal RGB-derived features, original instruction, time/step and internal noise only. GT/outcome/identity are training or audit metadata, never actor inputs or oracle routing.

## Active NAS and evidence
- nas-a800:/data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925/a4_hybrid_replay_20261006.
- Read RUN.json, EXPERIMENT_PLAN.md, data/support/index audits, CORE_DEVELOPMENT.json, REPLAY_LOADER_DEVELOPMENT.json, INFERENCE_DEVELOPMENT.json, ONLINE_DEVELOPMENT_SINGLE.json, ONLINE_DEVELOPMENT_DUAL.json and BATCH_DEVELOPMENT.json.
- All1732 original NPZ hashes and886589 actual latent/reward/component rows match8430 complete A3 traces. Reconstructed complete-episode MC/S labels; old replay unchanged.
- 4760 successful attempts /2914 ever-successful source episodes include568 original A2 failures. These are historical experience coverage, never a single policy's SR.
- FP16[16,256] features precede the trainable reader; actual total28D latent retained.8426 episodes have only first RGB; four have complete frames. No invented full RGB supervision.

## Real developer milestone
- Fresh MC value/outcome plus capped advantage-weighted actual-action regression and soft A2 residual penalty implemented. Both success/failure data retained; task/episode-balanced sampling.
- Freeze original A2 encoder/MetaQuery/Flow/heads/controller and fixed projector. New actor/value readers and optimizers; never restore A3 actor/Q/optimizer.
- Explicit exact CUDA z0(seed7+step); z=z0+zero-initialized residual. Hard rho0.5 supports0 archived actions;rho2 supports15/886589. No old-action clipping or reward relabeling.
- Zero-residual same-input serving:132/132 complete real-frame comparisons passed; final metadata recheck55/55 passed. Reset, GT-key rejection,1207 frozen parameter/buffer versions and trained reload passed.
- Single pipeline:2 offline +4 actual mixed online updates;2 complete adapt episodes/105steps, then2 frozen/reset eval episodes/114steps. Checkpoint100d067f7a943bce475fa395fed890ac8877169aeb7f40a3dbc732f95d88d1e7.
- Dual pipeline:2 offline +4 shared online updates;2 complete adapt episodes/123steps, then2 frozen/reset eval episodes/105steps. Checkpoint5d22c9c5e699720f98aa2d9be53fafdef2fbb1f7921a2685d76f224cfcf1ef4c.
- Independent CPU single/dual audits passed with explicit JPEG/feature-replay limits;96 tensors finite,105/123 actual features reload exactly.
- Each run is independent. Frozen update6 stayed unchanged; durable terminal commit precedes replay admission/update; exact checkpoint reload passed. AT3/DT28 are developer cases, not selection or efficacy.
- Batch256:two real updates/reload PASS;sample IO7.362/7.466s versus GPU update0.1865/0.01475s. Full archive x+a6.8566GiB;8GiB cache proposed. Core memory is not whole-job memory.

## Precision and evidence limits
- A3 executed geometry/auxiliary heads in FP32; original A2 uses BF16 autocast. New A4 restores A2 heads and keeps four-step Flow FP32.
- Nine real precision probes:two controller actions differed,max0.00437714. This does not establish the cause of A3 SR decline.
- Old A3_FP32_HEAD outcomes are historical supervision; new A2_BF16_HEAD execution has a different transition kernel. Preserve old labels, never claim counterfactual A4 outcomes.
- Current online developer first-frame evidence is JPEG hash equality to A2, not historical raw RGB proof. Formal entry must add raw initial RGB and feature hashes; do not backfill unavailable evidence.

## Next executable stage and formal gate
- FORMAL_PIPELINE_DESIGN.json is NOT_SUBMITTABLE:proposed4A800/72h,1024 offline+8430 online updates,one4215 adaptation pass,six frozen252 screens,max one promoted4215;9942 full rollouts upper budget.
- Implement separate formal cohort scheduling, drain/snapshot/restore, standalone frozen serving, optimizer/RNG/online-queue/cursor recovery and full audit/BEST_FULL integration. Preserve developer hard limits.
- Complete short interface checks, four-lane/resource safety, immutable manifests and actual submitter/environment/resource configuration before asking approval.
- Previous8A800/48h recollection submission was automatically rejected before execution;no Job. Old APPROVAL_BLOCK.json remains;no bypass via renamed/smaller/delegated/alternate-interface submission.
- Revised hybrid formal resources have not yet been requested or approved. No formal A4 updates, Job or full4215 result. No stale approval reminders.
- Fixed252 panel is84/task;baseline160/252 (STT68,DT55,AT37). Only fully audited4215 improves a4_sr_adaptation_20261006/BEST_FULL.json;partial peaks never qualify.

## Preserved references and reporting
- Original A2 SHA59cdfe10fc1efa92ae24568152e2c24baa69a1b91609fc4f823ec625204d713b. Representation bundle SHA7b523f1d10a1dad7407ec6aee3fca4016bc1cbb34460cc9ed39ab948ee7dbcc9;load projector only.
- A3_61020/71943 SUCCEEDED:8430adapt/886589transitions/221392updates,then2209/4215 SR52.408066%;STT1061,DT583,AT565. FINALc7269fa14ba53ec60f4f27fa80aa9b295f83464ca5e0124597c30e0c10fce435.
- A2 full STT1143,DT796,AT696;B60994 full STT1169,DT787,AT728. B63.677343% remains highest measured among A2/B/A3,checkpointd4987b0aa961d971c30095c1b914224fff4b4dcc4b604655bb6113cd0f6f2883.
- Preserve old A4 source3979/b87cfadaa2350892ad8eb34f70fa5165bf316cac9c13ddae88f8910a3b44c6a6,all failures/STOPPED results. No60766/59352/DA3 restart or repeated B jobs.
- NAS only;A800 preferred. Doc-only explicit Markdown commit/push;preserve dirty sources and both CSV ledgers.
- Automation every20min advances development and reports milestones/failures;during formal252/4215 eval reports actual partial success/completed SR and84/1405 coverage. No active goal to create or stop.
