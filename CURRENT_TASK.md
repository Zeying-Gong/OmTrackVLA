# Current task
Updated: 2026-10-04 11:10 China
Status: A3_RLT_DSRL_SAC_61020_RUNNING / B_FULL4215_60996_RUNNING / DA3_RETIRED

## Current authorization
- User approved A3 after reviewing two complete episode/reward/policy previews: “我同意你的方法，赶紧实验。”
- The approved RLT/eRLT-inspired representation + DSRL-SAC replaces the earlier PPO proposal and approval-wait gate.
- A3 may repeatedly train on4215 TEST episodes, then freeze/reset/evaluate all4215; label TEST_SCENE_ADAPTATION_NOT_HELDOUT.
- This measures fixed-budget scene adaptation, not heldout generalization or a proven global upper bound.
- WLA remains the route. No DA3 restart, old59352 restart, or stopping unrelated tasks.
- Original A2, A/B data/checkpoints/results remain immutable; historical A2 action-only/59752 is a separate old experiment.

## A3 active task
- Job61020/Task71943 RUNNING since11:10:52 China,4A800,96h.
- NAS package: WLA-EVT-20260925/a3_test_scene_rl_20261004.
- Frozen727-file source SHA265eeb843c73122d2b738b78792fcbf03079cf315e4191f2f60c7c067fe03124; no edits or duplicate submission.
- Initialize pristine A2_60058 step3114 SHA59cdfe10fc1efa92ae24568152e2c24baa69a1b91609fc4f823ec625204d713b.
- Frozen backbone/MetaQuery/Flow/geometry/XY/visibility/controller. Pool last16 MetaQuery layers;256-D projection/readout.
- TRAIN representation probe768train+96heldout,5epochs60updates; online readout/latent actor/twinQ SAC follows.
- One central learner,4simulator/model lanes; no B-style inversion. Actual control transitions, gamma1, replay50000.
- Approved reward F_after/300 + terminal(10S-2C-2Lost); GT reward/scoring-only, actor RGB/text/time.
- Two complete4215 adaptation passes(8430episodes), then frozen deterministic4215 with environment/history reset.
- Worker727source+210protocol hashes PASS;768TRAIN/96heldout representation warmup completed60updates,exact reload. Online SAC verified:1300 real transitions/11 complete episodes/70updates; no efficacy claim.
- Output: /data/nas_ray/project/md-ak/users/zeying.gong/job_61020/task_71943/wla_a3_rlt_dsrl_sac.
- Read run/STATE.json,run/warmup/COMPLETE.json,run/learner/progress.json,run/adapt/lane_*/progress.json.
- Then run/FROZEN_POLICY.json,run/eval/lane_*/progress.json,run/COMPLETION_AUDIT.json.
- Single/dual real developer checks each2adapt episodes123transitions27SACupdates, then2frozen re-evaluations; exact reload/nonzero gradients/initialRGB/reward/replay PASS.
- Developer losses/two-episode results are not efficacy evidence. Initial layer-interface/Xvfb/checker-assumption failures retained.
- Preview preserved: http://127.0.0.1:18785/a3_pretrain_20261004/ ; ssh -N -L 18785:127.0.0.1:18783 nas-a800.

## B active evaluation
- User stopped remaining inversion60766/71645 at09:23:27; do not resume full78769 route or fixed36 requirement.
- Partial B60994/71917 SUCCEEDED:63488 durable TRAIN windows,63127pass;2epochs124updates, noheldout validation.
- Checkpoint SHAd4987b0aa961d971c30095c1b914224fff4b4dcc4b604655bb6113cd0f6f2883.
- B60996/71919 full4215 RUNNING8A800 since09:28:40; keep running, no duplicate.
- Output: /data/nas_ray/project/md-ak/users/zeying.gong/job_60996/task_71919/wla_b_partial_full4215.
- Wrapper A directories belong to B_partial_cache_60994. Require1405each/source identity/seed7/initialRGB/fullhorizon/finite audit.

## Completed reference state
- A2_60058 full4215 evaluation60857 SUCCEEDED: STT1143/1405 SR81.3523%;AT696/1405 SR49.5374%;DT796/1405 SR56.6548%.
- A2 TR_macro86.6925/75.3419/68.4171;CR6.2633/9.1103/10.0356 (STT/AT/DT). Already reported; do not repeat completion notification.
- Iterative original-A60316:8798updates/2epochs and fixed36 diagnostic60349 complete/mixed; frozen.
- Historical A2 action-only/59752 confirmation DT regression remains distinct; no relabeling old runs.
- Three-release union70370train+8399heldout audited; no diagnostic/human labels in original training.
- Old60507 OOMKilled137;60317 SIGKILL cause unconfirmed; preserve logs/caches.

## Acceptance and records
- SR primary; per-task successes/1405,TR_macro(mean episode following_rate),CR and paired differences.
- No cause labels without actual human annotation. Late avoidance/reorientation comments scoped to reviewed videos only.
- Natural Lost/Collision finish=false allowed; never fabricate missing terminal logs in old results.
- Authoritative NAS RUN/source manifests, bothCSV ledgers and docs/wla_intervention_ab_20260929.md.
- Only explicit task Markdown is committed/pushed; existing dirty research source/CSV stays on NAS.
- A800 preferred; no unverified cross-NAS migration. Routine heartbeat quiet; notify milestones/failure/material changes.
Routine2026-10-04 14:41 China: A3_61020 RUNNING 104800transitions/25945SACupdates,1002/8430adapt episodes; exploratory online 591/1002 is not fixed-policy SR. Four frozen servers ~9.90GiB,evidence[0,0],no traceback. B60996 RUNNING 2995/4215; no complete audit. No model/reward/source/job changes; no validated A3 full4215 or peak SR yet.
