# Current task
Updated: 2026-10-06 09:20 China
Status: A3_RLT_DSRL_SAC_61020_FULL4215_AUDITED / B_FULL4215_60996_AUDITED / DA3_RETIRED

## Current authorization
- User approved A3 after reviewing two complete episode/reward/policy previews: “我同意你的方法，赶紧实验。”
- The approved RLT/eRLT-inspired representation + DSRL-SAC replaces the earlier PPO proposal and approval-wait gate.
- A3 may repeatedly train on4215 TEST episodes, then freeze/reset/evaluate all4215; label TEST_SCENE_ADAPTATION_NOT_HELDOUT.
- This measures fixed-budget scene adaptation, not heldout generalization or a proven global upper bound.
- WLA remains the route. No DA3 restart, old59352 restart, or stopping unrelated tasks.
- Original A2, A/B data/checkpoints/results remain immutable; historical A2 action-only/59752 is a separate old experiment.

## A3 completed test-scene adaptation evaluation
- Job61020/Task71943 SUCCEEDED on4A800;8430 TEST-scene adaptation episodes,886589transitions,221392SACupdates,then frozen/reset4215 deterministic evaluation.
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
- Final A3 checkpoint SHA c7269fa14ba53ec60f4f27fa80aa9b295f83464ca5e0124597c30e0c10fce435; each STT/DT/AT1405, strict A2 source/seed7/initialRGB/full-horizon/reward/431802-step trace audit PASS.
- A3 frozen SR2209/4215=52.408066%; STT1061/1405=75.516014%, DT583/1405=41.494662%, AT565/1405=40.213523%. A2=62.514828% (-10.106762pp); B=63.677343% (-11.269276pp).
- A3 TR_macro70.368704%,CR3.724792%; per-task metrics and audit: A800 a3_test_scene_rl_20261004/FINAL_INDEPENDENT_AUDIT_20261006.json. Only final checkpoint had full4215 evaluation; no intermediate optimum or heldout-generalization claim.

## B completed evaluation
- User stopped remaining inversion60766/71645 at09:23:27; do not resume full78769 route or fixed36 requirement.
- Partial B60994/71917 SUCCEEDED:63488 durable TRAIN windows,63127pass;2epochs124updates, noheldout validation.
- Checkpoint SHAd4987b0aa961d971c30095c1b914224fff4b4dcc4b604655bb6113cd0f6f2883.
- B60996/71919 SUCCEEDED17:24:52; independent4215 audit/strict A2 pairing PASS. No duplicate.
- Output: /data/nas_ray/project/md-ak/users/zeying.gong/job_60996/task_71919/wla_b_partial_full4215.
- Wrapper A directories belong to B_partial_cache_60994;1405each/source identity/seed7/initialRGB/fullhorizon/finite audit PASS.

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
- A800 preferred; no unverified cross-NAS migration. User requires progress and STT/DT/AT partial success rates every20min until full evaluation and audit complete.

2026-10-04 17:31 China: B partial-cache60994 full evaluation60996/71919 SUCCEEDED17:24:52; independent4215 strict pairs with A2_60058 PASS (each task1405 exactly once, source identity/instruction/seed7/initialRGB/full horizon, checkpoint/result/trace hashes and finite values). B STT1169/1405 SR83.202847 TR_macro87.186117 CR4.982206; AT728/1405 SR51.814947 TR_macro75.449817 CR9.537367; DT787/1405 SR56.014235 TR_macro68.297166 CR10.177936. Overall2684/4215 SR63.677343 versus A2_60058 2635/4215 SR62.514828, +1.162515pp; task SR delta STT+1.850534/AT+2.277580/DT-0.640569pp,307 success improvements/258 regressions.455740 contiguous frames,454984 finite policy records,756 Lost terminal omissions retained; failure categories UNANNOTATED. Checkpoint60994 epoch-2.pt SHAd4987b0aa961d971c30095c1b914224fff4b4dcc4b604655bb6113cd0f6f2883. Highest measured full frozen SR among current A2/B comparison is B63.677343%; not all historical methods or A3 upper bound. B trained63488 durable TRAIN prefix rows/noheldout, not full78769 or a single-factor ablation. Evidence intervention_b_full_20261004/INDEPENDENT_COMPLETION_AUDIT.json. A3_61020 remains RUNNING:190000transitions/47245SACupdates/1783adapt episodes at17:31;8430adapt and frozen4215 evaluation pending. No A3 frozen SR yet; keep model/reward/frozen727source unchanged, no duplicate jobs/resume60766.
Routine2026-10-05 17:03 China: A3_61020 remains in frozen evaluation; latest scheduler read16:59 RUNNING and208/4215 completed. Independent all8430 adaptation reward/terminal audit PASS across886589steps, including2173Lost terminals. Full durable replay audit PASS:1732chunks, all886589action IDs exactly once, executed latent/reward/done exact,phase tolerance1e-7,chunk hashes/finite/terminal next-feature zeros verified. Evidence ADAPTATION_REWARD_AUDIT_20261005.json and ADAPTATION_REPLAY_AUDIT_20261005.json. Final4215 coverage/paired metrics remain pending; no model/source/job changes. User now requires20min progress and per-task partialSR reports.
