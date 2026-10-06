# Current task
Updated: 2026-10-06 China
Status: A4_DEVELOPMENT_PASS_FORMAL_COLLECTION_APPROVAL_PENDING / A3_FULL4215_AUDITED / DA3_RETIRED

## Current user authorization
- User explicitly requests A4 to raise A-method SR on the same4215 TEST episodes, with repeated learning/tuning/model selection allowed. A2 plus RL or other WLA-compatible method is authorized.
- First target: exceed A2 2635/4215=62.514828%; next exceed B60994 2684/4215=63.677343%. Keep highest actually full-tested candidate.
- This is TEST_SCENE_ADAPTATION_NOT_HELDOUT, an empirical fixed-budget ceiling search; no heldout-generalization or global-optimum claim.
- User broadly authorized A4 SR improvement. Auto-review on2026-10-06 rejected specific8A80048h formal collection because exact resource authorization was not recognized;explicit question sent. Read A4 APPROVAL_BLOCK.json;do not retry/submit indirectly without clear approval.
- Inference remains causal RGB+original text+timestamp. GT is training labels/scoring only; no episode lookup or oracle policy routing.
- Old A2/B/A3 weights/results and failed/STOPPED runs remain immutable. No DA3,59352,60766 restart.

## Active A4 package and route
- A800: /data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925/a4_sr_adaptation_20261006.
- Read RUN.json,EXPERIMENT_PLAN.md,A3_LESSONS.json before acting; check scheduler before submissions.
- Initial checkpoint originalA2_60058 SHA59cdfe10fc1efa92ae24568152e2c24baa69a1b91609fc4f823ec625204d713b.
- Collect all4215 episodes independently: A2-success2635 forced originalstudent for behavior anchors; A2-failure1580 allows LightNav takeover/recovery/handback.
- Admit correction only from actual terminal SR1, complete0.7s expert-only/no-fallback windows. If intervention fails or yields0windows, reset a separate full LightNav rescue; preserve both attempts and selected pointer. Coverage4215, actual rollouts up to5795.
- Train action_expert only; freeze backbone/MetaQuery/geometry/XYvis. Correction flow supervision plus frozen-A2 velocity distillation at identical conditions/x_t/t.
- Balanced anchor/correction batches and task sampling; document repeats/coverage. SmallLR1e-6 proposed; no performance claim from training loss.
- Same real PNG132-step old/new serving RPC outputs bitwise PASS. Full3 replay endpoint metrics match;STT0/AT3 116steps exact,DT28 action drift remainsUNCONFIRMED despite same initialJPEG. Preserve initial failure;JPEG is not rawRGB proof.
- Fixed252 full-horizon panel(84/task,uniform deterministic hash) for multiple saved candidates;top2 receive full4215. Persist cross-campaign BEST_FULL.json;A2 stays incumbent until genuine higher fullSR.
- Keep original A2 incumbent and every checkpoint; never replace best using online SR,partial-panel peak or final-step assumption.
- Real single batch2/single batch4/dual batch4 each2updates PASS;initialanchor0,214frozen tensors exact,checkpoint reload exact,dualpeak17.219GiB. Formal batch4/GPU,global16. Actual newstep2 checkpoint serving/reset/full3episodes166frames PASS.3979collectioninputs frozen/checkPASS;auto-review rejected specific8A80048h submission before execution. No A4job;wait explicit resource approval.
- Expected resources8A800 collection/eval,4A800training; query live resources before submission. No cluster smoke.

## A3 completed result and lessons
- Job61020/Task71943 SUCCEEDED;8430 adaptation episodes,886589transitions,221392SACupdates,then frozen/reset4215.
- A3 final SHA c7269fa14ba53ec60f4f27fa80aa9b295f83464ca5e0124597c30e0c10fce435.
- Independent4215 strictA2 pairs and431802-step trace/reward/frozen policy audit PASS.
- A3 STT1061/1405 SR75.516014%;DT583/1405 SR41.494662%;AT565/1405 SR40.213523%.
- Overall2209/4215 SR52.408066%,TR_macro70.368704%,CR3.724792%;SR-10.106762pp versusA2.
- A3 actorzero latent was not A2seed7+step Gaussian behavior. Q target inflated182 despite episode reward<=11; final latent44.81% nearbound.
- A3 improved388 originalfailures but regressed814 originalsuccesses. Lost1331 vsA2 802; collision157 vs357. Cause labels remainUNANNOTATED.
- Evidence A800 a3_test_scene_rl_20261004/FINAL_INDEPENDENT_AUDIT_20261006.json;only finalA3 checkpoint received full4215.

## Preserved references and ongoing reporting
- A2_60058 full4215: STT1143,DT796,AT696 successes; SR62.514828%.
- B_partial_cache60994 full60996 audited:2684/4215 SR63.677343%, highest measured among A2/B/A3.
- Historical A2 action-only/59752 remains separate; originalA60316 fixed36mixed remains frozen.
- Use task+dataset_index,never scene+episodeID dictionary overwrite. Preserve full horizon and terminal omissions explicitly.
- Update NAS RUN,ledgers,CURRENT_TASK/PROGRESS/taskdoc on new evidence;doc-only explicit commit/push,preserve dirty source.
- Automation wla updated to A4. Every20min advance verified stage;notify formal starts/completions/failures. During evaluation report exact partial task counts/SR every20min.
- A3 monitoring goal complete;A4automation remainsactive. While specificcollectionapproval pending,no automatic retry or repeatednags;independent development complete,wait user reply.
