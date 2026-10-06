# Current task
Updated: 2026-10-06 China
Status: A4_DATA_INTERFACE_PASS_MODEL_DEVELOPMENT_PENDING / A3_FULL4215_AUDITED / DA3_RETIRED

## Latest user objective
- Raise A-method frozen full4215 SR by reusing A3 data and mixing offline learning with direct online RL from A2.
- User allows repeated TEST learning/tuning/selection. Label TEST_SCENE_ADAPTATION_NOT_HELDOUT; do not claim heldout generalization or global optimum.
- First target>A2 2635/4215=62.514828%; next>B60994 2684/4215=63.677343%. Preserve highest actually full-tested executable candidate.
- This latest request supersedes making full4215 expert recollection a prerequisite. The old flow-finetuning alternative and all developer evidence remain preserved.
- Inference causal RGB+original instruction+timestamp and internally generated noise only. No GT inputs, episode lookup or oracle policy routing.

## Active NAS package
- nas-a800:/data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925/a4_hybrid_replay_20261006.
- Read RUN.json,EXPERIMENT_PLAN.md,DATA_REUSE_AUDIT.json,then REPLAY_INDEX_AUDIT.json and RESIDUAL_SUPPORT_AUDIT.json when present.
- DATA_INTERFACE_PASS:all1732 NPZ hashes and886589 actual latent/reward/component rows versus8430 fulltraces independently verified;MC/SR index implemented. Hybrid model/optimizer/serving checks pending,noformaljob.
- A3 archive:8430 episodes,886589 transitions,1732 replay chunks(~13.63GiB);4760 successful attempts,2914 unique source episodes ever succeeded.
- Of originalA2 failures,568 had an A3 exploratory success(STT137,DT217,AT214). This is learnable experience coverage,not any policy's SR or an achievable oracle-combined score.
- Cached FP16[16,256] features precede trainable reader;actual28D total latent/reward/done/phase/action_id retained. All success/failure experience remains available.
- Only4 A3 trajectories have complete RGB;8426 have first frame only. Direct latent learning is possible; arbitrary RGB→Flow supervision is not.
- DATA_REUSE_AUDIT independently rechecks all manifests/source pairs and first/middle/last NPZ; full886589 numerical replay conclusion cites the preserved earlier full audit,not a new full read.

## Revised hybrid route and verification
- Freeze originalA2 encoder/MetaQuery/Flow/oldheads/controller and A3 fixedprojector. OriginalA2 SHA59cdfe10fc1efa92ae24568152e2c24baa69a1b91609fc4f823ec625204d713b.
- Projector SHA7b523f1d10a1dad7407ec6aee3fca4016bc1cbb34460cc9ed39ab948ee7dbcc9;reader/policy/value/optimizers initialized independently,not resumed from degradedA3.
- Exact CUDA Gaussian z0(seed7+prediction_step,shape1x7x4) plus zero-initialized residual;actor explicitly sees z0. Same-input zero-residual serving must reproduceA2.
- Full886589 residualsupport audited:rho0.5 supports0, rho2 only15 recorded28Dactions. Usezero-initialized unboundedresidualmean+softA2regularization;explicitz0 input,neverclipolda/keepoldreward.
- Reconstruct complete episodes by lane/action_id/done;success S is main MC target,old shapedreward separately retained. No cross-lane/chunk-blind return accumulation.
- Both successes/failures train value;advantage-weighted regression uses actual recordedactions,episode/task balancing and cappedweights. Historical MC advantage is not unbiased current-policy advantage.
- Mix archived data with new online complete episodes,small initial exploration,record actual totalz/basez/delta/policyversion/outcomes.
- ChangingFlow invalidates simple oldlatent-replay compatibility;old ActionExpert-finetuning branch must remain separate.
- Verify real data index/returns,exactnoise/interface,real offline+online short updates,frozenparams,finite,checkpointreload/reset and noGT actor inputs before formal work.
- Existing fixed252 full-horizon panel(84/task) screens multiple frozencheckpoints;promoted candidates get full4215 eachclass1405 and strictA2 pairing.
- Persistent fullbest remains a4_sr_adaptation_20261006/BEST_FULL.json. Never use onlineSR,partialpeak,Q/loss or cross-policy successunion as fullbest.

## Formal resource gate and preserved work
- Previous8A80048h fullcollection request rejected by automaticapproval review beforeSSH execution;noJob. Preserve old APPROVAL_BLOCK.json.
- Latest method question is not approval of that old resource request. No retry via anotherinterface,smallerresources,delegation or heartbeat.
- Continue independent hybrid development;prepare a concrete revised formal configuration and resource request after actual checks. Do not repeatedly ask the stale full-recollection question.
- Old3979 collectioninputs frozen SHA b87cfadaa2350892ad8eb34f70fa5165bf316cac9c13ddae88f8910a3b44c6a6 unchanged;old developer2updates/rescue/samePNG evidence remains separate.
- Old samePNG132-step equivalence PASS;DT28 historical action drift UNCONFIRMED. JPEG hash is not rawRGB proof.
- A800 priority;H100/4090 require actualreadiness checks,no blind crossNAS or cluster smoke.

## Completed references and reporting
- A3_61020/71943 SUCCEEDED:8430adapt/886589transitions/221392updates,then frozen4215;2209successes SR52.408066%(-10.106762pp vsA2).
- A3 STT1061,DT583,AT565 each/1405;finalSHA c7269fa14ba53ec60f4f27fa80aa9b295f83464ca5e0124597c30e0c10fce435.
- A3 initialnoise did not preserveA2;Q inflated andactor saturated;only finalcheckpoint fulltested. Preserve failedmethod and traces,do not resume.
- A2 full4215 STT1143,DT796,AT696;B60994 STT1169,DT787,AT728;B63.677343% highest measured amongA2/B/A3.
- Historical A2 action-only/59752 separate;A60316 fixed36mixed frozen. No60766/59352/DA3 restart.
- Update NAS RUN,twoledgers,these taskdocs on real evidence;doc-only commit/push,preserve dirtysource/CSV.
- Automationwla nowhybrid:advance every20min;notify milestones/failures. Duringevaluation report exact partialsuccess/completedSR and taskcoverage every20min.
