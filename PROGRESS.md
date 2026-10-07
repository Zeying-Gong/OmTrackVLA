## 2026-10-07 09:39 China — user-authorized16GPU formal launch
- User explicitly approved and required16A800. Active package a4_hybrid_16gpu_20261007; two independent8GPU jobs61597/72791 +61598/72792 bothRUNNING,72h each; one shared learner/campaign, not two methods.
- Final frozen6557 inputs19,139,486,370bytes SHA1fb3481f1e1f5a0d974a6c862b32d0af310bbe41d69a63b6930b44267455c15b includes all old6498 unchanged. No WLA Git; new NAS evidence/RUN/CSV authoritative.
- attempt02 independentPASS20completeepisodes/1317savedfeature-latent-reward rows,4adapt285steps8onlineupdates+2offline;7checkpoint reloads; two-group drain/shutdownPASS. Developer used4physicalGPUs;16lane full4215 plan and capacity separateCPUchecks.
- Preserve attempt01 missingmanifest failure, independent all-PNG checker assumption failure, and initial frozen candidate omission; final manifest corrected before formal submission, no input model/source mutation.
- BothRUNNING and16uniqueGPU UUIDs/private-network/lane0–15 passed; eachworker6557new+6498old+protocol hashesPASS, CPU120quota/RAM960GiBlimit/startupoom0. Offline1024updates complete,1732cachedchunks; first252panel offline_panel_000128 started, noA4SR yet. Masteroutput job61597/task72791/wla_a4_hybrid_16gpu/run; peerjob61598/task72792; coord experiments/wla_a4_hybrid16_campaign_20261007_r01/control.
- Learning/selection budget unchanged:1024offline +8430onlineupdates max,6x252 candidatepanels and strictpromotion4215, up to9942rollouts. FormalfreshA2, TEST_SCENE_ADAPTATION_NOT_HELDOUT, no efficacy claim.
- Earlier4GPU approval-waiting notes are historical superseded records; no approval pending now. A2 best62.514828%; measuredA2/B/A3 highestB63.677343%.

# Progress
Updated: 2026-10-07 China
Historical state: archive/2026-10/PROGRESS_before_a4_20261006.md and archive/2026-10/a4_before_hybrid_reuse_20261006/.

## 2026-10-07 user authorized16GPU
User explicitly approves starting A4 and requests16A800, not4. New a4_hybrid_16gpu_20261007 preserves originalfrozenpackage and learningbudget. Two independent8GPUJobs will shareoneNAScoordinator/learner; sameJobmulti-root approach exposed SDKlegacyserialization and is retainedasUNSUPPORTED. ActualSDK singleJobconfigs passed.16laneCPUownership/networkserialhandler and two-nodebus checksPASS. Firstfull developerattempt failed beforefirstrollout due missingnewmanifest; preserved,originalassetcopied; secondbounded4GPU/two-node fullpipelinecheck running. NoformalJob yet,noA4effect. Priorapprovalwaiting descriptions below are historical and superseded; nofurtherpermissionrequest.

## A4 complete pipeline technically ready; explicit4A800/72h approval pending
New package a4_hybrid_replay_20261006 reuses A3 experience and performs fresh MC terminal-success learning plus A2-preserving online residual adaptation. Formal Job=null,formal optimizer updates0,no A4 full4215 performance result.
Frozen6498 inputs/19,118,576,195 bytes SHAef76b4a5a7b27df476ae3354480985fe75514351a9a2343b60d32b943fc60c14;4199 original expected hashes matched. Source/data/checkpoint bytes were actually reread.
Main review and evidence:FORMAL_REVIEW_REQUEST.md,DEVELOPMENT_READY.json,FROZEN_INPUTS_REPORT.json,PRE_SUBMISSION_READY.json. New production config preserves all tested learning/protocol values.
Attempt03 four-GPU developer pipeline completed20 whole episodes/1281 feature-action records. Every stage passed artifact and independent saved-feature CPU neural replay. It exercised offline checkpoint selection/fallback,one four-lane online cohort,drain/freeze,separate frozen servers and final-path protocol;developer final does not update BEST_FULL.
Offline2 candidate updates were discarded for the online start;online started from zero-residual A2,then4 complete episodes/267transitions/8updates. Actual sampled rows1258 old+790 new. Developer runs are independent,not cumulative and not SR evidence.
All267 actual online actions passed CUDA bitwise mean/epsilon/total and RNG-end checks;4 served policy snapshots retained. Mean is same-input deterministic recomputation;epsilon is a clone of the pre-act CUDA draw recipe,with exact runtime checks,not algebraically fabricated noise.
Developer FINAL404d521753f5ad15b34ce27b181e3d8018492fa17d827ed58c74f6ea600a2ba6. Actual FINAL model/two optimizers/complete replay queue/sampler/exploration/commits/snapshot catalog restored exactly with0 updates/interactions;nextbatch256 and copied-generator action exact.
Completed campaign restore reuses both audit types and creates0 new episodes/updates. Recovery discards failed partial-cohort updates and uses a new output rooted at the last drained checkpoint.
All1732 NPZ current hashes/prewarm PASS:7,362,235,056bytes within8GiB,53.55s,cache-process peakRSS7,946,944,512bytes. Whole four-lane summed child RSS peaked154,752,921,600bytes;shared pages may be counted repeatedly. Short memory evidence does not guarantee long-run safety.
All20 new resets retain exact rawRGB per source;A2 historical pairing uses firstJPEG hash only. CPU neural replay is savedfeature→latent,not fullRGB encoder/Flow. CPU/CUDA numerical errors explicitly bounded;267 runtime CUDA reconstructions are bitwise exact.
Initial attempt01 failed before rollout because Xvfb PATH was missing;logs/source retained. Existing NAS Xvfb bundle fixed the launch environment. Attempt02 v1 andattempt03 v2 preserved separately.

## Concrete formal campaign
Default A800 selected without cross-NAS movement. Canonical baidu_bj_a800 had47 free GPUs at04:56UTC;SDK2.0.0 login/help/queue checked. SDK-supported MD_AK_CONFIG_PATH overlay resolves the canonical name without global SDK/auth changes.
Proposed4A800/72h task viafull_a800.yaml→pipeline.sh;SDK pure-render CPUrequest32/limit60,RAM240/480GiB,shm150GiB. Actual Pod resources remain a post-submit verification.
Offline up to1024batch256 updates,snapshots128/512/1024 each fixed252(84/task);strict panel improvement over160/252 seeds online,else zeroA2. Online4215 once,three cohorts351/351/703 per task,two updates per complete episode;each cohort drains then freezes for252.
Best of six actual panel results strictly above160/252 gets one full4215. Upperbudget9454updates/9942rollouts;no efficiency or improvement guarantee. Equal/lower full results cannot replace A2 BEST_FULL.
Automatic review previously rejected the old8A800/48h full-recollection submission before execution due to unclear concrete resource authorization. The new4A800/72h hybrid package is now reviewable but not yet approved or submitted. No alternate-interface/name/size bypass;ask once for this concrete new task.
After explicit approval,refresh RUN/queue/hash/login/resources,submit exactly once through the normal SDK plus canonical overlay,record IDs and inspect scheduler/worker checks/Pod resources. No cluster smoke. During formal252/4215 eval,report actual partial successes/completed SR and84/1405 coverage every20min.
Automation remains active while approval is pending;no repeated reminders unless changed. No new goal was created.

## Scientific boundaries and preserved results
A3 archive8430episodes/886589transitions/1732NPZ audited.4760 successful attempts/2914 ever-successful source episodes include568 A2 failures,which is mixed historical coverage,not a single policy's score.8426 episodes lack fullRGB history;no invented image windows.
Fresh MC value/outcome and capped advantage-weighted actual-total-latent regression,task/episode-balanced sampling,soft A2 anchor,no Q-max/bootstrap. MC mixed-policy supervision is not unbiased new-policy advantage. Original A2 encoder/MetaQuery/Flow/heads/controller and projector remain frozen;no A3 actor/Q/optimizer restore.
A3 heads were FP32,A2/A4 heads BF16;Flow remains FP32. Nine developer probes found2 controller differences(max0.00437714),not a causal explanation of A3 decline. Preserve old outcomes/rewards;no same-transition-kernel or A4 counterfactual claims.
A3_61020/71943 full4215:2209success SR52.408066%,STT1061/DT583/AT565;TR70.368704 CR3.724792;SR-10.106762ppvsA2,-11.269276ppvsB. Onlyfinalcheckpoint evaluated;no optimal-checkpoint claim.
A2_60058:2635/4215=62.514828%,STT1143/DT796/AT696. B60994:2684/4215=63.677343%,STT1169/DT787/AT728,highest measured A2/B/A3;checkpointd4987b0aa961d971c30095c1b914224fff4b4dcc4b604655bb6113cd0f6f2883.
A4 BEST_FULL stillA2. TEST_SCENE_ADAPTATION_NOT_HELDOUT,not heldout generalization/global optimum. HistoricalA2 action-only59752 remains distinct;all old failures/STOPPED preserved. No60766/59352/DA3/B restart.
