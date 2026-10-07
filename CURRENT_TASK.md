# Current task
Updated: 2026-10-07 09:44 China
Status: A4_FORMAL_16GPU_RUNNING_OFFLINE_PANEL_000128

## Objective and authorization
User explicitly approved the prepared A4 formal run and required 16 GPUs: “早就批准了赶紧搞啊”; “16张gpu，而不是4张。搞快点”.
This supersedes all previous A4 resource-approval waiting records. No further permission is pending for this run.
A4 reuses A3 experience plus offline and direct online adaptation; raise true frozen full4215 SR.
TEST_SCENE_ADAPTATION_NOT_HELDOUT: user permits repeated learning/selection on TEST; no heldout-generalization or global-upper-bound claim.

## Active formal run
- A800 node0 Job61597/Task72791 RUNNING,8 GPUs; node1 Job61598/Task72792 RUNNING,8 GPUs; timeout72h each.
- Two independent normal shell/K8s Jobs rendezvous into one16-lane campaign/shared learner. Same-Job two-task layout is not used due SDK legacy serialization.
- Package nas-a800: WLA-EVT-20260925/a4_hybrid_16gpu_20261007; source SHA1fb3481f1e1f5a0d974a6c862b32d0af310bbe41d69a63b6930b44267455c15b,6557files/19139486370bytes.
- Old4GPU package and6498 frozen inputs SHAef76b4a5... remain intact. WLA has noGit; hash manifest is source fact.
- Configs full_a800_node0.yaml/full_a800_node1.yaml -> pipeline_16gpu.sh; canonicalbaidu_bj_a800, no crossNAS or cluster smoke.
- Shared coord /data/nas_ray/home/zeying.gong/algorithm/experiments/wla_a4_hybrid16_campaign_20261007_r01/control.
- Master output /data/nas_ray/project/md-ak/users/zeying.gong/job_61597/task_72791/wla_a4_hybrid_16gpu/run.
- Peer output /data/nas_ray/project/md-ak/users/zeying.gong/job_61598/task_72792/wla_a4_hybrid_16gpu.
- BothRUNNING; actual16 uniqueA800 UUIDs, lane0–15 and differentPods, bidirectional private-IP peer checks PASS. Eachworker6557new+6498old+protocol hashes PASS; CPUquota120cores, RAMlimit960GiB, startupoom0.
- Offline1024actualupdates complete;1732 archivechunks prewarmed with0evictions; checkpoints128/512/1024 saved. First252panel offline_panel_000128 started; no frozenA4 SR yet.
- Evidence newpackage RUNTIME_ALLOCATION_AUDIT.json and OFFLINE_STAGE_AUDIT.json. Offline progress.json initial0 is stale; COMPLETE/updates.jsonl authoritative.

## Verified preparation
- New isolated distributed developer attempt02:20 fullrollouts,1317 actual feature/latent/reward records;4adapt285transitions8onlineupdates plus2offline.
- Independent artifact/neural audit PASS;7 distinct checkpoints reload with new core, freeze/drain/idempotency and bothnode clean exit.
- Four physical developer GPUs in two logical groups, not16 physical or crossPod proof;16-lane production ownership/4215 indices, HTTP serialization/capacity/drain separately PASS.
- Attempt01 missing manifest failed before rollout, preserved. Independent checker all-PNG assumption and initial freeze omission of historical A3/source manifest preserved; final6557 includes complete old6498.
- All runtime source unchanged during developer. Formal starts pristineA2/freshactor, not developer weights.

## Method and acceptance
- FrozenA2 encoder/MetaQuery/Flow/heads/controller and projector; fresh actor/value, terminal-success MC plus capped weighted latent regression and softA2 residual. No restoredA3 actor/Q/optimizer.
- Reuse8430episodes/886589transitions/1732NPZ; oldA3 FP32heads vs newA4 A2 BF16heads is explicit execution-domain difference.
- Offline1024updates snapshots128/512/1024 each252panel(84/task); online4215 in3cohorts,2updates/completeepisode,eachcohort drained then252.
- Best of6panels strictly aboveA2 160/252 promoted to one frozen4215; otherwise retainA2. Max9454updates/9942rollouts.16 lanes changes sampling interleave.
- Full acceptance:1405/task exactlyonce, identity/instruction/seed7/initialimage evidence/full naturalhorizon/finite/frozencheckpoint and reward/latent savedfeature CPUreplay audits.
- BEST_FULL staysA2 2635/4215=62.514828%; comparable A2/B/A3 highestB60994 2684/4215=63.677343%; A3 final2209/4215=52.408066%.
- No A4 SR yet. Only a strictly better audited full4215 result updatesbest; online/panel/oldexperience results are not fullSR.
- Every20min during frozen252/4215 eval report STT/DT/AT success/completed and partialSR plus84/1405 coverage.
- Do not resume60766/59352/DA3, repeat completedB jobs, modify frozen inputs, or stop unrelated jobs. HistoricalA2 action-only/59752 distinct.
