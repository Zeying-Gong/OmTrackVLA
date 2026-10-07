# Current task
Updated: 2026-10-07 09:58 China
Status: A4_LANEFIX_FORMAL_16GPU_RUNNING_OFFLINE_PANEL_000128

## Objective and authorization
User explicitly approved the prepared A4 formal run and required 16 GPUs: “早就批准了赶紧搞啊”; “16张gpu，而不是4张。搞快点”.
This supersedes all previous A4 resource-approval waiting records. No further permission is pending for this run.
A4 reuses A3 experience plus offline and direct online adaptation; raise true frozen full4215 SR.
TEST_SCENE_ADAPTATION_NOT_HELDOUT: user permits repeated learning/selection on TEST; no heldout-generalization or global-upper-bound claim.

## Active formal run
- A800 node0 Job61599/Task72793 RUNNING,8 GPUs; node1 Job61600/Task72794 RUNNING,8 GPUs; timeout72h each.
- Two independent normal shell/K8s Jobs rendezvous into one16-lane campaign/shared learner. Same-Job two-task layout is not used due SDK legacy serialization.
- Package nas-a800: WLA-EVT-20260925/a4_hybrid_16gpu_lanefix_20261007; source SHAbee40d997a860b5d276d6e91da42b4be36bb1234be3062fc80ee31ecad0ca080,6610files/19165834447bytes.
- Old4GPU package and6498 frozen inputs SHAef76b4a5... remain intact. WLA has noGit; hash manifest is source fact.
- Configs full_a800_node0.yaml/full_a800_node1.yaml -> pipeline_16gpu.sh; canonicalbaidu_bj_a800, no crossNAS or cluster smoke.
- Shared coord /data/nas_ray/home/zeying.gong/algorithm/experiments/wla_a4_hybrid16_campaign_20261007_r02/control.
- Master output /data/nas_ray/project/md-ak/users/zeying.gong/job_61599/task_72793/wla_a4_hybrid_16gpu_lanefix/run.
- Peer output /data/nas_ray/project/md-ak/users/zeying.gong/job_61600/task_72794/wla_a4_hybrid_16gpu_lanefix.
- NewtwoJobs RUNNING;16uniqueA800UUID/privatepeer/lane0–15/cgroup actualPASS, eachworker6610new+6498parent+210protocolchecksPASS. All16server_ready and16workerlogs exist, actualHabitat task initialization underway. Previous61597/61598 verified16physicalGPU/privatepeer/cgroup, then naturally FAILED09:44 due artifact validator residual8lane capacity;0episodes. Allfailure/source/output preserved.
- Reuse completed1024offlineupdates and exact128/512/1024checkpoints from61597; newofflineupdates0. Strict offline_reuse guards config/learnerconfig/normalized source semantics/SHA and zero completed rollout stages; no retraining or partialcohort adoption.
- NewDEVELOPMENT_READY PASS:5real production16lane assignments through actualworker/artifact validator,13invalid cases and5GTrequests rejected,checkpoint output exact after newcoreload. Only capacity8to16+validatedoffline reuse changed. Newrun/OFFLINE_REUSE_AUDIT PASS, no duplicateofflineupdates;16worker actualsimulator initialization passed; wait firstcompletedepisodes for partialSR.

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
