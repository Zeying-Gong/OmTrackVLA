# WA current task — STT target and UWB study
Updated: 2026-10-08 after explicit group-fit permission. Phase RUNNING_CPU_SELECTION, not formal training admission.
Existing Goal tool status remains BLOCKED (no resume API); work continues under the user's new request. Heartbeat wa remains ACTIVE every20minutes; not complete or user-paused.
Full predecessor: archive/2026-10/CURRENT_TASK_before_group_fit_permission_20261008_1656.md.
Paths: R=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928; C=R/checkout; J=/data/nas_ray/project/md-ak/users/zeying.gong. Resolve R/C/J prefixes literally below.

## Acceptance and retained best

- Same1405/task: STT>=1289 (>91.7%), DT>=1173, AT>=1203; then same-weight image/noUWB paired study. Best61609=1279/1178/1207 (3664/4215),invalid0;STT gap10,NOT_MET.
- 61715=1277/1166/1207 (3650/4215),invalid0,not promoted. Full results/failures retained;do not repeat completed4215 evaluations.
- LightNav=1273/1128/944;Oracle=1281/1210/1226. Different-input systems;explicit evaluation-set adaptation,not unseen-test/product generalization.
- Preserve JEPA/MetaQuery/ActionExpert,original loss/LR/controller/physics/criteria. RGB+episode0 GTBBox template+current ideal polarUWB;no text,laterGTBBox,futureGTpaths orRL;JEPA auxiliary during training.

## Current operation and next gates

- User explicitly approved the two group-fit files. C/wa/tools/failure_state_group_fit.py SHA d4bf544e837149b16a31973f546e5d86a7b3050c61649aef919f36127166e404;C/wa/tests/test_failure_state_group_fit.py SHA 37d865e0683b6478c1908f35c7d66d52204b77841688ca461413ccd8aa00c2c7.
- Initial CPU30tests had28PASS/2fixture-index failures;two indices corrected before rerun30PASS1.164s. Not a GPU/model-fit result.
- CPU select PID3966836/session43283 ondevpod-4090 still active at18:15Beijing (~10min/4.855GB NAS reads);output R/artifacts/failure_state_group_fit_selection_20261008_v1.json. No final selection orgroup-fit artifact accepted yet;no newGPU orformalJob/checkpoint/SR.
- Selection uses R/artifacts/teacher_group_fit_61377_20261007_v1.json plus the immutable v2cache/loader/dedup below;do not reselect based on fitting results.
- Next: finish and audit fixed selection;bounded group-fit diagnostic under current authorization;review evidence plus target-runtime gate before one complete8GPU training task. No automatic hyperparameter sweep orblind epoch extension.
- Formal recipe remains independent59866 MODEL+OPTIMIZER step22707,+1new/cumulative2epochs;never61609/61715 epoch3. Then full STT/DT/AT each8GPU/1405,24total if needed,followed by all3SR gates before UWB study.
- Resources17:54-56Beijing:A80088total85used3free;H100 SSH works but separateAliNAS lacks checkout/env/base/old/newcaches;bj4090 144total68used76free. Snapshots not allocations;recheck priority A800→H100-ready→4090;user permits24bj4090,no migration performed.
- Developer4090 at18:15:GPU7 free7490MiB/util0;GPU6 util100%/free9854MiB. No idle device met the4update training safety margin;clusterfree76 is not developerfree. A short GPU7 fit with5GiB cap is only planned,not run.
- Frozen training source R/source_failure_state_train_v1 commit199385cd9c826c8f21308ad99b6a8375396d9c90;241WA hashes unchanged. New fit tooling is outside this freeze;never edit frozen runtime.
- Eight-GPU draft C/wa/jobs/failure_state_train_4090_v1.yaml SHA b01bc53bb51cd0ccc7d539701aed44714285c0fa1940e829a43a416ab7749ff2 remains NOT_SUBMIT_READY. Intended output J/job_<JobID>/task_<TaskID>/wa_failure_state_train_4090_v1;IDs not assigned.
- Draft syntax/11flags/16paths/staticDDP passed;actual4090 recipe/kernel/NCCL/worker placement and fixedfit gates pending. Worker checks must prove8distinctRTX4090 plus exact exposure/3x73368 finiteheldout/1543logs/finalcheckpoint;not executed claims.
- Prior A800 developer check R/artifacts/failure_state_train_developer_20261008_v1 exited0:4updates22708..22711,finite loss/gradients,peak7.3089757GiB;16positions=7base+6oldteacher+3new,3modes2heldout each. Not formal8rank training orSR;do not rerun without a new reason.

## Immutable data and exposure anchors — completed gates, do not redo

- Collection61844/73066 SUCCEEDED2026-10-08 11:24:43 Beijing,8A800;output J/job_61844/task_73066/wa_failure_state_collect_a800_v3.90new+36frozen=126keys,96originalwinners/30none,not studentSR;old61833/61836 failures/partials preserved.
- Collection source R/source_failure_state_collect_v3 commit726544449a5e7ad10c7d246b2bf302d783210c7e;overlay R/artifacts/failure_state_continuation_61836_v1.json SHA e5fe2986517c171fd93f507ebe0f6d9937c420d3c37efd3fe7ff391899444641.
- R/artifacts/failure_state_collection_release_126_v1.json SHA73fe7d6b7cdcdfa144a5bcf540cbac33b0d2c0971b25d6aeca35e928d8af0fbd;94807sourcefiles,7396candidate→6864valid/532excluded,96episodes/90nonempty;training_released=false.
- C/wa/results/FAILURE_STATE_DEDUP_20261008.json SHA599d60ea2d40ff4f9ae69f87df03e3a5a192b749ce030504bf19177999262859;526same-key exactduplicates (182start+344mid),6338new-within-scope;not cross-key/within-new dedup. Keep all6864valid cache rows;duplicate extra exposure0.
- Cache R/artifacts/failure_state_se2_cache_20261008_v2 admissionSHA c690957761133f98e8925bf0f491373c9b0ca30d74f427834d9d4a155093bc69;completeSHA b96b90bca2a88413b4f618a6e5c280b8e262a3e2713fc9f63524402f4d77e6f9. Failedv1/failed_conversion.json retained;no overwrite.
- C/wa/results/FAILURE_STATE_CACHE_LOADER_20261008.json SHAa330f7d39ecd89f7f1f67e33f0a2d6cdb67bf62f7dd57758ea270ec6a0644755;all6864metadata,2127real10tensoritems/2039policy+231JEPA+308proprio crossings,90routing checks,15034sourcehashes;6heldoutlinks unchanged.
- Candidate R/artifacts/failure_state_sampling_candidate_20261008_v1:admissionSHA230079f1e99836dc7b3bf20942859e127e0b42dfa0b64c76b2c6ab67dd373902;canonicalplan11cb7150e33c8b66cbd3353a7f95b903c8b8b282ebe84faee4aa91e4b0beeaee;training_released=false.
- Real8rank CPU sampler+loader passed1233424positions exactlyonce/0drop:726631base+457641oldteacher per-window exact+49152recovery.6338windows/87episodes;window1..16,episode48..759;early<=1s23598/late25554,LN15932/Oracle33220. Same counts,not identical ordering/RNG/optimizer trajectory.
- Planned38545new updates (38544effective32+final16),end61252;not actualtraining. Preserve old executed1184272positions,exclude1151263 exactly. Old plan R/artifacts/hard_stt_candidate_61377_20261007_v1/plan.json SHAc5533396f8a454bf8dd7281ff73d2b2d2737598b7a2f7e08b97260897e296d0c.
- Old actual J/job_61609/task_72803/wa_hard_stt_train_a800_v1/actual_exposure_epoch1.npz SHA65bde6ac0f43835e09df12df413e935ba7f9538612c5c7738eb51631e7f8ba62;trusted R/artifacts/hard_stt_training_audit_61609_v1.json SHA59d5b468eec4283db1a1d87ad7b9a9d1bc2123739cffc38cebfc9a1e14dcc727.
- Base /data/nas_ray/home/zeying.gong/datasets/wla_evt_se2_cache_20260925_v2 completeSHAeb5a52d3478bab1391d12a990dcdc57ce2c77cef787109d29f7f286be797f901;index R/artifacts/robot_transition_audit_v2/audit.json SHA3138364dea80543fc83be476e1af8191a4eca3711a9b353591311fc8daf5c2d3.
- Oldteacher R/artifacts/dual_teacher_se2_cache_20261006_v1 completeSHA79e5a7dc43b61a23bb64d5676c4c1cac17b459a5e390b308377012e582ba1a8d;436816valid. No old dataset,heldout,loss,architecture orsampler reinterpretation.
- Keep episode0 template/allframes/originalindices;studentprefix only causal history,never action labels;teacher-owned0.1..0.7s SE2/JEPA futures,original now>=4/dt/displacement/yaw/badtransition masks. Failed/repeat/typed-error windows never demonstrations.
- Do not repeat collection/full126raw/release/dedup/loader gates by default,overwrite immutable cache/plan/checkpoints,or apply old local startup31/RUNNING patch. Full proof paths and past corrections remain in PROGRESS/archive.

## Checkpoints, reporting and services

- Parent J/job_59866/task_70728/wa_history_repeat_v1/checkpoint.pt SHAab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d;step22707.
- Best J/job_61609/task_72803/wa_hard_stt_train_a800_v1/checkpoint.pt SHAc510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52;step59716. Full report C/wa/results/STUDENT61609_FINAL_20261007.json.
- Rejected61715 J/job_61715/task_72909/wa_stt_anchor_train_a800_v1/checkpoint.pt SHA9631778c81992a349f8315b773f763217a5f387b5206b1bfb97b323839030a6a;report C/wa/results/STUDENT61715_FINAL_20261008.json;not promoted.
- Mixedzero sampling4 seed7+step;TR reference-normalized/macroTR separate;CR target-human distance ever<0.5m,not obstacle contact.
- Prior review18799/18800(devpod-4090),TB6006(devpod-a800),not revalidated now. Historical forwards:ssh -N -L 18799:127.0.0.1:18799 -L 18800:127.0.0.1:18800 devpod-4090;ssh -N -L 16006:127.0.0.1:6006 devpod-a800.
- Devpod-4090 hostname devpod-zeying-gong-4090-7f59ffb47c-rpkjh,GPFS mt-tutz0R-pfs-haZOBP;priorNo-route/kex failures preserved,no crossNAScopy. Current endpoint works;new permission resolved,not an ongoing approval blocker.
- GitHubwa5380b32e867d88bb42441e684de0a896ba8c4e53 is prior verified backup;new fitfiles/state pending explicit stage/push. Two raw JSON reports intentionally NAS-only/untracked;no weights/data/videos. Archive before growth;CURRENT<=80,PROGRESS<=150.
