# WA current task — STT target and UWB study
Updated: 2026-10-07 Beijing. Goal ACTIVE; heartbeat wa ACTIVE every 20 minutes.
Full previous CURRENT_TASK: archive/2026-10/CURRENT_TASK_before_stt_goal_20261007.md.

## Acceptance
- On the same 1405 episodes/task: STT >=1289 (>91.7%), DT >=1173, AT >=1203.
- Current student: STT1276 (90.818505%), DT1173 (83.487544%), AT1203 (85.622776%).
- STT needs at least +13 net successes; DT/AT must not regress. Collision recorded, not primary optimization gate.
- After this gate, quantify with/without-UWB effects; distinguish input removal from no-UWB training.

## Active 24GPU evaluation and completed training — do not duplicate
- 61609/72803 SUCCEEDED2026-10-07 15:50:25Beijing; actual8 A800-SXM4-80GB; finalstep59716/new37009updates/cumulative2epochs.
- Output:/data/nas_ray/project/md-ak/users/zeying.gong/job_61609/task_72803/wa_hard_stt_train_a800_v1
- Frozen source_hard_stt_train_v1 commit8d8efe3aa8a7ce9714913b6f65e5e3c8196eae06;clean;159Pythonfiles match developer diagnostic.
- Config wa/jobs/hard_stt_train_a800_v1.yaml SHA99f8fc7491b4f7c4c347c73b0ba4f1bf363ff292c76de0d4b29fd9850df19184;GitHubfda647a7.
- Parent59866 model+optimizer SHAab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d;independent1new epoch cumulative2;not61377epoch3.
- Final checkpoint.pt SHAc510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52. Full audit artifacts/hard_stt_training_audit_61609_v1.json SHA59d5b468eec4283db1a1d87ad7b9a9d1bc2123739cffc38cebfc9a1e14dcc727 PASS24198sourcehashes/optimizer/step/actualexposure/heldout;OFFLINE_ONLY,no newSR.
- Monitoring correction: earlier inline watcher used nonexistent run/console.log andsilently treatedmissing as empty;its nofatal claim iswithdrawn (step records remainvalid). Actual logs are task_72803/console.log andtask_72803/.md-ak/workload.log;fullscan15:24 andcorrectedwatch15:28 found0specifiedfatal signatures. Missingrequiredlog nowraises,notempty. Terminaltrainingaudit15:50+PASS;preservethiscorrection.
- Actual1184272exposures=726631base+457641teacher;10413hardwindows each3=31239/earlyhard9810;teacherSTT148547DT156048AT153046;8ranks148034each. Heldout73368/mode:ADE/FDE image.265734/.463479 point.253940/.442441 mixed.253890/.441945;notclosedloop.
- TensorBoard hard_stt_61609 is completed-training curves, not runningtraining; existing6006/exporter. ssh -N -L 16006:127.0.0.1:6006 devpod-a800 then http://127.0.0.1:16006/#scalars .
- 24GPU fullmixed preflight PASS:98CPUtestexecutions;actualcheckpoint shortRTX4090interface;24realHabitatdefinitionlanes1405/task,72hashes. Syntheticinterface is not simulatorreset/rollout/firstbenchmarkframe evidence.
- Active STT61653/72847,DT61654/72848,AT61655/72849 submitted2026-10-07 16:13Beijing;each8A800/1405,total4215,all24ready actualA800verified. Snapshot16:53:44 validcompleted559/460/481=1500unique;invalid0;all24shardsnotCOMPLETE. No finalSR.
- Frozen source_student61609_eval24_v1 commit192b57f5e270acfffd8c7c1a4590cb1b257d92a3 clean;configs wa/jobs/student61609_{stt,dt,at}_a800_v1.yaml backedupb27a39b9. Output job_<ID>/task_<ID>/wa_student61609_<task>_a800_v1. Noresume/reuse.
- Preflight wa/results/STUDENT61609_PREFLIGHT_20261007.json;startup STUDENT61609_STARTUP_20261007.json. Latestwatch STUDENT61609_WATCH_20261007T0843Z.json:13snapshots16:43..16:53 grew1089->1500(+411),all24lanesprogressed;actualtasklogs/no4specifiedfatals,alljobsRUNNING;warningsretained. Prior0823Zwatch preserved.
- Bounded24startpair PASS144teacherfiles,report STUDENT61609_STARTPAIR_BOUNDED24_20261007.json SHA34d3f00d2271d1988e744657ee6ff2f7586c888528da96013dcfaa164391b785. Fixedfirstrow/task/lane inclfailures;not4215audit. A800vs61377RTX4090notbitwiseequivalence.
- Bounded24media PASS:STUDENT61609_MEDIA_BOUNDED24_20261007.json SHA60006fe85ceffc83d2d68a6a28802f6420d988b0a53697b2022c3bd63cd9f6e1;fixedfirstrow/task/lane;frames=steps=total_step,duration=frames/20;no oldroot links. Not4215/everyframe/HTMLaudit.
- Explicit evaluation-set adaptation authorized. Never describe this as untouched-test generalization.

## Frozen baseline and evidence
- Training61377/72474 SUCCEEDED: independent59866 model+optimizer branch, one new epoch, cumulative2.
- Checkpoint: /data/nas_ray/project/md-ak/users/zeying.gong/job_61377/task_72474/wa_dual_teacher_train_a800_v1/checkpoint.pt
- SHA256 b5236a21f2d2695780029503c97e339c8350dc4f7337d05ec7e8692b9b9c327f; step59065.
- 61423/72526 STT,61424/72527 DT,61425/72528 AT SUCCEEDED; each8RTX4090,1405new rows.
- 24COMPLETE/4215unique/78sourcehashes verified; invalid0; full4215 initial pair audit PASS; all4215 firstJPEG hashes/video metadata PASS (not every-frame decode).
- Old60502 checkpoint and3627/4215 baseline unchanged. Student3652/4215; LightNav3345/4215, different inputs.
- Code/results backed up in GitHub wa20b94dc0; ledger wa/results/PROJECT_LEDGER_20261007.md.

## Latest completed work and active audit — do not duplicate
- Full merge/start audit session1145 exited0:4215 paired against both teachers; statusPASS.
- Merged: /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/student61377_full_audit_20261007_v1
- Summary SHA30dfdbd9263ca0ba3615cb1845938de235c7e1b15b2141edb5ac584622f6bbf3; pair audit SHAdc98149bdcfb6baf05d1fae167de572b476679f48384dd91ea92e57851645844.
- Media builder session20033 exited0 REVIEW_AUDITED4215: artifacts/student61377_review_20261007_v1.
- Review audit SHA7d2000efd58135753c9a4b5a8fe5bd3c228593ad2427f6278fc66841a293261d; 127.0.0.1:18798 HTML/audit/firstvideo HEAD200 verified.
- Access: ssh -N -L 18798:127.0.0.1:18798 devpod-4090 then http://127.0.0.1:18798/ ;old18797 unchanged.
- STT start coverage v2 PASS:1358teacher episodes;93student-failure demonstrations,91nonzero/10413valid windows.
- Of91nonzero failures,90have teacher windows within actual first2s;3270early windows. Only XNeHsjL6nBB/9 lacks early windows.
- Two more hard cases BHXhpBwSMLh/11 and XNeHsjL6nBB/4 have zero valid windows; filters unchanged.
- Coverage artifacts/stt_start_coverage_61377_20261007_v2.json SHA8559e02a411358ba6dd59d7f2cb7da968d81e1542398b08dc09271d6d62ae3ec; v1 preserved.
- Opt-in plan/runtime accounting integrated into train.py;40CPUtestsPASS plus real4update diagnosticPASS;originalmodel/loss unchanged.
- Candidate91hard/10413windows each3;allbase+teacher once. Exact8rank1184272actualsimulated;reportSHA663c66b1. Formal61609completed/fulltrainingauditPASS;closedloop pending.
- STT read-only failure analysis:18gains/18regressions;129failures=61targetCollision/49Lost/19Normal-no-success;93teacher-solvable.
- 51/61collisions ended<=40steps;58/61last5commands alreadybackward. Not evidence of general yaw saturation or wall collisions.
- SSH same-entry handshake intermittent; standard ControlMaster connection reuse enabled per-command only, no SSH config/credential change.

## Next bounded experiment gates
- Full pair/media and88window teacher-fit COMPLETE;hardCollisionADE.55880->.48717;successfulSTT.27892->.25227;label-fit notSR.
- RealRTX4090 diagnostic4updates22708..22711 consumed16samples10base6teacher3hard;peak7.309GiB;no checkpoint.
- Evidence:wa/results/HARD_STT_CANDIDATE_20261007.json. Trainingandactual_exposure_epoch1.json/npz finalauditPASS;do notduplicate training.
- Final tools: wa.tools.audit_hard_stt_training checks completed metadata/exact actual exposure/source+YAML pins; completed61609auditPASS;live-checkpoint guard previouslyPASS. Scheduler terminal status checked separately.
- wa.tools.audit_student_goal checks full4215 new rows and paired61377 baseline, fixed1289/1173/1203 thresholds and gains/regressions. 27 new audit tests PASS (plus6 existing runtime tests); not a new model result. Boolean false init/policy failures remain in denominator; regression fixture PASS.
- Goal-auditor real61377 self-check:81sourcehashes PASS,4215rows,NOT_MET with13/0/0shortfall whileLightNav superiority=True. Read-only/no new result artifact;61609 closedloop still pending.
- FullSTT/DT/AT24GPU closedloop nowrunning;monitor existing61653/61654/61655,do notduplicate. New61609SRunknown;actualtrainingperwindow exposure validated.
- Do not blindly continue epoch3, change architecture/loss/physics/success thresholds, or invent task-ID inputs.
- Teacher demonstrations: successful branch first; if both succeed choose higher tracking rate, exact tie LightNav.
- Failed/fallback branches excluded from demonstration labels. Record actual old/new/task/episode exposure.
- Formal training8GPU; evaluation STT/DT/AT each8GPU,total24; verify realtime resources and AGENTS preflight.
- Freeze source/config/hash; short developer checks then complete scheduled jobs, no cluster smoke/duplicate job.
- Input remains RGB+initialGTBBox+idealpolarUWB,no text/laterGTboxes; JEPA training auxiliary only, no onlineMPC.
- UWB interface audit remains historical. Explicit WA_EVAL_MODE=image now connected server/agent/ready/rows/markers/merge/review, mixed default unchanged.98CPUtestsPASS;real61377 mixed4215/78hashes metrics unchanged. Evidence wa/results/UWB_EVAL_MODE_PREPARATION_20261007.json; no actualimage trajectory/loaded-weight ablation.
- New fullmixed configs must explicitly set WA_EVAL_MODE=mixed. Formal UWB only after SR gate; final-weight image interface/observer perturbation/realHabitat checks and full paired4215 still pending. Never replace this with mixed zero-coordinates or no-UWB-training claims.
- Preserve originals and failed experiments. Update state/experiment records and GitHub; verified service links need SSH forwarding.
