# WA current task — STT target and UWB study
Updated: 2026-10-07 Beijing. Goal ACTIVE; heartbeat wa ACTIVE every 20 minutes.
Full previous CURRENT_TASK: archive/2026-10/CURRENT_TASK_before_stt_goal_20261007.md.

## Acceptance
- On the same 1405 episodes/task: STT >=1289 (>91.7%), DT >=1173, AT >=1203.
- Current student: STT1276 (90.818505%), DT1173 (83.487544%), AT1203 (85.622776%).
- STT needs at least +13 net successes; DT/AT must not regress. Collision recorded, not primary optimization gate.
- After this gate, quantify with/without-UWB effects; distinguish input removal from no-UWB training.

## Active formal run — do not duplicate
- 61609/72803 submitted2026-10-07 10:35:58Beijing; schedulerRUNNING; worker verified8 A800-SXM4-80GB.
- Output:/data/nas_ray/project/md-ak/users/zeying.gong/job_61609/task_72803/wa_hard_stt_train_a800_v1
- Frozen source_hard_stt_train_v1 commit8d8efe3aa8a7ce9714913b6f65e5e3c8196eae06;clean;159Pythonfiles match developer diagnostic.
- Config wa/jobs/hard_stt_train_a800_v1.yaml SHA99f8fc7491b4f7c4c347c73b0ba4f1bf363ff292c76de0d4b29fd9850df19184;GitHubfda647a7.
- Parent59866 model+optimizer SHAab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d;new1epoch cumulative2;planned37009updates/end59716.
- Firstworker read:dataset/8rank initialization andmodel loading;no verifiednewstep yet. xFormers/NCCLdevice warnings retained;no fatal signature in checkedlog.
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
- Candidate91hard/10413windows each3;allbase+teacher once. Exact8rank1184272actualsimulated;reportSHA663c66b1. Formal61609submitted;not complete.
- STT read-only failure analysis:18gains/18regressions;129failures=61targetCollision/49Lost/19Normal-no-success;93teacher-solvable.
- 51/61collisions ended<=40steps;58/61last5commands alreadybackward. Not evidence of general yaw saturation or wall collisions.
- SSH same-entry handshake intermittent; standard ControlMaster connection reuse enabled per-command only, no SSH config/credential change.

## Next bounded experiment gates
- Full pair/media and88window teacher-fit COMPLETE;hardCollisionADE.55880->.48717;successfulSTT.27892->.25227;label-fit notSR.
- RealRTX4090 diagnostic4updates22708..22711 consumed16samples10base6teacher3hard;peak7.309GiB;no checkpoint.
- Evidence:wa/results/HARD_STT_CANDIDATE_20261007.json. Monitor61609 steps/checkpoint andfinal actual_exposure_epoch1.json/npz;do notduplicate.
- FullSTT/DT/AT24GPU closedloop required before claiming gain;actualperwindow exposure recorded and validated after epoch.
- Do not blindly continue epoch3, change architecture/loss/physics/success thresholds, or invent task-ID inputs.
- Teacher demonstrations: successful branch first; if both succeed choose higher tracking rate, exact tie LightNav.
- Failed/fallback branches excluded from demonstration labels. Record actual old/new/task/episode exposure.
- Formal training8GPU; evaluation STT/DT/AT each8GPU,total24; verify realtime resources and AGENTS preflight.
- Freeze source/config/hash; short developer checks then complete scheduled jobs, no cluster smoke/duplicate job.
- Input remains RGB+initialGTBBox+idealpolarUWB,no text/laterGTboxes; JEPA training auxiliary only, no onlineMPC.
- Preserve originals and failed experiments. Update state/experiment records and GitHub; verified service links need SSH forwarding.
