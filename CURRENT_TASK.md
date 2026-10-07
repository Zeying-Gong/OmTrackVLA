# WA current task — STT target and UWB study
Updated: 2026-10-07 Beijing. Goal ACTIVE; heartbeat wa ACTIVE every 20 minutes.
Full previous CURRENT_TASK: archive/2026-10/CURRENT_TASK_before_stt_goal_20261007.md.

## Acceptance
- On the same 1405 episodes/task: STT >=1289 (>91.7%), DT >=1173, AT >=1203.
- Current student: STT1276 (90.818505%), DT1173 (83.487544%), AT1203 (85.622776%).
- STT needs at least +13 net successes; DT/AT must not regress. Collision recorded, not primary optimization gate.
- After this gate, quantify with/without-UWB effects; distinguish input removal from no-UWB training.
- Explicit evaluation-set adaptation authorized. Never describe this as untouched-test generalization.

## Frozen baseline and evidence
- Training61377/72474 SUCCEEDED: independent59866 model+optimizer branch, one new epoch, cumulative2.
- Checkpoint: /data/nas_ray/project/md-ak/users/zeying.gong/job_61377/task_72474/wa_dual_teacher_train_a800_v1/checkpoint.pt
- SHA256 b5236a21f2d2695780029503c97e339c8350dc4f7337d05ec7e8692b9b9c327f; step59065.
- 61423/72526 STT,61424/72527 DT,61425/72528 AT SUCCEEDED; each8RTX4090,1405new rows.
- 24COMPLETE/4215unique/78sourcehashes previously verified; invalid0; full start/media audit still PENDING.
- Old60502 checkpoint and3627/4215 baseline unchanged. Student3652/4215; LightNav3345/4215, different inputs.
- Code/results backed up in GitHub wa20b94dc0; ledger wa/results/PROJECT_LEDGER_20261007.md.

## Work in progress — do not duplicate
- Full merge/start audit started 2026-10-07 on devpod-4090, PID3216342, local control session1145.
- Read-only audit plus new report directory; NOT new rollout, training or GPU job.
- Output planned: /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/student61377_full_audit_20261007_v1
- Last process check elapsed3m41 CPU31s, stateDsl (I/O waiting); no completion output yet.
- Next poll existing process/session/output before invoking merger again; run build_student_review only after merge succeeds.
- devpod-4090 intermittent SSH handshake closures; bounded spaced retries same entry. No host/credential workaround.
- Read-only parallel STT failure and training-exposure audits are underway; no new experiment submitted.

## Next bounded experiment gates
- First preserve full audit and diagnose STT student failures/teacher successes and old-baseline regressions.
- Add and test task/teacher/failure-group diagnostics and deterministic exposure accounting before choosing sampling changes.
- Do not blindly continue epoch3, change architecture/loss/physics/success thresholds, or invent task-ID inputs.
- Teacher demonstrations: successful branch first; if both succeed choose higher tracking rate, exact tie LightNav.
- Failed/fallback branches excluded from demonstration labels. Record actual old/new/task/episode exposure.
- Formal training8GPU; evaluation STT/DT/AT each8GPU,total24; verify realtime resources and AGENTS preflight.
- Freeze source/config/hash; short developer checks then complete scheduled jobs, no cluster smoke/duplicate job.
- Input remains RGB+initialGTBBox+idealpolarUWB,no text/laterGTboxes; JEPA training auxiliary only, no onlineMPC.
- Preserve originals and failed experiments. Update state/experiment records and GitHub; verified service links need SSH forwarding.
