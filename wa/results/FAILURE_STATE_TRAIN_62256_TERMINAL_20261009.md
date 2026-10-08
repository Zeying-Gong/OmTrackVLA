# WA 62256 training completion and fixed sample comparison

Job62256/Task73511 completed on 2026-10-09 at 06:06:07 Beijing. Independent terminal auditing accepted its final checkpoint as an offline candidate. The fixed152 teacher-state comparison also completed. Neither result establishes closed-loop SR, and best61609 remains the retained best until all three complete1405 evaluations are audited.

The experiment is explicitly evaluation-set adaptation. Model architecture, physics, controller, loss, learning-rate method and success criteria were not changed. The next thresholds remain STT>=1289, DT>=1173 and AT>=1203 successes per1405, followed by a same-weight UWB ablation only after all three thresholds pass.

## Training identity and checkpoint

Paths below use `R=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928`, `C=R/checkout`, and `J=/data/nas_ray/project/md-ak/users/zeying.gong`.

| Item | Verified value |
| --- | --- |
| Scheduler | Job62256 and Task73511 SUCCEEDED, updated2026-10-09 06:06:07 Beijing |
| Hardware | 8 distinct A800-SXM4-80GB UUIDs, compute capability8.0 |
| Frozen training source | `R/source_failure_state_train_v1`, commit `199385cd9c826c8f21308ad99b6a8375396d9c90` |
| Job configuration | `C/wa/jobs/failure_state_train_a800_v1.yaml` |
| Configuration SHA256 | `08553eb73e8d2db4ae2fd37ab1c322bc7fbf77ac4a3ba3a7c59ae2508a2f2e69` |
| Output | `J/job_62256/task_73511/wa_failure_state_train_a800_v1` |
| Final checkpoint | `J/job_62256/task_73511/wa_failure_state_train_a800_v1/checkpoint.pt` |
| Checkpoint bytes and SHA256 | 4358277705; `40915b366ee5a2ef5967e2ce49a94d2b0f45f149955dd5cccf85ac3d6ab178fc` |
| Parent | `J/job_59866/task_70728/wa_history_repeat_v1/checkpoint.pt`, step22707 |
| Parent SHA256 | `ab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d` |
| Final step | 61252, after38545 additional optimizer updates; cumulative2epochs |

The model AND optimizer resumed from59866, not from61609 or61715. All480 full-continuation optimizer states reached61252. Of494 model state tensors,480 changed; conditional-parent and new optimizer-state counts were0. The recipe used world-size8, per-rank batch2, accumulation2 and seed42. There were38544 full effective-batch32 updates and one final effective-batch16 update. This is one new epoch, not a third epoch appended to best61609.

Actual exposure was1233424 positions exactly once:726631 base,457641 old-teacher and49152 new recovery. No position was unconsumed or repeated. Per-window old-teacher counts matched the prior actual exposure, including its omitted position1151263. Recovery exposure covered6338 nonduplicate windows across87 episodes;526 exact duplicate windows received zero additional exposure. Early/late counts were23598/25554 and LightNav/Oracle counts15932/33220. The per-rank154178 positions and77089 microbatches are reconstructed assignments, not independently recorded rank-local telemetry.

The terminal training log contains1543 finite scheduled rows, from22708 to61250. Final61252 is confirmed by the checkpoint and optimizer rather than an invented scheduled log row. Five fatal-pattern searches found0 hits. Retained warnings include24 xFormers UserWarning entries,16 device-id UserWarnings and8 NCCL startup mapping warnings. The48 xFormers literal occurrences include echoed warning source lines.

## Full heldout results

Each mode evaluated73368 windows using the same evaluate implementation and heldout inputs as61609. Values below are trajectory errors in metres, not closed-loop SR. All six listed errors are slightly worse than61609; this does not by itself determine closed-loop behaviour or justify another epoch.

| Mode | 61609 ADE | 62256 ADE | 61609 FDE | 62256 FDE |
| --- | ---: | ---: | ---: | ---: |
| image | 0.265734130 | 0.266651275 | 0.463478806 | 0.464677133 |
| point | 0.253939639 | 0.256589632 | 0.442441453 | 0.446753661 |
| mixed | 0.253890195 | 0.256096974 | 0.441944517 | 0.445404009 |

Candidate yaw errors are0.125396401,0.114885575 and0.114834528 respectively. The full metrics remain finite with `closed_loop=false` and unavailable SR. The evaluate function text/AST SHA is `94241e06b19839be44bc32143f49b0a2f1b8617f96f21d5568d1d8aa584a8602`. The metrics elapsed time19115.204254s includes the complete training/validation workflow, not pure GPU training time.

## Independent auditor correction and acceptance

The first real terminal audit failed before loading the checkpoint: `ValueError: exact complete worker launch input inventory`, session85550 exit1. No v1 output JSON was produced. The actual launch inventory had24 entries, exactly matching the frozen A800 YAML; the auditor incorrectly imposed the legacy4090 inventory of18. Six additional A800 developer-evidence files plus the already-bound developer environment explain the difference. This was an auditor profile-integration defect, not missing worker evidence.

The correction only changes `wa/tools/audit_failure_state_training.py` and its CPU tests. Explicit A800 now requires the fixed24-entry whitelist and all seven developer-evidence hashes. Legacy4090 still requires exactly18 entries and preserves the historical absent-profile default. Unknown, missing, extra, cross-profile or hash-changed inputs remain rejected. The expected whitelist is not inferred from the worker's reported inventory. The frozen trainer, job YAML, checkpoint and outputs were unchanged.

The final168-test CPU regression run passed in4.079s. A reviewer caught an intermediate compatibility error: requiring an explicit profile would have rejected historical4090 launches that omitted that field. The implementation now defaults only those legacy launches to `rtx4090`, with positive and negative tests. The old failure and this correction are retained rather than silently relabelled as a first-pass success.

| Evidence | SHA256 |
| --- | --- |
| Corrected auditor | `11e384825f9392e82845235337a666586cf2cef17ba9d6bd67c23d276c93b9f0` |
| Auditor tests | `6db9d3b8f0ff447aaa61329d8b9d8c1914f01f64b804a410368220907acdc07d` |
| `R/artifacts/failure_state_training_audit_62256_v2.json` | `1698dc68cdcc93c8aaca67c3de93bfd33557ae8f820db90455aa6b09825e3735` |

The v2 audit exited0 (session40885), output148970bytes, status `TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY`. It bound source, inputs, exposure, all three heldout modes, checkpoint and optimizer continuation, and rechecked source hashes for read-time stability. It does not certify closed-loop performance.

The exact CLI is the command in [the auditor report](FAILURE_STATE_TRAIN_AUDITOR_20261008.md), with `--hardware-profile a800`, this run/config/source, and output `R/artifacts/failure_state_training_audit_62256_v2.json`. Parent/config/plan pins are unchanged. Do not rerun into the existing output.

## Fixed152 candidate comparison

The unchanged152-window selection contains the original88 control windows plus32 early and32 late recovery windows. Candidate, parent59866 and best61609 each have304 unique predictions:152 normal-history and152 repeat-current-history, all finite7x4 trajectories. The same input and label hashes were used across models. Selection was not changed after viewing results.

| Group | Windows | Best normal ADE | Candidate normal ADE | Best normal FDE | Candidate normal FDE | Best repeat ADE | Candidate repeat ADE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STT collision | 24 | 0.467611 | 0.460156 | 0.807868 | 0.791991 | 0.488093 | 0.483697 |
| STT other | 16 | 0.383305 | 0.385286 | 0.557643 | 0.568087 | 0.406727 | 0.410342 |
| STT success control | 16 | 0.255145 | 0.251788 | 0.365834 | 0.358186 | 0.266653 | 0.262020 |
| DT control | 16 | 0.310269 | 0.303711 | 0.466993 | 0.452697 | 0.342154 | 0.326224 |
| AT control | 16 | 0.371023 | 0.330015 | 0.574820 | 0.518750 | 0.437934 | 0.422923 |
| Recovery early | 32 | 0.517808 | 0.365765 | 0.830163 | 0.571128 | 0.533759 | 0.393244 |
| Recovery late | 32 | 0.396540 | 0.328371 | 0.676199 | 0.567534 | 0.422109 | 0.356184 |

Six groups improved normal/repeat ADE; STT other regressed slightly. Yaw and first-point error did not improve in every group. These are label-fit measurements at fixed teacher states, not student rollouts or new SR. Candidate fit used A800 while baseline fit used4090, so no cross-GPU bitwise equivalence is claimed.

Independent array review recomputed all ADE/FDE/yaw/first-point errors in float64, with maximum candidate difference1.7765661408652988e-7 within the established tolerance. All group summaries and per-record/group deltas matched exactly. It verified912 label hashes, full152 selection identity, paired inputs/labels, and the selection/baseline/training-audit/checkpoint identities. It did not rehash or decode the3166 raw source files or call the production validation functions again.

The bounded developer run started06:22:03 and completed06:24:43 Beijing, elapsed160.060734499s, session94041 exit0. Only its owned PID4053777/timeout4053776 were used and are no longer running. GPU3 was A800 UUID `GPU-8feb0a89-838a-a61a-cbd3-ec9205b9c344`; at admission it showed0% utilization and79263MiB free, not exclusive ownership. Allocator cap5368709120bytes excludes CUDA context/external allocations. Timeout2400s was not reached. Peak allocated/reserved were1590791680/1608515584bytes;3 xFormers UserWarnings remain, five fatal-pattern hits0.

| Artifact | SHA256 |
| --- | --- |
| `R/artifacts/failure_state_group_fit_selection_20261008_v1.json` | `7e7db844c09f67d7b0e3b5de7376c0573c18fccc485dfa235b879ed4dab111a9` |
| `R/artifacts/failure_state_group_fit_pretrain_20261008_v1.json` | `6ecff4bda1c19d153e7e4be0742914d2b761698759a9157c17ec2920a6725cd9` |
| `R/artifacts/failure_state_group_fit_posttrain_62256_v1.json` | `bf35d861cf975dfa775cff9b7f23c75e326509c21823ba188d611838ead5d7dc` |
| `R/artifacts/failure_state_group_fit_posttrain_62256_v1.log` | `efa63cd6a02424c565821def4c37fc0b9bae3d7151a6bdc650f5c7280b702592` |

The output JSON is2060946bytes and records `FIXED152_TEACHER_STATE_LABEL_FIT_ONLY_NOT_SR`, `closed_loop=false`, `training_released=false`. The exact issued argv is log line2: `wa.tools.failure_state_group_fit run --model candidate`, selection/SHA above, candidate checkpoint/SHA/step61252, `--training-audit`/SHA above, `--baseline-fit`/SHA above, and `--output` to this new JSON. It used the unchanged fitter SHA `d4bf544e837149b16a31973f546e5d86a7b3050c61649aef919f36127166e404`. The bounded wrapper follows [the pretraining report](FAILURE_STATE_GROUP_FIT_20261008.md); no optimizer or environment trajectories ran in this fit.

## Terminal artifact pins and next decision

| Artifact under the training run | SHA256 |
| --- | --- |
| `metrics.json` | `ecedd7eb65960ef6a81190a933b8b31ad232c42439bbe7449d52f86253d0e500` |
| `worker_postcheck.json` | `32ee2fe6b9495b943ce11250f4e6ebe6bd9c595d8e58d4d6fb033df15dd0731a` |
| `actual_exposure_epoch1.json` | `7a9d810d146e807fb31042c5e9e3aa4cb81d2148d4b1d06023bc2425fda4fd1f` |
| `actual_exposure_epoch1.npz` | `3a5a515a5d56d48552da960fccd03dbd0c6af20a592ab1b68a50ef646160965d` |
| `train.jsonl` | `bc34f5de78a95fb53ffe44a2cdbb9678e5134868f867f3026e54923730b674bd` |
| Task-level `console.log` | `3353e5167008b2b4b2399a3495b7bade9539411b68ac26b5eacd028e3680f014` |

Next is complete mixed/zero STT, DT and AT, each1405 with8GPU, using fresh output roots and no reused student rows. The semantic PLY fix, seven BBox repairs, teacher initial-state pairing, model input contract and all denominators stay unchanged. The target candidate must pass the actual interface/environment checks; a configuration or free-GPU snapshot alone is not a submission or worker validation. No additional training epoch or success-threshold change is justified by these offline results.

At06:31 Beijing the scheduler inventory reported A8007free, H1000free and bj409072free. The known H100 development NAS lacked this project, model environment and62256 checkpoint. Main-thread devpod4090 identity/GPFS read succeeded; two later independent SSH sessions hit kex-close/reset and stopped without repeated attempts. Those errors are retained and require a successful known-entry check before relying on that endpoint for a new developer run or formal submission. No new evaluation Job was submitted during this report's initial recording.

The prior80-line CURRENT and120-line PROGRESS were copied and byte-compared before editing to `archive/2026-10/{CURRENT_TASK,PROGRESS}_before_62256_terminal_20261009_0620.md`. All prior failed experiments, old checkpoints and complete scores remain preserved.

## Full three-task evaluation preparation

Three complete1405-episode configs now use the independent clean worktree `R/source_student62256_eval24_v1`, commit `5a23a982c7ef01f7fdec58faf27f6ea623ed9eeb`. They preserve the successful61715 mixed/zero protocol, eight fixed shards, semantic repair, seven BBox repairs, teacher initial-state pairing, controller, video and denominator. Only candidate identity/prerequisite evidence and task names/output roots differ. No previous student rows are reused.

| Configuration | SHA256 |
| --- | --- |
| `wa/jobs/student62256_stt_4090_v1.yaml` | `0b4010b0d8781e7f8d349e6313acfe76536bb67b41edf8c7a22f5e30d711957f` |
| `wa/jobs/student62256_dt_4090_v1.yaml` | `8b1a211d2450bb8962388fb97335244a901c6a422d270ee47bf5569ca6129a03` |
| `wa/jobs/student62256_at_4090_v1.yaml` | `f17202a9c4c0190550854ff866454e426932fef6bad66515fcb2d397bf9bf5a9` |

Each config requests one8GPU K8s shell task on `baidu_4090`, image `x5-builder:cuda12.8-isaac5.0.0-v2.test1`, timeout86400. Full commands are stored in the YAML. Outputs are fresh `J/job_<job>/task_<task>/wa_student62256_{stt,dt,at}_4090_v1`; no output was precreated. Worker checks require eight real distinct RTX4090 UUIDs/cc8.9 and the fixed evidence before launching. Worker checks are not yet executed.

The actual CPU preflight passed against the terminal checkpoint, audit, candidate/baseline fit, selection, teacher index, semantic config and BBox plan. All three YAML/bash/Python ASTs and task contracts passed; their identical CPU program was executed once, not falsely counted as three independent runs. The first checker failed because its comparison normalized a `hard_stt` group label; correcting only the checker resolved it before actual execution. Independent static review found no must-fix. Final config changes after these checks only replace the draft comment; command bodies are unchanged.

At06:50:23 Beijing an authorized bounded developer process ran the exact frozen `wa.wm.eval_server --developer-check --mode mixed --noise-mode zero` with the62256 checkpoint on realRTX4090 GPU6, UUID `GPU-63ea1d9b-4c44-1c78-e545-dc93f6e962d4`. Admission showed24090MiB free and0% utilization, not ownership of the whole device. It ended06:54:09, exit0, elapsed225.276574723s. The5GiB PyTorch allocator cap excludes CUDA context/external allocations;900-second timeout was not reached. The complete argv and source identity are in the log's first line.

| Artifact under `R/artifacts/student62256_eval4090_developer_20261009_v1` | Bytes | SHA256 |
| --- | ---: | --- |
| `.json` | 2327 | `e552cb63a1796d4525f37f99f7b3286e58d8640fd92cabadf371125c6a7d4842` |
| `.ready.json` | 3334 | `ef4b60572284eb37235714be3a92545df972834c030e173ec99d5631d8709963` |
| `.log` | 2267 | `9b0dffbd37931dcb449e3854fb01b5b3d1c183f4c3f2b2fe1d02261cd1c655ae` |

Independent readback verified report/ready/log binding, checkpoint40915/step61252, mixed/zero/sampling4/seed7+step, two finite xy7x2/yaw7/geometry3 predictions and step0to1. Frame times were `[0,0,0,0]` then `[0,0,0,0.05]`. Peak allocated/reserved were1571047424/1595932672bytes;3xFormers warnings remain, checked fatal patterns0. This is synthetic directSession inference, not HTTP RPC, Habitat/Xvfb, real trajectories, negative later-BBox testing,8GPU worker/NCCL or newSR. The later `initial_bbox=None` positive path ran; rejecting a nonempty later box is only the unchanged source contract in this check.

At06:54:20-23, independent A800 read-only validation of the sameGPFS used the real Habitat interpreter and formal PYTHONPATH: each task has1405unique definitions, matching manifest instructions,101existing scene paths and shard counts176/176/176/176/176/175/175/175. Probe Python3.11.15/torch2.8.0+cu128 and Habitat Python3.9.19/torch2.5.0+cu124-overlay/Habitat+Habitat-Sim0.3.1 imported successfully. All10existing dependency pins passed. Xvfb installer/runner both0755,34debs13682356bytes;36file hashes/sizes matched the existing handoff manifest. Installer SHA `abec088b9ac251757fab98a659534aedb68f2861c5353bcb42b0ff2ec2b44ead`, runner `9db22b3b69f9d0b63dd9ba6bef3146225d37fd310702566ce9f6e5deb8eb0353`. No installer, GPU or trajectory ran in this static check; Gym's deprecation warning is retained. This does not establish a new worker's GLX/EGL behavior.

At06:49 the normal A800 submission chain was authenticated as the owner with md_ai_kit2.0.0. Resources were A8007free/H1000free/bj409072free, not reserved allocations. Own RUNNING62262/61847/61846 and SUBMITTED62404 were individually identified as other work; no student62256duplicate existed. A separate new4090SSH session again hit kex-close and was not retried; the already-running main developer session completed normally. The healthy original A800 submission/NAS path is unchanged, not a workaround to execute through the failed4090SSH connection.

GitHub `wa` terminal-report commit `56df1c03e444b3ab8aa07b4348cdf87cd4350eb1` was verified after the first push returned remote `fatal error in commit_refs`; one normal same-commit retry succeeded. The current preparation must be backed up before full submissions. CURRENT80/PROGRESS127 were copied and byte-compared to `archive/2026-10/*_before_62256_eval_preflight_20261009_0653.md`. No new modelSR or promotion is claimed.
