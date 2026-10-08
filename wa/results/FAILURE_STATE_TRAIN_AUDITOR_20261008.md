# Independent failure-state training terminal auditor — 2026-10-08

Status: CPU_DEVELOPMENT_PASS_NOT_REAL_TRAINING_AUDIT. No formal failure-state training Job, new checkpoint or closed-loop SR was produced by this work.

## Purpose and unchanged experiment

The new tool `wa/tools/audit_failure_state_training.py` is an independent end-of-training checker, not a trainer or a replacement for scheduler/worker checks. Its output is accepted only after the planned full run has actual terminal artifacts. Frozen training source remains `199385cd9c826c8f21308ad99b6a8375396d9c90`, with no changes to model, losses, optimizer recipe, controller, physics, data or success criterion.

Draft `wa/jobs/failure_state_train_4090_v1.yaml` remains NOT_SUBMIT_READY; SHA256 `33838ac5dc95bf1a00133cdcb32d28a47224bc73f53bb6ed5fd863ccbf01d035`. The target-4090 real four-update training compatibility check still needs safe development GPU capacity. Existing A800 four-update and completed fixed152 policy-fit evidence are preserved, not rerun or mislabeled as 4090 backward/NCCL evidence.

## Checks implemented

- Read complete terminal metrics before hashing/loading a checkpoint; require final61252 and finite image/point/mixed metrics with exactly73368 heldout windows each. Re-read pinned metrics and reject any change from the initial read.
- Bind the frozen source, full26 training flags, runtime config, eight distinct RTX4090 UUIDs, immutable input graph, fixed152 selection and608 pretraining predictions.
- Reconstruct the immutable three-source plan and compare all four recorded int64 exposure arrays per window, including the old omitted position. Require1233424 total=726631 base+457641 old teacher+49152 recovery.
- Reconstruct eight-rank sampler/DataLoader assignment (154178 positions and77089 microbatches per rank); this is not independent per-rank actual telemetry.
- Check1543 scheduled log rows, original loss composition, resumed five-group cosine LR, final61252 and38545 updates.
- Validate CPU model and AdamW tensor identity/finiteness, resumed parent model+optimizer, unchanged group/hyperparameters, at least one changed model tensor. Every parent optimizer state already at22707 must reach61252; conditional parent states have bounded increments and are explicitly counted, not silently certified as full-step states.
- Rehash pinned inputs before releasing the independent offline-only report. Worker postcheck is supplementary, never a substitute. No new closed-loop or untouched-test claim.

## Executed evidence

Tool SHA256: 862aa8a71693773c00006846297b3d00daa40f5feecda8aeb076228a159873f8
Test SHA256: 389ecf5476b4a10af2ed0c01522cfe862b54d0526618f8901798db962835c10a

Independent tool tests: 34 tests PASS (0.448 seconds). Main directly related regression: 151 tests PASS (3.421 seconds), command:

```bash
cd /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/checkout
PYTHONDONTWRITEBYTECODE=1 /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/probe_env/bin/python -m unittest wa.tests.test_audit_failure_state_training wa.tests.test_failure_state_sampling wa.tests.test_failure_state_sampling_runtime wa.tests.test_failure_state_train_args wa.tests.test_failure_state_train_gate wa.tests.test_failure_state_group_fit wa.tests.test_teacher_window_plan
```

Main real-file CPU integration (synthetic launch/runtime envelope, no submitted Job):
1. `validate_config` parsed the actual draft YAML's26 flags; `load_plan` reconstructed actual pinned1233424 positions and original counts. Eight pins; elapsed1.955407729s; session11321 exit0.
2. `bind_inputs` and `bind_fit` read actual NAS arrays/admissions and completed fit, verified6864 recovery rows,73368 original heldout rows,152 selected windows/608 predictions, and47 stable file pins. Elapsed2.642060791s; session48333 exit0.
3. No checkpoint was loaded, no PNG decoded, no policy/GPU inference or training executed by these integrations. The synthetic envelope is a code/data compatibility test, not worker provenance.

Independent review found and corrected two false-PASS possibilities before acceptance: relying only on maximum optimizer step could hide a stopped active parameter/group; first-read metrics could differ from the later pinned content. Regression cases retain these failures. Earlier30 tests and147 related tests passed before these stricter cases were added; they are superseded by the final counts above, not evidence that the original checks were sufficient.

## Resource status and remaining work

Read-only snapshot2026-10-08 19:51:59 Beijing: A8003free/88; bj409080free/144. These are cluster capacities, not allocations. H100's separately mounted NAS was previously checked this day and lacks this project's ready code/environment/data; no migration was done.

Development4090 snapshots19:52:04 and19:52:55: GPU7free7690MiB/util0; GPU6free16252MiB bututil93→88; GPUs0–5free881–2892MiB. No device met the safe short-training criterion (at least10240MiB free and utilization≤5), given prior measured allocated peak7.309GiB plus CUDA/context/reserved headroom. No busy card was borrowed and no other process stopped.

Next: when safe development capacity is available, run the unchanged four-update recipe on4090, record actual kernel/backward/optimizer evidence, bind it into the final config and this auditor's exact input inventory, refresh resources and duplicate-job checks, announce and submit one complete8GPU epoch. Do not submit a cluster smoke task. At terminal time run this auditor against real artifacts, plus scheduler/log review and fixed-fit; only then proceed to fullSTT/DT/AT each1405 on3×8 GPUs.

Current best remains61609: STT1279, DT1178, AT1207 /1405 each. STT still needs10 additional successes. This work changes neither SR nor the evaluation-set-adaptation boundary.

## 20:38 update: separate complete A800 task

The earlier RTX4090-capacity wait is retained as history. A safe alternative is available: submit the full8-A800 task to the scheduler queue, using the same frozen recipe's already completed realA800 four-update development gate. Fresh installed-client/dispatcher inspection found no instantaneous-free-GPU submission gate; Volcano ray-a800 Pending remains SUBMITTED and shell timeout begins on actual execution. External queue policy and wait duration remain unknown. Use canonical baidu_bj_a800 for queries; the compatibility alias baidu_a800 is valid for submission but can hide jobs when used as a list filter.

New config: wa/jobs/failure_state_train_a800_v1.yaml, SHA08553eb73e8d2db4ae2fd37ab1c322bc7fbf77ac4a3ba3a7c59ae2508a2f2e69. The original4090 draft remains byte-identical. Full26 training flags, frozen199385cd, image, source data, parent59866 MODEL+OPTIMIZER and one-new/cumulative-two budget are unchanged. The new config requires eight exact A800-SXM4-80GB GPUs, CC8.0, at least80000MiB each and uniqueUUIDs; launch pins18→24 bind the existing seven A800 diagnostic files (environment already in18). Single-GPU four-update and fixed-fit evidence are explicitly not eight-rank execution or newSR.

Auditor now accepts an explicit --hardware-profile a800; default remains rtx4090. Profile binds cluster, run suffix, exactmodel, capability, memory and unique physical UUIDs. A800 additionally requires exact developer evidence schema and its seven pinned files. Cross-profile and unknown-profile acceptance is rejected; other data, numerical, optimizer, terminal and TOCTOU checks are unchanged.

Final toolSHA1bf234bfda69047059013413ff69e9a3223a2ab81cf72d61847b2b4a427aab72;testSHA4e407009d42494889bb5770c1b11e32254e0a71e3e240fc06680c4b61e6ca8ae.41 tooltests PASS0.528s; main158 relatedtests PASS4.516s,session31033. Actual newYAML26flag parsing passed against a synthetic formal config envelope; it is not a worker result. Independent fresh18file hashes (8.80GB),241frozenPython/environment match and seven-small-file diagnostic schema checks passed. No model checkpoint loaded, no GPU/repeatedcollection/PNG audit, and no formal Job has been submitted at this report update.

## 20:57 formal submission record

Code/configuration backed up to GitHub wa at602c6aa3efb354388de2871f6614c350ff18d27c before submission. The first20:44:35attempt was rejectedHTTP429 because the account had10/10activejobs;noJobID was returned and a read-only check found no newWAjob. After the user cleared other queued tasks and explicitly requested continuation, fresh checks found only two unrelated runningWLAjobs and no queued/submittingWAjob. One renewed submission at20:56:43 succeeded:Job62256/Task73511,baidu_bj_a800,8GPUs;firstdetailcreated20:56:44/updated20:56:45,statusSUBMITTED.

Output:/data/nas_ray/project/md-ak/users/zeying.gong/job_62256/task_73511/wa_failure_state_train_a800_v1. NoworkerGPUallocation,optimizerstep,finalcheckpoint ornewSR is certified by this submission. Do notsubmit a4090duplicate. At completion use this auditor with explicit --hardware-profile a800 and the pinned A800YAML hash;the remaining heldout/actualexposure/checkpoint/groupfit/closed-loop gates are unchanged.

## Pending terminal command for Job 62256

This command has **not run**. At 2026-10-09 01:10 Beijing the job is training, with step26075 and no terminal metrics. The earlier unsubmitted/queued text describes historical stages, not the current job. See [live training evidence](FAILURE_STATE_TRAIN_62256_STARTUP_20261009.md).

First verify the same Job62256/Task73511 has reached scheduler success and inspect its complete worker logs. Require final metrics for all three modes, each73368 windows; a periodic checkpoint or `checkpoint.pt` alone is insufficient. Scheduled train logs may end at61250; final checkpoint/metrics must establish61252, so do not mistake the logging interval for a missing final update. Only then run the existing CPU auditor. Preserve any exception and partial evidence as a failure; do not relax checks or retry under the default4090 profile.

```bash
set -euo pipefail
WA_PROJECT=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928
cd "$WA_PROJECT/checkout"
AUDIT_OUTPUT="$WA_PROJECT/artifacts/failure_state_training_audit_62256_v1.json"
test ! -e "$AUDIT_OUTPUT"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$WA_PROJECT/probe_env/bin/python" -B -m wa.tools.audit_failure_state_training \
  --hardware-profile a800 \
  --run /data/nas_ray/project/md-ak/users/zeying.gong/job_62256/task_73511/wa_failure_state_train_a800_v1 \
  --source "$WA_PROJECT/source_failure_state_train_v1" \
  --source-commit 199385cd9c826c8f21308ad99b6a8375396d9c90 \
  --config "$WA_PROJECT/checkout/wa/jobs/failure_state_train_a800_v1.yaml" \
  --config-sha256 08553eb73e8d2db4ae2fd37ab1c322bc7fbf77ac4a3ba3a7c59ae2508a2f2e69 \
  --plan-root "$WA_PROJECT/artifacts/failure_state_sampling_candidate_20261008_v1" \
  --plan-admission-sha256 230079f1e99836dc7b3bf20942859e127e0b42dfa0b64c76b2c6ab67dd373902 \
  --output "$AUDIT_OUTPUT"
```

Recheck tool SHA `1bf234bfda69047059013413ff69e9a3223a2ab81cf72d61847b2b4a427aab72`, clean relevant source and fresh output before execution; a changed tool requires review, not blindly restoring or overwriting somebody else's edits. The tool validates terminal metrics before hashing/loading large weights. Its per-rank assignment is reconstructed, while actual exposure counters are merged: do not describe these as separately measured rank-local telemetry.

A PASS is `TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY`, not SR. Read back and hash its fresh report; use its actual candidate checkpoint hash with the unchanged selection and baseline in [the candidate-fit command](FAILURE_STATE_GROUP_FIT_20261008.md). Then validate304 finite7x4 predictions and fixed input/label identities. After that, refresh platform readiness and submit fullSTT/DT/AT evaluations, each1405 on8GPUs, with real candidate SHA and fresh outputs. No synthetic SHA, mid-training checkpoint or old student rows may substitute for the final candidate.
