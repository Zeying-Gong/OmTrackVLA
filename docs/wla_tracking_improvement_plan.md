# WLA tracking: architecture and paired review
Updated: 2026-09-28

## User decision and authorization
Preserve the WLA research paradigm and pursue competitive/SOTA performance; LightNav is a comparator and potential teacher.
The user explicitly permits WLA architecture changes and borrowing ideas from LightNav and USS.
The user authorized exactly the selected 16 pairs / 32 complete rollout recordings.
This permission does not assert any proposed architecture works, nor authorize unrelated DA3 jobs.
New architecture training/evaluation is a separate full experiment to specify before formal submission.

## Existing evidence
Completed full evaluations: WLA Job58346 STT/DT/AT SR80.2847/53.3096/49.2527%; LightNav Job58433 SR89.47/78.22/65.91%.
Both cover1405episodes/task under the existing text+RGB protocol.
WLA Job58638 is a separate new candidate under evaluation.
Among teacher-only wins, WLA Lost counts are STT66, DT206, AT171; collision counts67/102/80.
These outcomes locate useful diagnostic cases; Lost does not by itself establish target-identity failure rather than control failure.
WLA has four historical RGB frames, MetaQuery, layerwise flow action expert, seven SE(2) waypoints and a range/bearing auxiliary head.
Current tracking loss is action flow plus target geometry; it does not train future latent-state prediction for tracking.
The source snapshot already contains optional Belief modules and per-episode transactional caches, but the current tracking adapter does not use them. These are integration candidates, not validated tracking memory. Its existing World Expert rejects action_condition, so the proposed action-conditioned latent predictor needs an explicit new adapter/objective rather than a configuration toggle.
Current expert correction uses a single learner-to-teacher switch per episode and successful teacher futures; recovery-and-return is not yet implemented.

## External evidence
USS uses temporal visual memory, prompt-conditioned compact representations, presence prediction and an action-conditioned latent auxiliary objective.
Its bbox-DT ablation reports SR83.6 full,72.2 without memory and80.4 without world-model loss.
These are its own experiments, not predicted WLA gains.
Its spatial-prompt evaluation changes initialization to ensure target visibility and omits AT; its scores cannot establish SOTA under our text-only protocol.
Source: https://arxiv.org/html/2606.25880v1

LightNav exposes temporally aware history compression, dual-channel pointing, RVQ actions and staged embodied-reasoning/SFT/RL training.
Its released model also benefits from extensive training data; our system gap does not isolate an architecture cause.
Source: https://github.com/lightorigins/LightNav-0

## Proposed sequence (not yet trained)
1. **Diagnose paired videos.** Locate first divergence, apparent target confusion, occlusion recovery, turn lag, distance/speed error, collision and action oscillation. Include three WLA-only-win counterexamples. Human labels describe visible evidence and confidence, not guessed internal states.
2. **Train-only recovery data.** Replace one-way expert takeover with bounded takeover, recovery and hand-back, allowing repeat interventions; collect training-scene failures and preserve failed recovery attempts in the audit. Explicitly measure teacher availability, recovery rate, intervention count and autonomous duration. Do not train on these selected validation videos.
3. **Target state and temporal memory.** Keep the WLA backbone/MetaQuery/action-expert structure initially. Add compact persistent target tokens with identity, visibility confidence, location/motion estimates, and confidence-gated updates. Use text-grounded predicted evidence at text-only evaluation; ground truth may supervise training but must not enter inference. Control drift while target is absent; avoid overwriting memory with distractors.
4. **Action-conditioned latent prediction.** From current target tokens and executed actions predict next-step/multi-step target representations; use a stop-gradient EMA target encoder during training. Compare with equal-data/equal-compute memory-only baseline. Explicitly test action shuffling/removal to establish action dependence and monitor latent collapse.
5. **Action head/control ablation only if diagnosis supports it.** Compare current flow expert to a lightweight deterministic waypoint head, or add turn/forward structure and confidence-adaptive replanning. Hold perception/data/controller settings fixed where possible; report rollout SR/TR/CR and inference latency. A decoder change should not be confused with changing the entire WLA paradigm.
6. **Optional spatial interface as a separate track.** Initial bbox/point/mask prompting may be useful for intended deployment, but changes information available to the policy. Keep text-only benchmark and spatial-prompt results separate, including explicit initialization protocol.

## Experiment and acceptance design
Start with baseline versus recovery-data-only versus memory-only, then combine only if individual gains justify it.
Freeze checkpoint/data/controller/seed/protocol and record manifests/hashes. Compare paired outcomes across full STT/DT/AT, not the selected16.
Report per-task SR/TR/CR, per-scene paired uncertainty, target-reacquisition/visibility diagnostic metrics and inference wall-time.
Claims of SOTA require matching interface/protocol and comparison against relevant current systems; surpassing one selected-case score is insufficient.
No claim that WLA's backbone is unsuitable is currently established.

## Recording specification and status
Execution snapshot: /data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925 (Baidu NAS; no .git).
Base CODE_REVISION: cb78954213f5ca32494f13639b8340797144a664; added recording source tracked by SHA256 manifest.
Config/manifest: paired_review_20260928/full_a800.yaml and manifest.json.
WLA checkpoint: job58346/task69086/wla_teacher_train/training/checkpoints/step-0043203.pt
SHA256: 0b8f036fd282474d8c9efe9e34e1f4dc56b9282c44295bcda1f16edcf48460e1
LightNav checkpoint: algorithm/repos/LightNav-0/checkpoints/LightNav-0; original released controller/fallback preserved.
A8004GPU, two lanes per method, eight complete episodes per lane; original16scenes/text/seed7/termination unchanged.
Source manifest r3 SHA256: 8037a3636e4320c8900c10ef0102e798ba97c0e453579efd80850e4eb9adaffa.
- Job58917 / Task69684: FAILED before model loading; missing CUDA_VISIBLE_DEVICES assumption. GPU allocation and protocol hashes passed. Failure artifacts retained.
- Job58920 / Task69687: FAILED before rollouts; installed vLLM rejects GPU UUIDs. Retained.
- Job58925 / Task69692: SUCCEEDED2026-09-28 20:03:23 Asia/Shanghai; submitted19:55:02 with container CUDA ordinals matching original evaluation.
Output: /data/nas_ray/project/md-ak/users/zeying.gong/job_58925/task_69692/paired_review
Acceptance PASS:32complete episode records/frame sequences/videos;2612source frames;16initial-frame hashes match originals;maximum video timestamp error2.78e-13s;video SHA256 verified.
Outcome drift:WLA case_02 Lost/SR0 became Normal/SR1;case_12 Lost/SR0 became Normal/SR0. All other30outcomes match. No causal explanation of rerun variation established.
Full evidence: VERIFICATION.json and REPORT_RENDER.json in output. Browser verified actual side-by-side playback, case changes, outcome display and copyable JSON export. Native download event was not confirmed in IAB.
Videos preserve all pre-action RGB observations; last-frame hold is estimated from preceding step. No terminal post-action observation.
Paired HTML review supports playback, shared seeking, labels, confidence, timestamps and local JSON export.
Raw data/video remain on NAS. These selected validation episodes and human annotations must not be training data.

## Review access
Only the HTML review and32MP4files are served on loopback; raw frames/logs are not web-accessible.
The original broad-directory server request was blocked by automatic review; it was not run.
A narrower explicit33-file allowlist was approved and verified (unlisted paths return404).
Run:

```bash
ssh -N -L 18781:127.0.0.1:18781 nas-a800
```

Open http://127.0.0.1:18781 . Annotate first visible error time, failure type and confidence; export/copy JSON.
The service serves review_site/index.html with only display fields, excluding internal NAS paths.
Human review is pending; no new architecture training has been submitted.
