# Current task
Updated: 2026-10-06 13:00 China
Status: A4_FORMAL_READY_AWAITING_USER_RESOURCE_APPROVAL / A3_FULL4215_AUDITED / DA3_RETIRED

## Objective and protocol
- User authorized A4 reuse of A3 experience plus offline and direct online adaptation to raise actual frozen full4215 SR. Repeated TEST learning/tuning/selection is allowed.
- Label TEST_SCENE_ADAPTATION_NOT_HELDOUT; no heldout-generalization or global theoretical upper-bound claim.
- First exceed A2_60058 2635/4215=62.514828%, then B60994 2684/4215=63.677343%. Only a strictly better audited full result replaces BEST_FULL.
- Inference uses causal RGB-derived features, original instruction, time/step and internal noise. GT/outcome/identity remain training/audit metadata; no episode lookup or oracle routing.

## Authoritative package and readiness
- nas-a800:/data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925/a4_hybrid_replay_20261006.
- Read RUN.json, FORMAL_REVIEW_REQUEST.md, DEVELOPMENT_READY.json, PRE_SUBMISSION_READY.json, FROZEN_INPUTS_REPORT.json and formal_campaign_config.json.
- Independent full pipeline implemented: assignment ownership, complete-episode admission, shared four-lane updates, drain/freeze, standalone frozen serving, optimizer/RNG/online-queue recovery, artifact plus neural replay audits and BEST_FULL gate.
- Frozen6498 inputs/19,118,576,195 bytes SHAef76b4a5a7b27df476ae3354480985fe75514351a9a2343b60d32b943fc60c14. All selected bytes read;4199 prior expected hashes matched. Do not edit frozen files.
- WLA export has no Git. H100 Git is task documentation only; dirty trained_agent.py/CSV/untracked sources remain untouched.

## Latest real developer evidence
- Four actual development GPUs3/4/6/7;learner shares3. Attempt03 completed20 complete rollouts and1281 actual feature/action records; all artifact and independent CPU neural replay audits PASS.
- Offline2 candidate updates were tested, then online started zero-residual A2. Four online episodes/267transitions/8updates used1258 old+790 new sampled rows. These are developer fixtures, not efficacy or cumulative prior-run updates.
- All267 online mean/epsilon/total latent and CUDA RNG reconstructions were bitwise exact. Four actually served policy versions retained; every future executed feature carries snapshot and explicit exploration evidence.
- Developer FINAL404d521753f5ad15b34ce27b181e3d8018492fa17d827ed58c74f6ea600a2ba6; formal starts fresh, never from this checkpoint.
- Actual FINAL model/two optimizers/queue/sampler/exploration RNG/commits/snapshot catalog restored exactly. Next256 samples and copied-RNG actions exact;0 new updates/interactions.
- Completed campaign resume reuses both artifact and neural audits, with0 new rollouts/updates. Failed partial cohort restarts in a new output from the preceding drained checkpoint.
- All1732 old NPZ hashes and8GiB cache prewarm passed:7,362,235,056 cached bytes,53.55s. Whole child summed RSS peak154,752,921,600 bytes can double-count shared pages; not a long-run Pod memory guarantee.
- New raw initial RGB is identical per source across20 resets. Historical A2 pairing remains JPEG-only; no invented historical raw-RGB equality.
- Saved feature-to-latent CPU replay is not full RGB encoder/Flow replay. CPU/CUDA arithmetic tolerance is explicit; no CPU bitwise claim.
- Initial Xvfb PATH failure retained in attempt01. Attempt02 v1 and attempt03 v2 are separate, retained successful runs.

## Concrete formal task awaiting approval
- One normal K8s task on baidu_bj_a800,4A800,72h maximum. SDK proposalCPU32/60,RAM240/480GiB,shm150GiB; actual Pod specification needs post-submit verification.
- At04:56UTC A800 had47 free GPUs. SDK2.0.0 login and canonical queue checks passed. Canonical cluster uses the supported process-local submitter_canonical_overlay.yaml via MD_AK_CONFIG_PATH; no global SDK/auth changes.
- full_a800.yaml calls pipeline.sh with formal_campaign_config.json. Output job_<JOB_ID>/task_<TASK_ID>/wla_a4_hybrid_replay/run on the existing Baidu NAS; no cross-NAS migration.
- Offline snapshots128/512/1024 each get fixed252 full-horizon panel (84/task); A2 panel baseline160/252. Strictly better best offline seeds online; otherwise zero-residual A2.
- One full4215 online pass in task-balanced351/351/703 cohorts,2 mixed replay updates per complete episode. Each drained cohort gets another252 screen.
- Best of six panels strictly above160/252 gets one full4215. Maximum9454updates/9942complete rollouts; no runtime or improvement guarantee.
- Only strict4215 identity/seed/horizon/finite/checkpoint plus saved-feature neural replay auditing permits full SR claims and BEST_FULL updates. Formal evaluation reports actual partial success/completed counts every20min.
- Previous8A800/48h recollection submission was rejected by automatic approval review before execution for missing concrete resource authorization. Old APPROVAL_BLOCK.json remains; no bypass.
- This newly prepared4A800/72h hybrid task requires explicit user approval. Formal Job=null;formal updates0;no A4 full SR. Ask once with the concrete review package, then wait without repeated reminders.
- After approval, recheck RUN/queue/hash/login/resources, submit once through the normal SDK with canonical overlay, then record IDs and actual worker resources. No cluster smoke.

## Preserved science and references
- Fresh MC terminal-success value/outcome and capped weighted actual-total-latent regression; separate actor/value readers, soft A2 residual penalty, no Q-max/bootstrap. Both success/failure episodes retained; MC mixed behavior is not unbiased new-policy advantage.
- Freeze pristine A2 encoder/MetaQuery/Flow/heads/controller and projector. A2 SHA59cdfe10fc1efa92ae24568152e2c24baa69a1b91609fc4f823ec625204d713b; representation bundle SHA7b523f1d10a1dad7407ec6aee3fca4016bc1cbb34460cc9ed39ab948ee7dbcc9,projector only.
- Old A3_FP32_HEAD outcomes remain historical labels; new A2_BF16_HEAD is a different execution domain. No clipped latent/reward relabeling or A3 actor/Q/optimizer restore.
- A3_61020/71943 completed2209/4215=52.408066%,STT1061/DT583/AT565,FINALc7269fa14ba53ec60f4f27fa80aa9b295f83464ca5e0124597c30e0c10fce435.
- A2 fullSTT1143/DT796/AT696;B60994 fullSTT1169/DT787/AT728. B63.677343% remains highest measured A2/B/A3,checkpointd4987b0aa961d971c30095c1b914224fff4b4dcc4b604655bb6113cd0f6f2883. A4 BEST_FULL remains A2.
- Preserve historicalA2 action-only/59752, all failures/STOPPED and oldA4 source3979/b87cfadaa2350892ad8eb34f70fa5165bf316cac9c13ddae88f8910a3b44c6a6. No60766/59352/DA3 restart or repeated B jobs.
- NAS only; A800 preference. Only explicit Markdown task documents commit/push; both CSVs stay on NAS. Automation remains active; no active goal to create or stop.
