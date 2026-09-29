# WLA intervention A/B — 2026-09-29

Status: SHARED_COLLECTION_RUNNING. Formal A/B training and evaluation authorized; training waits for the full audited release.
Old Job59352/Task70125 STOPPED at user request; preserve all checkpoints and partial metrics. No automatic restart.

## Experimental question
Compare parameter-space weighted flow learning (A) against frozen-policy noise-space adaptation (B), retaining original WLA RGB+text, four causal frames, seven SE(2) action points and range_only controller.
Initial shared baseline: Job58346 step0043203. Current memory adapter is not the starting policy.
Do not claim the old one-way takeover experiment was a complete iterative DAgger evaluation.

## Collection
A new independent NAS package intervention_ab_20260929 uses the same audited 2043 training-corpus episode manifest and scene partitions (no official validation scene).
Student and LightNav both process every observation. Training-only range, true-target visibility/centroid and displacement gate selects which action executes.
No fixed64-step takeover; threshold/hysteresis, minimum1.4s teacher control and0.5s stable recovery allow repeat takeover/return.
This is an oracle-assisted training collector, not a deployable learned ThriftyDAgger risk policy and not an explicit wrong-person classifier.
A locally stalled motion trigger detects lack of progress, not predicted obstacle collision.
Only recovered segments contribute full0.7s teacher-owned, fallback-free labels. Retain failed episodes and bad-prefix observations; never supervise bad student actions.
Exclude collision-terminal recovery within1s of the last observation. Audit initial identity/camera/time, action ownership, scene partitions and exact interpolated labels.
All qualified correction windows plus deterministic task-stratified normal replay of approximately equal window count. Loss mass correction/replay=50/50 in each task.
A/B share immutable release hashes. Original replay remains preserved; selection seed20260929 and episode IDs are recorded.

## A
SIRIUS-inspired weighted flow matching; train action expert, existing LoRA/metaquery and geometry head from baseline.
Full2epochs,4A800 planned, batch size selected by real developer memory checks.
This is an adaptation of SIRIUS, not its original four-class behavioral-cloning implementation.

## B
FlowDAgger-inspired deterministic noise MLP on frozen WLA observation features; frozen base action decoder.
Invert expert7x4 chunks with the exact production4step Euler schedule. Run actual forward reconstruction; cannot substitute an action residual.
Current developer gate: >=23/24 fixed task/group-stratified samples have mean XY error<=1cm and mean yaw error<=2degrees; parameters truly update and frozen base gradients absent.
Default5-step fixed-point failed.20-step and damped variants under investigation; preserve every failed report. Do not relax gate or declare B runnable prematurely.
Normal replay uses inverted expert actions for matched-data comparison, differing from the paper's successful autonomous-noise buffer. Record this distinction.
Full dataset inversion must recheck reconstruction distribution; explicit failure if coverage fails.

## Iterations and evaluation
Round0 collects shared baseline distribution. A/B train in parallel after shared release and individual implementation checks pass.
Subsequent rounds collect current A and B students, pool new corrections with earlier rounds and matched replay, then update both. Record student checkpoint at every rollout.
Do not call round0 alone a completed DAgger loop. Round1 union collector and scheduling still require implementation.
Final independent WLA evaluation has no expert, panoptic labels or simulator state as policy inputs. Full4215 STT/DT/AT, same seeds and initial identities; report SR/TR/CR, recovery diagnostics and compute/data budgets.
Do not allocate GPUs waiting for upstream data. Check A800 first; H100 requires independent NAS/runtime validation; 4090 only after compatibility checks.

## Primary references
- DAgger: https://proceedings.mlr.press/v15/ross11a.html
- HG-DAgger: https://arxiv.org/abs/1810.02890
- ThriftyDAgger: https://arxiv.org/abs/2109.08273
- SIRIUS: https://arxiv.org/abs/2211.08416
- FlowDAgger: https://arxiv.org/abs/2607.08877


## Live evidence
- User-requested stop: scheduler confirms Job59352/Task70125 STOPPED.
- Shared collection Job59545/Task70402 RUNNING,4A800,2026-09-29 15:42:29 China time.
- Output: /data/nas_ray/project/md-ak/users/zeying.gong/job_59545/task_70402/wla_intervention_collection
- Frozen collection manifest SHA256:01da34132cb819e90d809a81fc7d466d69a0a6c77b451b66285eb4a7c1651585 (439 files).
- Developer real rollout:54frames,25student/29teacher actions,1actual takeover-return,0fallback,15audited complete teacher windows.
- Initial developer collector launch failed for absent Xvfb PATH; preserved, rerun with NAS runtime bundle passed.
- A real optimizer update verified; two-update check is ongoing.
- B5fixed-point reconstruction failed; B20 passed11/24; dampedBF16 passed12/24. Report serialization failed for numpy int and was fixed; raw per-window progress retained.
- FP32 action-decoder investigation is developmental only; if adopted it must be shared by A/B and independently evaluated reference. Existing benchmark scores would not alone verify a matched numerical protocol.


## Development verification update, 16:07 China time
A: single and dual A800 two-update runs passed; actual action-head update approximately5e-6 at first update; gradients verified for action expert, LoRA, output heads, target geometry and yaw.
B: the four-step decoder requires more than paper-default fixed-point inversion in this checkpoint. FP32+damped80 iterations passed21/24, still below the fixed23/24 gate.
Selected candidate: FP32 four-step decoder, FP20 initialization plus latent-only L-BFGS50 refinement.24/24fixed task/group-stratified train windows passed; mean reconstruction ADE0.003993m, worst sample mean0.009756m;2noise-policy updates verified, base gradients absent.
This is a FlowDAgger-inspired WLA adaptation with an alternate inversion optimizer, not a claim of exact paper reproduction or measured tracking improvement.
B full training entry passed a single-GPU developer run (2train+2heldout samples,2epochs/2updates); dual-GPU verification pending.
A/B checkpoint reload, three consecutive production Session predictions, reset reproducibility and rejection of privileged request keys passed.
Formal configurations: train_a_a800.yaml and train_b_a800.yaml,4A800 each,full2epochs of the common released dataset; neither submitted yet.
The immutable collection recipe remains its launch snapshot; current development status lives in DEVELOPMENT_STATUS.json. No running source was changed.
Initial full collection evidence:74completed,0exceptions,32takeovers,23returns. Partial counts do not establish a release or method efficacy.
At submission, estimate B inversion runtime from actual release size and developer throughput; allow sufficient timeout or improve batching with checks, never silently shrink the formal data.


## Ready to train after upstream data, 16:07 China time
- Both A/B single and dual GPU real2updates PASS. B dual-GPU inversion4train+4heldout samples all pass.
- Both branches' checkpoint reload, three-frame serving, reset reproducibility and privileged-input rejection PASS.
- DEVELOPMENT_READY.json=DEVELOPMENT_PASS_WAITING_SHARED_RELEASE. Training-source manifest SHA256:b2501ed49ca9c3169467e7aecf9c2d00f1e8453642e035ef94cb57fa2593e168.
- Full shared collection latest snapshot107/2043,0exceptions,50takeovers,34returns. No A/B formal training submitted yet.
- Known frozen release corner case: return_count records quality-qualified returns while the release assertion counts all physical returns. A collision within1s of a later return can fail final audit despite earlier valid segments. COLLECTION_AUDIT_ERRATA.md and separate release_after_collection.py preserve a corrected whole-corpus audit path without modifying running source or raw data. Never consume a partial release.
- prepare_round1.py and aggregate_rounds.py now prepare latest-student packages and a shared union of all three correction releases; static syntax checks only. Actual round1 serving/collector checks remain required once real round0 checkpoints exist.
- Updated heartbeat records actual59545/70402, waits for complete audited caches, verifies resources/source/checkpoints before each full A/B submission, and must continue updated-student aggregation before final evaluation.

## 2026-09-30 full release and formal A start
- Collection59545/70402 SUCCEEDED2026-09-29 23:56:11China;2043/2043 complete. Original release RECOVERY_RELEASE_AUDIT_PASS; no errata rebuild required.
- Eligible correction episodes596; training671episodes/24910windows; heldout72episodes/3204windows. Cache hashes verified, train/heldout scenes disjoint, no final-evaluation scene overlap.
- A59730/70592 submitted00:01:40China on4A800, full2epochs/3114updates.550actual updates verified; no tracking improvement claim.
- A output: /data/nas_ray/project/md-ak/users/zeying.gong/job_59730/task_70592/wla_intervention_a_round0 .
- Source manifest b2501ed49ca9c3169467e7aecf9c2d00f1e8453642e035ef94cb57fa2593e168; starting checkpoint SHA256 0b8f036fd282474d8c9efe9e34e1f4dc56b9282c44295bcda1f16edcf48460e1.
- A8002free after A launch; H100 missing this snapshot/cache. B 4090 developer compatibility check uses shared verified NAS; no cross-NAS copy. Full job pending check.
- B conservative inversion estimate22.13h from272s/24 developer samples across28114windows/4GPU; includes model startup, rough only.48h timeout retained. Morning completion is not promised.
- Latest sleep-time authorization: independently evaluate each completed candidate on fixed stratified development episodes against matching FP32 reference; preserve separate confirmation subset. Diagnose failures and make limited traceable fixes/retrain if needed. No diagnostic labels enter training; final4215 test remains independent. Continue latest-student round1 afterwards.
- First follow-up after2026-09-30 08:00China must report actual status/updates/coverage/pairedSR/TR/CR or explicit incomplete reason and persist report marker.
- Dashboard18784 now reads A actual updates and B full28114-window inversion denominator; diagnostic card explicitly pending until jobs exist.

### B formal submission00:12China
B59734/70596 RUNNING onbaidu_bj_4090,4RTX4090; train_b_4090.yaml invokes unchanged frozen train_b_pipeline.sh,48h timeout, full inversion and2epochs. Output /data/nas_ray/project/md-ak/users/zeying.gong/job_59734/task_70596/wla_intervention_b_round0 .
4090 developer compatibility PASS:4/4 real windows meet1cm/2degree;2optimizer updates and checkpoint epoch2 saved. Evidence /data/nas_ray/home/zeying.gong/algorithm/experiments/wla_b_4090_compat_20260930_gpu6/training/COMPLETE.json . Initial own GPU0 preflight stopped before model load because GPU was occupied; retained its output, reran on idle GPU6, no unrelated process stopped.
No cross-NAS data copy; shared paths and source hashes verified on4090. Hardware differs from A; final inference precision/protocol remain matched. B formal optimizer updates not yet verified (inversion first).
A now950/3114 real updates. Begin fixed matched development diagnosis once A completes, independently of B. Diagnostic subset and jobs not yet implemented; this remains the immediate next engineering step.
