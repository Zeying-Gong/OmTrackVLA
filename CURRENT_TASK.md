# WA-Mobile current task

Objective: diagnose and improve JEPA-WA RGB + polar UWB mixed tracking, NO TEXT.
User authorizes overnight code fixes / retraining / evaluation without repeat confirmation.
Heartbeat wa every15min; read wa/wm/OVERNIGHT_DIAGNOSIS.md before further action.
Do not blindly extend beyond2epochs; fixed development per-task SR>=80% is provisional "most" gate.
Report CR and all failures; confirmation scenes reserved; no product/generalization claim from small sample.

Training59566/70423 SUCCEEDED:8RTX4090,22707steps,1epoch. Offline metrics.json exists.
Image ADE/FDE0.2691/0.4697m; mixed0.2576/0.4487m. Not closed-loop metrics.
Full IMAGE59678/70540 SUCCEEDED02:40:39; all4215 unique episodes, all8 COMPLETE markers. wa/results/full_image_59678.json.
Full IMAGE SR STT63.42% DT18.36% AT33.52%; CR8.04/14.66/11.32%. No UWB; not mixed-mode results.
Small eval59674/70536 STOPPED_USER_SUPERSEDED; retained outputs.

Confirmed old closed-loop mode0/polarzero wrongly omits requested UWB; mixed interface now implemented.
Also offline zero vs closed-loop Gaussian initial flow state mismatch; causal role unverified.
Paired96 COMPLETE59726: image_random8/24 mixed_random14/24 image_zero7/24 mixed_zero16/24 successes; initialRGB pairs equal.
Mixed_zero taskSR STT87.5% DT75% AT37.5%; CR12.5% each. Gate unmet. No checkpoint extension yet.
Controller59737/70599 SUCCEEDED but REJECTED:14/24vs16/24, STT collisions increased. Use original learned_target_guard_v3.
Epoch2 59791/70653 SUCCEEDED06:15:01;45414steps/2epochs plus all3mode offline validation. Mixed ADE/FDE .25385/.44281m improves slightly but mixed closedloop14/24 regresses. No third epoch.
Final checkpoint SHA25639d47f885d303d491f4b97d36044f60def8f25af04ad5bf6ad07f95c85fcf728.
Epoch2 mixed24 evaluation59826/70688 SUCCEEDED05:42:21 but GATE_FAILED:14/24 vs epoch1 16/24; SR75/62.5/37.5%; CR25/12.5/12.5%; initialRGB matched; invalid0. Do not adopt as better or extend epoch3.
Next: audit action/time/control and failure-state coverage before evidence-backed correction. Turning/reverse labels are present; lack of these labels is not an established cause. Confirmation24 untouched.
Sampled128episodes/9000windows: firstXY cache vs raw reconstruction error<3e-8m; training range>=3m only49/4800 and>=5m0. Recovery coverage hypothesis needs full-distribution audit; not established causal explanation. See wa/results/coverage_audit_20260930_v1.json.
Full coverage audit COMPLETE: train726631 >=3m4031 >=4m318 >=5m12; heldout73368 >=3m510 >=4m30 >=5m0. Recovery states rare, but causal explanation unproven. Next inspect failure onset before far-range drift and determine train-only recovery data collection; no new training queued.
Queued4090 59748/70610 STOPPED02:56:50 without optimizer steps; A800 became available after full eval. Same NAS/environment/recipe, GPU-type change documented. New output job_59791/task_70653/wm_jepa_epoch2_a800_v1.
Read wa/wm/EPOCH2_CONTINUATION.md. When NEW checkpoint exists, verify complete/hash then evaluate fixed24 mixed_zero with original controller; don't accidentally use old checkpoint. No duplicate train job.
Plan wa/wm/mixed_diagnostic_plan_v1.json outcome-blind selection; separate24 confirmation episodes.
Sensor ideal_simulated_uwb matches training axes/time; no text/future/GT path fed to policy.
Developer10unit tests + actual mixed checkpoint2calls + actual worker-cwd imports PASS. Dependencies hash PASS.
Paired96 diagnostic59726/70588 SUCCEEDED00:17:26+08; archived wa/results/mixed_diagnostic_59726.json. Do not rerun.
Job59720/70582 STOPPED before worker launch due4GPU/CPU placement failure; outputs retained. Same96episodes on2GPUs, no protocol change.

Repo /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/checkout branch wa.
Development entry devpod-a800; GPFS /data/nas_ray. No cross-NAS needed for current diagnosis.
Priority A800 then readyH100 then4090; eval GPU count flexible to avoid queue; training8GPU.
Architecture unchanged officialJEPA auxiliary + DINOv2 +64MetaQueries + originalActionExpert.
TensorBoard existing devpod-a800:6006 via localhost16006; don't recreate.
Backup ModelScope private150shards and H100 receiver separate; completion must be rechecked.
H100 migration historical records in archive/2026-09; no H100 training submitted by this task.
