# WA-Mobile current task

Objective: diagnose and improve JEPA-WA RGB + polar UWB mixed tracking, NO TEXT.
User authorizes overnight code fixes / retraining / evaluation without repeat confirmation.
Heartbeat wa every15min; read wa/wm/OVERNIGHT_DIAGNOSIS.md before further action.
Do not blindly extend beyond2epochs; fixed development per-task SR>=80% is provisional "most" gate.
Report CR and all failures; confirmation scenes reserved; no product/generalization claim from small sample.

Training59566/70423 SUCCEEDED:8RTX4090,22707steps,1epoch. Offline metrics.json exists.
Image ADE/FDE0.2691/0.4697m; mixed0.2576/0.4487m. Not closed-loop metrics.
Full IMAGE eval59678/70540 runs on8A800 frozen8ebbb30f; do not edit running source.
Last23:29 snapshot1879/4215; STT complete SR63.42% CR8.04%; DT partial. Check fresh status.
Small eval59674/70536 STOPPED_USER_SUPERSEDED; retained outputs.

Confirmed old closed-loop mode0/polarzero wrongly omits requested UWB; mixed interface now implemented.
Also offline zero vs closed-loop Gaussian initial flow state mismatch; causal role unverified.
Next: complete paired diagnostic96episodes (24each image/mixed x random/zero).
Plan wa/wm/mixed_diagnostic_plan_v1.json outcome-blind selection; separate24 confirmation episodes.
Sensor ideal_simulated_uwb matches training axes/time; no text/future/GT path fed to policy.
Developer10unit tests + actual mixed checkpoint2calls + actual worker-cwd imports PASS. Dependencies hash PASS.
Paired96 diagnostic Job59720/Task70582 SUBMITTED4A800 23:46:33+08; sourceefa41778 configcca17305. Check fresh state before any retry.

Repo /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/checkout branch wa.
Development entry devpod-a800; GPFS /data/nas_ray. No cross-NAS needed for current diagnosis.
Priority A800 then readyH100 then4090; eval GPU count flexible to avoid queue; training8GPU.
Architecture unchanged officialJEPA auxiliary + DINOv2 +64MetaQueries + originalActionExpert.
TensorBoard existing devpod-a800:6006 via localhost16006; don't recreate.
Backup ModelScope private150shards and H100 receiver separate; completion must be rechecked.
H100 migration historical records in archive/2026-09; no H100 training submitted by this task.
