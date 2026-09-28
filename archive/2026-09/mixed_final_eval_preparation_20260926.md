> Archived DA3 evidence. All earlier continuation/retraining instructions and live status claims are superseded by the 2026-09-28 retirement decision; Job58613 is STOPPED. See docs/tracking_route_decision.md.

# Final mixed-checkpoint EVT paired-ten preparation

Status: PREPARED_WAITING_FINAL_CHECKPOINT; no evaluation GPU job submitted.

- Root: /data/nas_ray/home/zeying.gong/datasets/da3_mixed_eval_20260926_v1.
- Frozen code335files; source.sha256 SHA2562f3c9c26a4cedfddb643ab02e354a9fb9f5519a38450ee6d9f9bf6c58eb28ec2. Copies exact training policy model files plus separately audited controller v2/agent. Running training snapshot untouched.
- Lightweight Habitat/EVT Python+config code frozen; large models/runtime packages/data stay explicit external NAS dependencies, not copied. Worker path provenance in snapshot.json.
- Python imports from snapshot verified: trained_agent/model/habitat. Actual STT validation1405episode definition SHA2568a96fe3be38ab78b0b8985e71645eef337d396be126bbd78ce1c87ffef5a0c63.
- Only IDs119/25/297/136/130/291/16/40/22/273, scene2n8kARJN3HM, seed7. CPU dataset lookup confirms exactly one match per ID. No scene simulator or GPU run performed.
- Config ctrl_freq40 andac_freq_ratio4. Actual runtime physics timestep and world-clock agreement remain to be checked by v2duringrollout; no hardcoded.048s substitute.
- Both DA3 and official baseline will run anew; strict official model loading; common config andinitialRGBmatching enforced byexistingrun_evt_ten/summarize_evt_ten. No reuseofoldofficialresult. DA3 hasinitialbbox+missingUWB,officialtext;reportinputdifference.
- CPU preflight inputs.json andruntime_config.json underroot/preflight. YAML parserandbash-nPASS. seal_mixed_eval_checkpoint.py --verify-ready returnsWAITING_FOR_FINAL_TRAINING exit3asexpectedwhiletrainingisunfinished.
- New canonical scripts: prepare_mixed_eval_snapshot.py; seal_mixed_eval_checkpoint.py; baidu_a800_mixed_final_paired_ten.yaml. tools directoryunderrootcontainsadmission/YAML;code manifestdoesnotinclude tools. Freeze/recordtools hashesbeforeformal submission.
- Required finalcheckpoint: /data/nas_ray/project/md-ak/users/zeying.gong/job_58085/task_68820/da3_sage_evt_mixed/training/epoch_2.pt;requiresTRAINING_COMPLETE679878steps2epochs4ranksandstrictCPUmodelstate load. Never substitute intermediatecheckpoint.
- Futureproposedjob1A800K8s14400s runsbothmethodsthenpairedsummary;not submitted. Admissionreceiptmustexistandmatchsource/validation/checkpoint/training-completion hashes. Refreshresources/dependencyprovenanceandshowfullsubmissiondetailsfirst.
- Afterpairedmetrics: diagnoseSR/TR/CRandrolloutsbeforeconditionalEVTTRAIN-ONLYexpertcorrection/DAgger. No validationtrajectory/label aggregation. Noautomatic1405episodebenchmark.

## Live pre-submission recheck, 2026-09-28 01:07-01:10 Asia/Shanghai

- Same Baidu NAS mounted with196T available; md_ai_kit2.0.0 login valid; submit CLI accepts one config path. BeijingA80011free at this observation (not a reservation); intended1GPU paired evaluation unchanged.
- All335 frozen source hashes revalidated; YAML safe-load and bash-n pass; official checkpoint/Qwen/SigLIP/runtime/torch overlay/Xvfb bundle paths exist. Prior CPU import/config/ID tests remain separately recorded above.
- Exact task YAML SHA256:6d23af373369f65ed79b32234361484a71a43112e0b53c0d85867a7c6aabb63d; admission script SHA256:81d72964d05f784ff494584b8a08275095a6a883fcd3abb815a7954b9801a8ca. Both will be rechecked before submit; no tool edits made.
- Canonical base commit9ad4bd1554f97e3a43913bdcd3d4f30af797f6b7; existing uncommitted project changes preserved. No new commit or cross-NAS data migration.
- Job58085 still RUNNING; latest observed674425/679878epoch2. Final checkpoint and completion receipt still pending; no evaluation submitted and no SR/TR/CR result claimed.

## First full paired submission: Job58558 / Task69305

- Submitted2026-09-28 01:39:02Asia;RUNNING01:39:07;FAILED01:39:56. 1actualA800-SXM4-80GB driver550.163.01. Output /data/nas_ray/project/md-ak/users/zeying.gong/job_58558/task_69305/mixed_final_paired_ten. Runtime-v2 YAML SHAa73bd2dff45e7758813bf1379a8b9981f9a83b1cc472bd2d82c58757564af70a. CPUstrict final load passed with isolated dependencies;GPU finalmodel load passed. Initial missing-addict preflight failure retained in this history;32dependencyfiles frozenSHAe8c8e48c83099b5b9d942a13c3810e91b18b5bfb986523508f702a139b24c705.
- Actual failure: track_task.py opens cwd humanoid_infos.json, absent from frozen code. Scene initialized but task creation failed before anyrollout;no valid metrics or scene-clock evidence. Official not started. Preserve all logs/config;repair missing external cwd assets without changing frozen335source, then rerun complete paired workload. No validation training or protocol changes.
