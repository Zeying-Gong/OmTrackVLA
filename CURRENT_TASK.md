# WA current task — STT target and UWB study
Updated: 2026-10-08 14:32 Beijing. Existing Goal ACTIVE after user resume; heartbeat wa ACTIVE every20minutes.
Full previous task: archive/2026-10/CURRENT_TASK_before_failure90_terminal_20261008.md.

## Acceptance and retained best
- Same1405/task: STT>=1289 (>91.7%), DT>=1173, AT>=1203; then same-weight with/without UWB.
- Best61609: STT1279 (91.032028%), DT1178 (83.843416%), AT1207 (85.907473%);3664/4215;invalid0. STT gap10; goal NOT_MET.
- 61715:1277/1166/1207;3650/4215;invalid0;not promoted. All complete results and failures preserved.
- Explicit evaluation-set adaptation; not unseen-test generalization. WA and LightNav have different inputs.
- RGB + episode0 GTBBox template + current ideal polarUWB; no text, later GTBBox or future target path. JEPA training auxiliary only.

## Current failure-state collection — finished, do not resubmit
- Job61844/Task73066 SUCCEEDED at2026-10-08 11:24:43 Beijing;8actualA800-SXM4-80GB.
- Output:/data/nas_ray/project/md-ak/users/zeying.gong/job_61844/task_73066/wa_failure_state_collect_a800_v3
- Root COLLECTION_PROCESSES_COMPLETE_NOT_TRAINING_RELEASE;8COMPLETE;90unique records;lanes12/13/8/13/12/9/12/11.
- 90keys exactly match frozen remaining lanes, disjoint from old36;126total coverage. New72 candidates +18no_valid;combined96candidates +30no_valid. These are collection outcomes, not model SR.
- Source:/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/source_failure_state_collect_v3;commit726544449a5e7ad10c7d246b2bf302d783210c7e unchanged.
- Config wa/jobs/failure_state_collect_a800_v3.yaml SHA667ab19aa192802eb8029326730ec46a089a4094b17d12ff9d8d7f6e01ab4218.
- Baseplan artifacts/failure_state_plan_61609_v1.json SHA2eff9e83e89008ce1ee73632def406a27131fece6e77b01f6d4ca8c02624292b.
- Frozen36+90 overlay artifacts/failure_state_continuation_61836_v1.json SHAe5fe2986517c171fd93f507ebe0f6d9937c420d3c37efd3fe7ff391899444641;never overwrite.
- All24 owned role spawn/exit PIDs, wrapper launcher0/dependency0, frozen source clean,8ready c510/step59716 and checkpoint SHA verified by new terminal gate.
- One recorded typed LN error VLzqgDo317F/123 k34 MissingRVQ[0,1];60obs59actions;null terminal/0windows/no fallback executed. Do not interpret typed errors as successful demonstrations or claim all logs warning-free.

## Completed raw and numerical evidence; full126 admission pending
- Old61833 FAILED1 +61836 FAILED35 are retained. Old36 full raw audit PASS225branches/14985PNG/14984actions/18836files.
- Old raw:artifacts/failure_state_completed_search_audit_61836_v1.json SHAcbdc709aa4f8df67fb5edf2e06faeb2476f77de365ec9954c8e5b323ed957299.
- Old numeric:wa/results/FAILURE_STATE_36_NUMERIC_20261008.json SHA2965e8bd086169eb3c180dd92db72254c2d420d445d10f7586fc6e4be850340b;2039candidate ->1891valid/148excluded;21nonzeroepisodes.
- New90 numeric:wa/results/FAILURE_STATE_90_NUMERIC_20261008.json SHA4d0bf78345b46b8fb36e969662f7600c48f35370eb151823829ea0c9fb42a776;5357candidate ->4973valid/384excluded;69nonzeroepisodes.
- New72winner episodes:LN20 with1531valid;Oracle52 with3442valid;3zero-valid Oracle episodes.376transition-filter exclusions +8current<4.
- Combined numerical arithmetic:7396candidate ->6864valid/532excluded across90nonzeroepisodes. Repeat windows0;student future-action labels0;past student context allowed.
- New90 all5357 individual-window rederivations PASS;old-runtime effective-mask differences0;endpoint differences0;SE2/yawdelta0;17CPUtests PASS. Numeric status is NOT training admission.
- New90 full raw audit and readback PASS_NONRELEASE:637branches/64946PNG-observations/64945actions/75855files rehashed;52Oracle20LightNav,70mid2start;24ownedprocess exits matched;5fatal literals0 in checked24logs, warnings retained.
- Raw artifacts/failure_state_completed_search_audit_61844_v1.json SHA7759d30c623c3f6a3b989c4ba0efa3554344daa17e84df662d52650efb334aa1;provenance SHA2a27d5198f35feb95a670b3cbcd8be629eb03e634d17ddf82939318572c97dd7. Missing-final-l2 category not observed in this90.
- Main independently reran135CPUtests across raw/numeric/release/converter/loader/dedup. Real126 serial gate stopped only its own CPU PID3894656 to accelerate same hashes to4threads;no output/release existed, noGPU touched. Parallel-gate53tests then PASS0.742s;sole4CPU runtime PID3904735/session91363 active,37of126checked at14:40 snapshot;notrelease.
- No126 release/cache/real-loader admission/newtraining/newstudentSR yet. At least182/6864valid windows exactly duplicate old demonstration inputs/labels;remaining6682 not yet fully deduplicated. Do not call6864 unique additions.

## Next gates and authorized experiment scope
- Execute full126 independent release using finished90+frozen36 proofs and exact ownership/selected-original/repeat checks;do not redo allPNG decode. After true release SHA, run cache conversion and full6864 same-key old-data equivalence audit.
- Release rederives exact有效window sets for all96winners;old36 compact report remains unchanged. Exclude failed/repeat/typed-error labels;zero-valid episodes get0exposure.
- Converter preserves episode0 template and original indices/all frames;teacher-owned interpolated0.1..0.7s SE2;policy sparse4 causal history;JEPA continuous4/action/proprio-1/actualt+1.
- Preserve old now>=4/dt/displacement/yaw/badtransition masks and compare exact original-runtime mask. UWB current observation only.
- After source/image/array/index/admission hashes and actual loader checks:independent59866 MODEL+OPTIMIZER step22707 branch,8GPU,at most1new/cumulative2epochs.
- Never continue61609/61715 epoch3;no loss/LR/controller/physics/architecture/criterion changes orRL. Record unique windows, group fitting, new/old exposure and actual8rank consumption.
- Full evaluations STT/DT/AT each8GPU/1405,24total;user permits needed24bj4090. Check A800/H100/4090 priority and actual target NAS/environment/worker first.
- Only after all3SR gates:explicitimage-mode sameweight noUWB paired study with real interface/observer checks. Not mixedzero or noUWB-retraining claim.

## Checkpoints, completed comparisons and services
- Best61609 checkpoint:/data/nas_ray/project/md-ak/users/zeying.gong/job_61609/task_72803/wa_hard_stt_train_a800_v1/checkpoint.pt
- SHA c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52;step59716;mixedzero sampling4 seed7+step.
- Parent59866 SHAab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d;step22707.
- 61609 trained37009new updates with1184272actualexposures=726631base+457641teacher;8ranks148034each;independent1new/cumulative2epochs. Complete report STUDENT61609_FINAL_20261007.json.
- 61715 checkpoint:/data/nas_ray/project/md-ak/users/zeying.gong/job_61715/task_72909/wa_stt_anchor_train_a800_v1/checkpoint.pt;SHA9631778c81992a349f8315b773f763217a5f387b5206b1bfb97b323839030a6a.
- 61715 full4215/24COMPLETE/audits preserved in STUDENT61715_FINAL_20261008.json;no promotion or duplicate evaluation.
- LightNav1273/1128/944;Oracle1281/1210/1226;different-input systems. Product/generalization claims not established.
- Prior review18799(best61609)/18800(61715) ondevpod-4090;TB6006 ondevpod-a800. Not revalidated in current audit turn;not new collection pages.
- Prior forwards:ssh -N -L 18799:127.0.0.1:18799 -L 18800:127.0.0.1:18800 devpod-4090 ;ssh -N -L 16006:127.0.0.1:6006 devpod-a800.
- TR reference-normalized;macroTR separate;CR means target-human distance ever<0.5m,not obstacle contact.

## Network, repository and reporting
- This turn devpod-4090 restored;hostname devpod-zeying-gong-4090-7f59ffb47c-rpkjh;GPFS mt-tutz0R-pfs-haZOBP;no crossNAS copy.
- Prior No-route and kex failures preserved;earlier BLOCKED Goal explicitly resumed by user and now ACTIVE. No automatic BLOCKED-to-ACTIVE claim.
- GitHubwa backup0d9fe65b7007617493c35da0d52038a54325bf80 verified;newloader/dedup/real-loader tool and parallel-gate changes remain this turn's work. Raw90 final NAS report is not uploaded as data.
- Never apply old local startup31/RUNNING patch;61844 is terminal. No resubmit/restart/stop/migration of61844.
- ExistingGitHub wa only;announce beforepush;no weights/data/videos. CURRENT<=80,PROGRESS<=150;archive before expanding beyond limits.
