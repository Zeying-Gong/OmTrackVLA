# WA full mixed closed-loop validation — 2026-10-04

Status: COMPLETE_FULL_VALIDATION_AUDITED. Overall research: PARTIAL.

60989/71912 SUCCEEDED 2026-10-04 12:45:43 Beijing; submitted09:16:57, actual8 NVIDIA A800-SXM4-80GB.
Original60885/71808 intentionally stopped09:14:27 for authorized acceleration, not model failure.
Combined frozen2238 + continuation1977 =4215 unique(task,key); eight new COMPLETE files, exact manifest coverage, each class1405.

## Results

| Task | Success / 1405 | SR % | TR % | HumanCollision / 1405 | CR % | Invalid init |
|---|---:|---:|---:|---:|---:|---:|
| STT | 1204 | 85.693950 | 81.424200 | 71 | 5.053381 | 57 |
| DT | 1092 | 77.722420 | 74.058413 | 102 | 7.259786 | 57 |
| AT | 1124 | 80.000000 | 79.949682 | 79 | 5.622776 | 51 |

Overall success3420/4215 =81.138790%; HumanCollision252/4215 =5.978648%; invalid165/4215 remain in denominator.
DT SR remains below80%; this does not satisfy an all-three-task80% criterion.
TR =100*sum(following_step)/sum(max(total_step,reference_steps.get(key,0))).
Each task has52missing reference keys; existing implementation falls back to actual steps for them. This limitation is preserved, not silently corrected.
macro_TR (mean episode following_rate) is a different statistic: STT85.819082%,DT75.752646%,AT82.341487%.
CR means target-person distance ever<0.5m, not doorframe/wall/general obstacle collision.

## Model and protocol

RGB + first-frame BBox + ideal simulated polar UWB, noise0 delay0; no text.
60502/71381 checkpoint step45900; SHA20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331.
JEPA/MetaQuery/ActionExpert unchanged; world predictor training auxiliary only, no online MPC.
mixed/zero/seed7/learned_yaw_guard_v1; sampling_steps4; original physics and success criteria.
Existing full validation includes previous development/confirmation; NOT untouched test.
No full LightNav comparator in this run; small24 WA20vsLN17 is not full-set superiority and uses different inputs.
No real UWB, sensor noise/delay, real robot or edge-latency validation. No new training or tuning.

## Audit and provenance

Independent recomputation matched scheduler summary. No duplicates/new-old overlap; frozen parent hashes unchanged.
All8server_ready checkpoint/input contracts passed; all4215 initial JPEG hashes matched recorded hashes.
All4215 videos exist with positive duration and valid ffprobe video metadata; not a full-frame decode audit.
No fatal/OOM/Traceback in8worker logs; scheduler log contains FULL_MIXED_EVALUATION_COMPLETE.
Audit combined SHA: d2d69547ea47aab503bbac7807afcc60e46a9ead4bd61c3b4fd695e202435aad
Audit summary SHA: beee6bc5a012309031886eb70800d4f8cb4185d76c971736fa6d2be9d99ecb54
Source source_full_mixed_resume_8gpu_v1 commit5dcbeef5350dc55c345fa7688ec6c2b4ed422e35.
ConfigSHA1d12415f9f2ecf90f3a31d3b0e80d2cfd099be10d1b1dd7ff3d7c0cffa5a8196.
PlanSHA66bb0c34f3d44b0a238911e21b21a29502cc85d7076e9a59b2f9048aef1ed33b.

## Artifacts and reproduction

Combined JSONL and summary:
 /data/nas_ray/project/md-ak/users/zeying.gong/job_60989/task_71912/wa_full_mixed_resume_8gpu_v1/
Parent videos:
 /data/nas_ray/project/md-ak/users/zeying.gong/job_60885/task_71808/wa_full_mixed_learned_yaw_v2/
Each combined row retains artifact_root. Interrupted old partial recordings are not included.
HTML, audit.json, episodes.json:
 /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/full_mixed_review_60989
Audit/page generator: wa/tools/build_full_mixed_review_60989.py (CPU; exclusive new output directory, do not rerun over existing output).
Browser verified4215 records andATinvalid51 filter; videos linked directly to both NAS roots.
Forward: ssh -N -L 18794:127.0.0.1:18794 devpod-a800
Open http://127.0.0.1:18794/ . Server bound127.0.0.1 only.
Monitorwa PAUSED after completion/audit; no new GPU work submitted.

## Post-audit caveat (2026-10-04)
Structural/data-integrity audit remains valid, but a semantic-rendering/initialization validity issue was subsequently identified. See INIT_BBOX_DIAGNOSIS_20261004.md. Original metrics remain unchanged; do not interpret all165 invalid starts as ordinary model tracking failures. Exact renderer/asset mechanism remains unresolved.
