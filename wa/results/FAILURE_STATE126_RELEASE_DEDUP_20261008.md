# Failure-state collection release and exact deduplication — 2026-10-08

Status: **collection release PASS; exact same-key dedup PASS; cache v1 FAILED; training NOT RELEASED**.
This is an evaluation-set-adaptation experiment. None of the teacher results below is a new WA closed-loop score.

## Completed collection and independent admission

Job 61844 / Task 73066 completed successfully at 2026-10-08 11:24:43 Asia/Shanghai on 8 A800 GPUs. Its 90 unique completed search records are disjoint from the 36 previously audited records; the union exactly covers the frozen 126-case plan. The finished job must not be resubmitted.

| Quantity | Audited result |
| --- | ---: |
| Complete searches | 126 |
| Accepted original successful teacher branches | 96 |
| No valid repeated teacher recovery | 30 |
| Candidate windows | 7,396 |
| Valid windows under original numeric rules | 6,864 |
| Excluded windows | 532 |
| Accepted branches with nonempty valid windows | 90 |
| Pinned source files checked by release gate | 94,807 |

Only the original successful selected teacher branch supplies future labels. Repetition verifies eligibility but is not another demonstration. Student history before takeover can be causal input; failed student future actions and failed teachers are not labels. No physics, model, control, loss, or success threshold changed.

Release: `/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/failure_state_collection_release_126_v1.json` (23,419,823 bytes), SHA256 `73fe7d6b7cdcdfa144a5bcf540cbac33b0d2c0971b25d6aeca35e928d8af0fbd`. It explicitly retains `training_released=false`.

New-90 raw audit: `artifacts/failure_state_completed_search_audit_61844_v1.json`, SHA256 `7759d30c623c3f6a3b989c4ba0efa3554344daa17e84df662d52650efb334aa1`; 637 branches, 64,946 PNGs, 64,945 actions, 75,855 evidence files. New numeric audit: `wa/results/FAILURE_STATE_90_NUMERIC_20261008.json`, SHA256 `4d0bf78345b46b8fb36e969662f7600c48f35370eb151823829ea0c9fb42a776`; 5,357 candidates to 4,973 valid, added to the old 1,891 valid. All paths without a leading slash in this report are relative to the project root (artifacts) or checkout (wa).

## Deduplication against the best model's previous teacher data

All 6,864 valid windows were compared against **all old valid windows of the same task and episode key**, without restricting teacher, step, or time. Equality uses the actual ten consumed input/label tensor components, exact shape and bytes; not approximate numeric closeness or only pose labels.

| Outcome | Windows |
| --- | ---: |
| Exact previous-data duplicates | 526 |
| Of these: from-start takeover | 182 |
| Of these: mid-trajectory takeover | 344 |
| Different non-image components | 5,257 |
| Same non-image components but different image context | 168 |
| No old matching task/key | 913 |
| Eligible for additional exposure within this comparison scope | 6,338 |

Full NAS report: `/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/checkout/wa/results/FAILURE_STATE_DEDUP_20261008.json` (10,634,245 bytes), SHA256 `599d60ea2d40ff4f9ae69f87df03e3a5a192b749ce030504bf19177999262859`. It verified 1,900 consumed PNGs, 1,104 decoded tensors and 2,565 source pins. This is **not global cross-key or new-versus-new deduplication**. All 6,864 valid rows remain evidence; 526 duplicates should receive no additional recovery exposure. The report pins the v1 converter helper (`5bdc9fd3fe0fef78b3250081596b7999c557b0fec9a3c80aed2ebee2072aaea7`), which must remain unchanged.

## Preserved conversion failure and next gate

`artifacts/failure_state_se2_cache_20261008_v1` failed with `incomplete original evidence`, exit 1, before arrays/admission were written. The failed directory retains `failed_conversion.json` and an empty index directory. The converter and synthetic fixture expected nonexistent `first_start_pair.json` and `takeover_pair.json`. Actual producer evidence is **`first_start.json`, `pair_start.json`, `takeover.json`**; no source evidence was lost. Independently checking all 96 original branches found all three pinned files, first-start/pair-start equality and correct takeover step. Preserve v1 and its failed record; use a separate converter v2 and fresh output directory for the repair, not another rollout.

The pure sampling candidate preserves all 1,184,272 old executed positions (the old DDP omission is position 1,151,263), then proposes 49,152 added recovery exposures, maximum 16/window and 1,024/episode. The 6,338 eligible windows span 87 nonempty added-data episodes and have capacity 59,792. This is feasibility, **not a released training plan**. It balances extra early/late takeover-relative exposure after including each eligible window once. Old per-window counts are retained, not ordering, random augmentation, rank assignment, or the stepwise optimizer trajectory.

Required next: v2 real-cache conversion; independent hash-pinned loader audit including true timestamps, causal history, teacher-owned future labels and unchanged heldout; exact raw-row-to-loader-row/dedup binding; real eight-rank exposure verification; training integration and short developer checks. No new training or student SR has been produced. Retained best remains STT 1279/1405, DT 1178/1405, AT 1207/1405. Any next training branches independently from 59866 model **and optimizer**, for at most one new epoch/cumulative two, not a third epoch from 61609.

Original failures, successful collection artifacts, frozen source and checkpoint are preserved. Large raw audit inventories, images, checkpoints and videos remain on NAS; this compact report is the GitHub evidence index.
