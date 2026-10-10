# Student62256 image-failure visibility diagnostic

Status: `STORED_OBSERVER_ONLY_DIAGNOSTIC_NOT_CAUSAL_NOT_TRAINING_RELEASE`. This is a read-only analysis of completed, same-checkpoint image and mixed evaluations. It did not run another rollout, alter the policy, or use observer state as a policy input.

## Inputs and selection

- Project root: `/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928`.
- Paired comparison: `artifacts/student62256_uwb_comparison_20261010_v1/comparison.json`, SHA256 `785e3799db25bca2699edc9d740ca5204b615585da023ec3ecd21eef1d1344a4`, status `PASS_STORED_EVIDENCE_COMPARISON_ONLY` and `ablation_release=false`.
- Image episode rows: `artifacts/student62256_image_full_audit_20261010_v1/combined_episodes.jsonl`, SHA256 `d01d36417c29801a211b213d3d232727b0c08da5d9bca1a05653c1f2fefea0ef`.
- The compared model is the same checkpoint SHA256 `40915b366ee5a2ef5967e2ce49a94d2b0f45f149955dd5cccf85ac3d6ab178fc`, step 61252. For each task, select the comparator's `paired_keys.mixed_only_success`: the mixed rollout succeeded and the paired image rollout failed. These are subsets of the complete 1405 episodes per task, not new denominator-adjusted SR estimates.
- For every selected image row, read its `artifact_root/<task>/_review/<scene>/<episode>/steps.jsonl`; `scene/episode` comes from that row's key. The stored review recorder writes pre-action semantic target visibility for offline observation only. All selected episode rows and every per-step visibility field were present.

## Exact counting rule and results

For each selected episode, use the integer `target_visibility.target_pixels` in each stored pre-action observer record. “Any zero” means at least one record has zero target pixels; “>25% zero” means strictly more than one quarter of that episode's records have zero target pixels; “last zero” means the final stored **pre-action** record has zero target pixels. Status counts come from the image episode result; `Normal` can still be an unsuccessful episode. Counts are per episode except “observer steps.”

| Task | Mixed-success / image-failure episodes | Observer steps read | Image status Normal / Lost / Collision | Any zero-pixel frame | >25% zero-pixel frames | Last pre-action frame zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| STT | 182 | 19,235 | 89 / 44 / 49 | 102 | 64 | 77 |
| DT | 739 | 57,262 | 188 / 429 / 122 | 638 | 412 | 473 |
| AT | 527 | 48,692 | 152 / 296 / 79 | 147 | 40 | 75 |

The three status counts sum to each row's selected-episode denominator. The diagnostic confirms no missing selected review telemetry (0 of 1,448 episodes missing). The last-frame observation is not a post-terminal simulator state. Zero semantic pixels can mean out-of-view or occluded target, or a semantic-label limitation; it is not proof of the model's internal belief or the cause of failure. DT has frequent zero-visibility intervals among these paired regressions, while many AT `Lost` outcomes occur without long zero-visibility intervals. This is an association to investigate with the paired videos and action traces, not evidence that adding frames, fine-tuning, or RL will improve SR.

The target position, semantic pixels and third-person view used by the review recorder are privileged **observer-only** diagnostic data. They were not sent in the prediction RPC, not supplied to the student policy, and must not become online inputs or future-action labels. The current image goal remains RGB plus the episode-0 bounding-box template, without UWB or language. This report does not replace the full 4215-pair audit, video authenticity checks, or an independently audited new training/evaluation result.
