# WA / LightNav independent confirmation — 2026-10-03
Status: COMPLETE_COMPARATOR_PHASE; broader research/product requirements remain PARTIAL / UNVERIFIED.

## Outcome
| Task | WA success | LightNav success | WA collision | LightNav collision | Invalid starts WA/LN |
|---|---:|---:|---:|---:|---:|
| STT | 6/8 | 7/8 | 1/8 | 1/8 | 0/0 |
| DT | 7/8 | 5/8 | 0/8 | 0/8 | 0/0 |
| AT | 7/8 | 5/8 | 1/8 | 2/8 | 0/0 |
| Total | 20/24 (83.33%) | 17/24 (70.83%) | 2/24 (8.33%) | 3/24 (12.50%) | 0/0 |

All selected episodes are in the denominator. Difference: +3 successes (+12.5 percentage points) and one fewer benchmark collision. WA is ahead overall on this set, but STT is worse. The prior provisional each-task80% threshold is NOT met. This is not evidence of statistically established superiority.

Paired outcomes: WA-only success4, LN-only success1, both-success16, both-fail3.
- WA-only: DT GLAQ4DNUx5U/20; DT JmbYfDe2QKZ/10; AT GLAQ4DNUx5U/20; AT cvZr5TUy5C5/3.
- LN-only: STT GLAQ4DNUx5U/20.
- Both-fail: SUHsP6z2gcJ/13 under STT/DT/AT.

## Jobs, weights and source
- WA60770/71649 SUCCEEDED at2026-10-03 11:48:40 Beijing; 2A800.
- LN60771/71650 SUCCEEDED at11:49:56; 2A800.
- Frozen source: source_yaw_confirmation_v1 commit0a12e09aa249b6bcfac5e56ea4b5c82152d49b4b, clean during independent audit.
- WA config: wa/jobs/learned_yaw_confirmation_a800_v1.yaml, SHA4b98c7bf39e3cb0c70fb439f446dab4a8431de36326365f26a0e4ada3e7f1a49.
- LN config: wa/jobs/lightnav_confirmation_a800_v1.yaml, SHAd8313f0b6c53cef2c716d3cba2e5d29c8a99574bd2cf4d806946b540383d1ce5.
- WA checkpoint: /data/nas_ray/project/md-ak/users/zeying.gong/job_60502/task_71381/wa_recovery_mix_a800_v1/checkpoint.pt.
- CheckpointSHA20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331; step45900,cumulative2epochs. Same as development60767 and pre-fix60509.
- Ready markers and scheduler's origin_job_config.yaml independently agree. No extra training, weight selection or tuning after this confirmation.
- WA root: /data/nas_ray/project/md-ak/users/zeying.gong/job_60770/task_71649/wa_learned_yaw_confirmation_v1.
- LN root: /data/nas_ray/project/md-ak/users/zeying.gong/job_60771/task_71650/wa_lightnav_confirmation_v1/lightnav.

## Diagnosis and what changed
Legacy near-target heading guard overwrote learned yaw completely at range<=1.5m and attenuated it at1.5..3m. Previous development had572/1987 full overwrites. Retaining the model's clipped yaw while keeping legacy x/y guard fixed improved the same-weight development24 from15 to17 successes, two doorway STT/DT gains and no regressions; collisions stayed3. Development LightNav also17/24 with3collisions.
The present confirmation uses that frozen correction learned_yaw_guard_v1. It does not add angular gain, privileged obstacle geometry, RL, a new loss or a new backbone.
JEPA/MetaQuery/ActionExpert remain; JEPA latent predictor is used as a training auxiliary, not inference-time MPC.
Earlier44teacher-window fitting audit improved ADE.56266->.42632; this suggests teacher learning, not proof that recovery data alone caused gains (there was also original-data training).

## Pairing, isolation and integrity
- Plan wa/wm/mixed_diagnostic_plan_v1.json SHAe7d9b8597435f9e7b3208a7f74abe092e66c83fb94b3c5ad1a82e0e750cf61dd was first committed in efa41778 on2026-09-29 15:45:53UTC, before this candidate.
- Exactly the24 predeclared keys (8per task) occurred in each method; no missing/duplicate/cherry-picked episodes; seed7.
- 8confirmation scenes have zero intersection with8development scenes, originaltrain625scenes/9657episodes, recovery11scenes, and collection91scenes/96episodes.
- This confirmation is reserved from this iteration's training/tuning, not an untouched external test: the prior full image-only evaluation covered the existing validation benchmark.
- Saved initial JPEG hashes independently recomputed and match for24pairs. This does not prove equality of all hidden simulation state or subsequent trajectories.
- Four lane COMPLETE markers12each/splitconfirmation and both root completions validated. WA summary baseline=null alone is not cross-method pairing evidence.
- 48video files/header probes validated: H264,768x432,20fps; container frame counts agree with complete markers and steps (WA2824,LN2425). Full pixel-by-pixel video decoding was not performed.
- 2824WA logged actions satisfy exact finalXY==legacyXY and finalyaw==clippedrawyaw; 2689yaw outputs differ from legacy. Independent CPU recomputation maxerror6.84e-13 passes1e-12 tolerance; logs themselves match exactly.
- LightNav fallback0; allinitializations valid.
- Observer-only visibility coverage WA2824/2824,LN2425/2425, pre-action only, no finalpost-action claim. Observer state is not model input.
- Shared semantic-descriptor load warnings (21per method group) and X11 listener warning in both shard1 remain in logs. No Traceback, RuntimeError or OOM detected; complete artifacts verified. Shared semantic warnings and occasional full-frame pixel counts limit visual-causality claims; completion alone does not prove semantic rendering error absent.

## Yaw and residual failures
Normalized action statistics (not physical rad/s, not safety certification):
| Statistic | WA | LightNav |
|---|---:|---:|
| Absolute yaw variation per within-episode transition | .0642878 | .2217721 |
| Active-sign changes per100steps (abs(yaw)>.02) | 4.21388 | 18.02062 |
| Saturation abs(yaw)>=.99 | 5/2824 (.177%) | 304/2425 (12.536%) |

No artificial transition is formed across episode boundaries. Different rollout states/durations limit causal comparison.
STT GLAQ4DNUx5U/20 is the remaining WA-only failure: steps24..64 (1.168..3.112s) remain at<=3000targetpixels; 26measured intervals have commanded translation>3cm but measured displacement<25%. LN has0 such intervals and succeeds. Supports visibility loss with obstructed motion; does not identify exact doorframe geometry or establish a simple yaw-only cause.
SUHsP6z2gcJ/13 fails for both methods: STT/ATcollision, DTlost. A failed teacher rollout remains ineligible for positive demonstration labels.
HumanCollision is target-person base distance ever<0.5m, latched historically (OmTrackVLA-da3-polar-20260924/evt_bench/additional_metric.py:151-157; DistanceToLeader:184-190). It is not a general obstacle contact detector; these fractions are not product static/dynamic collision rates. SUHsP6z2gcJ STT/AT have0 strongly blocked-motion intervals: recorded retreat was executed while target inward radial motion was greater. Do not relabel this as doorframe-blocked retreat.

## Modality and scope limits
WA: RGB + initialBBox + ideal simulated polar UWB (noise0,delay0), no text. LN: RGB + text. This is a different-input system comparison, not equal-sensor pure-vision fairness.
24episodes reuse8scenes across3tasks, one seed. STT gap remains; no universal/generalization/safety/real-UWB/Thor<=200ms claim. ProductSR>=90% remains unmet.
Do not train on either development or confirmation trajectories. Do not tune repeatedly to this confirmation and relabel it untouched. Further changes need train-scene diagnostics plus fresh independent validation.
Capped recovery sampling plan971unique/15536exposures (4..32each) is prepared but unused; no blindepoch3, LRincrease, RL or newarchitecture. If future evidence justifies training, start independent parent59866 model+optimizer branch on8GPU, cumulative<=2epochs.

## Reproduction and review
Audit: python3 -m wa.tools.compare_confirmation --output <new-result.json> (refuses overwrite).
JSON: wa/results/confirmation_60770_60771.json; residual details: wa/results/confirmation_failure_audit_60770.json.
Video generator: wa/tools/build_confirmation_review.py --wa <WA-root> --lightnav <LN-root> --output <new-NAS-folder>.
NAS page: /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/closedloop_confirmation_60770.
48links HTTP200; page rendered with realvideo firstframe and filter behavior verified in browser. Mobilewidth stacks columns; larger viewports show side-by-side.
Forward: ssh -N -L 18793:127.0.0.1:18793 -L 18792:127.0.0.1:18792 -L 16006:127.0.0.1:6006 devpod-a800
Confirmation: http://127.0.0.1:18793/ ; development: http://127.0.0.1:18792/ ; TensorBoard: http://127.0.0.1:16006/ (recovery_mix_60502 final45900).
Phase fulfills overall same-set LightNav comparator and collision-not-worse condition. The wa phase monitor has been PAUSED through the app tool and local configuration rechecked. Final Git sync completes this handoff; broader research is not finished.
