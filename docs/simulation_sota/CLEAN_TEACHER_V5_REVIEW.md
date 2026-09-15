# V5 independent CPU acceptance

The full frozen V5 run and its unchanged **final success = 1** are independently verified. Sustained teacher quality remains unqualified: following occupies 43/123 post-action observations, visibility is worse than V4, and both longest following and invisible runs are worse. No optimizer, formal-training, teacher-quality, or deployed-search admission is granted.

## Seals and execution

- Supervisor SHA256: `516d9b45d9388826f33aece33a4202dab46272251a42e3dc9a962d96a37f6de3`.
- Plan file SHA256: `34de3ad63a4c12bfbb18061d476583bae7431fc1812fbe7b738a16b350060ecd`.
- 123 issued/returned actions, 124 observations; natural terminal, worker exited successfully and was reaped. Only then were run files read for audit.
- All 262 sealed files and 122,148,188 bytes match SHA256, size and exact file set. All RGB/panoptic labels, causal input boundaries, initialization and action/frame/time/state alignments pass.
- All audits were CPU only. No Simulator, renderer, GPU run, recollection or frozen evidence mutation occurred. Large journals remain remote.

## Control and navigation verification

The guard was independently reimplemented, without importing producer guard code. All 3,379 candidates, path queries, corrections, scores and selections match; 106 nonnavigable candidates were rejected and 12 actions modified. All 124 actual robot observations are navigable and have a route to the current target, on island pair 0/0. Every selected prediction has exactly zero base-position residual; maximum yaw residual is 8.00414e-7 rad. There are no missing approach waypoints.

The new area feedback and qualified reframe history were independently reimplemented, without importing producer area code. All 123 current raw target observations, goals, eligibility reasons, counters, counter resets, effective thresholds, original V6 branch/request, saved before/after controller states, overlay requests and final issued requests match; maximum request difference is zero. The control goal never falls below 1.30 m.

There are 44 untruncated low-area-but-far action observations, 42 with area above the low threshold, 31 invisible and six qualifying border-truncated observations. Maximum qualifying streak is two; resets occur at actions 20, 60 and 83. No reframe occurs in this real run. Positive clipped and close reframe behavior was established separately in frozen CPU counterexamples; absence in this run is not evidence that the branch is removed.

The first area audit had a leftover 1.2 m filter in a diagnostic summary field; all actual feedback/control assertions already used 1.30 m and passed. The original audit remains preserved. `area_v2_verification.json` is the authoritative repeated audit with that summary filter corrected to 1.30 m; its request reconstruction also passes exactly.

All 124 actual target positions and world clocks match V4 exactly, as do initial robot/RGB/panoptic states. The first robot-position divergence is frame 25. No recorded target-motion or timing confound appears in this paired case.

## Terminal success and sustained following are separate

The actual pinned MainHumanoidDetector method was replayed on every raw frame. The strict area gate is unchanged: 10,000 < target pixels < 44,236.8, followed by distance <=3 m. Main target semantic ID is correctly 1060; no ID or metric-frame mismatch exists. The original HumanFollowingSuccess methods were separately replayed on all 124 observations and also match exactly: stop-called, current following and distance in [1,3] m are required. Only terminal frame 123 has success=1.

| Measure | V4 | V5 |
|---|---:|---:|
| Final success | 0 | 1 |
| Final target pixels | 8,107 | 18,272 |
| Final range | 1.795 m | 1.30004 m |
| Following post-action samples | 26/123 (21.14%) | 43/123 (34.96%) |
| Visible observations | 99/124 | 93/124 |
| Number of following runs | 6 | 7 |
| Longest following run | 20 samples | 17 samples |
| Longest run first-to-last sampled span | 0.928 s | 0.784 s |
| Longest invisible run | 12 samples | 16 samples |
| Following after actual target movement | 20/95 (21.05%) | 29/95 (30.53%) |
| Following after stationary target transition | 6/28 | 14/28 |
| Following during final eight wait observations | 0/8 | 8/8 |

Both actual world-observation windows are 5.984 s. Assigning each post-action sample to its preceding interval gives V5 following coverage 2.104 s / 35.16%, versus V4 1.264 s / 21.12%. Using each pre-action sample on its following interval gives V5 2.064 s. These are explicit discrete sampling conventions, not measurements of continuous between-frame tracking or hardware speed. A target-moving transition means actual target displacement >0.1 mm.

The terminal V5 following run is observations 107–123 (17 samples, observed first-to-last span 0.784 s; 0.832 s under the preceding-interval convention). The last eight wait observations 116–123 all follow. The final policy uses `track_distance` with small translational requests and converges near 1.30 m. This addresses V4's terminal lateral-reframe loop, while the overall visibility and run-length regressions still need evaluation.

## Contacts and admission limits

The distance-collision proxy and saved physical target/robot contact are zero on every observation. Contact geometry/manifolds were not saved, so full contact cannot be independently reconstructed. Minimum actual range is 0.88826 m, unchanged from V4; a 1.30 m setpoint is not a hard physical separation guarantee. Full all-actor guard snapshots were not saved, only before/after hashes and runtime equality assertions.

This is one repeatedly inspected training episode and a GT teacher, not a general evaluation, trained policy or deployed active-search result. The next bounded step is to evaluate the frozen V5 teacher on all original fixed 48 members with every failure retained, using the separate proposal. Do not admit action labels merely because this episode now terminates successfully.
