# Project progress
Updated: 2026-09-28

## Current route
WLA is the primary tracking route by explicit user decision. DA3 is retired.
Official OmTrackVLA and modular baselines remain reference implementations.
No further DA3 collection, training, evaluation or automatic DAgger retraining is authorized.

## Verified evidence
| Result | Evidence | Status |
|---|---|---|
| DA3 mixed training | Job58085, 4 A800, 679878 updates / 2 epochs, final epoch_2 checkpoint verified | TRAINING_COMPLETE |
| DA3 paired diagnostic | Job58582: 0/10 successes, 40% collisions; valid-init subset 0/7 | EFFICACY_FAILED |
| WLA matched diagnostic | Job58346: 9/10 successes, 0 collisions; same 10 initial image hashes; valid-DA3-init subset 6/7 | VERIFIED_SYSTEM_COMPARISON |
| WLA completed full evaluation | Job58346: 1405 unique episodes/task; STT SR80.28%, DT53.31%, AT49.25% | COMPLETE |
| DA3 offline heldout | Job58633: 1003 episodes / 78958 windows; ADE0.75114m, FDE1.27461m | COMPLETE, not closed-loop success |
| WLA new candidate | Job58638, full STT SR81.28%; DT/AT incomplete at last check | EVALUATION_RUNNING |
| DA3 correction collection | Job58613 / Task69363, 8 A800 | STOPPED_USER_RETIRED |

The matched-ten comparison is one scene. WLA uses text; DA3 uses initial bbox with UWB missing.
Backbone, decoder, training and controller differ. Do not describe this as an encoder ablation.
DA3 diffusion terminal noise mismatch is confirmed in code, but its causal contribution remains UNVERIFIED.

## Cancellation
Initial stop attempts were rejected for expired authentication; no credentials were copied.
After user reauthentication, Job58613 and Task69363 were verified STOPPED at
2026-09-28 19:06:21 Asia/Shanghai.
All-cluster RUNNING returned WLA58638 only; SUBMITTED/SUBMITTING/SCHEDULED returned none.
No DA3 jobs remain active or queued. Logs and NAS outputs were retained.

## Storage and publication
All raw data, logs, checkpoints and partial correction rollouts are retained on NAS.
Partial correction data are not an audited training release; no correction retraining was run.
The code baseline is 9ad4bd1554f97e3a43913bdcd3d4f30af797f6b7 plus uncommitted experimental source.
This cleanup publishes documentation and records only, not those unreviewed source changes.
WLA execution snapshot is on Baidu NAS at algorithm/repos/WLA-EVT-20260925;
this document does not claim its source has been integrated into this repository.

## References
- [Decision and limitations](docs/tracking_route_decision.md)
- [Retirement record](archive/2026-09/da3_retirement/README.md)
- [Full pre-retirement progress](archive/2026-09/da3_retirement/pre_retirement_PROGRESS.md)
- [Original experiment ledger](EXPERIMENTS.csv)
