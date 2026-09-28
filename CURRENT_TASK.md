# Current task
Updated: 2026-09-28
Status: WLA_PRIMARY / DA3_RETIRED / ALL_DA3_JOBS_STOPPED

## Decision
- Adopt WLA as the primary tracking route.
- DA3 collection, training and evaluation are retired. Earlier conditional retraining authorizations are superseded.
- Do not submit, restart or automatically resume DA3 jobs without a new explicit user instruction.
- Preserve all NAS data, checkpoints, logs and failure evidence.

## Outstanding work
- DA3 Job58613 / Task69363 verified STOPPED at 2026-09-28 19:06:21 Asia/Shanghai after user reauthentication.
- All-cluster active/queued listings contain no DA3 jobs; only WLA58638 remains RUNNING.
- No SUBMITTED, SUBMITTING or SCHEDULED jobs were returned in the all-cluster check.
- WLA Job58638 continues its existing evaluation; do not stop it.
- GitHub update is authorized for the documentation cleanup. Existing uncommitted experimental source changes remain on NAS.

## Acceptance
- PASS: decision and comparative evidence recorded in docs/tracking_route_decision.md.
- PASS: current state compacted and prior state preserved under archive/2026-09/da3_retirement/.
- PASS: scheduler-confirmed DA3 cancellation.
- PASS: documentation/CSV validation; this change is the authorized GitHub documentation publication.

## Boundaries
WLA currently uses text+RGB, whereas DA3 used initial bbox+RGB with UWB missing.
This selects the current complete system, not a proven backbone-only winner.
A bbox/UWB WLA interface is separate, unimplemented work.
