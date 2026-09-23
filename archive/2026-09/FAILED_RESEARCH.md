# Archived Failed Research Routes

The active project was narrowed on 2026-09-23 to the official OmTrackVLA baseline and the modular person-following baseline.

## End-to-end policy

The `e2e_policy` route combined DINO features, frame history, 3D mRoPE, mixed Sage3D/TpT/NavDP data, and auxiliary dynamics losses. Smoke tests and short optimization runs succeeded, but those checks did not establish accepted STT/DT/AT benchmark gains. The route is classified `FAILED` for project purposes.

Relevant commits include `a9195aa`, `d4a64f3`, `bc1f35d`, and `7a02284`. Source and notes are preserved under `failed_routes/`.

## Hybrid FLUX and online RL

The Hybrid FLUX/PPO route attempted residual recovery control on top of imitation learning. V26 stopped at update 11 without validation, best checkpoint, or final checkpoint. V27 repaired reward and observation protocol issues, but no accepted full benchmark improvement was produced. The route is classified `FAILED` for project purposes.

The last pre-cleanup implementation is commit `16733f1`. Source, the unit test, launchers, and diagnosis note are preserved under `failed_routes/`.

## Archive rule

Files in this directory are evidence and recovery material, not maintained code. Active modules and launchers must not import from or execute this archive unless the user explicitly reopens a route.
