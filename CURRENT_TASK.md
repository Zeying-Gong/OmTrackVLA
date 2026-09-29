# WA-Mobile current task
Status: SUBMITTED; full JEPA-WM and DINO-WM robot-domain training authorized and dispatched.
JEPA Job59566 Task70423; DINO Job59568 Task70425; each8RTX4090 on baidu_bj_4090.
Source frozen atab3ed46d; configurations pinned in0ace1c0c. Worker training evidence pending.
Complete provenance/output paths: wa/jobs/wm_robot_v1_run.json.
WLA-compatible WA retains64 MetaQueries, original7x4 SE2 ActionExpert and target head.
USS-inspired image/BBox target fusion + UWB polar; no Qwen/text at deployment.
Official world interiors retained; robot command/dt and prior-command interfaces trained.
Full data audit:726631 train /73368 heldout; no parse errors; original scene split preserved.
Abnormal transitions excluded; original data and superseded audit retained.
13 tests and both2A800 real-data optimizer/NCCL diagnostics passed; peak8.67/8.23GiB.
Recipe: wa/wm/ROBOT_TRAINING.md;1epoch22707 updates; batch2x8xaccum2=32; world weight0.1.
New source/dependency snapshots on Beijing NAS; no cross-NAS migration.
External8H100 full launcher and portable result exchange instructions published.
Next: verify worker/NAS optimizer logs then completed checkpoints and full offline metrics.
Model effectiveness, closed-loop SR/CR and Thor/RDK latency remain UNVERIFIED.
