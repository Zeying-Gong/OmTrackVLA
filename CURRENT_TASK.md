# WA-Mobile current task
Status: READY_TO_SUBMIT; user authorized complete training on2026-09-29.
WLA-compatible WA: retain64 MetaQueries and original7x4 SE2 ActionExpert/target head.
JEPA-WM main + Meta DINO-WM reproduction comparison; UWB polar / image / mixed modes.
Robot command/time adapter implemented; data contract evt_normalized_command3_actual_dt_v1.
Official world interiors retained; new4->10 command and4->4 prior-command adapters learned.
World loss only training; inference retains existing WLA flow action path, no Qwen/text.
Full data audit v2:726631 train /73368 heldout; zero parsing errors; scene split unchanged.
Invalid history/transitions excluded, not relabeled. Raw records and audit v1 preserved.
13 regression tests and both2A800 real-data optimizer/DDP checks passed.
Peak allocated memory8.67GiB JEPA /8.23GiB DINO; original WLA checkpoint strict-loaded.
Recipe: wa/wm/ROBOT_TRAINING.md; each model8RTX4090,1epoch,batch2x8xaccum2=32.
22707 optimizer updates, full three-mode heldout, save checkpoint and portable results.
Source and WLA dependency frozen on Beijing NAS; no cross-NAS migration.
H100 reachable but project assets absent; A800 insufficient8-card availability;4090 ready.
Formal job IDs pending actual submission; do not claim training until worker evidence.
Closed-loop SR/CR and Thor/RDK latency remain UNVERIFIED; formal run is offline training/eval.
