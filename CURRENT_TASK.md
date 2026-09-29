# WA-Mobile current task
Status: WORKER_STARTING; retry Job 59519 / Task 70376; first run 59125 FAILED.
Prepare reproducible WA research on branch `wa`, independently of existing WLA.
Current deliverable: ResNet18 temporal policy, three prompt modes, DDP trainer.
Seven CPU tests and real-data single/two-process diagnostic runs passed.
User explicitly authorized formal training on 2026-09-28; source commit 4a73fbd6.
User selected Beijing 4090: 8 GPUs x batch4; full epoch; global batch32 unchanged.
Keep source 4a73fbd6 and all previous failed-run evidence; confirm actual steps.
All prompts passed; NCCL and pretrained GPU diagnostics passed. Effectiveness UNVERIFIED.
Acceptance: see wa/VALIDATION.md and wa/RUNBOOK.md.
