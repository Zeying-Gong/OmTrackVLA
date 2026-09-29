# WA-Mobile current task
Status: RETRY_PREPARING; Job 59125 / Task 69896 FAILED before training.
Prepare reproducible WA research on branch `wa`, independently of existing WLA.
Current deliverable: ResNet18 temporal policy, three prompt modes, DDP trainer.
Seven CPU tests and real-data single/two-process diagnostic runs passed.
User explicitly authorized formal training on 2026-09-28; source commit 4a73fbd6.
Retry on Beijing A800 shared NAS, 1 GPU x batch32; same model/global batch.
Keep source 4a73fbd6 and all previous failed-run evidence; confirm actual steps.
All prompts passed; NCCL and pretrained GPU diagnostics passed. Effectiveness UNVERIFIED.
Acceptance: see wa/VALIDATION.md and wa/RUNBOOK.md.
