# WA-Mobile current task
Status: CPU_REAL_BATCH_AND_DDP_PASS
Prepare reproducible WA research on branch `wa`, independently of existing WLA.
Current deliverable: ResNet18 temporal policy, three prompt modes, DDP trainer.
Seven CPU tests and real-data single/two-process diagnostic runs passed.
No formal training or effectiveness claim; all 10,660 episode prompts passed audit.
Next: provision pretrained weights/data on target workers and approve full-run config.
Formal training/evaluation requires an approved experiment configuration.
Acceptance: see wa/VALIDATION.md and wa/RUNBOOK.md.
