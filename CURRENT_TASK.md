# WA-Mobile current task
Status: TRAINING + PRIVATE_BACKUP_RUNNING. Both ModelScope logins verified; no credential copying.
JEPA59566 verified4875/22707 steps; do not stop or launch duplicate H100 training.
Private other-license repo a597836509/wa-evt-jepa-private-backup-20260929:150shards /163832726558sourcebytes; first shard uploaded. H100 downloader started.
DINO Job59568 Task70425 verified STOPPED on2026-09-29; outputs retained.
JEPA Job59566 Task70423 RUNNING on8RTX4090;775/22707 optimizer steps verified. H100 switch now interrupts valid training.
Target: aliyun_sh_h100 using verified platform KubeRay/TorchTrainer capability, not Baidu shell/K8s.
Central resource snapshot:26 free H100. This is not guaranteed8GPU node placement.
Legacy aliyun Ray Dashboard DNS failed; current central jobs/list works; KubeRay needs final validation.
Alibaba root: /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928.
Source/dependency/index copy completed; data and WLA checkpoint transfers stopped. Partial targets are NOT usable.
Exact data inventory:10660episodes /1050874files; original scene split preserved.
Dataset target: data_migration_v1; use data-root/source-prefix mapping, never edit original manifest.
Checkpoint target: pretrained_migration_v1; hashes must match before training.
Direct devpod-to-devpod SSH denied; authorized local SSH stream relay used, no local dataset copy.
H100 isolated h100_env: torch2.7.0+cu128 / torchvision0.22.0 / timm1.0.30.
Baidu verified environment uses torch2.8; H100 full import/model/data checks still required.
Ray initialization guard added to wa/wm/train.py;13 developer regression tests PASS.
Source baseline ab3ed46d; new guard must be committed and frozen for H100 before submission.
Architecture unchanged: official JEPA dynamics;64 MetaQueries; original7x4 ActionExpert; USS polar goals.
Recipe unchanged:726631 train /73368 heldout;1epoch22707updates;batch2x8xaccum2;worldweight0.1.
No H100 formal job submitted. No H100 optimizer-step evidence. Training effectiveness UNVERIFIED.
Next: finish/check transfer hashes; freeze source; validate KubeRay config/env; submit full8H100 only.
After successful H100 submission cancel old JEPA queue and register IDs/config/output evidence.
