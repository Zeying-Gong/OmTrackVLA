# Fixed failure-state group fit — 2026-10-08

## Terminal status at 19:15 Beijing

The bounded policy-fit diagnostic finished with session52695 exit0. This compares the existing parent59866 and best61609 checkpoints; it does not create a new trained model, optimizer update, checkpoint or closed-loop result. The fixed selection and fitting gates are complete and must not be repeated by default.

- Result: /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/failure_state_group_fit_pretrain_20261008_v1.json
- Result SHA256: 6ecff4bda1c19d153e7e4be0742914d2b761698759a9157c17ec2920a6725cd9; 3,049,079 bytes; elapsed 1169.6806585s; file mtime 18:43:58 Beijing.
- Log: same prefix with .log; SHA256 1de08bf57a9a35552e6fd69b406587f09355d7e23dd5aefd7b6fe62f5d6abc46; 2,425 bytes. Three xFormers warnings retained; five checked fatal-pattern counts are zero. No related process remains.
- Peak PyTorch allocated/reserved memory: 1,590,791,680 / 1,608,515,584 bytes. This inference peak and the 5GiB allocator limit do not establish training memory safety or exclusive GPU ownership.
- Independent persisted-array audit PASS: each checkpoint has 304 records (152 normal-history + 152 repeated-current-history), all finite 7x4 predicted/target trajectories; cross-model input hashes and labels are identical. Float64 recomputation maximum metric difference 3.017611525e-7; stored group summaries/deltas match exactly. The result records 3,166 source pins; this audit did not rescan the old full collection.

### Normal-history ADE (meters)

| Fixed group | Parent59866 | Best61609 | Reading |
|---|---:|---:|---|
| collision | 0.558798 | 0.467611 | lower |
| other | 0.408546 | 0.383305 | lower |
| successful | 0.278920 | 0.255145 | lower |
| DT | 0.304162 | 0.310269 | worse |
| AT | 0.449542 | 0.371023 | lower |
| early recovery | 0.593261 | 0.517808 | lower |
| late recovery | 0.471090 | 0.396540 | lower |

These are offline label-fit comparisons on predefined groups, not a new training improvement, an SR estimate, or proof of JEPA loss/backward/optimizer/NCCL compatibility. DT worsens; no aggregate claim hides that result. Normal and repeated-history records are both retained in the JSON; this compact table shows normal-history values only.

## Historical status at 18:25 Beijing

The user explicitly approved the two previously blocked diagnostic files. The original Goal tool still reports BLOCKED; this is not a completion or a new permission blocker. Work is continuing under the user's new request. Existing heartbeat wa remains active.

CPU tests: initial 28/30 passed; two test fixtures assumed interleaved normal/repeated-history list order. They now address the selected window identity. Rerun: all 30 tests passed in 1.164s, without weakening product checks. Code and tests were backed up in wa commit 6bc034b28003e5e0471caa74ed16cac5c01b8883.

## Fixed selection

- NAS selection: /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/failure_state_group_fit_selection_20261008_v1.json
- SHA256: 7e7db844c09f67d7b0e3b5de7376c0573c18fccc485dfa235b879ed4dab111a9
- Exit 0; 977.593435954s; 786143 bytes; 2865 source hashes.
- 88 original windows retained verbatim, plus 32 early and 32 late recovery windows. The new 64 are unique and have no intersection with the 526 exact old-window duplicates.
- All new windows are STT, covering 40 episodes in total. Each stratum uses 32 different episodes; cross-stratum overlap is allowed.
- Early: 0.440–0.488s after takeover; Oracle 23 / LightNav 9. Late: 1.016–2.000s; Oracle 24 / LightNav 8.
- Selection uses fixed episode hash order and proximity to predefined times, not observed fit scores. The independent compact JSON check confirmed identities, fields and sets; it did not rerun images or independently recompute nearest-time eligibility.

## Historical startup: no result was claimed at that time

The pretrain-fit command below started at 18:23:50 Beijing, unified session52695, Python PID3969666. At 18:25 the startup guard was recorded and source validation was still running. No completed-prediction count, final result, new checkpoint or model SR is claimed.

The existing RTX4090 GPU7 is shared, not exclusively owned. At startup it showed 0% utilization and 7690MiB free. The wrapper sets a 5GiB PyTorch allocator ceiling and a 2400s process timeout. The ceiling excludes CUDA context and external allocations. No other process was stopped.

The diagnostic makes 608 window/mode/model predictions, using batch2, no_grad and one GPU model instance loaded successively from parent59866 and best61609. Only RGB/template/current polar UWB/timestamps enter prediction. Full 7x4 predicted and target trajectories plus input hashes are retained; no rollout or training occurs.

This tests policy inference, not JEPA loss, backward, optimizer updates or multi-GPU NCCL. The original new-recipe A800 four-update check is already complete; a real target4090 training compatibility gate and submission preflight still remain. The frozen training source stays at 199385cd9c826c8f21308ad99b6a8375396d9c90.

## Exact issued developer command

Historical command run on devpod-4090; the existing log and result are now complete. Preserve this issued command for reproducibility; do not rerun it or overwrite either output.

```bash
set -euo pipefail
cd /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/checkout
test ! -e /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/failure_state_group_fit_pretrain_20261008_v1.json
test ! -e /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/failure_state_group_fit_pretrain_20261008_v1.log
test "$(git rev-parse HEAD)" = 6bc034b28003e5e0471caa74ed16cac5c01b8883
set -o noclobber
CUDA_VISIBLE_DEVICES=GPU-3c8e0886-91d1-d6dc-0f7b-1ef3f757f4fd OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=20s 2400s /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/probe_env/bin/python -B - <<'PY' > /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/failure_state_group_fit_pretrain_20261008_v1.log 2>&1
import csv,json,os,runpy,subprocess,sys,torch
rows=list(csv.reader(subprocess.check_output(["nvidia-smi","-i","GPU-3c8e0886-91d1-d6dc-0f7b-1ef3f757f4fd","--query-gpu=uuid,name,utilization.gpu,memory.free","--format=csv,noheader,nounits"],text=True).splitlines()))
assert len(rows)==1
row=[x.strip() for x in rows[0]]
assert row[0]=="GPU-3c8e0886-91d1-d6dc-0f7b-1ef3f757f4fd" and "RTX 4090" in row[1]
assert int(row[2])<=5 and int(row[3])>=7000, ("developer shared capacity changed",row)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
cap=5*2**30
props=torch.cuda.get_device_properties(0)
torch.cuda.set_per_process_memory_fraction(cap/props.total_memory,device=0)
print(json.dumps(dict(stage="BOUNDED_POLICY_FIT_START_NOT_TRAINING",pid=os.getpid(),gpu=row,torch=torch.__version__,allocator_cap_bytes=cap,cap_scope="PyTorch allocator only; CUDA context and external allocations excluded",timeout_s=2400)),flush=True)
sys.argv=["wa.tools.failure_state_group_fit","run","--selection","/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/failure_state_group_fit_selection_20261008_v1.json","--selection-sha256","7e7db844c09f67d7b0e3b5de7376c0573c18fccc485dfa235b879ed4dab111a9","--model","pretrain","--output","/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/failure_state_group_fit_pretrain_20261008_v1.json"]
runpy.run_module("wa.tools.failure_state_group_fit",run_name="__main__")
print(json.dumps(dict(stage="BOUNDED_POLICY_FIT_EXIT",peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),closed_loop=False,training=False)),flush=True)
PY
tail -n 4 /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/failure_state_group_fit_pretrain_20261008_v1.log
```

## Next

The draft `wa/jobs/failure_state_train_4090_v1.yaml` is now 276 lines, SHA256 `33838ac5dc95bf1a00133cdcb32d28a47224bc73f53bb6ed5fd863ccbf01d035`. Its 18 external pins retain the original16 and add the fixed selection and completed fit. CPU assertions passed for actual fit schema/status/false flags, 152 selected windows, seven groups, model SHA/step and each model's 304 paired records; launch evidence records these checks. `bash -n` and two Python AST checks passed; training argv and the original postcheck are byte-for-byte unchanged. It remains NOT_SUBMIT_READY until real target4090 four-update compatibility and refreshed full-task preflight pass. No formal job or data change results from binding this evidence.

Final arrays and summaries have been audited. The remaining gate is a real unchanged-recipe four-update target4090 developer training check, including loss/backward/optimizer behavior, followed by formal submission preflight; this policy-fit run cannot substitute for it. Do not repeat the completed fit/data gates, lower batch, alter the objective, stop another task, or submit a separate cluster smoke task.

At 19:09:33 and 19:10:16 Beijing, developer GPU7 had 7690MiB free/0% utilization, insufficient for the observed 7.309GiB training allocation peak plus CUDA-context safety headroom. GPU6 had 16252MiB free but 87→88% utilization and was not borrowed or stopped. A800 had 3 cluster GPUs free and bj4090 had 76; cluster availability is not developer allocation. Jobs61846/61847 belong to other work, not this task; there is no task-owned SUBMITTED/SUBMITTING job. Permission is resolved; safe target-runtime capacity is the current blocker. The Goal tool remains BLOCKED (no resume API), while the existing heartbeat and newly authorized work continue.

Formal training remains an independent one-new-epoch continuation from59866 model AND optimizer, not a third epoch from61609. No formal job was submitted in this diagnostic stage. Best closed-loop counts remain1279/1178/1207 out of1405 each; the STT target is1289. Results are evaluation-set adaptation, not unseen-test generalization.
