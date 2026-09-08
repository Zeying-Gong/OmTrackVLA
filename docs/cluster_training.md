# 8×H100 单节点训练流水线

`scripts/run_pipeline_8xh100.sh`面向已经分配好的单节点8×H100环境。它不申请集群资源；协作者登录节点、激活环境后直接运行脚本。

## 当前状态

集群编排、环境预检、三阶段衔接、断点续跑、日志、benchmark、可视化和gate契约已经确定。Phase 1的真实训练、B1-ID/B1-GEO/B1-PROBE评测、`viz_val`渲染和gate入口已实现并通过单卡小样本闭环；Phase 2/3模块和配置仍未实现，因此`--phase all`会在preflight阶段明确失败，不会用占位模型伪造成功结果。

Phase 1已接通、Phase 2/3仍需扩展的Python模块：

```text
omtrackvla.training.train
omtrackvla.evaluation.evaluate
omtrackvla.evaluation.render
omtrackvla.evaluation.gate
```

配置状态：

```text
configs/phases/phase1_pretrain.yaml
configs/phases/phase2_following_sft.yaml
configs/phases/phase3_recovery.yaml
configs/benchmarks/phase1.yaml
configs/benchmarks/phase2.yaml
configs/benchmarks/phase3.yaml
configs/gates/phase1.yaml
configs/gates/phase2.yaml
configs/gates/phase3.yaml
```

其中三个`phase1*.yaml`已经实现，其余六个Phase 2/3配置仍待实现。

模块名、配置位置和额外参数集中在`configs/pipeline/h100_8gpu.env`，后续无需改集群脚本主体。

## 直接运行

```bash
cd OmTrackVLA
export OMTRACKVLA_DATA_ROOT=/path/to/shared/data
bash scripts/run_pipeline_8xh100.sh \
  --config configs/pipeline/h100_8gpu.env \
  --phase all \
  --run-id baseline_v1
```

默认要求：

- 当前进程恰好能看到8张GPU；
- 8张卡的名称均包含`H100`；
- Git工作区干净；
- 数据根目录、Phase配置和Python入口均存在。

首次运行应先按[`phase1_minimal_loop.md`](phase1_minimal_loop.md)执行单卡小样本验证。开发smoke使用显式样本/step覆盖参数，不能据此创建`GATE_PASSED`；正式gate仍要求冻结benchmark规模和阈值。

开发机可以只检查将要执行的命令：

```bash
bash scripts/run_pipeline_8xh100.sh \
  --dry-run \
  --allow-dirty \
  --allow-non-h100 \
  --run-id dry_run
```

真实环境预检：

```bash
bash scripts/run_pipeline_8xh100.sh --preflight-only
```

## Phase与入口契约

训练模块由`torch.distributed.run`启动8个进程，接收：

```text
--phase <1|2|3>
--config <phase_train_config>
--data-root <shared_data_root>
--output-dir <run/phase_x>
--run-manifest <run_manifest.json>
[--resume-from <last.ckpt> | --init-checkpoint <parent_best.ckpt>]
```

训练成功必须生成：

```text
phase_x/checkpoints/last.ckpt
phase_x/checkpoints/best.ckpt
```

评测模块也以8进程启动，接收checkpoint与benchmark配置，并必须生成`metrics.json`和`report.md`。渲染模块运行固定的`viz_val`，只画模型诊断输出`PRED`和评测真值`GT`；第1帧以后不能向模型提供外部bbox。

gate模块单进程运行。返回码为0表示通过；任何非0返回码都会停止流水线，不进入下一Phase。

## 断点续跑与单阶段训练

使用相同`run-id`即可自动恢复`last.ckpt`，已经存在`GATE_PASSED`的Phase会被跳过：

```bash
bash scripts/run_pipeline_8xh100.sh \
  --phase all \
  --run-id baseline_v1 \
  --resume auto
```

从指定checkpoint单独启动Phase 2：

```bash
bash scripts/run_pipeline_8xh100.sh \
  --phase 2 \
  --run-id phase2_ablation \
  --from-checkpoint /path/to/phase_1/checkpoints/best.ckpt
```

`--skip-gate`只适合调试；正式训练不得跳过gate。工作区确需保留未提交修改时可显式传入`--allow-dirty`，该状态仍会记录在`run_manifest.json`中。

## 产物

默认输出到`outputs/training/<run_id>`：

```text
run_manifest.json
pipeline.env
phase_1/
  checkpoints/
  logs/{train,eval,render,gate}.log
  metrics.json
  report.md
  gate.json
  visualizations/
  failures/
phase_2/
phase_3/
PIPELINE_COMPLETE
```

脚本异常时会写入`FAILED`，其中记录失败Phase、退出码和时间。每个Phase通过gate后写入`GATE_PASSED`。

## 多任务并发注意事项

- 每个任务使用不同`run-id`和`master-port`。
- 通过`CUDA_VISIBLE_DEVICES`控制当前任务看到的8张卡。
- 数据、checkpoint和视频保存在共享存储，不提交到GitHub。
- 正式对比必须固定代码commit、数据manifest、Phase配置、benchmark episode/seed和父checkpoint。
