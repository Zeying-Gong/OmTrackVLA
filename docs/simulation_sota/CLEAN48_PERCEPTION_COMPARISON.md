> 2026-09-14 17:46 纠正：真实 Habitat 探针已确认，人物的 visual_scene_nodes 为空，旧编号分配循环没有执行任何写入。旧感知框、可见性和依赖它们的跟随指标暂不能认定对应唯一物理目标。已暂停新增训练、数据准入和权重晋升，正在验证根/骨骼节点修复并准备重新采集。原结果和失败记录均保留；没有 SOTA 成绩。详见 docs/SIMULATION_SOTA_STATUS_20260914.md。

# Fixed development comparison

Four fixed cases come from two unique scenes. All groups use the same 225 raw RGB observations, including 48 visible boxes and 177 absent targets. The old report called these four scenes; its frozen bytes remain preserved.

Both the training scene count (1 to 34) and optimizer budget per group (256 to 2048) changed. This comparison does not isolate either factor. No threshold or checkpoint was selected from validation outcomes.

| Variant / training run | Mean IoU | IoU >= 0.5 / 48 | Confidence >= 0.9 and IoU >= 0.5 / 48 | Missed visible / 48 at 0.5 | False present / 177 at 0.5 |
|---|---:|---:|---:|---:|---:|
| B0 pooled SmoothL1; 1 scene, 256 steps | 0.1428 | 2 | 2 | 18 | 5 |
| B0 pooled SmoothL1; 34 scenes, 2048 steps | 0.1050 | 1 | 1 | 4 | 103 |
| B1 pooled L1/GIoU; 1 scene, 256 steps | 0.1454 | 1 | 1 | 26 | 5 |
| B1 pooled L1/GIoU; 34 scenes, 2048 steps | 0.4132 | 19 | 19 | 4 | 9 |
| D1 dense; 1 scene, 256 steps | 0.3406 | 15 | 6 | 16 | 7 |
| D1 dense; 34 scenes, 2048 steps | 0.3813 | 18 | 16 | 7 | 34 |

See comparison.json for the new evaluator's exact train/validation scene-intersection metadata. These results are development perception measurements; they do not establish closed-loop navigation, arbitrary-target identity recovery, or SOTA.

Old metrics SHA256: `9af3af371cb1f516973fb6e054467a6186ddaa81f1468a61a125ae00806a837b`.
New metrics SHA256: `feac2947a8a40e0b2bb74a3d62aa27949305a93eff0879dba2225c74a47326d4`.
