# 首帧BBox无效诊断 — 2026-10-04

Status: PARTIAL_ROOT_CAUSE_ISOLATED_NOT_FIXED. No training, formal evaluation, policy repair, metric change or result replacement.

## 已确认

- 4215条中的165条首帧初始化无效：STT57/DT57/AT51。原始原因全部为BBox[0,0,0,0]。
- 对全部165条读取首帧语义observer与动作记录：目标像素0，整条所有机器人动作均为0。
- 其中107条后续重新出现目标语义像素：STT35/DT37/AT35；首次可见时间0.048..1.752s。永久零动作来自WA首帧失败锁定，不是后续模型推理。
- 所有165条首帧目标身体坐标bearing绝对值<=20.112度、range1.554..2.962m。该几何值不能单独证明无遮挡或相机可见。
- 抽查STT oLBMNvg9in8/55、oLBMNvg9in8/2、2n8kARJN3HM/299、E9uDoFAP3SH/10原始录像首帧，RGB中均有人可见。因此不能把语义0直接解释成RGB中没有人；未声称完成165条人工目标身份标注。
- 短时开发机复现STT oLBMNvg9in8/55：预期目标ID1098，语义图147456/147456像素全部ID160，BBox为零；同一静态状态连续渲染两次不变。
- RGB和语义相机实际4x4矩阵完全相等（不只是YAML相同）；ID160元数据类别ceiling。因此不能解释为相机姿态不匹配或3000像素阈值。
- 同场景有效对照STT oLBMNvg9in8/12：ID1088有6730像素，BBox[227,92,291,281]。排除全场景语义传感器不可用的笼统解释。
- 已定位到特定起点的RGB/语义渲染不一致，具体语义资产几何、遮挡/背面渲染或载入变换机制尚未隔离。尚不能把该单例根因外推全部165条。

## 代码链路

1. evt_bench/additional_sensor.py MainHumanoidDetectorSensor读取episode目标semantic_id，在jaw_panoptic中>0像素才生成BBox；3000阈值仅控制facing，不控制BBox。
2. wa/wm/eval_agent.py在sim_step0检查BBox；零框设置policy_failure_reason，此后act开头永久返回[0,0,0]。不是服务打不开，也不是正常推理后的短期跟踪失败。
3. trained_agent.py保存失败原因并强制success=False，仍让仿真自然结束。
4. evt_bench/additional_metric.py HumanFollowing也依赖同一传感器facing。因此语义问题可能同时影响初始化和跟随指标，不能只改输入后忽略评测可比性。

## 对成功率的影响边界

165/4215=3.914591%样本被初始化分支锁定；占全部795条失败的20.754717%。
STT最多涉及4.05694个百分点，DT4.05694，AT3.62989；不是已实现增益。
极端假设只将165条全部变成功、其他不变，总SR上限3585/4215=85.053381%，当前仍为81.138790%。
不要删样本、移动起点、补任意中心框、用后续GT框追溯冒充首帧输入，或把原成绩改成修复后成绩。

## 下一步建议，尚未执行

- 在独立诊断环境隔离RGB/semantic场景资产与遮挡几何；固定同一相机/时刻/人形姿态，检查全59个无效task-key并集对应的起点及有效对照。
- 若证实渲染/资产缺陷，独立版本修复后需配对验证RGB、语义、BBox及HumanFollowing，保留旧4215结果并标新协议/运行版本。
- 首帧缺框的恢复是另一项策略选择。延迟使用GT框会改变首帧BBox协议，不应偷偷加入；真实RGB目标检测+UWB关联也需单独设计验证。不要先盲目重训。

## 证据与过程

Machine-readable: wa/results/init_bbox_diagnosis_20261004.json（165条路径及统计）。
NAS artifacts/init_bbox_probe_20261004_v1、init_bbox_probe_20261004_v3_55、init_bbox_probe_20261004_v3_12含原始RGB、semantic.npy、report.json。
短检查只reset并渲染2帧，没有动作rollout、模型推理或正式集群任务。
首次Xvfb路径缺失失败，日志init_bbox_probe_xvfb_20261004.log保留；复用现有解压依赖后通过。
v2完成渲染但numpy.float32报告序列化失败，保留v2输出；v3修正诊断序列化并使用独立目录。
原checkpoint、源码冻结目录、数据、metrics和已有视频均未修改；自动监控仍暂停。
