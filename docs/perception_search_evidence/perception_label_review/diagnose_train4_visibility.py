"""Read-only causal visibility audit of four fixed completed train-source runs.

No model/simulator imports, no GPU work, no threshold fitting. Writes only the
requested new diagnostic output files; frozen rollout sources remain read-only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

REMOTE_ROOT = "/data/nfs/share/wam_tracking/OmTrackVLA"
SOURCES = [
    {"source": "AT401", "run": "at_0401_128steps_gpu3_v2", "task": "at", "index": 401, "episode": "9", "scene": "9hJwm8k7Gka", "steps": 85, "sha256": "61e1f27a3778e36fdfce747ba787450a6b03b29ed3db4398a7a63a8286aa2f81"},
    {"source": "DT200", "run": "dt_0200_128steps_gpu3_v2", "task": "dt", "index": 200, "episode": "7", "scene": "XxbS57Z6PDU", "steps": 104, "sha256": "c0f880bb42b84c77c508cb54489c318e5f164c1ee0190aaef20520510eba0737"},
    {"source": "STT0", "run": "stt_0000_128steps", "task": "stt", "index": 0, "episode": "4", "scene": "16tymPtM7uS", "steps": 84, "sha256": "351be447338a4b304e77c16a5c0f28de6fbbd9136670ff0dbf945c880468c6e6"},
    {"source": "STT3100", "run": "stt_3100_128steps_gpu3_v2", "task": "stt", "index": 3100, "episode": "2", "scene": "qxwfVS8MQ67", "steps": 93, "sha256": "f7c6e76402bc77ee706716edb37ed0422d2d3966797a45312aef970aa50a8317"},
]
THRESHOLD = 0.90
WINDOW_RADIUS = 3
CHECKPOINT_SHA = "32c8f2f73277cfab8c3c7c22a53994061f7178498cfa003e1635b9576641bc9c"


def confusion_name(predicted, observed):
    return ("TP" if predicted else "FN") if observed else ("FP" if predicted else "TN")


def diagnostic(repository_root):
    cases = []
    for source in SOURCES:
        relative = "outputs/takeover/phase2_train4_long_rollouts_v1/" + source["run"] + "/result.json"
        data = (repository_root / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == source["sha256"]
        result = json.loads(data)
        rows = result["steps"]
        assert len(rows) == source["steps"] and [r["step"] for r in rows] == list(range(1, len(rows) + 1))
        assert result["task"] == source["task"] and result["dataset_index"] == source["index"]
        assert str(result["episode_id"]) == source["episode"] and source["scene"] in result["scene_id"]
        assert result["split"] == "train" and result["summary"]["episode_finished"] is True
        assert result["summary"]["termination_reason"] == "episode_over"
        assert result["loading"]["checkpoint_sha256"] == CHECKPOINT_SHA
        initialization = result["initialization"]
        assert initialization["source"] == "first_frame_panoptic_mask"
        assert initialization["environment_step"] == 0 and initialization["used_frames"] == [0]
        x1, y1, x2, y2 = initialization["bbox_xyxy"]
        assert x2 > x1 and y2 > y1
        observed_gt = True  # Explicit reset-frame target mask box, not a later GT value.
        aligned = []
        gt_events = []
        prediction_events = []
        prior_prediction = None
        for row in rows:
            step = row["step"]
            policy = row["policy"]
            probability = policy["visibility_probability"]
            assert math.isfinite(probability) and 0 <= probability <= 1
            assert policy["uwb_valid"] is False
            high_confidence = probability >= THRESHOLD
            action = policy["action"]
            full_stop = all(v == 0 for v in action.values())
            uncertain_stop = policy["mode"] == "visual_uncertain_stop"
            assert uncertain_stop == (not high_confidence)
            assert not uncertain_stop or full_stop
            post_gt = row["evaluation_only_after_action"]["gt_visible"]
            assert type(post_gt) is bool
            box = policy["predicted_bbox_xyxy_norm"]
            assert len(box) == 4 and all(math.isfinite(v) for v in box)
            aligned.append({
                "decision_step": step, "input_observation_index": step - 1,
                "input_gt_source": "reset_initialization_mask" if step == 1 else f"record_{step-1}_post_action_gt",
                "input_gt_visible_anypixel": observed_gt,
                "predicted_visibility_probability": probability,
                "predicted_visible_at_fixed_threshold": high_confidence,
                "confusion": confusion_name(high_confidence, observed_gt),
                "predicted_bbox_xyxy_norm": box,
                "predicted_bbox_height_norm": box[3] - box[1],
                "predicted_stop_probability": policy["stop_probability"],
                "mode": policy["mode"], "action": action,
                "uwb_valid": policy["uwb_valid"], "full_zero_action": full_stop,
                "post_action_gt_visible_anypixel": post_gt,
                "post_action_gt_observation_index": step,
            })
            if prior_prediction is not None and high_confidence != prior_prediction:
                prediction_events.append({"decision_step": step, "input_observation_index": step - 1,
                                          "kind": "confidence_regained" if high_confidence else "confidence_lost"})
            if post_gt != observed_gt:
                gt_events.append({"kind": "gt_reappearance" if post_gt else "gt_loss",
                                  "post_action_observation_index": step,
                                  "first_decision_that_can_observe_change": step + 1 if step < len(rows) else None,
                                  "terminal_without_next_decision": step == len(rows)})
            observed_gt = post_gt
            prior_prediction = high_confidence

        matrix = {key: sum(r["confusion"] == key for r in aligned) for key in ("TP", "TN", "FP", "FN")}
        assert sum(matrix.values()) == len(rows)
        stops = [r for r in aligned if r["mode"] == "visual_uncertain_stop"]
        first_low = next((r for r in aligned if not r["predicted_visible_at_fixed_threshold"]), None)
        first_loss = next((event for event in gt_events if event["kind"] == "gt_loss"), None)
        windows = []
        for event in gt_events:
            # Center on the first causally eligible decision; last observation has no such decision.
            center = event["post_action_observation_index"] + 1
            windows.append({"event": event, "center_decision_step": center,
                            "radius_decisions": WINDOW_RADIUS,
                            "rows": [r for r in aligned if center - WINDOW_RADIUS <= r["decision_step"] <= center + WINDOW_RADIUS]})
        cases.append({"source": source["source"], "result_path": REMOTE_ROOT + "/" + relative,
            "result_sha256": source["sha256"], "partition_role": "train_source_development_diagnostic",
            "checkpoint_sha256": CHECKPOINT_SHA, "initialization": initialization,
            "scene_id": result["scene_id"], "episode_id": str(result["episode_id"]),
            "decision_count": len(rows), "observation_indices_used": [0, len(rows) - 1],
            "excluded_terminal_post_action_observation_index": len(rows),
            "confusion_matrix": matrix,
            "visual_uncertain_stop_count": len(stops),
            "visual_uncertain_stop_while_input_gt_visible": sum(r["input_gt_visible_anypixel"] for r in stops),
            "visual_uncertain_stop_while_input_gt_invisible": sum(not r["input_gt_visible_anypixel"] for r in stops),
            "first_low_confidence_decision": first_low,
            "first_gt_loss": first_loss,
            "first_low_confidence_lead_decisions_before_first_gt_loss_observed":
                first_loss["first_decision_that_can_observe_change"] - first_low["decision_step"]
                if first_loss and first_loss["first_decision_that_can_observe_change"] and first_low else None,
            "prediction_confidence_transitions": prediction_events, "gt_transitions": gt_events,
            "gt_transition_windows": windows,
            "first_low_confidence_window": [r for r in aligned if first_low and abs(r["decision_step"] - first_low["decision_step"]) <= WINDOW_RADIUS],
            "terminal_gt_visible_anypixel_not_used_as_extra_label": rows[-1]["evaluation_only_after_action"]["gt_visible"],
            "aligned_decisions": aligned})

    aggregate = {"decision_count": sum(c["decision_count"] for c in cases),
                 "confusion_matrix": {k: sum(c["confusion_matrix"][k] for c in cases) for k in ("TP", "TN", "FP", "FN")},
                 "visual_uncertain_stop_count": sum(c["visual_uncertain_stop_count"] for c in cases),
                 "visual_uncertain_stop_while_input_gt_visible": sum(c["visual_uncertain_stop_while_input_gt_visible"] for c in cases),
                 "visual_uncertain_stop_while_input_gt_invisible": sum(c["visual_uncertain_stop_while_input_gt_invisible"] for c in cases),
                 "reset_initialized_labels_included": 4, "terminal_observations_excluded": 4,
                 "cold_start_cases": 0, "gt_loss_events": sum(sum(e["kind"] == "gt_loss" for e in c["gt_transitions"]) for c in cases),
                 "gt_reappearance_events": sum(sum(e["kind"] == "gt_reappearance" for e in c["gt_transitions"]) for c in cases)}
    return {"status": "fixed_train4_readonly_causal_diagnostic_complete", "schema_version": 1,
        "threshold": THRESHOLD, "threshold_policy": "Fixed existing 0.90 threshold; no threshold search or adaptation.",
        "method": "Hash-bound completed train4 source JSON only; no GPU/model/simulator execution; no source writes.",
        "alignment": "Decision j (1-based) uses observation j-1. GT for j=1 is the reset target mask; j>1 uses record j-1 post-action GT. Terminal post-action observation N has no next decision and is excluded as a label; each N-step source contributes N decisions, not N+1 or N-1.",
        "positive_class": "Input GT target mask contains any pixel; predicted positive iff policy visibility_probability >= 0.90.",
        "scope": "Only AT401(85), DT200(104), STT0(84), STT3100(93), all already-initialized train-source rollouts. No validation-output training, cold-start addition, new rollout or waypoint training.",
        "limitations": ["GT anypixel visibility is not a visible-fraction estimate, an occlusion category, or evidence the model identified the correct target.",
            "GT bbox/visible fraction/occlusion cause are not stored in these result files; windows show predicted bbox only and cannot diagnose box correctness.",
            "A false-negative against anypixel does not by itself mean motion was safe or the tracking confidence was miscalibrated; no threshold lowering is justified here.",
            "These four model-visited training scenes are a descriptive diagnostic, not held-out validation or project acceptance.",
            "Zero reappearance events means these sources alone do not supply initialized-target reappearance or same-identity recovery examples."],
        "aggregate": aggregate, "sources": cases}


def fmt(value):
    if isinstance(value, bool): return "1" if value else "0"
    if isinstance(value, float): return f"{value:.6g}"
    if isinstance(value, list): return "[" + ", ".join(fmt(v) for v in value) + "]"
    return str(value)


def window_table(rows):
    lines = ["| decision j / 输入 obs | 输入 GT | visibility | 预测 bbox(x1,y1,x2,y2) | stop p | 模式 | 动作(f,l,y) | 动作后 GT |",
             "|---|---:|---:|---|---:|---|---|---:|"]
    for row in rows:
        lines.append("| " + " | ".join([f"{row['decision_step']} / {row['input_observation_index']}", fmt(row["input_gt_visible_anypixel"]),
            fmt(row["predicted_visibility_probability"]), fmt(row["predicted_bbox_xyxy_norm"]), fmt(row["predicted_stop_probability"]),
            row["mode"], fmt([row["action"][key] for key in ("forward", "lateral", "yaw")]), fmt(row["post_action_gt_visible_anypixel"])]) + " |")
    return lines


def markdown(report):
    a = report["aggregate"]
    lines = ["# 固定 train4 感知置信度因果对齐诊断", "",
        f"四个已初始化训练来源共 **{a['decision_count']} 次决策**，在固定阈值 **0.90** 下，TP/TN/FP/FN 为 **{a['confusion_matrix']['TP']}/{a['confusion_matrix']['TN']}/{a['confusion_matrix']['FP']}/{a['confusion_matrix']['FN']}**。低置信停车共 {a['visual_uncertain_stop_count']} 次，其中 {a['visual_uncertain_stop_while_input_gt_visible']} 次输入 GT 仍有目标像素、{a['visual_uncertain_stop_while_input_gt_invisible']} 次输入 GT 无目标像素。结果用于指导感知标签补齐，不触发阈值调整或新的路点训练。", "",
        "decision j 使用 observation j−1：j=1 仅用 reset 的初始化目标框确认可见；j>1 用前一条 record 的 post-action GT。每例终末 observation N 没有下一次 decision，不能再配一条标签。因此样本量为 85+104+84+93=366，其中包括 4 个 reset 标签，排除 4 个终末观测。未引入冷启动或验证集输出。", "",
        "TP 表示预测置信度≥0.90且输入 GT 可见；TN 为低置信且 GT 不可见；FP 为高置信但 GT 不可见；FN 为低置信但 GT 可见。GT 可见仅表示指定目标 panoptic mask 至少有 1 个像素。", "",
        "| 来源 | 决策数 | TP | TN | FP | FN | 低置信停车：GT可见 / 不可见 | 首次低置信 decision(obs) | 首次 GT 丢失 obs → 可感知 decision | 提前决策数 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    for source in report["sources"]:
        m = source["confusion_matrix"]
        low, loss = source["first_low_confidence_decision"], source["first_gt_loss"]
        lines.append("| " + " | ".join([source["source"], str(source["decision_count"]), *[str(m[k]) for k in ("TP", "TN", "FP", "FN")],
            f"{source['visual_uncertain_stop_while_input_gt_visible']} / {source['visual_uncertain_stop_while_input_gt_invisible']}",
            f"{low['decision_step']}({low['input_observation_index']})" if low else "无",
            f"{loss['post_action_observation_index']} → {loss['first_decision_that_can_observe_change']}" if loss else "未丢失",
            str(source["first_low_confidence_lead_decisions_before_first_gt_loss_observed"]) if loss else "不适用"]) + " |")
    lines.extend(["", "“提前决策数”比较的是首次低置信 decision 与首次可以看到 GT 丢失的 decision；不是直接减同一行 post-action GT 的 step。下表 bbox 全是模型预测，未将其冒充真实目标框；动作是归一化控制值。", ""])
    for source in report["sources"]:
        lines.extend(["## " + source["source"], "",
            f"输入结果 SHA-256：`{source['result_sha256']}`。", "",
            "首次失去置信前后 3 个 decision：", "", *window_table(source["first_low_confidence_window"]), ""])
        if source["gt_transition_windows"]:
            for window in source["gt_transition_windows"]:
                lines.extend([f"GT 变化：{window['event']['kind']}，observation {window['event']['post_action_observation_index']}；中心为首次可观察该变化的 decision {window['center_decision_step']}，前后各 3 个 decision：", "", *window_table(window["rows"]), ""])
        else:
            lines.extend(["本例所有决策输入和终末 GT 都有目标像素，没有 GT 可见性变化；首次低置信不能称为真实 GT 丢失。", ""])
    lines.extend(["## 标签补齐的含义与限制", "",
        "优先补齐与策略输入同一因果时刻的目标 mask 面积、可见比例或代理量、真实 bbox 和 bbox 有效性，并将 anypixel 可见性与身份可信度分开。可见性变化附近需核对出视野、几何遮挡和目标尺寸变化；现有二值 GT 无法区分这些原因。随后补充已初始化目标重新出现及同身份确认的训练来源，因为当前四个来源的 GT 再可见事件为 0。", "",
        "模型在目标仍有像素时低置信，说明存在值得核查的感知标签与任务语义差距；不能据此认定阈值过高、bbox 一定错误或继续运动安全。固定 0.90，不对这些来源搜索阈值。这里的 GT 也不能证明模型锁定了正确身份，FP=0 不代表身份误认率为零。", "",
        "四个结果文件的 SHA-256 与已冻结长源完成记录一致；仅读取固定训练来源。没有 GPU、模型或模拟器调用，没有改动冻结源文件，也没有新增路点训练或把 val 输出变成训练数据。JSON 保存全部 366 条因果对齐记录和完整精度窗口。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = diagnostic(args.repository_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {"train4_visibility_diagnostic.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                 "train4_visibility_diagnostic.md": markdown(report)}
    for name, text in artifacts.items():
        path = args.output_dir / name
        payload = text.encode("utf-8")
        if path.exists():
            assert path.read_bytes() == payload, "Refuse overwrite of a differing report: " + str(path)
        else:
            path.write_bytes(payload)
    print(json.dumps({"aggregate": report["aggregate"], "sources": [{"source": s["source"], "matrix": s["confusion_matrix"], "first_low_decision": s["first_low_confidence_decision"]["decision_step"] if s["first_low_confidence_decision"] else None, "first_loss": s["first_gt_loss"], "lead_decisions": s["first_low_confidence_lead_decisions_before_first_gt_loss_observed"]} for s in report["sources"]],
                      "artifacts": {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in artifacts.items()}}))


if __name__ == "__main__":
    main()
