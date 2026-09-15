"""Read-only localization audit of already-admitted train4 perception labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

VERIFICATION_SHA = 'aec0690b2d2a3727d7e3ba80afce07670876568e49ed2028e48d6545d70cfc5d'
THRESHOLD = 0.90
RUNS = ('at_0401_128steps', 'dt_0200_128steps', 'stt_0000_128steps', 'stt_3100_128steps')


def pinned(path, expected):
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, str(path)
    return data


def iou(predicted, target):
    """Continuous area of normalized inclusive-extrema coordinates; no +1 pixel."""
    assert len(predicted) == len(target) == 4
    assert all(type(v) in (int, float) and math.isfinite(v) for v in predicted + target)
    intersection = max(0., min(predicted[2], target[2])-max(predicted[0], target[0])) * max(
        0., min(predicted[3], target[3])-max(predicted[1], target[1]))
    pred_area = max(0., predicted[2]-predicted[0]) * max(0., predicted[3]-predicted[1])
    gt_area = (target[2]-target[0]) * (target[3]-target[1])
    assert gt_area > 0
    return intersection/(pred_area+gt_area-intersection), pred_area > 0


def summarize(rows):
    values = [r['iou'] for r in rows]
    return {'count': len(values), 'mean_iou': statistics.mean(values) if values else None,
        'median_iou': statistics.median(values) if values else None,
        'iou_ge_0_5_count': sum(v >= .5 for v in values),
        'iou_ge_0_5_fraction': sum(v >= .5 for v in values)/len(values) if values else None,
        'zero_overlap_count': sum(v == 0 for v in values),
        'zero_overlap_fraction': sum(v == 0 for v in values)/len(values) if values else None,
        'predicted_degenerate_box_count': sum(not r['predicted_bbox_positive_area'] for r in rows)}


def groups(rows):
    return {'all_eligible': summarize(rows),
        'visibility_ge_0_90': summarize([r for r in rows if r['predicted_visibility'] >= THRESHOLD]),
        'visibility_lt_0_90': summarize([r for r in rows if r['predicted_visibility'] < THRESHOLD])}


def diagnose(repository):
    verification_path = repository/'outputs/takeover/perception_sidecar_independent_admission_v1/verification.json'
    verified = json.loads(pinned(verification_path, VERIFICATION_SHA))
    assert verified['status'] == 'verified_development_perception_labels_pending_loader_integration'
    plan_path = repository/'outputs/takeover/perception_label_sidecar_train4_v1/frozen_plan.json'
    plan = json.loads(pinned(plan_path, verified['plan_file_sha256']))
    assert plan['plan_sha256'] == verified['plan_sha256']
    assert [e['run_id'] for e in plan['entries']] == list(RUNS)
    assert [e['run_id'] for e in verified['sources']] == list(RUNS)
    source_reports, all_rows = [], []
    for entry, approved in zip(plan['entries'], verified['sources']):
        assert entry['permanent_partition_role'] == 'train'
        source = json.loads(pinned(entry['source']['path'], entry['source']['sha256']))
        labels_path = Path(entry['output_dir'])/'labels.jsonl'
        labels = [json.loads(line) for line in pinned(labels_path, approved['labels_sha256']).decode().splitlines()]
        assert len(labels) == len(source['steps']) + 1 == entry['observation_count']
        rows, excluded_invisible, excluded_invalid = [], 0, 0
        for k, label in enumerate(labels):
            assert label['environment_step'] == label['policy_call_index'] == k
            assert label['terminal_observation'] is (k == len(source['steps']))
            if label['terminal_observation']:
                assert label['next_source_action_step'] is None
                continue
            assert source['steps'][k]['step'] == label['next_source_action_step'] == k+1
            target = label['target']
            assert target['visible'] is (True if k == 0 else source['steps'][k-1]['evaluation_only_after_action']['gt_visible'])
            if not target['visible']:
                excluded_invisible += 1
                continue
            if not target['bbox_label_valid']:
                excluded_invalid += 1
                continue
            policy = source['steps'][k]['policy']
            probability = policy['visibility_probability']
            assert math.isfinite(probability) and 0 <= probability <= 1
            value, valid = iou(policy['predicted_bbox_xyxy_norm'], target['bbox_xyxy_norm'])
            rows.append({'run_id': entry['run_id'], 'observation_index': k, 'source_action_step': k+1,
                'predicted_visibility': probability, 'mode': policy['mode'],
                'predicted_bbox_xyxy_norm': policy['predicted_bbox_xyxy_norm'],
                'gt_bbox_xyxy_norm': target['bbox_xyxy_norm'], 'gt_mask_area_pixels': target['mask_area_pixels'],
                'iou': value, 'predicted_bbox_positive_area': valid})
        assert len(rows) + excluded_invisible + excluded_invalid == len(source['steps'])
        all_rows.extend(rows)
        source_reports.append({'run_id': entry['run_id'], 'source': entry['source'],
            'labels': {'path': str(labels_path), 'sha256': approved['labels_sha256']},
            'nonterminal_decisions': len(source['steps']), 'excluded_terminal_observations': 1,
            'excluded_gt_invisible': excluded_invisible, 'excluded_visible_but_invalid_gt_bbox': excluded_invalid,
            'groups': groups(rows)})
    return {'status': 'fixed_train4_bbox_localization_readonly_diagnostic_complete',
        'verification_sha256': VERIFICATION_SHA, 'plan_file_sha256': verified['plan_file_sha256'],
        'plan_sha256': verified['plan_sha256'], 'threshold': THRESHOLD,
        'alignment': 'sidecar observation k corresponds to source.steps[k].policy, source action step k+1; four terminal observations have no policy and are excluded.',
        'iou_definition': 'Continuous normalized bbox area (x2-x1)*(y2-y1), with nonnegative predicted dimensions; GT coordinates are inclusive pixel extrema divided by W/H. No +1 pixel correction or clipping to improve predictions. Boundary-only contact counts as zero overlap.',
        'eligibility': 'GT target visible=true and bbox_label_valid=true only; do not drop low-confidence or degenerate predicted boxes.',
        'total_sidecar_observations': 370, 'nonterminal_decisions': 366, 'excluded_terminal_observations': 4,
        'excluded_gt_invisible': sum(s['excluded_gt_invisible'] for s in source_reports),
        'excluded_visible_but_invalid_gt_bbox': sum(s['excluded_visible_but_invalid_gt_bbox'] for s in source_reports),
        'groups': groups(all_rows), 'sources': source_reports, 'eligible_frames': all_rows,
        'highlighted_frames': [r for r in all_rows if (r['run_id'], r['observation_index']) in
                               [('at_0401_128steps', 25), ('stt_3100_128steps', 23)]],
        'limitations': ['GT anypixel visibility does not establish correct model identity, localization, motion safety, or cause of invisibility.',
            'This is localization on already-used train sources, not held-out model evaluation or threshold calibration.',
            '43 low-confidence/GT-anypixel-positive decisions must not be interpreted as 43 safe opportunities to relax the stop threshold.'],
        'threshold_changed': False, 'training_started': False, 'validation_sources_used': False,
        'formal_training_eligible': False, 'product_acceptance_evidence': False}


def markdown(report):
    lines = ['# 固定 train4 预测框定位诊断', '',
        '将已独立通过验收的 sidecar observation k 与原 source.steps[k].policy 对齐，排除 4 个没有后续 policy 的终末观测。固定视觉置信度阈值 0.90，只统计 GT 可见且 bbox_label_valid=true 的帧，低置信预测保留在统计中。', '',
        'IoU 使用归一化坐标的连续面积：(x2−x1)×(y2−y1)。GT 框由包含端点的像素极值除以图宽/高得到，此处不另加 1 像素，不裁剪预测框以改善结果；仅边界接触归入零重叠。', '',
        '| 来源 | 置信度组 | 帧数 | 平均 IoU | 中位 IoU | IoU≥0.5 数量/比例 | 零重叠数量 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name, grouped in [('总体', report['groups']), *[(s['run_id'], s['groups']) for s in report['sources']]]:
        for group, values in grouped.items():
            def number(value): return '无样本' if value is None else f'{value:.6f}'
            lines.append(f"| {name} | {group} | {values['count']} | {number(values['mean_iou'])} | {number(values['median_iou'])} | {values['iou_ge_0_5_count']} / {number(values['iou_ge_0_5_fraction'])} | {values['zero_overlap_count']} |")
    lines += ['', f"共 366 次非终末决策；排除 GT 不可见 {report['excluded_gt_invisible']} 帧、GT 可见但 bbox 无效 {report['excluded_visible_but_invalid_gt_bbox']} 帧。", '',
              '图中重点帧的数值核验：', '']
    for row in report['highlighted_frames']:
        lines.append(f"- {row['run_id']} obs {row['observation_index']}（action {row['source_action_step']}）：visibility={row['predicted_visibility']:.6f}，IoU={row['iou']:.6f}；预测框 {row['predicted_bbox_xyxy_norm']}，GT 框 {row['gt_bbox_xyxy_norm']}。")
    lines += ['', '目标仍有 GT 像素时，模型可能已经框错位置。低置信停车与定位错误同时存在时，不能把 43 个可见性 FN 当成应该放宽停车门的证据。这里既没有更改阈值，也没有训练、使用 val 或更改原始标签。', '',
              f"独立验收报告 SHA-256：`{VERIFICATION_SHA}`；同时重新校验冻结 plan、四个 labels JSONL 及四个原 source 的哈希。逐帧完整精度指标与绑定信息见配套 JSON。", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.repository)
    args.output_dir.mkdir(exist_ok=True)
    for name, data in [('bbox_localization_diagnostic.json', json.dumps(report, indent=2, ensure_ascii=False)+'\n'),
                       ('bbox_localization_diagnostic.md', markdown(report))]:
        with (args.output_dir/name).open('x', encoding='utf-8') as stream: stream.write(data)
    print(json.dumps({'groups': report['groups'], 'sources': [{'run_id':s['run_id'], 'groups':s['groups']} for s in report['sources']],
                      'highlighted_frames': report['highlighted_frames'],
                      'excluded_visible_but_invalid_gt_bbox': report['excluded_visible_but_invalid_gt_bbox']}))


if __name__ == '__main__':
    main()
