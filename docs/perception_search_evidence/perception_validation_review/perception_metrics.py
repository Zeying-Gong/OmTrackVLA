"""CPU metrics for independently admitted permanent-val perception observations.

This pure module does not certify raw sidecars, run a policy, tune thresholds, or
admit labels to training. A caller must pin the independent raw-data admission
report and provide all fixed cases. Missing/invalid predictions remain failures.
"""
from __future__ import annotations

import math
import re


METRIC_SPEC = {
    'primary_visibility_threshold': 0.9,
    'secondary_visibility_threshold': 0.5,
    'probability_domain': 'finite_numeric_0_to_1; bool_is_invalid',
    'calibration_bins': [index / 10 for index in range(11)],
    'ece_definition': 'sum_valid_bin_weight_times_abs_mean_probability_minus_positive_frequency',
    'ece_scope': 'valid_probabilities_only; always_report_coverage_and_invalid_count',
    'log_loss_clip_epsilon': 1e-6,
    'invalid_probability_brier_penalty': 1.0,
    'invalid_probability_log_loss_penalty': '-log(epsilon)',
    'bbox_iou_thresholds': [0.5, 0.75],
    'bbox_population': 'all_GT_visible_and_bbox_label_valid_nonterminal_observations; do_not_filter_by_prediction_visibility',
    'bbox_convention': 'continuous_area_on_inclusive_pixel_extrema_divided_by_width_height; no_plus_one_after_normalization',
    'invalid_predicted_bbox_iou': 0.0,
    'predicted_bbox_clipping': False,
    'missing_prediction': 'invalid_probability_and_bbox; kept_in_original_denominators',
    'duplicate_or_extra_prediction_index': 'protocol_failure; never_select_first_or_best',
    'world_time_match_tolerance_s': 1e-8,
    'report_aggregation': 'each_case; pooled_micro; equal_weight_case_macro_with_defined_case_count',
    'visible_is_identity_accuracy': False,
    'terminal_observation_is_scored_prediction': False,
    'automatic_checkpoint_promotion': False,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def valid_probability(value):
    return finite(value) and 0 <= value <= 1


def valid_box(box):
    return isinstance(box, (list, tuple)) and len(box) == 4 and all(finite(v) for v in box) \
        and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1


def bbox_iou(predicted, target):
    require(valid_box(target), 'GT bbox must be independently valid')
    if not valid_box(predicted):
        return 0.0
    overlap = max(0., min(predicted[2], target[2]) - max(predicted[0], target[0])) \
        * max(0., min(predicted[3], target[3]) - max(predicted[1], target[1]))
    union = (predicted[2]-predicted[0])*(predicted[3]-predicted[1]) \
        + (target[2]-target[0])*(target[3]-target[1]) - overlap
    return overlap/union


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def validate_labels(labels, expected_count, target_semantic_id):
    require(type(expected_count) is int and expected_count > 0, 'positive fixed prediction count required')
    require(len(labels) == expected_count + 1, 'all reset..natural-terminal labels required')
    previous = -1.
    for step, row in enumerate(labels):
        require(type(row.get('environment_step')) is int and row['environment_step'] == step,
                'noncontiguous or reordered GT observations')
        require(type(row.get('terminal_observation')) is bool
                and row['terminal_observation'] == (step == expected_count), 'wrong natural terminal label')
        require(row.get('visibility_label_valid') is True, 'invalid GT cannot be silently dropped')
        require(finite(row.get('world_time_s')) and row['world_time_s'] >= 0
                and row['world_time_s'] > previous, 'invalid/nonmonotonic GT time')
        previous = row['world_time_s']
        require(isinstance(row.get('rgb_array_sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', row['rgb_array_sha256']), 'GT RGB identity missing')
        target = row['target']
        require(type(target.get('semantic_id_label_side_only')) is int
                and target['semantic_id_label_side_only'] == target_semantic_id, 'GT target identity changed')
        require(type(target.get('visible')) is bool and type(target.get('bbox_label_valid')) is bool,
                'GT masks must be actual booleans')
        area = target.get('mask_area_pixels')
        require(type(area) is int and area >= 0 and target['visible'] == (area > 0), 'GT visibility/area mismatch')
        if target['bbox_label_valid']:
            require(target['visible'] and valid_box(target.get('bbox_xyxy_norm')), 'invalid GT box marked valid')


def aligned_rows(labels, predictions, expected_count, target_semantic_id):
    validate_labels(labels, expected_count, target_semantic_id)
    by_index = {}
    for row in predictions:
        step = row.get('observation_environment_step')
        require(type(step) is int and 0 <= step < expected_count, 'prediction contains future/terminal/invalid index')
        require(step not in by_index, 'duplicate prediction index')
        by_index[step] = row
    aligned = []
    for step in range(expected_count):
        label, prediction = labels[step], by_index.get(step)
        error = None
        if prediction is None:
            error = 'missing_prediction'
        elif prediction.get('source_rgb_array_sha256') != label['rgb_array_sha256']:
            error = 'wrong_source_rgb'
        elif not finite(prediction.get('source_world_time_s')) or abs(prediction['source_world_time_s']-label['world_time_s']) > METRIC_SPEC['world_time_match_tolerance_s']:
            error = 'wrong_source_time'
        probability = None if error else prediction.get('visibility_probability')
        predicted_bbox = None if error else prediction.get('predicted_bbox_xyxy_norm')
        aligned.append({'gt': label['target'], 'probability': probability,
                        'predicted_bbox': predicted_bbox, 'input_binding_error': error})
    return aligned


def confusion(rows, threshold):
    counts = {'tp':0, 'tn':0, 'fp':0, 'fn':0, 'invalid_gt_positive':0, 'invalid_gt_negative':0}
    for row in rows:
        target = row['gt']['visible']
        p = row['probability']
        if not valid_probability(p):
            counts['invalid_gt_positive' if target else 'invalid_gt_negative'] += 1
        elif p >= threshold:
            counts['tp' if target else 'fp'] += 1
        else:
            counts['fn' if target else 'tn'] += 1
    positives = counts['tp'] + counts['fn'] + counts['invalid_gt_positive']
    negatives = counts['tn'] + counts['fp'] + counts['invalid_gt_negative']
    return dict(counts, threshold=threshold, total=len(rows), gt_positive=positives, gt_negative=negatives,
        accuracy_including_invalid_failures=ratio(counts['tp']+counts['tn'], len(rows)),
        recall_including_invalid_failures=ratio(counts['tp'], positives),
        specificity_including_invalid_failures=ratio(counts['tn'], negatives),
        precision_among_positive_outputs=ratio(counts['tp'], counts['tp']+counts['fp']))


def score_rows(rows):
    bins = [dict(lower=i/10, upper=(i+1)/10, count=0, positive_count=0, probability_sum=0.) for i in range(10)]
    brier = log_loss = iou_sum = 0.
    invalid_probability = eligible_bbox = invalid_bbox = gt_degenerate_visible = joint_success = 0
    hit05 = hit075 = 0
    failures = {}
    epsilon = METRIC_SPEC['log_loss_clip_epsilon']
    for row in rows:
        target, probability = row['gt'], row['probability']
        if row['input_binding_error']:
            failures[row['input_binding_error']] = failures.get(row['input_binding_error'], 0)+1
        if valid_probability(probability):
            y = int(target['visible'])
            brier += (probability-y)**2
            clipped = max(epsilon, min(1-epsilon, probability))
            log_loss += -(y*math.log(clipped)+(1-y)*math.log(1-clipped))
            bin_row = bins[min(9, int(probability*10))]
            bin_row['count'] += 1
            bin_row['positive_count'] += y
            bin_row['probability_sum'] += probability
        else:
            invalid_probability += 1
            brier += 1.
            log_loss += -math.log(epsilon)
        if target['visible'] and not target['bbox_label_valid']:
            gt_degenerate_visible += 1
        if target['visible'] and target['bbox_label_valid']:
            eligible_bbox += 1
            box = row['predicted_bbox']
            invalid_bbox += int(not valid_box(box))
            iou = bbox_iou(box, target['bbox_xyxy_norm'])
            iou_sum += iou
            hit05 += int(iou >= .5)
            hit075 += int(iou >= .75)
            joint_success += int(iou >= .5 and valid_probability(probability) and probability >= .9)
    valid_count = len(rows)-invalid_probability
    calibration_error_sum = 0.
    for row in bins:
        row['mean_probability'] = ratio(row['probability_sum'], row['count'])
        row['observed_positive_rate'] = ratio(row['positive_count'], row['count'])
        if row['count']:
            calibration_error_sum += row['count']*abs(row['mean_probability']-row['observed_positive_rate'])
    return {
        'prediction_denominator':len(rows), 'visibility_at_0_9':confusion(rows,.9), 'visibility_at_0_5':confusion(rows,.5),
        'calibration':{'valid_probability_count':valid_count, 'invalid_probability_count':invalid_probability,
            'valid_probability_coverage':ratio(valid_count,len(rows)),
            'brier_with_invalid_penalty':ratio(brier,len(rows)),
            'log_loss_with_invalid_penalty':ratio(log_loss,len(rows)),
            'ece_valid_probability_only':ratio(calibration_error_sum,valid_count),
            'bins':bins},
        'bbox':{'visible_valid_gt_denominator':eligible_bbox, 'visible_degenerate_gt_count':gt_degenerate_visible,
            'invalid_prediction_count':invalid_bbox, 'mean_iou_including_invalid_failures':ratio(iou_sum,eligible_bbox),
            'iou_at_least_0_5_fraction':ratio(hit05,eligible_bbox), 'iou_at_least_0_75_fraction':ratio(hit075,eligible_bbox),
            'joint_visibility_0_9_and_iou_0_5_fraction':ratio(joint_success,eligible_bbox),
            'prediction_visibility_filter_applied':False},
        'input_binding_failures':failures,
        'identity_accuracy_claimed':False,
    }


def score_case(labels, predictions, expected_count, target_semantic_id):
    return score_rows(aligned_rows(labels, predictions, expected_count, target_semantic_id))


def score_suite(cases):
    """Exactly four already-bound cases; missing predictions may be an empty list."""
    require(len(cases) == 4 and len({c['case_id'] for c in cases}) == 4, 'four distinct fixed cases required')
    expected = [('dt_1300_episode3',90), ('at_1700_episode0',46), ('dt_1700_episode0',45), ('stt_1700_episode0',44)]
    require([(c['case_id'], c['expected_count']) for c in cases] == expected, 'fixed val4 identities/counts changed')
    pooled, output = [], []
    for case in cases:
        rows = aligned_rows(case['labels'], case['predictions'], case['expected_count'], case['target_semantic_id'])
        pooled.extend(rows)
        output.append({'case_id':case['case_id'], **score_rows(rows)})
    metrics = {
        'accuracy':lambda r:r['visibility_at_0_9']['accuracy_including_invalid_failures'],
        'recall':lambda r:r['visibility_at_0_9']['recall_including_invalid_failures'],
        'brier':lambda r:r['calibration']['brier_with_invalid_penalty'],
        'bbox_mean_iou':lambda r:r['bbox']['mean_iou_including_invalid_failures'],
    }
    macro = {}
    for name, getter in metrics.items():
        values = [getter(row) for row in output if getter(row) is not None]
        macro[name] = {'value':ratio(sum(values),len(values)), 'defined_case_count':len(values)}
    return {'case_denominator':4, 'prediction_denominator':225, 'label_observation_denominator':229,
            'cases':output, 'pooled_micro':score_rows(pooled), 'equal_weight_case_macro':macro,
            'optimizer_input_allowed':False, 'product_acceptance_evidence':False,
            'automatic_promotion':False, 'identity_accuracy_claimed':False}
