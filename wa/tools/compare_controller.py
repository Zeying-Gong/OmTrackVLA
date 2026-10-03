"""Strict fixed-24 learned-yaw comparison; offline diagnostics, never causal proof."""
import argparse
import hashlib
import json
import math
from pathlib import Path

JOBS = Path('/data/nas_ray/project/md-ak/users/zeying.gong')
BASELINE = JOBS / 'job_60509/task_71388/wa_recovery_mix_eval_v1'
LIGHTNAV = JOBS / 'job_59846/task_70708/wa_lightnav_review_v1/lightnav'
TASKS = ('stt', 'dt', 'at')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(value):
    require(isinstance(value, list) and len(value) == 3, 'expected action[3]')
    require(all(isinstance(x, (int, float)) and math.isfinite(x) and abs(x) <= 1.000000001 for x in value),
            'nonfinite or unclipped action')
    return [float(x) for x in value]


def yaw_stats(actions):
    y = [a[2] for a in actions]
    n = len(y)
    active = [x for x in y if abs(x) > .02]
    flips = sum(a*b < 0 for a, b in zip(active, active[1:]))
    return dict(steps=n, abs_mean=sum(map(abs, y))/n if n else None,
                total_variation_per_transition=sum(abs(b-a) for a, b in zip(y, y[1:]))/(n-1) if n > 1 else None,
                sign_flips_active_sequence=flips,
                sign_flip_definition='remove abs(yaw)<=0.02 then count sign changes; deadband may span steps',
                sign_flips_per_100_steps=100*flips/n if n else None,
                adjacent_sign_flips=sum(abs(a) > .02 and abs(b) > .02 and a*b < 0 for a, b in zip(y, y[1:])),
                saturated_steps=sum(abs(x) >= .99 for x in y),
                saturation_fraction=sum(abs(x) >= .99 for x in y)/n if n else None)


def visibility_stats(rows):
    covered = lower = upper = either = 0
    for row in rows:
        info = row.get('target_visibility')
        if info is None:
            continue
        count, shape, fraction = info['target_pixels'], info['image_shape'], info['target_pixel_fraction']
        require(isinstance(shape, list) and len(shape) == 2 and all(isinstance(x, int) and x > 0 for x in shape), 'invalid semantic image_shape')
        pixels = shape[0]*shape[1]
        require(isinstance(count, int) and 0 <= count <= pixels, 'invalid target pixel count')
        require(math.isfinite(fraction) and abs(fraction-count/pixels) < 1e-10, 'pixel fraction mismatch')
        require(info.get('usage') == 'observer_only_not_policy_input', 'unmarked privileged visibility')
        require(info.get('timing') == 'pre_action_not_final_post_action', 'visibility timing changed')
        covered += 1
        lo, hi = count <= 3000, fraction >= .3
        lower += lo
        upper += hi
        either += lo or hi
    n = len(rows)
    return dict(total_pre_action_steps=n, covered_steps=covered, missing_steps=n-covered,
                coverage_fraction=covered/n if n else None,
                lower_pixels_le_3000_steps=lower, upper_fraction_ge_0_3_steps=upper, either_bound_steps=either,
                lower_fraction_of_covered=lower/covered if covered else None,
                upper_fraction_of_covered=upper/covered if covered else None,
                either_fraction_of_covered=either/covered if covered else None,
                interpretation='benchmark-specific observer diagnostics; missing is not invisible; no causal, edge-location, or post-terminal claim')


def candidate_invariants(trace, review, initialization_valid):
    require(len(trace) == len(review), 'candidate trace/review count mismatch')
    checked = failed = changed = 0
    raw_actions, legacy_actions = [], []
    for event, row in zip(trace, review):
        require(event['step'] == row['step']+1, 'candidate trace/review step mismatch')
        final = vector(event['action'])
        require(final == vector(row['action']), 'candidate trace/video action mismatch')
        if event.get('failure'):
            require(not initialization_valid and final == [0., 0., 0.], 'unexpected policy failure')
            failed += 1
            continue
        d = event['diagnostics']
        info = d['control_info']
        require(d['controller'] == 'learned_yaw_guard_v1', 'wrong candidate controller')
        require(d['normalized_action'] == final, 'candidate diagnostics/final mismatch')
        raw, legacy = vector(info['raw_pose_action']), vector(info['legacy_guard_action'])
        require(info.get('learned_yaw_retained') is True and info.get('ablation_only') is True, 'missing ablation flags')
        require(final[:2] == legacy[:2], 'legacy translation invariant violated')
        require(final[2] == raw[2], 'learned yaw invariant violated')
        raw_actions.append(raw)
        legacy_actions.append(legacy)
        checked += 1
        changed += final[2] != legacy[2]
    require(checked > 0 or not initialization_valid, 'no candidate controller steps checked')
    return dict(checked_steps=checked, failure_zero_action_steps=failed, changed_yaw_steps=changed,
                translation_violations=0, learned_yaw_violations=0,
                raw_yaw=yaw_stats(raw_actions), legacy_guard_yaw=yaw_stats(legacy_actions))


def load_method(root, candidate=False):
    root, rows, inputs = root.resolve(), {}, []
    for path in sorted(root.glob('*/episodes.jsonl')):
        inputs.append(dict(path=str(path), sha256=sha(path)))
        for result in read_jsonl(path):
            task, key = result['task'], result['key']
            identity = (task, key)
            require(identity not in rows, f'duplicate episode {identity}')
            require(task in TASKS and isinstance(key, str) and len(key.split('/')) == 2
                    and '..' not in key and not key.startswith('/'), 'invalid task/key')
            require(result['success'] in (0, 1) and result['collision'] in (0, 1), 'nonbinary result')
            require(isinstance(result['policy_init_valid'], bool), 'missing initialization validity')
            digest = result['initial_rgb_sha256']
            require(isinstance(digest, str) and len(digest) == 64 and all(x in '0123456789abcdef' for x in digest), 'invalid image hash')
            rows[identity] = (result, path.parent)
    require(len(rows) == 24, f'INCOMPLETE: {root} has {len(rows)}/24 completed unique episodes')
    require(all(sum(k[0] == task for k in rows) == 8 for task in TASKS), 'expected 8 per task')
    normalized, all_review = {}, []
    for identity, (result, lane) in sorted(rows.items()):
        task, key = identity
        review_path = lane/task/'_review'/key/'steps.jsonl'
        review = read_jsonl(review_path)
        require(len(review) == int(result['total_step']), f'review/total_step mismatch {identity}')
        require([r['step'] for r in review] == list(range(len(review))), 'noncontiguous review')
        times = [r['timestamp_s'] for r in review]
        require(all(math.isfinite(t) for t in times) and all(b > a for a, b in zip(times, times[1:])), 'nonmonotonic times')
        out = {k: result[k] for k in ('status', 'success', 'collision', 'policy_init_valid', 'following_rate', 'total_step', 'initial_rgb_sha256')}
        out.update(yaw=yaw_stats([vector(r['action']) for r in review]), visibility=visibility_stats(review),
                   steps_path=str(review_path), steps_sha256=sha(review_path), video_path=str(review_path.with_name('review.mp4')))
        if candidate:
            trace_path = lane/task/(key.replace('/', '_')+'.trace.jsonl')
            out['controller_invariants'] = candidate_invariants(read_jsonl(trace_path), review, result['policy_init_valid'])
            out['trace_path'], out['trace_sha256'] = str(trace_path), sha(trace_path)
        normalized[identity] = out
        all_review.extend(review)
    return normalized, inputs, visibility_stats(all_review)


def rates(rows):
    n = len(rows)
    success, collision = sum(bool(r['success']) for r in rows), sum(bool(r['collision']) for r in rows)
    invalid = sum(not r['policy_init_valid'] for r in rows)
    return dict(episodes=n, successful=success, success_rate=success/n if n else None,
                collisions=collision, collision_rate=collision/n if n else None,
                invalid_initializations=invalid, invalid_rate=invalid/n if n else None,
                denominator='all selected episodes; invalid starts retained')


def compare(candidate, baseline=BASELINE, lightnav=LIGHTNAV):
    methods, provenance, coverage = {}, {}, {}
    for label, root in [('candidate', candidate), ('wa_baseline_60509', baseline), ('lightnav_59846', lightnav)]:
        methods[label], provenance[label], coverage[label] = load_method(Path(root), candidate=label == 'candidate')
    keys = set(methods['candidate'])
    require(all(set(rows) == keys for rows in methods.values()), 'three-method episode set mismatch')
    for key in sorted(keys):
        require(len({rows[key]['initial_rgb_sha256'] for rows in methods.values()}) == 1, f'initial RGB mismatch: {key}')
    summary = {}
    for label, rows in methods.items():
        summary[label] = dict(overall=rates(list(rows.values())),
                              per_task={task: rates([r for k, r in rows.items() if k[0] == task]) for task in TASKS},
                              visibility=coverage[label])
    paired = []
    for task, key in sorted(keys):
        states = {label: rows[(task, key)] for label, rows in methods.items()}
        old, new = bool(states['wa_baseline_60509']['success']), bool(states['candidate']['success'])
        paired.append(dict(task=task, key=key, initial_rgb_three_way_match=True,
                           versus_wa='gain' if new and not old else 'regression' if old and not new else 'unchanged', methods=states))
    inv = [r['controller_invariants'] for r in methods['candidate'].values()]
    return dict(schema_version=1, status='COMPLETE_FIXED24_COMPARISON',
                protocol=dict(initial_rgb_three_way_matches=24, all_unique_episode_sets_match=True,
                              same_physics='required by frozen job protocol; offline tool does not independently inspect simulator configuration',
                              modalities='WA: RGB+initialBBox+ideal simulated polar UWB, no text; LightNav: RGB+text; different modalities',
                              warning='fixed development set; no product/generalization claim; action statistics descriptive, not causal or real-robot jitter',
                              yaw_metric_units='normalized command, not physical rad/s'),
                inputs=provenance, summary=summary,
                paired_vs_wa={name: sum(r['versus_wa'] == label for r in paired) for name, label in
                              [('gains', 'gain'), ('regressions', 'regression'), ('unchanged', 'unchanged')]},
                candidate_invariants={field: sum(r[field] for r in inv) for field in
                    ('checked_steps', 'failure_zero_action_steps', 'changed_yaw_steps', 'translation_violations', 'learned_yaw_violations')},
                episodes=paired)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--baseline', type=Path, default=BASELINE)
    p.add_argument('--lightnav', type=Path, default=LIGHTNAV)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), 'output exists; refusing overwrite')
    report = compare(a.candidate, a.baseline, a.lightnav)
    with a.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(output=str(a.output), summary=report['summary'],
                          paired_vs_wa=report['paired_vs_wa'], invariants=report['candidate_invariants']), indent=2))


if __name__ == '__main__':
    main()
