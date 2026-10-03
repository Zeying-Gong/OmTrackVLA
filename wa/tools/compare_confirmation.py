"""Audit two complete, predeclared confirmation sets without tuning or policy inputs."""
import argparse
import json
from pathlib import Path

try:
    from .compare_controller import (TASKS, load_method, rates, read_jsonl, require, sha)
except ImportError:
    from compare_controller import (TASKS, load_method, rates, read_jsonl, require, sha)

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
JOBS = Path('/data/nas_ray/project/md-ak/users/zeying.gong')
WA = JOBS/'job_60770/task_71649/wa_learned_yaw_confirmation_v1'
LIGHTNAV = JOBS/'job_60771/task_71650/wa_lightnav_confirmation_v1/lightnav'
PLAN = PROJECT/'source_yaw_confirmation_v1/wa/wm/mixed_diagnostic_plan_v1.json'
MANIFEST = PROJECT.parent/'WLA-EVT-20260925/evt_full_20260926/manifest.json'


def aggregate_yaw(rows, field='yaw'):
    metrics = [row[field] for row in rows]
    n = sum(m['steps'] for m in metrics)
    transitions = sum(max(m['steps']-1, 0) for m in metrics)
    flips = sum(m['sign_flips_active_sequence'] for m in metrics)
    saturated = sum(m['saturated_steps'] for m in metrics)
    valid = [m for m in metrics if m['steps']]
    return dict(episodes=len(metrics), steps=n, within_episode_transitions=transitions,
                step_weighted_abs_mean=sum(m['abs_mean']*m['steps'] for m in valid)/n if n else None,
                mean_episode_abs_mean=sum(m['abs_mean'] for m in valid)/len(valid) if valid else None,
                total_variation_per_transition=sum((m['total_variation_per_transition'] or 0)*max(m['steps']-1, 0) for m in metrics)/transitions if transitions else None,
                active_sequence_sign_flips=flips, sign_flips_per_100_steps=100*flips/n if n else None,
                adjacent_sign_flips=sum(m['adjacent_sign_flips'] for m in metrics),
                saturated_steps=saturated, saturation_fraction=saturated/n if n else None,
                definition='normalized command; abs(yaw)>0.02 active sequence, abs(yaw)>=0.99 saturated; no artificial transitions across episode boundaries')


def check_complete(root):
    markers = []
    fallback_count = 0
    files = sorted(root.glob('*/episodes.jsonl'))
    require(len(files) == 2, 'expected exactly two complete confirmation lanes')
    for path in files:
        rows = read_jsonl(path)
        complete = path.with_name('COMPLETE.json')
        value = json.loads(complete.read_text())
        require(value.get('episodes') == len(rows) == 12 and value.get('split') == 'confirmation', 'wrong or incomplete lane marker')
        markers.append(dict(path=str(complete), sha256=sha(complete), content=value))
        for row in rows:
            require(row.get('released_fallback_count', 0) >= 0, 'invalid fallback count')
            fallback_count += row.get('released_fallback_count', 0)
    return markers, fallback_count


def compare(wa_root, lightnav_root, plan_path=PLAN, manifest_path=MANIFEST):
    plan = json.loads(plan_path.read_text())
    require(sha(manifest_path) == plan['manifest_sha256'], 'predeclared manifest hash mismatch')
    expected = {(task, key) for task in TASKS for key in plan['confirmation'][task]}
    require(len(expected) == 24 and all(len(plan['confirmation'][task]) == 8 for task in TASKS), 'bad confirmation plan')
    confirmation_scenes = {key.split('/')[0] for _, key in expected}
    development_scenes = {key.split('/')[0] for task in TASKS for key in plan['development'][task]}
    require(len(confirmation_scenes) == 8 and not confirmation_scenes & development_scenes, 'confirmation/development scene leakage')
    methods, provenance, coverage, markers, fallbacks = {}, {}, {}, {}, {}
    for label, root in [('wa', Path(wa_root)), ('lightnav', Path(lightnav_root))]:
        markers[label], fallbacks[label] = check_complete(root)
        methods[label], provenance[label], coverage[label] = load_method(root, candidate=(label == 'wa'))
        require(set(methods[label]) == expected, f'{label} keys differ from predeclared confirmation24')
        for result in methods[label].values():
            video = Path(result['video_path'])
            complete = video.with_name('complete.json')
            require(video.is_file() and video.stat().st_size > 0, 'missing review video')
            require(json.loads(complete.read_text())['frames'] == result['total_step'], 'incomplete review video frame marker')
    require(all(methods['wa'][key]['initial_rgb_sha256'] == methods['lightnav'][key]['initial_rgb_sha256'] for key in expected), 'cross-method initial RGB mismatch')
    summaries = {}
    for label, rows in methods.items():
        summaries[label] = dict(overall=rates(list(rows.values())),
            per_task={task: rates([row for key, row in rows.items() if key[0] == task]) for task in TASKS},
            yaw=aggregate_yaw(list(rows.values())), visibility=coverage[label],
            released_fallback_count=fallbacks[label])
    paired = []
    for task, key in sorted(expected):
        w, l = methods['wa'][(task, key)], methods['lightnav'][(task, key)]
        outcome = 'wa_only_success' if w['success'] and not l['success'] else 'lightnav_only_success' if l['success'] and not w['success'] else 'both_success' if w['success'] else 'both_fail'
        paired.append(dict(task=task, key=key, initial_rgb_match=True, outcome=outcome, wa=w, lightnav=l))
    inv = [row['controller_invariants'] for row in methods['wa'].values()]
    control = {key: sum(i[key] for i in inv) for key in
               ('checked_steps', 'failure_zero_action_steps', 'changed_yaw_steps', 'translation_violations', 'learned_yaw_violations')}
    control['raw_pose_yaw'] = aggregate_yaw(inv, 'raw_yaw')
    control['legacy_yaw_same_candidate_states'] = aggregate_yaw(inv, 'legacy_guard_yaw')
    control['legacy_interpretation'] = 'counterfactual command recomputed on WA candidate states, not an independent old-controller rollout'
    pair_counts = {label: sum(row['outcome'] == label for row in paired) for label in
                   ('wa_only_success', 'lightnav_only_success', 'both_success', 'both_fail')}
    return dict(schema_version=1, status='COMPLETE_PREDECLARED_CONFIRMATION24_AUDIT',
        protocol=dict(split='confirmation', seed=plan['seed'], initial_rgb_matched=24,
            predeclared_plan_path=str(plan_path), predeclared_plan_sha256=sha(plan_path),
            manifest_path=str(manifest_path), manifest_sha256=sha(manifest_path),
            confirmation_scenes=sorted(confirmation_scenes), development_scene_overlap=[],
            modalities='WA RGB+initial BBox+ideal simulated polar UWB no text; LightNav RGB+text; different inputs',
            same_physics='same benchmark configuration required by frozen job source; initial RGB equality is not proof of identical future trajectories',
            scope='24 episodes across 8 scenes repeated in STT/DT/AT, not 24 independent scenes; one seed; no tuning after reading confirmation',
            collision_metric='reported benchmark collision flag, not independently instrumented static-wall or rear-doorframe contact rate',
            limits='small independent confirmation set, no statistical superiority/product SR90/real-UWB/edge-latency claim; preserved architecture and checkpoint'),
        source_files=provenance, lane_completion=markers, summary=summaries,
        paired_counts=pair_counts, controller_invariants=control, episodes=paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wa', type=Path, default=WA)
    parser.add_argument('--lightnav', type=Path, default=LIGHTNAV)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'refusing to overwrite existing result')
    report = compare(args.wa, args.lightnav, args.plan, args.manifest)
    payload = json.dumps(report, indent=2, allow_nan=False)+'\n'
    with args.output.open('x') as handle:
        handle.write(payload)
    print(json.dumps(dict(output=str(args.output), summary=report['summary'],
                          paired_counts=report['paired_counts'], controller_invariants=report['controller_invariants']), indent=2))


if __name__ == '__main__':
    main()
