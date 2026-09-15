"""Offline measurement only: normalized saved yaw versus observed camera/world time."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT/'yaw_source_snapshot'
BATCH = SNAPSHOT/'outputs/takeover/perception_label_sidecar_train4_v1'
REMOTE = '/data/nfs/share/wam_tracking/OmTrackVLA/'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def heading(raw):
    # _matrix4_list serializes Magnum's (column,row) tuple access as nested
    # outer rows, so its wire matrix is transposed relative to NumPy's normal
    # column-vector transform. This conversion applies only to this wire field.
    world_from_camera = np.asarray(raw, dtype=np.float64).T
    if not np.isfinite(world_from_camera).all():
        raise ValueError('nonfinite camera transform')
    if not np.allclose(world_from_camera[3], [0., 0., 0., 1.], atol=1e-7, rtol=0):
        raise ValueError('wrong wire matrix convention')
    rotation = world_from_camera[:3, :3]
    if not np.allclose(rotation.T@rotation, np.eye(3), atol=2e-5, rtol=0) or abs(np.linalg.det(rotation)-1) > 2e-5:
        raise ValueError('invalid camera rotation')
    forward = -rotation[:, 2]  # Magnum/OpenGL camera looks along local -Z.
    horizontal_norm = float(np.hypot(forward[0], forward[2]))
    if horizontal_norm < 1e-6:
        raise ValueError('camera heading cannot be defined near vertical')
    # Positive delta follows Habitat's +Y axis, matching this base action.
    return float(np.arctan2(forward[0], forward[2]))


def main():
    plan = json.loads((BATCH/'frozen_plan.json').read_text())
    status = json.loads((BATCH/'batch_status.json').read_text())
    if status['successful_source_count'] != 4:
        raise ValueError('the complete four-source batch is required')
    registry = {ref['path']: ref['sha256'] for ref in plan['artifacts']}
    for relative in ('habitat-lab/habitat/tasks/rearrange/actions/actions.py',
                     'habitat-lab/habitat/tasks/rearrange/rearrange_sim.py',
                     'habitat-lab/habitat/core/embodied_task.py',
                     'habitat-lab/habitat/core/track_env.py',
                     'habitat-lab/habitat/tasks/track/track_task.py'):
        if sha(SNAPSHOT/relative) != registry[REMOTE+relative]:
            raise ValueError('source differs from frozen collector: ' + relative)
    results = []
    for entry, execution in zip(plan['entries'], status['entries']):
        if entry['run_id'] != execution['run_id']:
            raise ValueError('source order mismatch')
        path = BATCH/entry['run_id']/'labels.jsonl'
        if sha(path) != execution['worker_result']['labels_sha256']:
            raise ValueError('label artifact differs from completed worker')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != entry['observation_count'] or any(r['environment_step'] != i for i, r in enumerate(rows)):
            raise ValueError('noncontiguous sidecar sequence')
        if not all(r['source_audit']['passed'] for r in rows):
            raise ValueError('failed source observation audit')
        times = np.array([row['world_time_s'] for row in rows], dtype=np.float64)
        dt = np.diff(times)
        if not np.isfinite(dt).all() or not (dt > 0).all():
            raise ValueError('invalid world clock')
        theta = np.array([heading(row['source_audit']['camera_after_render']) for row in rows])
        dtheta = np.arctan2(np.sin(np.diff(theta)), np.cos(np.diff(theta)))
        actions = np.array([row['source_audit']['action_just_replayed'] for row in rows[1:]], dtype=np.float64)
        if not np.isfinite(actions).all() or not (np.abs(actions) <= 1).all():
            raise ValueError('invalid normalized actions')
        action_config = entry['resolved_config']['habitat']['task']['actions']['agent_1_base_velocity']
        simulator_config = entry['resolved_config']['habitat']['simulator']
        expected = float(action_config['ang_speed']/simulator_config['ctrl_freq'])
        u = actions[:, 2]
        selected = np.abs(u) >= 1e-4
        denominator = float(u[selected]@u[selected])
        if denominator == 0:
            raise ValueError('no yaw excitation')
        step_gain = float(u[selected]@dtheta[selected]/denominator)
        world_gain = float(u[selected]@(dtheta[selected]/dt[selected])/denominator)
        results.append(dict(run_id=entry['run_id'], labels_sha256=sha(path), action_count=len(u),
            yaw_samples_above_1e_minus_4=int(selected.sum()),
            observed_world_dt_s_min=float(dt.min()), observed_world_dt_s_median=float(np.median(dt)),
            observed_world_dt_s_max=float(dt.max()),
            observed_world_dt_histogram={str(float(value)): int(count) for value, count in
                zip(*np.unique(np.round(dt, 9), return_counts=True))},
            expected_radians_per_action_per_normalized_yaw=expected,
            fitted_radians_per_action_per_normalized_yaw=step_gain,
            fitted_radians_per_world_second_per_normalized_yaw=world_gain,
            maximum_abs_error_against_configured_per_action_yaw_rad=float(np.max(np.abs(dtheta-expected*u))),
            normalized_yaw_min=float(u.min()), normalized_yaw_max=float(u.max()),
            command_vector_order=['forward','lateral','yaw'],
            pure_yaw_action_count=int(((np.abs(actions[:, :2]) < 1e-8).all(axis=1) & selected).sum()),
            audit_only=True, control_input_allowed=False))
    report = dict(status='offline_measurement_complete_not_a_controller',
        frozen_plan_file_sha256=sha(BATCH/'frozen_plan.json'),
        source_count=4, observation_count=sum(e['observation_count'] for e in plan['entries']),
        sources=results, no_gpu_environment_created=True, no_gt_used_for_control=True)
    (ROOT/'SIMULATOR_YAW_MEASUREMENT.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
